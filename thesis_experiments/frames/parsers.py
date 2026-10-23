"""Parse Metrica and Dataset A records into the Stage 4 frame schema."""

from __future__ import annotations

import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from thesis_experiments.events.converters import (
    convert_metrica_events,
    dataset_a_event_to_unified,
    metrica_event_to_unified,
)
from thesis_experiments.events.schema import UnifiedEvent
from thesis_experiments.frames.schema import UnifiedBall, UnifiedFrame, UnifiedPlayer


def _number(value: object) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _player_columns(path: Path) -> list[tuple[str, int, int]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        headers = [next(csv.reader(handle)) for _ in range(3)]
    output = []
    for column in range(3, len(headers[2]) - 2, 2):
        player_id = headers[2][column].strip().replace(" ", "")
        if player_id and player_id.lower() not in {"nan", "ball"}:
            output.append((player_id, column, column + 1))
    return output


def _tracking_rows(path: Path) -> Iterator[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for _ in range(3):
            next(reader)
        yield from reader


def _players(row: list[str], columns: list[tuple[str, int, int]]) -> list[UnifiedPlayer]:
    output = []
    for player_id, x_column, y_column in columns:
        x, y = _number(row[x_column]), _number(row[y_column])
        if x is not None and y is not None:
            output.append(UnifiedPlayer(player_id, x, y))
    return output


def parse_metrica_csv_frames(game_dir: Path) -> Iterator[UnifiedFrame]:
    """Stream a standard Metrica CSV game (Sample Games 1 or 2)."""
    name = game_dir.name
    home_path = game_dir / f"{name}_RawTrackingData_Home_Team.csv"
    away_path = game_dir / f"{name}_RawTrackingData_Away_Team.csv"
    home_columns, away_columns = _player_columns(home_path), _player_columns(away_path)
    events_by_frame: dict[int, list[UnifiedEvent]] = defaultdict(list)
    for event in convert_metrica_events(game_dir):
        if event.frame_id is not None:
            events_by_frame[event.frame_id].append(event)
    segment_id, previous_roster = -1, None
    for home, away in zip(_tracking_rows(home_path), _tracking_rows(away_path), strict=True):
        if home[1] != away[1]:
            raise ValueError(f"Metrica home/away frame mismatch: {home[1]} != {away[1]}.")
        left, right = _players(home, home_columns), _players(away, away_columns)
        roster = tuple(player.player_id for player in left + right)
        if roster != previous_roster:
            segment_id += 1
            previous_roster = roster
        ball_x, ball_y = _number(home[-2]), _number(home[-1])
        ball = UnifiedBall(ball_x, ball_y) if ball_x is not None and ball_y is not None else None
        frame_id = int(home[1])
        frame = UnifiedFrame(
            source="metrica",
            match_id=name,
            frame_id=frame_id,
            timestamp=home[2],
            left_team_id="Home",
            right_team_id="Away",
            left_players=left,
            right_players=right,
            ball=ball,
            events_at_frame=events_by_frame.get(frame_id, []),
            valid=len(left) == 11 and len(right) == 11,
            segment_id=segment_id,
            raw={"period": home[0]},
        )
        frame.validate()
        yield frame


@dataclass(frozen=True, slots=True)
class _EPTSFormat:
    start_frame: int
    end_frame: int
    players: tuple[tuple[str, str], ...]


def _xml_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _epts_metadata(path: Path) -> tuple[list[str], float, tuple[_EPTSFormat, ...]]:
    root = ET.parse(path).getroot()
    teams = list(dict.fromkeys(
        element.attrib["id"]
        for element in root.iter()
        if _xml_name(element) == "Team" and element.attrib.get("id")
    ))
    frame_rate = next(
        float(element.text)
        for element in root.iter()
        if _xml_name(element) == "FrameRate"
    )
    player_teams = {
        element.attrib["id"]: element.attrib["teamId"]
        for element in root.iter()
        if _xml_name(element) == "Player"
        and element.attrib.get("id")
        and element.attrib.get("teamId")
    }
    x_channel_players = {
        element.attrib["id"]: element.attrib["playerId"]
        for element in root.iter()
        if _xml_name(element) == "PlayerChannel"
        and element.attrib.get("channelId") == "x"
        and element.attrib.get("id")
        and element.attrib.get("playerId")
    }
    formats = []
    for element in root.iter():
        if _xml_name(element) != "DataFormatSpecification":
            continue
        channel_ids = []
        for child in element.iter():
            if _xml_name(child) != "PlayerChannelRef":
                continue
            channel_id = child.attrib.get("playerChannelId") or child.attrib.get("ref")
            if channel_id in x_channel_players:
                channel_ids.append(channel_id)
        if not channel_ids:
            continue
        players = tuple(
            (x_channel_players[channel_id], player_teams[x_channel_players[channel_id]])
            for channel_id in channel_ids
        )
        if len(players) != 22:
            raise ValueError(
                f"EPTS format {element.attrib.get('startFrame')}–{element.attrib.get('endFrame')} "
                f"defines {len(players)} x-coordinate player channels instead of 22."
            )
        formats.append(_EPTSFormat(
            start_frame=int(element.attrib["startFrame"]),
            end_frame=int(element.attrib["endFrame"]),
            players=players,
        ))
    if not formats:
        raise ValueError("EPTS metadata does not define any player data formats.")
    formats.sort(key=lambda item: item.start_frame)
    return teams, frame_rate, tuple(formats)


def _metrica_epts_events(path: Path, match_id: str) -> dict[int, list[UnifiedEvent]]:
    output: dict[int, list[UnifiedEvent]] = defaultdict(list)
    rows = json.loads(path.read_text(encoding="utf-8"))["data"]
    for row in rows:
        start, end = row.get("start") or {}, row.get("end") or {}
        subtypes = row.get("subtypes") or {}
        if isinstance(subtypes, list):
            subtype = ":".join(str(item.get("name")) for item in subtypes if isinstance(item, dict) and item.get("name")) or None
        else:
            subtype = subtypes.get("name")
        raw = {
            "Event ID": row.get("index"),
            "Type": (row.get("type") or {}).get("name"),
            "Subtype": subtype,
            "Period": row.get("period"),
            "Start Frame": start.get("frame"),
            "End Frame": end.get("frame"),
            "Start Time [s]": start.get("time"),
            "Team": (row.get("team") or {}).get("id"),
            "From": (row.get("from") or {}).get("id"),
            "To": (row.get("to") or {}).get("id"),
            "Start X": start.get("x"),
            "Start Y": start.get("y"),
            "End X": end.get("x"),
            "End Y": end.get("y"),
        }
        event = metrica_event_to_unified(raw, match_id=match_id, fallback_id=row.get("index", 0))
        if event.frame_id is not None:
            output[event.frame_id].append(event)
    return output


def parse_metrica_epts_frames(game_dir: Path) -> Iterator[UnifiedFrame]:
    """Stream Metrica Sample Game 3's FIFA EPTS representation."""
    metadata_path = game_dir / "Sample_Game_3_metadata.xml"
    tracking_path = game_dir / "Sample_Game_3_tracking.txt"
    event_path = game_dir / "Sample_Game_3_events.json"
    teams, frame_rate, formats = _epts_metadata(metadata_path)
    if len(teams) != 2:
        raise ValueError("Metrica EPTS metadata must define two teams.")
    events_by_frame = _metrica_epts_events(event_path, game_dir.name)
    format_index = 0
    with tracking_path.open(encoding="utf-8") as handle:
        for line in handle:
            frame_text, player_text, ball_text = line.rstrip("\n").split(":")
            frame_id = int(frame_text)
            while format_index + 1 < len(formats) and frame_id > formats[format_index].end_frame:
                format_index += 1
            current_format = formats[format_index]
            if not current_format.start_frame <= frame_id <= current_format.end_frame:
                raise ValueError(f"EPTS frame {frame_id} is not covered by a metadata data format.")
            positions = [position.split(",") for position in player_text.split(";")]
            players_by_team: dict[str, list[UnifiedPlayer]] = {team: [] for team in teams}
            for (player_id, team_id), position in zip(current_format.players, positions, strict=True):
                x, y = _number(position[0]), _number(position[1])
                if x is not None and y is not None:
                    players_by_team[team_id].append(UnifiedPlayer(player_id, x, y))
            ball_values = ball_text.split(",")
            ball_x, ball_y = _number(ball_values[0]), _number(ball_values[1])
            ball = UnifiedBall(ball_x, ball_y) if ball_x is not None and ball_y is not None else None
            left, right = players_by_team[teams[0]], players_by_team[teams[1]]
            frame = UnifiedFrame(
                source="metrica",
                match_id=game_dir.name,
                frame_id=frame_id,
                timestamp=f"{frame_id / frame_rate:.3f}",
                left_team_id=teams[0],
                right_team_id=teams[1],
                left_players=left,
                right_players=right,
                ball=ball,
                events_at_frame=events_by_frame.get(frame_id, []),
                valid=len(left) == 11 and len(right) == 11,
                segment_id=format_index,
            )
            frame.validate()
            yield frame


def parse_metrica_frames(game_dir: Path) -> Iterator[UnifiedFrame]:
    if (game_dir / f"{game_dir.name}_RawEventsData.csv").exists():
        yield from parse_metrica_csv_frames(game_dir)
    else:
        yield from parse_metrica_epts_frames(game_dir)


def _dataset_a_sides(record: dict, teams: list[str]) -> tuple[str, str]:
    for event in record.get("context", {}).get("events") or []:
        details = event.get("details") or {}
        if details.get("TeamLeft") in teams and details.get("TeamRight") in teams:
            return str(details["TeamLeft"]), str(details["TeamRight"])
    medians = {
        team: sorted(float(player["x"]) for player in record["state"]["players"] if str(player["team_id"]) == team)
        for team in teams
    }
    midpoint = {team: values[len(values) // 2] for team, values in medians.items() if values}
    return min(teams, key=midpoint.get), max(teams, key=midpoint.get)


def parse_dataset_a_frames(path: Path) -> Iterator[UnifiedFrame]:
    """Stream Dataset A's provider frame JSONL produced by the raw XML helpers."""
    sides: tuple[str, str] | None = None
    segment_id, previous_roster = -1, None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            teams = sorted({str(player["team_id"]) for player in record["state"]["players"]})
            if len(teams) != 2:
                raise ValueError(f"Dataset A frame {record['frame']} does not contain two teams.")
            if sides is None:
                sides = _dataset_a_sides(record, teams)
            left_team, right_team = sides
            by_team: dict[str, list[UnifiedPlayer]] = {left_team: [], right_team: []}
            for player in record["state"]["players"]:
                x, y = _number(player.get("x")), _number(player.get("y"))
                team = str(player["team_id"])
                if team in by_team and x is not None and y is not None:
                    by_team[team].append(UnifiedPlayer(str(player["person_id"]), x, y))
            for players in by_team.values():
                players.sort(key=lambda player: player.player_id)
            left, right = by_team[left_team], by_team[right_team]
            roster = tuple(player.player_id for player in left + right)
            if roster != previous_roster:
                segment_id += 1
                previous_roster = roster
            raw_ball = record["state"].get("ball") or {}
            ball_x, ball_y = _number(raw_ball.get("x")), _number(raw_ball.get("y"))
            ball = UnifiedBall(ball_x, ball_y) if ball_x is not None and ball_y is not None else None
            frame_id = int(record["frame"])
            events = [
                dataset_a_event_to_unified(event, match_id=str(record["match_id"]), fallback_frame=frame_id)
                for event in record.get("context", {}).get("events") or []
                if event.get("event_type") != "Delete"
            ]
            frame = UnifiedFrame(
                source="dataset_a",
                match_id=str(record["match_id"]),
                frame_id=frame_id,
                timestamp=str(record["timestamp"]),
                left_team_id=left_team,
                right_team_id=right_team,
                left_players=left,
                right_players=right,
                ball=ball,
                events_at_frame=events,
                valid=len(left) == 11 and len(right) == 11,
                segment_id=segment_id,
                raw={"game_section": record.get("context", {}).get("game_section")},
            )
            frame.validate()
            yield frame
