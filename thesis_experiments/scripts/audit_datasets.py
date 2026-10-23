"""Stream full Dataset A and Metrica sources into a Stage 3 audit report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class Range2D:
    x_min: float = math.inf
    x_max: float = -math.inf
    y_min: float = math.inf
    y_max: float = -math.inf

    def add(self, x: object, y: object) -> bool:
        try:
            x_value, y_value = float(x), float(y)
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(x_value) and math.isfinite(y_value)):
            return False
        self.x_min = min(self.x_min, x_value)
        self.x_max = max(self.x_max, x_value)
        self.y_min = min(self.y_min, y_value)
        self.y_max = max(self.y_max, y_value)
        return True

    def as_dict(self) -> dict[str, list[float] | None]:
        if self.x_min == math.inf:
            return {"x": None, "y": None}
        return {
            "x": [round(self.x_min, 5), round(self.x_max, 5)],
            "y": [round(self.y_min, 5), round(self.y_max, 5)],
        }


@dataclass
class MatchAudit:
    match_id: str
    frames: int
    valid_22_player_frames: int
    missing_player_frames: int
    missing_ball_frames: int
    events: int
    player_range: Range2D
    ball_range: Range2D
    team_ids: list[str] = field(default_factory=list)
    player_ids: int = 0
    sampling_hz: float = 25.0
    notes: list[str] = field(default_factory=list)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def finite_pair(x: str, y: str) -> bool:
    try:
        return math.isfinite(float(x)) and math.isfinite(float(y))
    except (TypeError, ValueError):
        return False


def audit_dataset_a(zip_path: Path) -> list[MatchAudit]:
    results: list[MatchAudit] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        metadata_files = sorted(name for name in members if "matchinformation" in name)
        for metadata_name in metadata_files:
            match_id = Path(metadata_name).stem.rsplit("_", 1)[-1]
            event_name = next(name for name in members if "events_raw" in name and match_id in name)
            position_name = next(name for name in members if "positions_raw_observed" in name and match_id in name)
            with archive.open(metadata_name) as handle:
                metadata = ET.parse(handle).getroot()
            teams = {
                element.attrib["TeamId"]
                for element in metadata.iter()
                if local_name(element.tag) == "Team" and element.attrib.get("TeamId")
            }
            players = {
                element.attrib["PersonId"]
                for element in metadata.iter()
                if local_name(element.tag) == "Player" and element.attrib.get("PersonId")
            }
            event_count = 0
            with archive.open(event_name) as handle:
                for _event, element in ET.iterparse(handle, events=("end",)):
                    if local_name(element.tag) == "Event":
                        event_count += 1
                    element.clear()

            all_frames: set[int] = set()
            player_counts: Counter[int] = Counter()
            ball_frames: set[int] = set()
            player_range, ball_range = Range2D(), Range2D()
            current_kind = "other"
            with archive.open(position_name) as handle:
                for action, element in ET.iterparse(handle, events=("start", "end")):
                    tag = local_name(element.tag)
                    if action == "start" and tag == "FrameSet":
                        team_id = element.attrib.get("TeamId", "")
                        person_id = element.attrib.get("PersonId", "")
                        if team_id in teams and person_id in players:
                            current_kind = "player"
                        elif "ball" in team_id.lower() or "ball" in person_id.lower():
                            current_kind = "ball"
                        else:
                            current_kind = "other"
                    elif action == "end" and tag == "Frame":
                        frame = int(element.attrib["N"])
                        all_frames.add(frame)
                        if current_kind == "player":
                            if player_range.add(element.attrib.get("X"), element.attrib.get("Y")):
                                player_counts[frame] += 1
                        elif current_kind == "ball":
                            if ball_range.add(element.attrib.get("X"), element.attrib.get("Y")):
                                ball_frames.add(frame)
                        element.clear()
                    elif action == "end" and tag == "FrameSet":
                        current_kind = "other"
                        element.clear()
            valid = sum(player_counts[frame] == 22 for frame in all_frames)
            results.append(
                MatchAudit(
                    match_id=match_id,
                    frames=len(all_frames),
                    valid_22_player_frames=valid,
                    missing_player_frames=len(all_frames) - valid,
                    missing_ball_frames=len(all_frames - ball_frames),
                    events=event_count,
                    player_range=player_range,
                    ball_range=ball_range,
                    team_ids=sorted(teams),
                    player_ids=len(players),
                )
            )
    return results


def metrica_columns(path: Path) -> list[tuple[int, int]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        headers = [next(csv.reader(handle)) for _ in range(3)]
    pairs = []
    for column in range(3, len(headers[2]) - 2, 2):
        name = headers[2][column].strip()
        if name and name.lower() not in {"nan", "ball"}:
            pairs.append((column, column + 1))
    return pairs


def csv_rows(path: Path) -> Iterable[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for _ in range(3):
            next(reader)
        yield from reader


def count_csv_events(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def audit_metrica_csv(game_dir: Path) -> MatchAudit:
    name = game_dir.name
    home_path = game_dir / f"{name}_RawTrackingData_Home_Team.csv"
    away_path = game_dir / f"{name}_RawTrackingData_Away_Team.csv"
    event_path = game_dir / f"{name}_RawEventsData.csv"
    home_pairs, away_pairs = metrica_columns(home_path), metrica_columns(away_path)
    frames = valid = missing_players = missing_ball = 0
    player_range, ball_range = Range2D(), Range2D()
    notes: list[str] = []
    for home, away in zip(csv_rows(home_path), csv_rows(away_path), strict=True):
        frames += 1
        if home[1] != away[1]:
            notes.append(f"Home/away frame mismatch at {home[1]} and {away[1]}.")
        present = 0
        for row, pairs in ((home, home_pairs), (away, away_pairs)):
            for x_column, y_column in pairs:
                if player_range.add(row[x_column], row[y_column]):
                    present += 1
        if present == 22:
            valid += 1
        else:
            missing_players += 1
        if not ball_range.add(home[-2], home[-1]):
            missing_ball += 1
    return MatchAudit(
        match_id=name,
        frames=frames,
        valid_22_player_frames=valid,
        missing_player_frames=missing_players,
        missing_ball_frames=missing_ball,
        events=count_csv_events(event_path),
        player_range=player_range,
        ball_range=ball_range,
        team_ids=["Home", "Away"],
        player_ids=len(home_pairs) + len(away_pairs),
        notes=notes,
    )


def audit_metrica_epts(game_dir: Path) -> MatchAudit:
    tracking_path = game_dir / "Sample_Game_3_tracking.txt"
    events_path = game_dir / "Sample_Game_3_events.json"
    metadata_path = game_dir / "Sample_Game_3_metadata.xml"
    metadata = ET.parse(metadata_path).getroot()
    teams = sorted(
        element.attrib["id"]
        for element in metadata.iter()
        if local_name(element.tag) == "Team" and element.attrib.get("id")
    )
    players = [
        element.attrib["id"]
        for element in metadata.iter()
        if local_name(element.tag) == "Player" and element.attrib.get("id")
    ]
    frame_rate = next(
        float(element.text)
        for element in metadata.iter()
        if local_name(element.tag) == "FrameRate"
    )
    frames = valid = missing_players = missing_ball = 0
    player_range, ball_range = Range2D(), Range2D()
    with tracking_path.open(encoding="utf-8") as handle:
        for line in handle:
            frames += 1
            _frame, player_text, ball_text = line.rstrip("\n").split(":")
            positions = [position.split(",") for position in player_text.split(";")]
            present = sum(player_range.add(*position) for position in positions)
            if present == 22:
                valid += 1
            else:
                missing_players += 1
            ball = ball_text.split(",")
            if not ball_range.add(*ball):
                missing_ball += 1
    events = json.loads(events_path.read_text(encoding="utf-8"))["data"]
    return MatchAudit(
        match_id="Sample_Game_3",
        frames=frames,
        valid_22_player_frames=valid,
        missing_player_frames=missing_players,
        missing_ball_frames=missing_ball,
        events=len(events),
        player_range=player_range,
        ball_range=ball_range,
        team_ids=teams,
        player_ids=len(players),
        sampling_hz=frame_rate,
    )


def audit_metrica(root: Path) -> list[MatchAudit]:
    return [
        audit_metrica_csv(root / "data" / "Sample_Game_1"),
        audit_metrica_csv(root / "data" / "Sample_Game_2"),
        audit_metrica_epts(root / "data" / "Sample_Game_3"),
    ]


def totals(matches: list[MatchAudit]) -> dict[str, int]:
    return {
        "matches": len(matches),
        "frames": sum(match.frames for match in matches),
        "valid": sum(match.valid_22_player_frames for match in matches),
        "missing_players": sum(match.missing_player_frames for match in matches),
        "missing_ball": sum(match.missing_ball_frames for match in matches),
        "events": sum(match.events for match in matches),
    }


def range_union(matches: list[MatchAudit], attribute: str) -> dict[str, list[float] | None]:
    combined = Range2D()
    for match in matches:
        values: Range2D = getattr(match, attribute)
        if values.x_min != math.inf:
            combined.add(values.x_min, values.y_min)
            combined.add(values.x_max, values.y_max)
    return combined.as_dict()


def table(matches: list[MatchAudit]) -> str:
    lines = [
        "| Match | Frames | Valid 22-player frames | Missing-player frames | Missing-ball frames | Events | Player x/y range | Ball x/y range |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for match in matches:
        player, ball = match.player_range.as_dict(), match.ball_range.as_dict()
        lines.append(
            f"| {match.match_id} | {match.frames} | {match.valid_22_player_frames} | "
            f"{match.missing_player_frames} | {match.missing_ball_frames} | {match.events} | "
            f"x={player['x']}, y={player['y']} | x={ball['x']}, y={ball['y']} |"
        )
    return "\n".join(lines)


def build_report(dataset_a: list[MatchAudit], metrica: list[MatchAudit]) -> str:
    dataset_a_totals, metrica_totals = totals(dataset_a), totals(metrica)
    dataset_a_teams = sorted({team for match in dataset_a for team in match.team_ids})
    return f"""# Stage 3 Dataset Audit

Generated by `python -m thesis_experiments.scripts.audit_datasets` from the
complete raw inputs. A valid 22-player frame has exactly 22 finite player
coordinate pairs. A missing-player frame is any frame that does not meet that
criterion; it is counted once regardless of how many players are absent.

## Dataset A (BASSEK)

Provider format: DFL XML, with separate match-information, event, and observed
position files. Tracking is 25 Hz (40 ms frame spacing). Team IDs use stable
`DFL-CLU-*` identifiers and player IDs use stable `DFL-OBJ-*` identifiers.
Player and ball tracking use metres with the origin at pitch centre on a
105 m by 68 m pitch. Event locations use a 0..105 by 0..68 metre convention.

{table(dataset_a)}

Totals: **{dataset_a_totals['matches']} matches**, **{dataset_a_totals['frames']} frames**,
**{dataset_a_totals['valid']} valid 22-player frames**, **{dataset_a_totals['missing_players']}
missing-player frames**, **{dataset_a_totals['missing_ball']} missing-ball frames**, and
**{dataset_a_totals['events']} events**. Observed player range:
`{range_union(dataset_a, 'player_range')}`; observed ball range:
`{range_union(dataset_a, 'ball_range')}`. The seven matches contain
{sum(match.player_ids for match in dataset_a)} match-roster player records and
{len(dataset_a_teams)} distinct team IDs.

## Metrica Sports Sample Data

Games 1 and 2 use synchronized home/away tracking CSV plus event CSV. Columns
start with `Period`, `Frame`, and `Time [s]`; player columns use provider labels
such as `Player11`, and the final coordinate pair is the ball. Game 3 uses FIFA
EPTS metadata XML, colon/semicolon-delimited tracking text, and JSON events.
All three games are sampled at 25 Hz. Games 1 and 2 use `Home`/`Away`; Game 3
uses `FIFATMA`/`FIFATMB` and stable `P*` player IDs. Coordinates are normalized
to 0..1 with `(0, 0)` at top-left and `(0.5, 0.5)` at pitch centre.

{table(metrica)}

Totals: **{metrica_totals['matches']} matches**, **{metrica_totals['frames']} frames**,
**{metrica_totals['valid']} valid 22-player frames**, **{metrica_totals['missing_players']}
missing-player frames**, **{metrica_totals['missing_ball']} missing-ball frames**, and
**{metrica_totals['events']} events**. Observed player range:
`{range_union(metrica, 'player_range')}`; observed ball range:
`{range_union(metrica, 'ball_range')}`.

## Known limitations and usability

- Missing-player counts include substitutions, dismissals, provider gaps, and
  pre/post-match frames; downstream builders must explicitly filter for 22
  finite player positions rather than assume every frame is usable.
- Missing-ball frames are expected when the provider cannot observe the ball.
- Dataset A coordinates for players/ball and events use different origins;
  event coordinates require the existing Dataset A normalization step.
- Both providers retain tracked positions slightly outside nominal pitch
  bounds (for example, substituted players leaving the field); conversion code
  must distinguish off-pitch observations from malformed coordinates.
- Metrica Games 1/2 and Game 3 have different file formats. Game 3 must use its
  EPTS metadata and cannot be parsed as the earlier CSV layout.
- Metrica is anonymized sample data and is appropriate for parser/replay
  validation, while Dataset A supplies the seven official-match experiments.
- The audit validates structural availability and finite coordinates. It does
  not yet segment possession, resolve substitutions, or construct learning
  transitions; those belong to later stages.

Usable frames for the strict unified pipeline are the valid 22-player frames;
tasks requiring the ball must additionally exclude missing-ball frames.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-a", type=Path, default=Path("data/raw/dataset_a/datasetA.zip"))
    parser.add_argument("--metrica", type=Path, default=Path("data/raw/metrica_sample"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/dataset_audit.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_a = audit_dataset_a(args.dataset_a)
    metrica = audit_metrica(args.metrica)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(dataset_a, metrica), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
