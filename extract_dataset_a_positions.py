from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ZIP_PATH = BASE_DIR / ".venv" / "datasets" / "datasetA.zip"


def build_match_file_map(zip_path: Path) -> dict[str, dict[str, str]]:
    match_map: dict[str, dict[str, str]] = {}

    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".xml"):
                continue

            stem = Path(name).stem
            if "_DFL-MAT-" not in stem:
                continue

            prefix, match_id = stem.rsplit("_", 1)
            slots = match_map.setdefault(match_id, {})

            if "matchinformation" in prefix:
                slots["matchinfo"] = name
            elif "positions_raw_observed" in prefix:
                slots["positions"] = name

    return match_map


def parse_match_metadata(zip_path: Path, xml_name: str) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    team_names: dict[str, str] = {}
    players: dict[str, dict[str, str]] = {}

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(xml_name) as xml_file:
            root = ET.parse(xml_file).getroot()

    match_info = next(iter(root), root)
    teams_node = match_info.find("Teams")
    if teams_node is None:
        return team_names, players

    for team in teams_node.findall("Team"):
        team_id = team.attrib.get("TeamId", "")
        team_names[team_id] = team.attrib.get("TeamName", "")

        players_node = team.find("Players")
        if players_node is None:
            continue

        for player in players_node.findall("Player"):
            person_id = player.attrib.get("PersonId")
            if not person_id:
                continue

            players[person_id] = {
                "team_id": team_id,
                "team_name": team_names.get(team_id, ""),
                "shirt_number": player.attrib.get("ShirtNumber", ""),
                "short_name": player.attrib.get("Shortname", ""),
                "first_name": player.attrib.get("FirstName", ""),
                "last_name": player.attrib.get("LastName", ""),
                "position": player.attrib.get("PlayingPosition", ""),
                "starting": player.attrib.get("Starting", ""),
                "captain": player.attrib.get("TeamLeader", ""),
            }

    return team_names, players


def write_positions_csv(
    zip_path: Path,
    match_id: str,
    positions_xml: str,
    output_csv: Path,
    team_names: dict[str, str],
    players: dict[str, dict[str, str]],
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0

    headers = [
        "match_id",
        "game_section",
        "frame",
        "match_time_minute",
        "timestamp",
        "entity_type",
        "team_id",
        "team_name",
        "person_id",
        "shirt_number",
        "short_name",
        "x",
        "y",
        "z",
        "speed",
        "acceleration",
        "direction",
        "motion_status",
        "ball_possession",
        "ball_status",
    ]

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(positions_xml) as xml_file, output_csv.open("w", newline="", encoding="utf-8") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=headers)
            writer.writeheader()

            current_frameset: dict[str, str] | None = None

            for event, element in ET.iterparse(xml_file, events=("start", "end")):
                if event == "start" and element.tag == "FrameSet":
                    current_frameset = dict(element.attrib)
                    continue

                if event == "end" and element.tag == "Frame" and current_frameset is not None:
                    team_id = current_frameset.get("TeamId", "")
                    person_id = current_frameset.get("PersonId", "")
                    is_ball = team_id == "BALL"
                    is_player = team_id in team_names

                    if not is_ball and not is_player:
                        element.clear()
                        continue

                    player_meta = players.get(person_id, {})

                    writer.writerow(
                        {
                            "match_id": match_id,
                            "game_section": current_frameset.get("GameSection", ""),
                            "frame": element.attrib.get("N", ""),
                            "match_time_minute": element.attrib.get("M", ""),
                            "timestamp": element.attrib.get("T", ""),
                            "entity_type": "ball" if is_ball else "player",
                            "team_id": team_id,
                            "team_name": "BALL" if is_ball else team_names.get(team_id, ""),
                            "person_id": person_id,
                            "shirt_number": "" if is_ball else player_meta.get("shirt_number", ""),
                            "short_name": "Ball" if is_ball else player_meta.get("short_name", ""),
                            "x": element.attrib.get("X", ""),
                            "y": element.attrib.get("Y", ""),
                            "z": element.attrib.get("Z", ""),
                            "speed": element.attrib.get("S", ""),
                            "acceleration": element.attrib.get("A", ""),
                            "direction": element.attrib.get("D", ""),
                            "motion_status": element.attrib.get("BallStatus" if is_ball else "M", ""),
                            "ball_possession": element.attrib.get("BallPossession", ""),
                            "ball_status": element.attrib.get("BallStatus", ""),
                        }
                    )
                    row_count += 1
                    element.clear()
                    continue

                if event == "end" and element.tag == "FrameSet":
                    current_frameset = None
                    element.clear()

    return row_count


def resolve_match_files(zip_path: Path, match_id: str) -> tuple[str, str]:
    match_map = build_match_file_map(zip_path)
    files = match_map.get(match_id)
    if not files:
        known = ", ".join(sorted(match_map))
        raise SystemExit(f"Unknown match id '{match_id}'. Available match ids: {known}")

    if "matchinfo" not in files or "positions" not in files:
        raise SystemExit(f"Match '{match_id}' is missing matchinfo or positions XML in {zip_path}")

    return files["matchinfo"], files["positions"]


def iter_complete_matches(zip_path: Path) -> list[tuple[str, str, str]]:
    match_map = build_match_file_map(zip_path)
    complete_matches: list[tuple[str, str, str]] = []

    for match_id in sorted(match_map):
        files = match_map[match_id]
        if "matchinfo" in files and "positions" in files:
            complete_matches.append((match_id, files["matchinfo"], files["positions"]))

    if not complete_matches:
        raise SystemExit(f"No complete matchinfo + positions pairs found in {zip_path}")

    return complete_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract player and ball positions from Dataset A into a flat CSV."
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=DEFAULT_ZIP_PATH,
        help=f"Path to datasetA.zip (default: {DEFAULT_ZIP_PATH})",
    )
    parser.add_argument(
        "--match-id",
        help="Match id to extract, e.g. DFL-MAT-J03WN1. If omitted, extract all matches in the zip.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "outputs",
        help="Output CSV path for single-match mode, or output directory for batch mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.match_id:
        matchinfo_xml, positions_xml = resolve_match_files(args.zip_path, args.match_id)
        output_path = args.output
        if output_path.suffix.lower() != ".csv":
            output_path = output_path / f"dataset_a_positions_{args.match_id}.csv"

        team_names, players = parse_match_metadata(args.zip_path, matchinfo_xml)
        row_count = write_positions_csv(
            zip_path=args.zip_path,
            match_id=args.match_id,
            positions_xml=positions_xml,
            output_csv=output_path,
            team_names=team_names,
            players=players,
        )
        print(f"Wrote {row_count} rows to {output_path}")
        return

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    complete_matches = iter_complete_matches(args.zip_path)
    for match_id, matchinfo_xml, positions_xml in complete_matches:
        team_names, players = parse_match_metadata(args.zip_path, matchinfo_xml)
        output_path = output_dir / f"dataset_a_positions_{match_id}.csv"
        row_count = write_positions_csv(
            zip_path=args.zip_path,
            match_id=match_id,
            positions_xml=positions_xml,
            output_csv=output_path,
            team_names=team_names,
            players=players,
        )
        total_rows += row_count
        print(f"{match_id}: wrote {row_count} rows to {output_path}")

    print(f"Finished {len(complete_matches)} matches with {total_rows} total rows.")


if __name__ == "__main__":
    main()
