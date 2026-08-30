from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = BASE_DIR / ".venv" / "datasets" / "statsbomb-open-data" / "data"


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_match_metadata(data_root: Path, match_id: int) -> tuple[dict, Path] | tuple[None, None]:
    matches_root = data_root / "matches"
    for path in matches_root.rglob("*.json"):
        rows = load_json(path)
        for row in rows:
            if row.get("match_id") == match_id:
                return row, path
    return None, None


def inspect_match(data_root: Path, match_id: int) -> None:
    events_path = data_root / "events" / f"{match_id}.json"
    lineups_path = data_root / "lineups" / f"{match_id}.json"
    three_sixty_path = data_root / "three-sixty" / f"{match_id}.json"

    if not events_path.exists():
        raise SystemExit(f"Missing events file: {events_path}")

    events = load_json(events_path)
    lineups = load_json(lineups_path) if lineups_path.exists() else []
    three_sixty = load_json(three_sixty_path) if three_sixty_path.exists() else []
    match_row, match_source = find_match_metadata(data_root, match_id)

    print(f"=== StatsBomb Match {match_id} ===")
    print(f"Events source: {events_path}")
    print(f"Lineups source: {lineups_path if lineups_path.exists() else 'missing'}")
    print(f"Three-sixty source: {three_sixty_path if three_sixty_path.exists() else 'missing'}")

    if match_row:
        print("\nMatch metadata:")
        print(f"  Source file: {match_source}")
        print(f"  Competition: {match_row.get('competition', {}).get('competition_name')}")
        print(f"  Season: {match_row.get('season', {}).get('season_name')}")
        print(f"  Home: {match_row.get('home_team', {}).get('home_team_name')}")
        print(f"  Away: {match_row.get('away_team', {}).get('away_team_name')}")
        print(f"  Kickoff: {match_row.get('kick_off')}")

    print("\nDataset summary:")
    print(f"  Events: {len(events)}")
    print(f"  Lineup teams: {len(lineups)}")
    print(f"  Three-sixty rows: {len(three_sixty)}")

    type_counts = Counter(event.get("type", {}).get("name", "Unknown") for event in events)
    print("\nTop event types:")
    for event_type, count in type_counts.most_common(15):
        print(f"  {event_type}: {count}")

    team_counts = Counter(event.get("team", {}).get("name", "Unknown") for event in events if event.get("team"))
    print("\nEvents by team:")
    for team_name, count in team_counts.most_common():
        print(f"  {team_name}: {count}")

    sample = events[:5]
    print("\nSample events:")
    for event in sample:
        event_type = event.get("type", {}).get("name")
        print(
            "  "
            f"index={event.get('index')} period={event.get('period')} "
            f"time={event.get('timestamp')} type={event_type}"
        )

    if lineups:
        print("\nLineup teams:")
        for team in lineups:
            print(f"  {team.get('team_name')}: {len(team.get('lineup', []))} players")

    if three_sixty:
        first = three_sixty[0]
        print("\nThree-sixty sample:")
        print(f"  Event UUID: {first.get('event_uuid')}")
        print(f"  Freeze frame rows: {len(first.get('freeze_frame', []))}")
        print(f"  Visible area points: {len(first.get('visible_area', []))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one StatsBomb open-data match.")
    parser.add_argument("--match-id", type=int, default=16317, help="StatsBomb match id")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"StatsBomb data root (default: {DEFAULT_ROOT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_match(args.data_root, args.match_id)


if __name__ == "__main__":
    main()
