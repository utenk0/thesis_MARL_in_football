"""Create a match-disjoint manifest for one repeatedly observed team."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis_experiments.training.team_bc import inspect_team_chunk


def _path(row: dict, source_manifest: Path) -> Path:
    value = Path(row["path"])
    if value.is_absolute() or value.exists():
        return value
    candidate = source_manifest.parent / value
    return candidate if candidate.exists() else value


def _evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    if len(rows) <= count:
        return list(rows)
    indices = np.linspace(0, len(rows) - 1, count, dtype=np.int64)
    return [rows[int(index)] for index in np.unique(indices)]


def _rows_for_team(source_manifest: Path, team_id: str) -> dict[str, list[dict]]:
    source = json.loads(source_manifest.read_text())
    matches: dict[str, list[dict]] = {}
    for original_split, rows in source["splits"].items():
        for original in rows:
            path = _path(original, source_manifest)
            try:
                metadata = inspect_team_chunk(path, team_id)
            except ValueError:
                continue
            row = dict(original)
            row.update(
                {
                    "path": str(path),
                    "original_split": original_split,
                    "focal_team_id": team_id,
                    "focal_team_side": metadata["side"],
                    "focal_player_indices": metadata["player_indices"],
                }
            )
            matches.setdefault(str(metadata["match_id"]), []).append(row)
    for rows in matches.values():
        rows.sort(key=lambda row: (int(row.get("start_frame", 0)), int(row.get("chunk_index", 0))))
    return matches


def build_manifest(
    source_manifest: Path,
    team_id: str,
    *,
    train_matches: list[str] | None = None,
    validation_matches: list[str] | None = None,
    test_matches: list[str] | None = None,
    single_match: str | None = None,
    purge_chunks: int = 1,
) -> dict[str, object]:
    matches = _rows_for_team(source_manifest, team_id)
    if not matches:
        raise ValueError(f"No chunks contain team {team_id!r}.")

    if single_match is not None:
        if single_match not in matches:
            raise ValueError(
                f"Team {team_id!r} does not occur in requested match {single_match!r}."
            )
        rows = matches[single_match]
        usable = len(rows) - 2 * purge_chunks
        if usable < 3:
            raise ValueError("Not enough chunks for train/validation/test after purging boundaries.")
        train_count = max(1, int(usable * 0.6))
        validation_count = max(1, int(usable * 0.2))
        test_count = usable - train_count - validation_count
        if test_count < 1:
            validation_count -= 1
            test_count += 1
        train_end = train_count
        validation_start = train_end + purge_chunks
        validation_end = validation_start + validation_count
        test_start = validation_end + purge_chunks
        split_rows = {
            "train": rows[:train_end],
            "validation": rows[validation_start:validation_end],
            "test": rows[test_start : test_start + test_count],
        }
        strategy = {
            "type": "single_match_chronological_60_20_20",
            "match_id": single_match,
            "purged_chunks_per_boundary": purge_chunks,
            "limitation": (
                "All splits share the same match, opponent and player identities; "
                "this measures within-game temporal reconstruction, not new-game generalization."
            ),
        }
    else:
        available = sorted(matches)
        if train_matches is None and validation_matches is None and test_matches is None:
            if len(available) < 3:
                raise ValueError("At least three matches are required for an automatic match split.")
            train_matches = available[:-2]
            validation_matches = [available[-2]]
            test_matches = [available[-1]]
            assignment = "automatic_lexicographic"
        elif not (train_matches and validation_matches and test_matches):
            raise ValueError(
                "Provide train, validation and test matches together, or omit all for automatic assignment."
            )
        else:
            assignment = "explicit"
        requested = list(train_matches) + list(validation_matches) + list(test_matches)
        if len(requested) != len(set(requested)):
            raise ValueError("Train, validation and test match assignments must be disjoint.")
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Requested matches do not contain team {team_id!r}: {missing}")
        split_rows = {
            "train": [row for match in train_matches for row in matches[match]],
            "validation": [row for match in validation_matches for row in matches[match]],
            "test": [row for match in test_matches for row in matches[match]],
        }
        strategy = {
            "type": "match_disjoint",
            "assignment": assignment,
            "train_matches": list(train_matches),
            "validation_matches": list(validation_matches),
            "test_matches": list(test_matches),
        }

    return {
        "version": "single_team_bc_v1",
        "source_manifest": str(source_manifest),
        "team_id": team_id,
        "available_team_matches": {
            match_id: len(rows) for match_id, rows in sorted(matches.items())
        },
        "split_strategy": strategy,
        "splits": split_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot-output", type=Path)
    parser.add_argument("--pilot-chunks-per-match", type=int, default=2)
    parser.add_argument("--train-matches", nargs="+")
    parser.add_argument("--validation-matches", nargs="+")
    parser.add_argument("--test-matches", nargs="+")
    parser.add_argument("--single-match")
    parser.add_argument("--purge-chunks", type=int, default=1)
    args = parser.parse_args()
    if args.pilot_chunks_per_match < 1 or args.purge_chunks < 0:
        parser.error("Pilot chunk count must be positive and purge chunks non-negative.")
    manifest = build_manifest(
        args.source_manifest,
        args.team_id,
        train_matches=args.train_matches,
        validation_matches=args.validation_matches,
        test_matches=args.test_matches,
        single_match=args.single_match,
        purge_chunks=args.purge_chunks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))
    result = {"manifest": str(args.output)}
    if args.pilot_output:
        pilot = dict(manifest)
        pilot["version"] = "single_team_bc_v1_pilot"
        pilot["parent_manifest"] = str(args.output)
        pilot["scope"] = "Evenly spaced chunks for pipeline verification; not final evidence."
        pilot["splits"] = {}
        for split, rows in manifest["splits"].items():
            selected = []
            for match_id in sorted({row["match_id"] for row in rows}):
                match_rows = [row for row in rows if row["match_id"] == match_id]
                selected.extend(_evenly_spaced(match_rows, args.pilot_chunks_per_match))
            pilot["splits"][split] = selected
        args.pilot_output.parent.mkdir(parents=True, exist_ok=True)
        args.pilot_output.write_text(json.dumps(pilot, indent=2))
        result["pilot_manifest"] = str(args.pilot_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
