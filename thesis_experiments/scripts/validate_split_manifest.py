"""Fail fast when a training manifest is incomplete or observation-incompatible."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis_experiments.transitions.observations import LOCAL_FEATURES, OBSERVATION_VERSION


REQUIRED_SPLITS = ("train", "validation", "test")


def _resolve_path(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def validate_manifest(
    manifest_path: Path,
    project_dir: Path,
    *,
    expected_team_id: str | None = None,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(f"{manifest_path}: missing object field 'splits'.")

    declared_team_id = manifest.get("team_id")
    if expected_team_id is not None and declared_team_id != expected_team_id:
        raise ValueError(
            f"{manifest_path}: expected team_id {expected_team_id!r}, "
            f"found {declared_team_id!r}."
        )

    summaries: dict[str, dict[str, int]] = {}
    checked_files = 0
    total_bytes = 0
    for split in REQUIRED_SPLITS:
        records = splits.get(split)
        if not isinstance(records, list) or not records:
            raise ValueError(f"{manifest_path}: required split {split!r} is empty or missing.")

    for split, records in splits.items():
        if not isinstance(records, list):
            raise ValueError(f"{manifest_path}: split {split!r} must be a list.")
        split_transitions = 0
        split_bytes = 0
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise ValueError(f"{manifest_path}: invalid record in split {split!r}.")
            if declared_team_id is not None and record.get("focal_team_id") != declared_team_id:
                raise ValueError(
                    f"{manifest_path}: {split!r} record {record.get('path')!r} does not "
                    f"belong to declared team_id {declared_team_id!r}."
                )
            path = _resolve_path(project_dir, record["path"])
            if not path.is_file():
                raise FileNotFoundError(f"{manifest_path}: referenced file does not exist: {path}")
            with np.load(path, allow_pickle=False) as data:
                version = (
                    str(data["observation_version"])
                    if "observation_version" in data.files
                    else "local_v1_spatial_24"
                )
                if version != OBSERVATION_VERSION:
                    raise ValueError(
                        f"{path} uses {version!r}; training requires {OBSERVATION_VERSION!r}. "
                        "Select or rebuild a current transition dataset."
                    )
                if "local_observations" not in data.files:
                    raise ValueError(f"{path}: missing local_observations.")
                feature_count = int(data["local_observations"].shape[-1])
                if feature_count != LOCAL_FEATURES:
                    raise ValueError(
                        f"{path}: found {feature_count} observation features; expected {LOCAL_FEATURES}."
                    )
            size = path.stat().st_size
            checked_files += 1
            total_bytes += size
            split_bytes += size
            split_transitions += int(record.get("transitions", 0))
        summaries[split] = {
            "files": len(records),
            "transitions": split_transitions,
            "bytes": split_bytes,
        }

    return {
        "manifest": str(manifest_path),
        "team_id": declared_team_id,
        "observation_version": OBSERVATION_VERSION,
        "feature_count": LOCAL_FEATURES,
        "checked_files": checked_files,
        "total_bytes": total_bytes,
        "splits": summaries,
        "ok": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--expected-team-id",
        help="Reject a manifest that is not exclusively bound to this focal team.",
    )
    args = parser.parse_args()
    result = validate_manifest(
        args.manifest.resolve(),
        args.project_dir.resolve(),
        expected_team_id=args.expected_team_id,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
