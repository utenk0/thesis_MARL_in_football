"""List a portable freeze manifest and every file it protects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _project_path(project_dir: Path, value: str) -> Path:
    path = (project_dir / value).resolve()
    try:
        path.relative_to(project_dir)
    except ValueError as error:
        raise ValueError(f"Frozen path escapes the project directory: {value}") from error
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    manifest = args.manifest.resolve()
    try:
        manifest_relative = manifest.relative_to(project_dir)
    except ValueError as error:
        raise ValueError("Freeze manifest must be inside the project directory.") from error
    payload = json.loads(manifest.read_text())
    paths = [manifest]
    paths.extend(
        _project_path(project_dir, str(record["path"]))
        for record in payload.get("files", [])
    )

    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = manifest_relative if path == manifest else path.relative_to(project_dir)
        print(relative.as_posix())


if __name__ == "__main__":
    main()
