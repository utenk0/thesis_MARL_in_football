"""List a split manifest and every data file it references, relative to a project root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    manifest = args.manifest.resolve()
    payload = json.loads(manifest.read_text())
    paths = [manifest]
    for records in payload.get("splits", {}).values():
        for record in records:
            path = Path(record["path"])
            paths.append(path if path.is_absolute() else project_dir / path)

    for path in paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(project_dir)
        except ValueError as error:
            raise ValueError(f"Manifest path escapes the project directory: {resolved}") from error
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        print(relative.as_posix())


if __name__ == "__main__":
    main()
