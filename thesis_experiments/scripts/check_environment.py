"""Check whether the active Python environment matches a project target."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["grf", "methods"], required=True)
    args = parser.parse_args()
    report = check_environment(args.target)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


def check_environment(target: str) -> dict[str, object]:
    packages = _package_versions(
        [
            "pip",
            "setuptools",
            "wheel",
            "gym",
            "gymnasium",
            "gfootball",
            "torch",
            "stable-baselines3",
            "imitation",
            "d3rlpy",
        ]
    )
    checks = _grf_checks(packages) if target == "grf" else _method_checks(packages)
    return {
        "target": target,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": packages,
        "checks": checks,
        "ok": all(item["ok"] for item in checks),
    }


def _package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _grf_checks(packages: dict[str, str | None]) -> list[dict[str, object]]:
    return [
        _present(packages, "gfootball"),
        _exact(packages, "gym", "0.21.0"),
        _exact(packages, "pip", "24.0"),
        _exact(packages, "setuptools", "65.5.0"),
        _exact(packages, "wheel", "0.38.4"),
        _absent_or_note(packages, "d3rlpy", "Keep d3rlpy out of the GRF environment because it requires gym>=0.26."),
    ]


def _method_checks(packages: dict[str, str | None]) -> list[dict[str, object]]:
    return [
        _present(packages, "gymnasium"),
        _present(packages, "torch"),
        _present(packages, "stable-baselines3"),
        _present(packages, "imitation"),
        _present(packages, "d3rlpy"),
    ]


def _present(packages: dict[str, str | None], name: str) -> dict[str, object]:
    version = packages.get(name)
    return {"name": name, "ok": version is not None, "version": version, "expected": "installed"}


def _exact(packages: dict[str, str | None], name: str, expected: str) -> dict[str, object]:
    version = packages.get(name)
    return {"name": name, "ok": version == expected, "version": version, "expected": expected}


def _absent_or_note(packages: dict[str, str | None], name: str, note: str) -> dict[str, object]:
    version = packages.get(name)
    return {"name": name, "ok": version is None, "version": version, "expected": "not installed", "note": note}


if __name__ == "__main__":
    main()
