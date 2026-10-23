"""Validate a method's frozen test summary against the shared trajectory contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_experiments.evaluation.trajectory_contract import (
    load_contract,
    validate_test_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    summary = json.loads(args.summary.read_text())
    validate_test_summary(contract, summary)
    print(
        json.dumps(
            {
                "passed": True,
                "contract": contract["contract"],
                "summary": str(args.summary),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
