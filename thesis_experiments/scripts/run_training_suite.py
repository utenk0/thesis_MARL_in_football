"""Run the training-stage method suite on one or more seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from thesis_experiments.methods import available_methods

DEFAULT_METHODS = ["bc", "gail", "ppo", "cql", "dt", "ghosting", "ctde"]


@dataclass(slots=True)
class TrainingRun:
    seed: int
    method: str
    command: list[str]
    status: str
    log: str
    returncode: int | None = None


def _dataset_contract(manifest_path: Path, manifest: dict) -> dict[str, object]:
    """Create immutable provenance shared by every method in this suite run."""
    return {
        "version": "closed_loop_dataset_contract_v1",
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "team_id": manifest.get("team_id"),
        "splits": {
            split: {
                "matches": sorted({str(row.get("match_id")) for row in rows}),
                "paths": [str(row["path"]) for row in rows],
            }
            for split, rows in manifest["splits"].items()
        },
    }


def _check_team_binding(
    parser: argparse.ArgumentParser,
    manifest_path: Path,
    manifest: dict,
    expected_team_id: str | None,
) -> None:
    if expected_team_id is None:
        return
    declared_team_id = manifest.get("team_id")
    if declared_team_id != expected_team_id:
        parser.error(
            f"{manifest_path} declares team_id {declared_team_id!r}; "
            f"expected {expected_team_id!r}"
        )
    for split, rows in manifest["splits"].items():
        for row in rows:
            if row.get("focal_team_id") != expected_team_id:
                parser.error(
                    f"{manifest_path} contains a {split} record for focal_team_id "
                    f"{row.get('focal_team_id')!r}; expected {expected_team_id!r}"
                )


def _method_command(args: argparse.Namespace, *, seed: int, method: str, output_dir: Path) -> list[str]:
    command = [
        args.python,
        "-m",
        "thesis_experiments.scripts.run_method",
        "--method",
        method,
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--hidden-size",
        str(args.hidden_size),
        "--total-timesteps",
        str(args.total_timesteps),
        "--device",
        args.device,
        "--actor-epochs",
        str(args.actor_epochs),
        "--disc-updates",
        str(args.disc_updates),
        "--data",
        *[str(path) for path in args.data],
    ]
    if method == "bc" and args.bc_backend:
        command.extend(["--backend", args.bc_backend])
    for flag, value in [("--env-backend", args.env_backend), ("--rollout-steps", args.rollout_steps),
                        ("--episode-steps", args.episode_steps), ("--context-length", args.context_length)]:
        command.extend([flag, str(value)])
    if args.grf_python:
        command.extend(["--grf-python", str(args.grf_python)])
    return command


def _run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return int(process.returncode)


def _completed_result(output_dir: Path, seed: int, method: str) -> bool:
    """Return true only when a completed manifest entry still has its artifact."""
    result_path = output_dir / f"seed_{seed}" / "methods" / method / "result.json"
    try:
        result = json.loads(result_path.read_text())
        artifact = result["artifacts"].get("checkpoint", result["artifacts"].get("policy"))
        return result["status"] == "trained" and bool(artifact) and Path(artifact).is_file()
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _run_ctde_dependency(args: argparse.Namespace, *, seed: int, seed_dir: Path, log_dir: Path) -> Path:
    dependency_output = seed_dir / "_dependencies" / "ctde_bc"
    command = _method_command(args, seed=seed, method="bc", output_dir=dependency_output)
    command.extend(["--backend", "torch"])
    log_path = log_dir / f"seed_{seed}_ctde_bc.log"
    if args.dry_run:
        return dependency_output / "bc" / "shared_actor_bc.pt"
    returncode = _run_command(command, log_path)
    if returncode != 0:
        raise RuntimeError(f"CTDE BC dependency failed for seed {seed}; see {log_path}")
    return dependency_output / "bc" / "shared_actor_bc.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument(
        "--expected-team-id",
        help="Require every split-manifest record to belong to this focal team.",
    )
    parser.add_argument("--validation-data", nargs="+", type=Path)
    parser.add_argument("--test-data", nargs="+", type=Path)
    parser.add_argument("--external-test-data", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/university_training"))
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-backend", choices=["grf", "replay"], default="grf")
    parser.add_argument("--grf-python", type=Path)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--episode-steps", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--eval-grf-steps", type=int, default=100)
    parser.add_argument("--actor-epochs", type=int, default=10)
    parser.add_argument("--disc-updates", type=int, default=4)
    parser.add_argument("--bc-backend", default="auto", choices=["auto", "imitation", "torch"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed seed/method runs and retry incomplete entries in an existing output directory.",
    )
    args = parser.parse_args()
    contract: dict[str, object] | None = None
    if args.split_manifest:
        if args.data or args.validation_data or args.test_data or args.external_test_data:
            parser.error("Use --split-manifest or explicit data paths, not both")
        manifest_document = json.loads(args.split_manifest.read_text())
        splits = manifest_document.get("splits")
        if not isinstance(splits, dict):
            parser.error(f"{args.split_manifest} does not contain a splits object")
        _check_team_binding(
            parser, args.split_manifest, manifest_document, args.expected_team_id
        )
        contract = _dataset_contract(args.split_manifest, manifest_document)
        for split, rows in splits.items():
            for row in rows:
                path = Path(row["path"])
                if not path.is_file():
                    parser.error(f"Missing {split} dataset: {path}")
                expected = row.get("sha256")
                if expected:
                    actual = hashlib.sha256(path.read_bytes()).hexdigest()
                    if actual != expected:
                        parser.error(f"Checksum mismatch for {path}: expected {expected}, got {actual}")
        args.data = [Path(row["path"]) for row in splits["train"]]
        args.validation_data = [Path(row["path"]) for row in splits["validation"]]
        args.test_data = [Path(row["path"]) for row in splits["test"]]
        args.external_test_data = [Path(row["path"]) for row in splits.get("external_test", [])]
    if not args.data:
        parser.error("Provide --data or --split-manifest")
    from thesis_experiments.scripts.evaluate_policy import validate_split
    for evaluation_paths in (args.validation_data, args.test_data, args.external_test_data):
        if evaluation_paths:
            validate_split(args.data, evaluation_paths)
    if args.validation_data and args.test_data:
        validate_split(args.validation_data, args.test_data)
    if args.external_test_data:
        for comparison_paths in (args.validation_data, args.test_data):
            if comparison_paths:
                validate_split(comparison_paths, args.external_test_data)

    methods = DEFAULT_METHODS if args.methods == ["all"] else args.methods
    unknown = sorted(set(methods) - set(available_methods()))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {available_methods()}")

    manifest: list[TrainingRun] = []
    log_dir = args.output_dir / "logs"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "training_manifest.json"
    if contract is not None:
        contract_path = args.output_dir / "dataset_contract.json"
        if contract_path.exists():
            try:
                existing_contract = json.loads(contract_path.read_text())
            except (OSError, ValueError) as exc:
                parser.error(f"Cannot read dataset contract {contract_path}: {exc}")
            if existing_contract != contract:
                parser.error(
                    f"{args.output_dir} is already bound to a different dataset contract; "
                    "choose a new --output-dir"
                )
        elif manifest_path.exists():
            parser.error(
                f"{args.output_dir} contains a legacy training manifest without a "
                "dataset contract; choose a new --output-dir"
            )
        else:
            contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    if manifest_path.exists():
        if not args.resume:
            parser.error(f"{manifest_path} already exists; choose a new --output-dir or pass --resume")
        try:
            manifest = [TrainingRun(**item) for item in json.loads(manifest_path.read_text())]
        except (OSError, ValueError, TypeError) as exc:
            parser.error(f"Cannot resume invalid manifest {manifest_path}: {exc}")
    runs_by_key = {(item.seed, item.method): item for item in manifest}

    def persist():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps([asdict(item) for item in manifest], indent=2))
        temporary.replace(manifest_path)

    failed = False
    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed_{seed}"
        ctde_checkpoint: Path | None = None
        for method in methods:
            method_output = seed_dir / "methods"
            command = _method_command(args, seed=seed, method=method, output_dir=method_output)
            previous = runs_by_key.get((seed, method))
            if previous and previous.status == "completed" and _completed_result(args.output_dir, seed, method):
                print(f"Skipping completed {method}, seed {seed}", flush=True)
                continue
            if method == "ctde":
                candidate = method_output / "bc" / "shared_actor_bc.pt"
                if "bc" in methods[:methods.index("ctde")] and args.bc_backend == "torch" and candidate.exists():
                    ctde_checkpoint = candidate
                else:
                    try:
                        ctde_checkpoint = _run_ctde_dependency(args, seed=seed, seed_dir=seed_dir, log_dir=log_dir)
                    except RuntimeError:
                        dependency_log = str(log_dir / f"seed_{seed}_ctde_bc.log")
                        run = previous or TrainingRun(seed, method, command, "dependency-failed", dependency_log, 1)
                        run.command, run.status, run.log, run.returncode = command, "dependency-failed", dependency_log, 1
                        if previous is None:
                            manifest.append(run)
                            runs_by_key[(seed, method)] = run
                        persist()
                        failed = True
                        if not args.continue_on_error:
                            raise SystemExit(1)
                        continue
                command.extend(["--bc-checkpoint", str(ctde_checkpoint)])
            log_path = log_dir / f"seed_{seed}_{method}.log"
            if args.dry_run:
                run = previous or TrainingRun(seed, method, command, "dry-run", str(log_path))
                run.command, run.status, run.log, run.returncode = command, "dry-run", str(log_path), None
                if previous is None:
                    manifest.append(run)
                    runs_by_key[(seed, method)] = run
                persist()
                continue
            run = previous or TrainingRun(seed, method, command, "running", str(log_path))
            if previous is None:
                manifest.append(run)
                runs_by_key[(seed, method)] = run
            result_path = method_output / method / "result.json"
            reuse_trained_artifact = previous is not None and _completed_result(
                args.output_dir, seed, method
            )
            if reuse_trained_artifact:
                print(f"Reusing trained {method}, seed {seed}; retrying evaluation", flush=True)
                run.command, run.status, run.log, run.returncode = (
                    command, "trained", str(log_path), 0
                )
                returncode = 0
                persist()
            else:
                print(f"Training {method}, seed {seed}; log: {log_path}", flush=True)
                run.command, run.status, run.log, run.returncode = (
                    command, "running", str(log_path), None
                )
                persist()
                returncode = _run_command(command, log_path)
                if returncode == 0:
                    try:
                        result = json.loads(result_path.read_text())
                        artifact = result["artifacts"].get("checkpoint", result["artifacts"].get("policy"))
                        if result["status"] != "trained" or not artifact or not Path(artifact).is_file():
                            returncode = 1
                    except (OSError, ValueError, KeyError):
                        returncode = 1
                run.status = "trained" if returncode == 0 else "failed"
                run.returncode = returncode
                persist()
            if returncode == 0:
                for split, paths in [
                    ("validation", args.validation_data),
                    ("test", args.test_data),
                    ("external_test", args.external_test_data),
                ]:
                    if not paths:
                        continue
                    eval_command = [args.python, "-m", "thesis_experiments.scripts.evaluate_policy",
                        "--result", str(result_path), "--output", str(method_output / method / f"{split}_evaluation.json"),
                        "--grf-steps", str(args.eval_grf_steps if split == "test" else 0),
                        "--seed", str(seed), "--device", args.device,
                        "--data", *map(str, paths)]
                    if args.grf_python:
                        eval_command.extend(["--grf-python", str(args.grf_python)])
                    eval_log = log_dir / f"seed_{seed}_{method}_{split}.log"
                    returncode = _run_command(eval_command, eval_log)
                    if returncode:
                        run.status, run.returncode = f"{split}-evaluation-failed", returncode
                        break
                if returncode == 0:
                    run.status = "completed" if args.test_data else "trained"
            persist()
            print(f"{method}: {run.status}", flush=True)
            if returncode:
                failed = True
                if not args.continue_on_error:
                    raise SystemExit(returncode)
    print(json.dumps({"manifest": str(manifest_path), "runs": len(manifest)}, indent=2))
    if not args.dry_run:
        from thesis_experiments.scripts.summarize_training import summarize
        summarize(args.output_dir)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
