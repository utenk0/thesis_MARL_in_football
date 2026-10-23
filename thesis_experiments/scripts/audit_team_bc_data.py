"""Audit labels, identities, roles and orientation for a one-team BC manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from thesis_experiments.datasets.action_mapping import GRF_ACTIONS
from thesis_experiments.scripts.extract_dataset_a_positions import (
    build_match_file_map,
    parse_match_metadata,
)


ACTION_NAMES = {identifier: name for name, identifier in GRF_ACTIONS.items()}


def _resolve(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    candidate = manifest_path.parent / path
    return candidate if candidate.exists() else path


def _distribution_distance(reference: np.ndarray, other: np.ndarray) -> dict[str, float]:
    reference = reference.astype(np.float64)
    other = other.astype(np.float64)
    p = reference / max(1.0, reference.sum())
    q = other / max(1.0, other.sum())
    midpoint = 0.5 * (p + q)

    def kl(left, right):
        keep = left > 0
        return float(np.sum(left[keep] * np.log2(left[keep] / right[keep])))

    return {
        "total_variation": float(0.5 * np.abs(p - q).sum()),
        "jensen_shannon_bits": float(0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)),
    }


def _metadata(dataset_a: Path, matches: set[str]) -> dict[str, dict[str, dict[str, str]]]:
    files = build_match_file_map(dataset_a)
    output = {}
    for match_id in matches:
        if match_id not in files or "matchinfo" not in files[match_id]:
            output[match_id] = {}
            continue
        _, players = parse_match_metadata(dataset_a, files[match_id]["matchinfo"])
        output[match_id] = players
    return output


def audit(manifest_path: Path, dataset_a: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    team_id = str(manifest["team_id"])
    split_matches = {
        split: {str(row["match_id"]) for row in rows}
        for split, rows in manifest["splits"].items()
    }
    all_matches = set().union(*split_matches.values())
    player_metadata = _metadata(dataset_a, all_matches)
    split_reports = {}
    match_accumulators = defaultdict(lambda: {
        "counts": np.zeros(19, dtype=np.int64),
        "samples": 0,
        "changes": 0,
        "chunks": 0,
        "sides": Counter(),
        "rosters": set(),
        "player_slots": defaultdict(set),
        "slot_players": defaultdict(set),
        "event_targets": {},
        "event_collisions": 0,
        "unmatched_focal_events": 0,
    })

    for split, rows in manifest["splits"].items():
        split_counts = np.zeros(19, dtype=np.int64)
        split_samples = split_changes = split_transitions = 0
        split_sides = Counter()
        split_event_targets = {}
        split_event_collisions = split_unmatched = 0
        for row in rows:
            path = _resolve(str(row["path"]), manifest_path)
            match_id = str(row["match_id"])
            with np.load(path, allow_pickle=False) as data:
                team_ids = np.asarray(data["team_ids"]).astype(str)
                selected = np.flatnonzero(team_ids == team_id)
                if len(selected) != 11:
                    raise ValueError(f"{path}: focal team does not occupy exactly 11 slots")
                side_offset = 0 if np.all(selected < 11) else 11
                side = "left" if side_offset == 0 else "right"
                player_ids = np.asarray(data["player_ids"]).astype(str)[selected]
                actions = np.asarray(data["actions"][:, selected], dtype=np.int64)
                frames = np.asarray(data["frames"], dtype=np.int64)
            counts = np.bincount(actions.reshape(-1), minlength=19)
            previous = np.zeros_like(actions)
            if len(actions) > 1:
                previous[1:] = actions[:-1]
            changes = int((actions != previous).sum())
            samples = int(actions.size)
            split_counts += counts
            split_samples += samples
            split_changes += changes
            split_transitions += len(actions)
            split_sides[side] += len(actions)

            current = match_accumulators[match_id]
            current["counts"] += counts
            current["samples"] += samples
            current["changes"] += changes
            current["chunks"] += 1
            current["sides"][side] += len(actions)
            roster = tuple(player_ids.tolist())
            current["rosters"].add(roster)
            for slot, player_id in enumerate(player_ids.tolist()):
                current["player_slots"][player_id].add(slot)
                current["slot_players"][slot].add(player_id)

            provenance_path = _resolve(str(row.get("label_provenance", "")), manifest_path)
            if provenance_path.is_file():
                for event in json.loads(provenance_path.read_text()):
                    if str(event.get("actor_team_id")) != team_id:
                        continue
                    if event.get("status") != "applied":
                        split_unmatched += 1
                        current["unmatched_focal_events"] += 1
                        continue
                    global_index = event.get("joint_action_player_index")
                    frame = event.get("anchor_frame")
                    action = event.get("grf_action")
                    if global_index is None or frame is None or action is None:
                        continue
                    if int(global_index) not in selected:
                        raise ValueError(f"{provenance_path}: focal event points outside focal slots")
                    key = (str(path), int(frame), int(global_index))
                    record = {
                        "action": int(action),
                        "event_type": str(event.get("event_type")),
                        "frame_in_chunk": bool(int(frame) in set(frames.tolist())),
                    }
                    if key in split_event_targets:
                        split_event_collisions += 1
                    if key in current["event_targets"]:
                        current["event_collisions"] += 1
                    # Conversion applies events in source order, so the last one is the final label.
                    split_event_targets[key] = record
                    current["event_targets"][key] = record

        event_counts = np.zeros(19, dtype=np.int64)
        event_types = Counter()
        for record in split_event_targets.values():
            event_counts[record["action"]] += 1
            event_types[record["event_type"]] += 1
        split_reports[split] = {
            "matches": sorted(split_matches[split]),
            "chunks": len(rows),
            "joint_transitions": split_transitions,
            "focal_player_samples": split_samples,
            "action_counts": split_counts.tolist(),
            "observed_actions": [int(i) for i in np.flatnonzero(split_counts)],
            "action_change_samples": split_changes,
            "action_change_rate": split_changes / split_samples,
            "sides_by_transition": dict(split_sides),
            "event_override_targets": len(split_event_targets),
            "event_override_counts": event_counts.tolist(),
            "event_types": dict(sorted(event_types.items())),
            "event_target_collisions": split_event_collisions,
            "unmatched_focal_events": split_unmatched,
        }

    match_reports = {}
    all_slot_positions = defaultdict(set)
    unstable_players = []
    for match_id, values in sorted(match_accumulators.items()):
        metadata = player_metadata.get(match_id, {})
        player_slots = {
            player: sorted(slots) for player, slots in values["player_slots"].items()
        }
        match_unstable = {
            player: slots for player, slots in player_slots.items() if len(slots) > 1
        }
        for player, slots in match_unstable.items():
            unstable_players.append({"match_id": match_id, "player_id": player, "slots": slots})
        slot_positions = {}
        for slot, players in values["slot_players"].items():
            positions = {
                str(metadata.get(player, {}).get("position", "unknown") or "unknown")
                for player in players
            }
            slot_positions[str(slot)] = sorted(positions)
            all_slot_positions[slot].update(positions)
        event_counts = np.zeros(19, dtype=np.int64)
        for record in values["event_targets"].values():
            event_counts[record["action"]] += 1
        match_reports[match_id] = {
            "chunks": values["chunks"],
            "focal_player_samples": values["samples"],
            "action_counts": values["counts"].tolist(),
            "action_change_rate": values["changes"] / values["samples"],
            "sides_by_transition": dict(values["sides"]),
            "unique_rosters": len(values["rosters"]),
            "unique_players": len(values["player_slots"]),
            "players_changing_slot": match_unstable,
            "slot_playing_positions": slot_positions,
            "event_override_targets": len(values["event_targets"]),
            "event_override_counts": event_counts.tolist(),
            "event_target_collisions": values["event_collisions"],
            "unmatched_focal_events": values["unmatched_focal_events"],
        }

    train_counts = np.asarray(split_reports["train"]["action_counts"], dtype=np.int64)
    support = {
        str(action): {
            "name": ACTION_NAMES[action],
            "train": int(split_reports["train"]["action_counts"][action]),
            "validation": int(split_reports["validation"]["action_counts"][action]),
            "test": int(split_reports["test"]["action_counts"][action]),
        }
        for action in range(19)
    }
    missing_for_evaluation = [
        action for action in range(19)
        if train_counts[action] == 0
        and (
            split_reports["validation"]["action_counts"][action] > 0
            or split_reports["test"]["action_counts"][action] > 0
        )
    ]
    under_supported = [int(action) for action in np.flatnonzero((train_counts > 0) & (train_counts < 100))]
    disjoint = all(
        not split_matches[left] & split_matches[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    position_sets = {str(slot): sorted(values) for slot, values in sorted(all_slot_positions.items())}
    arbitrary_slots = [slot for slot, positions in position_sets.items() if len(positions) > 1]
    checks = {
        "match_disjoint": disjoint,
        "training_contains_both_sides": set(split_reports["train"]["sides_by_transition"]) == {"left", "right"},
        "all_evaluation_actions_supported_in_train": not missing_for_evaluation,
        "all_observed_train_actions_have_at_least_100_samples": not under_supported,
        "stored_role_index_is_tactical_role": False,
        "players_keep_one_slot_within_each_match": not unstable_players,
    }
    return {
        "team_id": team_id,
        "manifest": str(manifest_path),
        "split_strategy": manifest["split_strategy"],
        "checks": checks,
        "splits": split_reports,
        "matches": match_reports,
        "action_support": support,
        "evaluation_actions_missing_from_train": missing_for_evaluation,
        "train_actions_with_fewer_than_100_samples": under_supported,
        "distribution_shift_from_train": {
            split: _distribution_distance(
                train_counts, np.asarray(split_reports[split]["action_counts"], dtype=np.int64)
            )
            for split in ("validation", "test")
        },
        "role_audit": {
            "stored_feature": "own_in_team / 10",
            "semantics": (
                "Roster-order slot: goalkeeper first, remaining active players ordered by "
                "provider player ID. It is not a tactical playing-position label."
            ),
            "players_changing_slot_within_match": unstable_players,
            "slot_playing_positions_across_matches": position_sets,
            "slots_with_multiple_playing_positions": arbitrary_slots,
            "recommended_correction": (
                "Zero feature index 7 for the current experiment. A future representation "
                "may replace it with a stable role embedding derived from PlayingPosition."
            ),
        },
    }


def write_markdown(result: dict[str, object], path: Path) -> None:
    lines = [
        "# Single-team BC label and role audit",
        "",
        f"Focal team: `{result['team_id']}`",
        "",
        "## Contract checks",
        "",
    ]
    for name, passed in result["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend([
        "",
        "## Split summary",
        "",
        "| Split | Matches | Chunks | Player samples | Change rate | Event overrides | Sides |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for split, values in result["splits"].items():
        sides = ", ".join(f"{key}:{value}" for key, value in values["sides_by_transition"].items())
        lines.append(
            f"| {split} | {len(values['matches'])} | {values['chunks']} | "
            f"{values['focal_player_samples']:,} | {values['action_change_rate']:.4f} | "
            f"{values['event_override_targets']} | {sides} |"
        )
    lines.extend([
        "",
        "## Action support",
        "",
        "| ID | Action | Train | Validation | Test |",
        "|---:|---|---:|---:|---:|",
    ])
    for action, values in result["action_support"].items():
        lines.append(
            f"| {action} | {values['name']} | {values['train']:,} | "
            f"{values['validation']:,} | {values['test']:,} |"
        )
    role = result["role_audit"]
    lines.extend([
        "",
        "## Role finding",
        "",
        role["semantics"],
        "",
        f"- Players changing stored slot within a match: {len(role['players_changing_slot_within_match'])}",
        f"- Slots spanning multiple PlayingPosition values: {', '.join(role['slots_with_multiple_playing_positions'])}",
        f"- Correction: {role['recommended_correction']}",
        "",
        "## Decision",
        "",
        "The match split and side coverage are usable. Run full BC only after removing the "
        "misnamed roster-slot feature. Keep rare/unsupported action results diagnostic and "
        "select the final checkpoint from validation, not test.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--dataset-a", type=Path, default=Path("data/raw/dataset_a/datasetA.zip")
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.manifest, args.dataset_a)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "team_bc_data_audit.json"
    report_path = args.output_dir / "TEAM_BC_DATA_AUDIT.md"
    json_path.write_text(json.dumps(result, indent=2))
    write_markdown(result, report_path)
    print(json.dumps({"audit": str(json_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
