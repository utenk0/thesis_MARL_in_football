"""Convert Metrica and Dataset A events into :class:`UnifiedEvent`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from thesis_experiments.datasets.action_mapping import (
    dataset_a_event_to_grf_action,
    metrica_event_to_grf_action,
)
from thesis_experiments.datasets.coordinates import dataset_a_to_grf, metrica_to_grf
from thesis_experiments.events.schema import UnifiedEvent


def convert_metrica_events(game_dir: Path, *, start_frame: int | None = None, end_frame: int | None = None) -> list[UnifiedEvent]:
    """Load one Metrica game CSV and return normalized events."""
    path = game_dir / f"{game_dir.name}_RawEventsData.csv"
    rows = pd.read_csv(path)
    output = []
    for index, row in rows.iterrows():
        frame = _int(row.get("Start Frame"))
        if start_frame is not None and (frame is None or frame < start_frame):
            continue
        if end_frame is not None and (frame is None or frame > end_frame):
            continue
        output.append(metrica_event_to_unified(row.to_dict(), match_id=game_dir.name, fallback_id=index))
    return output


def metrica_event_to_unified(raw: dict[str, Any], *, match_id: str, fallback_id: int | str = 0) -> UnifiedEvent:
    source_type = _text(raw.get("Type")) or ""
    subtype = _text(raw.get("Subtype"))
    event_type = _metrica_type(source_type, subtype)
    action = metrica_event_to_grf_action(source_type, subtype)
    if event_type == "PENALTY":
        action = 12
    if event_type in {"OTHER", "FOUL", "OFFSIDE", "CHALLENGE", "PERIOD_START", "PERIOD_END"}:
        action = None
    start = _metrica_position(raw.get("Start X"), raw.get("Start Y"))
    end = _metrica_position(raw.get("End X"), raw.get("End Y"))
    event = UnifiedEvent(
        source="metrica", match_id=match_id,
        event_id=_text(raw.get("Event ID")) or f"{match_id}_{fallback_id}",
        event_type=event_type, source_event_type=source_type,
        frame=_int(raw.get("Start Frame")), end_frame=_int(raw.get("End Frame")),
        period=_text(raw.get("Period")), timestamp_seconds=_float(raw.get("Start Time [s]")),
        subtype=subtype, team_id=_text(raw.get("Team")), player_id=_text(raw.get("From")),
        recipient_id=_text(raw.get("To")), start_grf=start, end_grf=end,
        outcome=_metrica_outcome(source_type, subtype), grf_action=action,
        raw=_json_safe(raw),
    )
    event.validate()
    return event


def convert_dataset_a_events(path: Path, *, start_frame: int | None = None, end_frame: int | None = None) -> list[UnifiedEvent]:
    """Stream embedded Dataset A events from one frame JSONL file."""
    output, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            frame = int(record["frame"])
            if start_frame is not None and frame < start_frame:
                continue
            if end_frame is not None and frame > end_frame:
                break
            for raw in record.get("context", {}).get("events") or []:
                # Dataset A uses Delete records to invalidate earlier provider
                # annotations; they are not football events or training labels.
                if raw.get("event_type") == "Delete":
                    continue
                event_id = str(raw.get("event_id") or f"{record['match_id']}_{frame}_{len(output)}")
                if event_id in seen:
                    continue
                seen.add(event_id)
                output.append(dataset_a_event_to_unified(raw, match_id=str(record["match_id"]), fallback_frame=frame))
    return output


def dataset_a_event_to_unified(raw: dict[str, Any], *, match_id: str, fallback_frame: int | None = None) -> UnifiedEvent:
    source_type = str(raw.get("event_type") or "")
    details = raw.get("details") or {}
    event_type = _dataset_a_type(source_type, details)
    event = UnifiedEvent(
        source="dataset_a", match_id=match_id,
        event_id=str(raw.get("event_id") or f"{match_id}_{fallback_frame}"),
        event_type=event_type, source_event_type=source_type,
        frame=_int(raw.get("anchor_frame")) or fallback_frame,
        end_frame=_int(raw.get("end_frame")), period=_text(details.get("GameSection")),
        timestamp=_text(raw.get("event_time")), subtype=_dataset_a_subtype(details),
        team_id=_first(details, "Team", "TeamLeft", "WinnerTeam", "LoserTeam"),
        player_id=_first(details, "Player", "Winner", "Loser"),
        recipient_id=_text(details.get("Recipient")),
        start_grf=_dataset_a_event_position(raw.get("x_position"), raw.get("y_position")),
        outcome=_normalize_outcome(_first(details, "Evaluation", "WinnerResult", "ChanceEvaluation")),
        grf_action=(12 if event_type == "PENALTY" else dataset_a_event_to_grf_action(source_type, details)), raw=_json_safe(raw),
    )
    event.validate()
    return event


def write_events_jsonl(events: Iterable[UnifiedEvent], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def _metrica_type(event: str, subtype: str | None) -> str:
    event_upper, sub = event.upper(), (subtype or "").upper()
    if "GOAL" in sub and "GOAL KICK" not in sub:
        return "GOAL"
    if event_upper == "PASS": return "PASS"
    if event_upper == "SHOT": return "SHOT"
    if event_upper == "CHALLENGE": return "TACKLE" if "TACKLE" in sub else "CHALLENGE"
    if event_upper == "BALL LOST": return "BALL_LOSS"
    if event_upper == "RECOVERY": return "BALL_RECOVERY"
    if event_upper == "FAULT RECEIVED": return "FOUL"
    if event_upper == "SET PIECE":
        if "PENALTY" in sub: return "PENALTY"
        if "CORNER" in sub: return "CORNER"
        if "GOAL KICK" in sub: return "GOAL_KICK"
        if "FREE KICK" in sub: return "FREE_KICK"
        if "THROW" in sub: return "THROW_IN"
        if "KICK OFF" in sub: return "KICK_OFF"
    return "OTHER"


def _dataset_a_type(event: str, details: dict[str, Any]) -> str:
    mapping = {"Play": "PASS", "ShotAtGoal": "SHOT", "CornerKick": "CORNER", "FreeKick": "FREE_KICK", "Penalty": "PENALTY", "GoalKick": "GOAL_KICK", "ThrowIn": "THROW_IN", "KickOff": "KICK_OFF", "TacklingGame": "TACKLE", "BallClaiming": "BALL_CLAIM", "OtherBallAction": "DRIBBLE", "PossessionLossBeforeGoal": "BALL_LOSS", "Offside": "OFFSIDE", "Foul": "FOUL", "FinalWhistle": "PERIOD_END"}
    if event == "ShotAtGoal" and "goal" in str(details.get("Evaluation", "")).lower():
        return "GOAL"
    return mapping.get(event, "OTHER")


def _metrica_position(x, y):
    values = (_float(x), _float(y))
    if None in values: return None
    converted = metrica_to_grf(*values)
    return float(converted[0]), float(converted[1])


def _dataset_a_event_position(x, y):
    values = (_float(x), _float(y))
    if None in values: return None
    # Dataset A tracking is centred metres, but event positions are offset to
    # [0,105] x [0,68]. Midfield is therefore (52.5,34).
    converted = dataset_a_to_grf(values[0] - 52.5, values[1] - 34.0)
    return float(converted[0]), float(converted[1])


def _metrica_outcome(event, subtype):
    text = f"{event} {subtype or ''}".upper()
    if "INCOMPLETE" in text or "LOST" in text or "OFF TARGET" in text: return "failure"
    if "COMPLETE" in text or "GOAL" in text or "ON TARGET" in text: return "success"
    return None


def _normalize_outcome(value):
    if value is None: return None
    text = str(value).lower()
    if text in {"successful", "success", "won", "goal"}: return "success"
    if text in {"unsuccessful", "failure", "failed", "lost"}: return "failure"
    return text


def _dataset_a_subtype(details):
    values = [_text(details.get(key)) for key in ("Type", "Height", "ExecutionMode", "Distance")]
    return ":".join(value for value in values if value) or None


def _first(mapping, *keys):
    for key in keys:
        value = _text(mapping.get(key))
        if value: return value
    return None


def _text(value):
    if value is None or (isinstance(value, float) and np.isnan(value)): return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def _float(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError): return None


def _int(value):
    number = _float(value)
    return int(number) if number is not None else None


def _json_safe(value):
    if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and np.isnan(value): return None
    return value
