# Unified Frame and Event Schema

Stage 4 separates provider parsing from GRF coordinate conversion. Coordinates
in this schema remain in their provider-native systems; Stage 5 performs the
GRF mapping.

## Frame

`UnifiedFrame` contains:

```text
source, match_id, frame_id, timestamp
left_team_id, right_team_id
left_players, right_players
ball, events_at_frame
valid, segment_id, raw
```

A player contains `player_id`, finite `x` and `y`, and optional paired velocity
components. The ball is either a finite position or `null`. A frame marked
`valid` must contain exactly 11 unique players for each team. `segment_id`
changes whenever the active roster changes, allowing stable player IDs to be
enforced within a contiguous lineup segment.

Left/right are stable logical slots in the normalized record. For Metrica CSV
they correspond to Home/Away; they do not claim that the team physically
occupies that side of the pitch in every period.

## Event

`UnifiedEvent` contains the required provider-independent fields:

```text
source, match_id, event_id, frame_id, timestamp
team_id, player_id, event_type, subtype
x, y, raw
```

It also retains end-frame, recipient, outcome, and GRF action/conversion fields
used by later existing code. The `x`/`y` fields are provider-native. An event
embedded in a frame must have the same `frame_id` as that frame.

## Parsers

- Metrica Games 1/2: synchronized home/away tracking CSV and event CSV.
- Metrica Game 3: FIFA EPTS metadata, tracking text, and JSON events.
- Dataset A: provider frame JSONL created by the raw XML extraction helpers.

Metrica Game 3 tracking rows do not identify players per position. Its parser
therefore uses stable anonymized team-slot IDs instead of incorrectly assigning
roster identities. Event actor IDs remain preserved in each event's raw data.

## Validation

`FrameStreamValidator` checks schema validity, monotonic frame IDs, exactly 11
players per team for valid frames, finite positions, explicit missing balls,
stable rosters within segments, and frame-aligned events. The
`build_unified_data` command validates while streaming so full matches do not
need to be held in memory.

## Method Input

Learning methods do not consume raw provider frames directly. They consume the
Stage 8 `JointTransitionDataset` `.npz` files created by `build_transitions`:

```text
local_observations       [T, 22, L]
global_states            [T, G]
actions                  [T, 22]
team_rewards             [T, 2]
next_local_observations  [T, 22, L]
next_global_states       [T, G]
dones                    [T]
frames                   [T]
```

The unified method framework exposes two views of this input:

```text
joint view      used by CTDE and project-owned multi-agent logic
player view     flattened to [T * 22] for Gymnasium, imitation, SB3, and d3rlpy
```

The player view uses each player's local observation, one GRF `Discrete(19)`
action, that player's team reward, the next local observation, and a terminal
flag copied from the source transition. This keeps all seven methods comparable
even when their third-party libraries expect single-agent data.
