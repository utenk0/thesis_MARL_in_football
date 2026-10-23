# Implementation Plan

Current implementation/training audit: [TRAINING_STAGE_STATUS.md](TRAINING_STAGE_STATUS.md).
The stage descriptions below are the original roadmap; the report distinguishes
completed initial training from the remaining full-scale thesis experiments.

Thesis theme:

```text
A Comparative Analysis of Multi-Agent Reinforcement Learning Methods
for Data-Driven Football Player Simulation
```

Core idea:

```text
real football tracking/event data
-> cleaned project structure
-> unified dataset format
-> GRF coordinate/action conversion
-> replay validation in Google Research Football
-> learning-ready transition datasets
-> selected MARL methods
-> comparative evaluation
```

The plan below is written as staged work. Each stage should be completed,
checked, and documented before moving to the next one.

## Stage 0: Folder Preparation And Project Cleanup

Goal: make the repository understandable before implementing more logic.

Important rule: do not delete files immediately. First classify them, confirm
whether they are used, then move or remove them in a controlled cleanup commit.

### 0.1 Create A Project Inventory

List every top-level file and folder.

Current important folders appear to be:

```text
football/              vendored Google Research Football source
thesis_experiments/    project-owned thesis implementation
data/                  dataset metadata and later raw/processed data
thesis_plan/           implementation planning documents
```

Current top-level loose files should be reviewed:

```text
main.py
GRF_SETUP.md
README.md
requirements-grf.txt
football_loop_snippet.py
build_dataset_a_batch.py
build_dataset_a_frame_jsonl.py
extract_dataset_a_positions.py
sort_positions_by_frame.py
```

For each file, record:

```text
file path
purpose
is it imported by project code?
is it a one-time helper script?
is it documented in README?
keep / move / archive / remove
reason
```

Deliverable:

```text
thesis_plan/PROJECT_INVENTORY.md
```

### 0.2 Define The Target Folder Structure

Use a stable structure before adding more data and experiments.

Recommended structure:

```text
.
├── README.md
├── GRF_SETUP.md
├── requirements-grf.txt
├── thesis_plan/
│   ├── IMPLEMENTATION_PLAN.md
│   └── PROJECT_INVENTORY.md
├── thesis_experiments/
│   ├── data/
│   ├── datasets/
│   ├── events/
│   ├── transitions/
│   ├── simulation/
│   ├── policies/
│   ├── training/
│   ├── scripts/
│   └── tests/
├── data/
│   ├── dataset_sources.yaml
│   ├── raw/
│   │   ├── metrica_sample/
│   │   └── dataset_a/
│   └── processed/
├── artifacts/
│   ├── audits/
│   ├── replays/
│   ├── transitions/
│   ├── checkpoints/
│   └── reports/
└── football/
```

### 0.3 Classify Files

Use these categories:

```text
KEEP_ROOT
  Files that belong at repository root.

MOVE_TO_SCRIPTS
  Useful scripts that should live under thesis_experiments/scripts/.

MOVE_TO_DOCS
  Notes or examples that should become documentation.

ARCHIVE
  Old exploratory files that might still be useful but should not clutter root.

REMOVE_CANDIDATE
  Files that are unused, duplicated, or misleading.
```

Suggested initial classification:

```text
KEEP_ROOT
  README.md
  GRF_SETUP.md
  requirements-grf.txt
  main.py

MOVE_TO_DOCS
  football_loop_snippet.py

REVIEW_SCRIPT
  build_dataset_a_batch.py
  build_dataset_a_frame_jsonl.py
  extract_dataset_a_positions.py
  sort_positions_by_frame.py

```

### 0.4 Check Whether Files Are Used

Before moving or removing anything, search references.

Example commands:

```bash
rg "build_dataset_a_batch|build_dataset_a_frame_jsonl|extract_dataset_a_positions|sort_positions_by_frame"
rg "football_loop_snippet"
```

If a file has no references and is not documented, it is probably an archive or
remove candidate.

### 0.5 Create An Archive Folder If Needed

If unsure whether a helper script is still useful, move it to:

```text
archive/
```

Recommended archive structure:

```text
archive/
└── exploratory_scripts/
```

Only delete files after the friend confirms they are not needed.

### 0.6 Clean Generated Outputs

Generated outputs should not be mixed with source code.

Expected generated folders:

```text
artifacts/
data/processed/
football/gfootball.egg-info/
football/gfootball_engine/
football/third_party/gfootball_engine/CMakeFiles/
football/third_party/gfootball_engine/CMakeCache.txt
football/third_party/gfootball_engine/Makefile
```

Before deleting generated files, check `.gitignore`. If these files are not
ignored, update `.gitignore` first.

### 0.7 Update README After Cleanup

After cleanup, the root README should answer:

```text
what the project does
how to install GRF
where raw data goes
how to run a 10-step GRF smoke test
how to run a 100-frame replay
how to build transitions
where experiment outputs are saved
```

Deliverable for Stage 0:

```text
clean repository layout
PROJECT_INVENTORY.md
updated README if paths changed
no accidental deletion of useful files
```

## Stage 1: Environment Setup

Goal: make Google Research Football and the local project runnable.

### 1.1 Create Separate Virtual Environments

Use separate project-local environments:

```bash
python3.11 -m venv .venv-grf
python3.11 -m venv .venv-methods
```

Reason:

```text
.venv-grf keeps Google Research Football on gym 0.21.
.venv-methods keeps Gymnasium, SB3, imitation, d3rlpy, and PyTorch together.
d3rlpy requires gym>=0.26, which is incompatible with vendored GRF.
```

### 1.2 Install Pinned Python Packaging Tools

GRF depends on old Gym metadata, so keep pip pinned in `.venv-grf`.

```bash
.venv-grf/bin/python -m pip install pip==24.0 setuptools==65.5.0 wheel==0.38.4
```

Do not upgrade pip in `.venv-grf` unless GRF compatibility is re-tested.

### 1.3 Install GRF Python Requirements

```bash
.venv-grf/bin/python -m pip install -r requirements-grf.txt
.venv-grf/bin/python -m pip install -r football/requirements.txt
.venv-grf/bin/python -m pip install -r requirements-project.txt
```

### 1.4 Install System Build Dependencies

On macOS, GRF needs CMake, SDL2, Boost, and Boost.Python.

```bash
brew install cmake sdl2 sdl2_image sdl2_ttf sdl2_gfx boost boost-python3
```

### 1.5 Build And Install Local GRF

```bash
GFOOTBALL_USE_PREBUILT_SO=1 .venv-grf/bin/python -m pip install -e football --no-deps
```

### 1.6 Smoke Test

```bash
.venv-grf/bin/python -m thesis_experiments.scripts.check_environment --target grf
.venv-grf/bin/python main.py --steps 10
```

Expected result:

```text
GRF environment resets
observation shape is printed
several random actions are executed
episode does not crash
```

Deliverable:

```text
working local GRF installation
documented setup result
```

### 1.7 Install Method Environment

Install method dependencies only in `.venv-methods`:

```bash
.venv-methods/bin/python -m pip install -r requirements-project.txt
.venv-methods/bin/python -m pip install -r requirements-methods.txt
.venv-methods/bin/python -m thesis_experiments.scripts.check_environment --target methods
```

Deliverable:

```text
Gymnasium method environment independent from the GRF environment
```

## Stage 2: Dataset Acquisition

Goal: place all required raw datasets in a consistent location.

### 2.1 Metrica Sample Data

Source:

```text
https://github.com/metrica-sports/sample-data
```

Recommended local path:

```text
data/raw/metrica_sample/
```

Metrica is useful for:

```text
early parser testing
coordinate conversion debugging
small replay validation
prototype experiments
```

### 2.2 Dataset A (BASSEK) Data

Source:

```text
https://doi.org/10.6084/m9.figshare.28196177
```

Recommended local path:

```text
data/raw/dataset_a/
```

Dataset A is useful for:

```text
main thesis experiments
larger-scale movement data
official event data
professional match trajectories
```

### 2.3 License And Citation Notes

Create:

```text
thesis_plan/DATASET_NOTES.md
```

Include:

```text
dataset name
source URL
license
citation requirements
number of matches
tracking format
event format
coordinate system
```

Deliverable:

```text
raw datasets downloaded and documented
```

## Stage 3: Dataset Understanding And Audit

Goal: understand the data before writing conversion logic.

### 3.1 Inspect Metrica Files

Check:

```text
tracking file names
event file names
home/away team columns
player ID format
ball columns
frame/timestamp columns
coordinate ranges
missing values
```

### 3.2 Inspect Dataset A Files

Check:

```text
match metadata
event files
position files
tracking provider format
sampling frequency
team identifiers
player identifiers
ball availability
coordinate origin
coordinate units
```

### 3.3 Produce Dataset Audit Report

Create:

```text
artifacts/reports/dataset_audit.md
```

Include:

```text
number of matches
number of frames
number of valid 22-player frames
event count
missing player frames
missing ball frames
coordinate min/max
known limitations
```

Deliverable:

```text
clear knowledge of what data is available and usable
```

## Stage 4: Unified Data Schema

Goal: convert both datasets into one common internal representation.

### 4.1 Define Frame Schema

Each normalized frame should contain:

```text
source
match_id
frame_id
timestamp
left_team_id
right_team_id
left_players[11]
right_players[11]
ball
events_at_frame
```

Each player should contain:

```text
player_id
x
y
optional velocity_x
optional velocity_y
```

### 4.2 Define Event Schema

Each normalized event should contain:

```text
source
match_id
event_id
frame_id
timestamp
team_id
player_id
event_type
subtype
x
y
raw
```

### 4.3 Implement Dataset Parsers

Metrica parser:

```text
raw Metrica CSV/JSON
-> normalized frames
-> normalized events
```

Dataset A parser:

```text
raw Dataset A metadata/events/positions
-> normalized frames
-> normalized events
```

### 4.4 Validate The Unified Schema

Checks:

```text
exactly 11 left players and 11 right players when frame is marked valid
finite x/y positions
ball position present or explicitly missing
monotonic frame IDs
stable player IDs within each match segment
events can be joined to frame IDs
```

Deliverable:

```text
dataset-independent frames and events
```

## Stage 5: Coordinate Conversion To GRF

Goal: map real match coordinates into Google Research Football coordinates.

### 5.1 Define Coordinate Systems

Metrica:

```text
x, y in [0, 1]
origin at top-left
pitch dimensions 105m x 68m
```

Dataset A:

```text
centred metre coordinates on a 105m x 68m pitch
confirm provider metadata and orientation during audit
```

GRF:

```text
x approximately in [-1, 1]
y approximately in [-0.42, 0.42]
centered at midfield
```

### 5.2 Convert Metrica To GRF

Steps:

```text
normalize Metrica coordinates
center coordinates around midfield
scale x to GRF field length
scale y to GRF field width
handle attacking direction
convert ball coordinates
```

### 5.3 Convert Dataset A To GRF

Steps:

```text
confirm Dataset A coordinate units
center coordinates around midfield
scale to GRF
handle team direction
convert ball coordinates
```

### 5.4 Test Coordinate Conversion

Tests:

```text
center spot maps near (0, 0)
left goal maps near negative x
right goal maps near positive x
all players stay inside expected GRF bounds
team mirroring is consistent
```

Deliverable:

```text
real tracking data converted into GRF coordinate space
```

## Stage 6: Action Extraction

Goal: infer approximate GRF discrete actions from real movement and events.

### 6.1 Movement-Based Actions

For each player:

```text
position at frame t
position at frame t+1
delta x/y
movement angle
movement speed
GRF direction action
```

Possible movement actions:

```text
idle
left
top_left
top
top_right
right
bottom_right
bottom
bottom_left
```

### 6.2 Speed-Based Actions

Use speed thresholds to infer:

```text
sprint
release_sprint
```

Thresholds must be documented and tuned on validation clips.

### 6.3 Event-Based Actions

Use event labels to infer:

```text
short_pass
high_pass
shot
dribble
slide
```

Important limitation:

```text
tracking data gives movement directly, but technical football actions such as
pass or shot require event data and are approximate.
```

### 6.4 Validate Action Distribution

Produce counts:

```text
action id
action name
count
percentage
```

Watch for suspicious results:

```text
too many idle actions
almost no passes
almost no sprint actions
impossible action values
```

Deliverable:

```text
one inferred GRF action per player per frame transition
```

## Stage 7: GRF Replay Validation

Goal: prove that converted real trajectories can be placed and followed inside
Google Research Football.

### 7.1 Build Custom Replay Scenarios

Each replay scenario should:

```text
read initial state
set game duration
place ball
place left-team players
place right-team players
disable unnecessary randomness
```

### 7.2 Run 100-Frame Metrica Replay

Command shape:

```bash
.venv-grf/bin/python -m thesis_experiments.scripts.track_trajectory \
  --dataset metrica \
  --input data/raw/metrica_sample/data/Sample_Game_1 \
  --output-dir artifacts/replays/metrica_100 \
  --start-frame 1 \
  --frames 100 \
  --live \
  --record \
  --fps 10
```

### 7.3 Run 100-Frame Dataset A Replay

Command shape:

```bash
.venv-grf/bin/python -m thesis_experiments.scripts.track_trajectory \
  --dataset dataset_a \
  --input data/processed/dataset_a_frames_MATCH.jsonl \
  --output-dir artifacts/replays/dataset_a_100 \
  --start-frame 10000 \
  --frames 100 \
  --live \
  --record \
  --fps 10
```

The exact Dataset A input path depends on the parser output.

### 7.4 Save Replay Metrics

Metrics:

```text
mean position error
median position error
p95 position error
max position error
per-player error
action counts
number of executed frames
video path
```

Deliverable:

```text
replay report proving that data-to-GRF conversion is usable
```

## Stage 8: Learning Transition Dataset

Goal: create files that learning methods can use.

### 8.1 Define Transition Format

Each transition:

```text
local_observations_t
global_state_t
joint_actions_t
team_rewards_t
local_observations_t+1
global_state_t+1
done
frame_id
```

Expected array shapes:

```text
local_observations:      [T, 22, L]
global_states:           [T, G]
actions:                 [T, 22]
team_rewards:            [T, 2]
next_local_observations: [T, 22, L]
next_global_states:      [T, G]
dones:                   [T]
frames:                  [T]
```

### 8.2 Define Local Observation

Possible player-local features:

```text
own position
ball relative position
ball absolute position
team side
player role/index
nearest teammates
nearest opponents
```

### 8.3 Define Global State

Possible global features:

```text
all 22 player positions
ball position
optional game phase features
optional possession/event features
```

### 8.4 Define Reward

Initial simple reward:

```text
left reward = ball x-progress
right reward = negative ball x-progress
goal event = large positive/negative reward
```

This is not perfect, but it is explainable and easy to compare.

### 8.5 Build Dataset Files

Example output:

```text
data/processed/metrica_ctde.npz
data/processed/dataset_a_ctde.npz
```

Deliverable:

```text
learning-ready transition datasets
```

## Stage 9: Unified Gymnasium Method Framework

Goal: run every selected method through one project interface, using the same
data contract and Gymnasium-compatible replay environment.

The selected comparison now contains seven method entries:

```text
1. BC        -> imitation backend, with local PyTorch fallback
2. GAIL      -> imitation adversarial trainer + stable-baselines3 PPO generator
3. PPO       -> stable-baselines3 PPO over the Gymnasium replay environment
4. CQL       -> d3rlpy discrete CQL over flattened offline player transitions
5. DT        -> external Decision Transformer backend later, placeholder now
6. Ghosting  -> small PyTorch next-position baseline
7. CTDE      -> existing centralized-critic/shared-actor baseline
```

The GitHub repositories listed during planning are useful as reference
implementations, but source code should not be copied into this project until
license compatibility and attribution are reviewed. Prefer maintained library
APIs for the thesis framework:

```text
Gymnasium              environment interface
stable-baselines3      PPO
imitation              BC and GAIL
d3rlpy                 CQL
project PyTorch code   Ghosting and CTDE
external backend later Decision Transformer
```

Repository triage note:

```text
alexstaravoitau/behavioral-cloning   relevant as a BC reference only
nikhilbarhate99/PPO-PyTorch          relevant as a PPO reference only
aviralkumar2907/CQL                  relevant as the original CQL codebase
kzl/decision-transformer             relevant as the official DT codebase
GailGithub/GAIL_Dev                  not adversarial imitation learning GAIL
hasherezade/process_ghosting         not sports/player ghosting
```

### 9.1 Data Layer

Keep `JointTransitionDataset` as the method input contract:

```text
local_observations:      [T, 22, L]
global_states:           [T, G]
actions:                 [T, 22]
team_rewards:            [T, 2]
next_local_observations: [T, 22, L]
next_global_states:      [T, G]
dones:                   [T]
frames:                  [T]
```

For single-agent libraries, flatten the joint data into player transitions:

```text
observations:      [T * 22, L]
actions:           [T * 22] in GRF Discrete(19)
rewards:           [T * 22]
next_observations: [T * 22, L]
terminals:         [T * 22]
```

This keeps every method tied to the same cleaned football data.

### 9.2 Simulation Layer

Use Gymnasium for the learning-facing replay environment:

```text
FootballReplayEnv
  observation_space = Box(local_observation_dim)
  action_space      = Discrete(19)
  step(action)      = returns next recorded observation
```

The environment is teacher-forced: actions do not change the recorded next
state. It is used to make PPO/GAIL and other library APIs runnable against the
same data. Google Research Football remains the separate validation simulator
for closed-loop replay and rollout inspection.

### 9.3 RL Method Layer

Define a central interface:

```text
FootballMethod.train()    -> MethodResult
FootballMethod.evaluate() -> metric dictionary
FootballMethod.imitate()  -> predicted GRF actions
```

All methods are created through:

```text
MethodConfig
build_method(config)
```

The command-line entry point is:

```bash
python -m thesis_experiments.scripts.run_method \
  --method bc \
  --data data/processed/dataset_a_ctde_100.npz data/processed/metrica_ctde_100.npz \
  --output-dir artifacts/methods \
  --backend torch
```

### 9.4 Method-Specific Notes

BC:

```text
Primary target backend is imitation.algorithms.bc.
The existing PyTorch trainer is retained as a fallback and smoke-test baseline.
```

GAIL:

```text
Use imitation.algorithms.adversarial.gail with an SB3 PPO generator.
Expert demonstrations are created from flattened player transitions.
```

PPO:

```text
Use stable-baselines3 PPO on FootballReplayEnv.
This is a replay-environment baseline, not proof of true online football control.
```

CQL:

```text
Use d3rlpy discrete CQL on the same flattened player transitions.
```

Decision Transformer:

```text
Keep a placeholder interface and manifest until the backend is selected.
The future backend must consume sequence-packed football trajectories.
```

Ghosting:

```text
Train a small local-observation -> next-position model.
Convert predicted movement back to GRF directional actions for comparison.
```

CTDE:

```text
Keep the existing centralized critic and shared actor as the seventh method.
It uses global state during training and local observations during execution.
```

Deliverable:

```text
common method registry
Gymnasium replay environment
run_method command
method artifacts under artifacts/methods/
clear result JSON for every method
```

Current status on 2026-09-15:

```text
Stage 9 smoke implementation is complete.
Built Dataset A and Metrica 100-frame transition .npz files.
Installed the optional method stack in the local method environment.
Created .venv-grf for live GRF replay and .venv-methods for method training.
Ran BC, GAIL, PPO, CQL, DT placeholder, Ghosting, and CTDE.
Saved method artifacts and result JSON under artifacts/methods/.
Recorded smoke metrics in thesis_plan/METHOD_SMOKE_RESULTS.md.
Recorded the final Stage 9 environment conclusion in
thesis_plan/STAGE_9_CONCLUSION.md.
```

Important environment note:

```text
d3rlpy requires gym>=0.26, while vendored GRF expects gym<=0.21.
Keep live GRF validation in .venv-grf and method training in .venv-methods.
Both environments pass thesis_experiments.scripts.check_environment.
PPO and GAIL directly use Gymnasium through FootballReplayEnv; BC uses the
imitation backend with Gymnasium spaces for offline demonstrations; CQL,
Ghosting, CTDE, and the Decision Transformer placeholder do not directly use
Gymnasium.
```

## Stage 10: Evaluation Design

Goal: compare methods fairly.

Current status:

```text
Stage 10 is now the next active stage.
The smoke metrics prove the framework runs; they are not final thesis results.
```

### 10.1 Split Data

Preferred split:

```text
train matches
validation match
test match
```

Avoid testing on the same exact frames used for training.

### 10.2 Offline Metrics

Use metrics that do not require running GRF:

```text
action prediction accuracy
cross-entropy loss
trajectory next-step error
action distribution similarity
team width/depth error
average distance-to-ball error
nearest teammate distance error
nearest opponent distance error
```

### 10.3 Simulation Metrics

Use metrics from GRF rollouts:

```text
episode reward
score reward
ball progression
simulation stability
position realism
team compactness
action entropy
video inspection
```

### 10.4 Qualitative Evaluation

Save:

```text
replay videos
policy rollout videos
example trajectory plots
formation plots
failure case screenshots
```

Deliverable:

```text
evaluation protocol and result tables
```

## Stage 11: Results And Thesis Writing

Goal: turn implementation into thesis evidence.

### 11.1 Required Result Tables

Tables:

```text
dataset statistics
conversion/replay validation metrics
model comparison metrics
training hyperparameters
runtime/compute summary
```

### 11.2 Required Plots

Plots:

```text
training curves
action distribution comparison
trajectory error over time
team shape metrics over time
example player trajectories
```

### 11.3 Thesis Chapter Structure

Suggested chapters:

```text
1. Introduction
2. Background
3. Datasets
4. Methodology
5. Implementation
6. Experiments
7. Results
8. Discussion
9. Conclusion
```

Deliverable:

```text
final thesis-ready experimental evidence
```

## Minimum Viable Completion Path

If the schedule becomes tight, complete only this:

```text
1. Clean project folder structure.
2. Finish environment setup.
3. Obtain Metrica and Dataset A (BASSEK).
4. Convert both datasets to a common schema.
5. Convert coordinates to GRF.
6. Validate 100-frame replay for both datasets.
7. Build transition datasets.
8. Train Behavioral Cloning.
9. Train one selected MARL/CTDE method.
10. Compare both methods with offline and simulation metrics.
```

This is enough for a focused, defensible implementation.
