# Project Inventory

This inventory records the Stage 0 repository audit. Reference searches were
performed before any files were moved. No project file was deleted.

## Top-level inventory

| Path | Purpose | Imported or referenced? | Documentation | Classification | Decision and reason |
|---|---|---|---|---|---|
| `.git/` | Git metadata | Used by Git | N/A | KEEP_ROOT | Required repository metadata. |
| `.gitignore` | Excludes environments, data, outputs, and generated files | Used by Git | N/A | KEEP_ROOT | Keeps generated and local-only files out of commits. |
| `.venv-grf/`, `.venv-methods/` | Project-local Python environments for GRF and methods | Used by setup/run commands | `README.md`, `GRF_SETUP.md` | GENERATED | Keep locally; ignored by Git. |
| `README.md` | Project overview and run commands | User entry point | Self-documenting | KEEP_ROOT | Primary project documentation. |
| `GRF_SETUP.md` | Local Google Research Football setup | Referenced by `README.md` | `README.md` | KEEP_ROOT | Specialized installation instructions. |
| `requirements-grf.txt` | Legacy-compatible GRF package-tool pins | Used during environment setup | `GRF_SETUP.md` | KEEP_ROOT | Required reproducibility input. |
| `main.py` | Ten-step headless GRF smoke test | Standalone entry point | `README.md`, `GRF_SETUP.md` | KEEP_ROOT | Small, documented repository smoke test. |
| `thesis_plan/` | Implementation plan and repository inventory | Referenced by `README.md` | `README.md` | KEEP_ROOT | Central home for planning documents. |
| `thesis_experiments/` | Thesis-owned datasets, conversion, simulation, MARL, and tests | Main Python package | `README.md` | KEEP_ROOT | Core implementation. |
| `data/` | Dataset source metadata plus ignored raw/processed data | Used by scripts and examples | `README.md` | KEEP_ROOT | Stable project data layout. |
| `artifacts/` | Generated audits, replays, transitions, checkpoints, and reports | Output target when created | `README.md` | GENERATED | Ignored; create subfolders as experiments require. |
| `football/` | Vendored Google Research Football source | Loaded through `thesis_experiments/env.py` | `GRF_SETUP.md` | KEEP_ROOT | Local simulator dependency. |
| `archive/` | Preserved exploratory code not in the active pipeline | Not imported | This inventory | ARCHIVE | Retained for provenance without cluttering root. |
| `__pycache__/` | Generated Python bytecode | Runtime-generated | N/A | GENERATED | Ignored; safe to regenerate. |

## Reviewed loose files

| Original path | Purpose | Imported by project code? | One-time helper? | In README before cleanup? | Classification | Stage 0 action |
|---|---|---:|---:|---:|---|---|
| `main.py` | Headless GRF smoke test | No | No | Yes | KEEP_ROOT | Kept at root. |
| `GRF_SETUP.md` | GRF environment notes | N/A | No | No | KEEP_ROOT | Kept and linked from README. |
| `README.md` | Project documentation | N/A | No | N/A | KEEP_ROOT | Updated for the cleaned layout. |
| `requirements-grf.txt` | Packaging-tool version pins | No | No | No | KEEP_ROOT | Kept at root. |
| `build_dataset_a_batch.py` | Orchestrates Dataset A extraction, sorting, and JSONL conversion | Imports three helper scripts | Yes | No | MOVE_TO_SCRIPTS | Moved to `thesis_experiments/scripts/`; imports changed to package imports. |
| `build_dataset_a_frame_jsonl.py` | Combines sorted tracking frames with Dataset A events | Imported by batch helper | Yes | No | MOVE_TO_SCRIPTS | Moved to `thesis_experiments/scripts/`. |
| `extract_dataset_a_positions.py` | Extracts tracking positions and match metadata from Dataset A XML | Imported by batch helper | Yes | No | MOVE_TO_SCRIPTS | Moved to `thesis_experiments/scripts/`. |
| `sort_positions_by_frame.py` | External/chunked sort for extracted tracking CSV | Imported by batch helper | Yes | No | MOVE_TO_SCRIPTS | Moved to `thesis_experiments/scripts/`. |
| `football_loop_snippet.py` | Incomplete example using undefined legacy wrappers and teams | No | Yes | No | MOVE_TO_DOCS / ARCHIVE | Preserved as `archive/exploratory_scripts/football_loop_snippet.py`. |
| `inspect_statsbomb_open_data.py` | Inspected StatsBomb open-data JSON | No | Yes | No | REMOVE_CANDIDATE | Removed after the thesis owner confirmed that only Dataset A (BASSEK) and Metrica are in scope. |

## Generated-output policy

The root `.gitignore` excludes `.venv*`, `artifacts/`, `outputs/`, Python
caches, egg-info, `data/raw/*`, `data/processed/*`, and a root-level generated
`football/gfootball_engine/`. The vendored `football/.gitignore` already
excludes generated CMake files under `football/third_party/gfootball_engine/`.
Raw and processed data directories retain only tracked `.gitkeep` placeholders.

## Resulting layout

```text
.
├── README.md
├── GRF_SETUP.md
├── requirements-grf.txt
├── main.py
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
├── artifacts/                 # generated when needed
├── archive/exploratory_scripts/ # preserved legacy football example
└── football/
```
