# Local and university container runs

There are two container boundaries:

- `Dockerfile.trajectory` is the current offline trajectory BC/CQL/CTDE
  runtime. It contains Python, NumPy, PyTorch, and project code only.
- `Dockerfile` is the closed-loop 19-action GRF runtime. It contains separate
  legacy Gym/GRF and modern Gymnasium/method environments for BC, GAIL, PPO,
  CQL, DT, ghosting, and CTDE.

Do not use a trajectory service when the experimental claim is interactive
GRF control, and do not use the legacy action services for the frozen
real-world trajectory comparison.

## Accepted trajectory BC

The accepted multi-horizon BC has a separate lightweight image and does not
use the legacy 19-action training command.  Its data and checkpoints remain on
the host: processed data and frozen model directories are mounted read-only,
while only the run-output directory is writable.

Build and reproduce the frozen three-seed check locally with:

```bash
docker compose -f docker/compose.yaml --profile trajectory build trajectory-bc-verify
docker compose -f docker/compose.yaml --profile trajectory run --rm trajectory-bc-verify
```

If Docker Hub is temporarily unavailable but the existing full project image
is already present, the trajectory image can be derived from that local image
without changing the runtime contract:

```bash
THESIS_TRAJECTORY_BASE_IMAGE=thesis-marl-football:local \
THESIS_TRAJECTORY_PYTHON=/workspace/.venv-methods/bin/python \
docker compose -f docker/compose.yaml --profile trajectory build trajectory-bc-verify
```

The report is written to:

```text
artifacts/container_runs/trajectory_bc_docker_portability_v1/portability_report.json
```

On the university server, set `THESIS_DATA_ROOT`,
`THESIS_FROZEN_ARTIFACT_ROOT`, `THESIS_OUTPUT_ROOT`, `THESIS_UID`, and
`THESIS_GID` in `docker/server.env`, then run:

```bash
docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml \
  --profile trajectory build trajectory-bc-verify

docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml \
  --profile trajectory run --rm trajectory-bc-verify
```

This service performs inference only.  It never modifies the frozen
checkpoints and it does not load test data for model selection.

## Trajectory CQL

Trajectory CQL uses the same lightweight image and read-only processed-data
mount as trajectory BC. It writes checkpoints and validation reports only to
the external output mount. Training builds trajectory-plan prototypes from the
train split, selects the Q-network checkpoint on validation ADE, and never
loads the held-out test contents.

Run one default pretest seed locally with:

```bash
docker compose -f docker/compose.yaml \
  --profile trajectory-cql build trajectory-cql-train
docker compose -f docker/compose.yaml \
  --profile trajectory-cql run --rm trajectory-cql-train
```

On the university server, reuse `docker/server.env`:

```bash
docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml \
  --profile trajectory-cql build trajectory-cql-train

docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml \
  --profile trajectory-cql run --rm trajectory-cql-train
```

The defaults can be overridden with `CQL_SEEDS`, `CQL_EPOCHS`,
`CQL_BATCH_SIZE`, `CQL_ACTION_COUNT`, `CQL_ALPHA`, `CQL_GAMMA`,
`CQL_TARGET_UPDATE_INTERVAL`, `CQL_PROTOTYPE_SAMPLES`, and
`CQL_KMEANS_ITERATIONS`. The output is stored under
`artifacts/container_runs/trajectory_cql_pretest_v1` on the host.

## Trajectory CTDE

Trajectory CTDE uses the same lightweight image, read-only processed-data
mount, and external results mount. It additionally mounts the accepted
three-seed BC directory read-only because each CTDE actor starts from the BC
checkpoint with the same seed. The centralized critic is used only during
training; saved-actor validation still uses only local causal observations.

Run one default pretest seed locally with:

```bash
docker compose -f docker/compose.yaml \
  --profile trajectory-ctde build trajectory-ctde-train
docker compose -f docker/compose.yaml \
  --profile trajectory-ctde run --rm trajectory-ctde-train
```

On the university server, reuse `docker/server.env`:

```bash
docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml \
  --profile trajectory-ctde build trajectory-ctde-train

docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml \
  --profile trajectory-ctde run --rm trajectory-ctde-train
```

The defaults can be overridden with `CTDE_SEEDS`, `CTDE_ACTOR_EPOCHS`,
`CTDE_CRITIC_EPOCHS`, `CTDE_BATCH_SIZE`, `CTDE_LEARNING_RATE`,
`CTDE_ACTOR_HIDDEN_SIZE`, `CTDE_CRITIC_HIDDEN_SIZE`, `CTDE_GAMMA`,
`CTDE_TEMPERATURE`, `CTDE_MAX_WEIGHT`, and `CTDE_BC_COEFFICIENT`. Output is
stored under `artifacts/container_runs/trajectory_ctde_pretest_v1` on the
host. The test split is not loaded by this training service.

The full `smoke`, `trial`, and `full` image reproduces the VM's important Linux
boundary with Python 3.11, a legacy Gym/GRF environment, and a separate modern
Gymnasium/method environment. The lightweight trajectory services use their
own Python 3.12 image with only NumPy and PyTorch. Training data is deliberately
excluded from both images. Prepared data is mounted read-only and results are
mounted read-write, so data changes do not rebuild the image and checkpoints
survive container replacement.

The default mounts are:

```text
host data/processed          -> /workspace/data/processed       (read-only)
host artifacts/container_runs -> /workspace/artifacts/vm_runs   (read-write)
```

They can be changed with `THESIS_DATA_ROOT` and `THESIS_OUTPUT_ROOT`. The default is the current
61-feature pilot selected by `final_stage_v1/pilot_split_manifest.json`. Change its directory with
`TRAINING_DATASET` and its manifest filename with `TRAINING_MANIFEST`.

On an Apple Silicon Mac the image deliberately uses `linux/amd64`, matching the
recommended VM. Docker Desktop runs it through architecture emulation, so the
build and GRF simulation are slower than native macOS or an x86 VM. Docker
Desktop does not expose the Mac GPU as CUDA; local container training is CPU
only.

## 1. Start Docker Desktop

Wait until `docker version` shows both a Client and a Server. Give Docker at
least 8 GB RAM and 4 CPUs in Docker Desktop settings. The image is tagged as
`thesis-marl-football:local`. The first build compiles
GRF and downloads the learning libraries, so it can take a while.

The Debian package step uses HTTPS and retries transient mirror failures. If a
build still ends with a connection timeout, rerun the same build command;
completed layers remain cached.

The trajectory image likewise gives pip longer network timeouts and retries.
It does not upgrade pip/setuptools/wheel implicitly, keeping the packaging
toolchain supplied by the pinned base image stable. On a machine that already
has the full project image, the local-base command near the top of this file
avoids downloading a second PyTorch installation.

## 2. Run the smallest complete smoke test

From the repository root:

```bash
docker compose --progress=plain -f docker/compose.yaml build smoke
docker compose -f docker/compose.yaml up \
  --abort-on-container-exit \
  --exit-code-from smoke \
  smoke
```

This runs all seven methods with seed 0, one offline epoch, 704 online/CQL
steps, and 25-step GRF evaluations. It verifies packaging and integration; it
is not a meaningful learning experiment.

For a background run:

```bash
docker compose -f docker/compose.yaml up -d smoke
docker compose -f docker/compose.yaml logs -f smoke
docker compose -f docker/compose.yaml ps
```

Results appear at:

```text
artifacts/container_runs/container_smoke/
```

Success means `training_manifest.json` contains seven `completed` entries and
the container exits with code 0.

## 3. Run the same trial planned for the VM

```bash
docker compose -f docker/compose.yaml --profile trial up -d trial
docker compose -f docker/compose.yaml --profile trial logs -f trial
```

This uses seed 0, five epochs, 20,000 online/CQL steps, and 100-step test
rollouts. Results appear in `artifacts/container_runs/container_trial/`.

## 4. Full profile

The full three-seed profile exists for reproducibility:

```bash
docker compose -f docker/compose.yaml --profile full up -d full
docker compose -f docker/compose.yaml --profile full logs -f full
```

Running it under AMD64 emulation on this Mac will be much slower than the VM.
Use the container smoke/trial to verify the Linux image, then use `vm/` for the
full experiment.

Re-running a profile uses the training suite's resume support. Completed
method/seed pairs are preserved in the mounted result directory. To force a
completely new run, use a new `RUN_NAME` or move the old result directory.

## 5. University workstation deployment

PC12041 is a Docker workstation rather than a Slurm/Apptainer cluster. Do not run Docker cleanup
commands on it: the daemon and its images/volumes may be shared with other users. Wait until the
administrator provides enough free space before building or pulling the image.

First copy the example configuration and edit its three absolute paths plus
the numeric UID/GID:

```bash
cp docker/server.env.example docker/server.env
nano docker/server.env
mkdir -p data/processed/final_stage_v1 artifacts/container_runs
```

For the current trajectory comparison, transfer the fixed focal-team manifest
and its referenced chunks. Select `ctde` to include the frozen BC checkpoints
and protected metadata automatically; select `cql` when they are not needed:

```bash
SYNC_PROCESSED_DATASET=final_stage_v1 \
SYNC_SPLIT_MANIFEST=team_00000P_manifest.json \
SYNC_METHOD=ctde \
bash vm/sync_to_vm.sh pc12041 thesis_MARL_in_football
```

This transfers the 169 referenced NPZ chunks (about 2.2 GiB), not unrelated
processed data or raw provider data. With `SYNC_METHOD=ctde`, the same command
also transfers the three hash-frozen BC initializations. The separate
`sync_trajectory_bc_to_vm.sh` command remains available for a BC-only
portability bundle.

For the closed-loop seven-method GRF pilot instead, select the legacy manifest
explicitly:

```bash
SYNC_PROCESSED_DATASET=final_stage_v1 \
SYNC_SPLIT_MANIFEST=pilot_split_manifest.json \
SYNC_METHOD=legacy \
bash vm/sync_to_vm.sh pc12041 thesis_MARL_in_football
```

That legacy pilot transfers 20 NPZ files (about 264 MiB).

On PC12041, validate the resolved configuration before starting anything:

```bash
cd thesis_MARL_in_football
docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml config
```

Build the shared lightweight trajectory image and verify the frozen BC on CPU:

```bash
docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml \
  --profile trajectory build trajectory-bc-verify
docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml \
  --profile trajectory run --rm trajectory-bc-verify
```

Then run one CQL or CTDE pretest seed with the commands above. For the legacy
closed-loop track, build and run its CPU smoke test separately:

```bash
docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml build smoke
docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml up \
  --abort-on-container-exit --exit-code-from smoke smoke
```

The server override runs the container as the calling user's UID/GID, preventing root-owned result
files. The launcher also checks that the split manifest exists and that the output filesystem has
the configured minimum free space before training begins.

After NVIDIA container support has been confirmed, add the GPU override. It
exposes the GPU and selects CUDA for trajectory CQL/CTDE as well as the legacy
`trial` and `full` profiles. Each launcher probes PyTorch CUDA before loading
the dataset and fails clearly rather than silently using CPU:

```bash
docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml -f docker/compose.gpu.yaml \
  --profile trajectory-cql run --rm trajectory-cql-train

docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml -f docker/compose.gpu.yaml \
  --profile trajectory-ctde run --rm trajectory-ctde-train
```

The legacy CUDA trial remains:

```bash
docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml -f docker/compose.gpu.yaml \
  --profile trial up --abort-on-container-exit --exit-code-from trial trial
```

If Docker cannot provide the GPU, omit `docker/compose.gpu.yaml`; the experiment will remain fully
functional on CPU. If CUDA is requested but unavailable to PyTorch, the launcher exits before
training instead of silently falling back to CPU.

For a larger **legacy closed-loop** experiment, keep
`TRAINING_DATASET=final_stage_v1` and change
`TRAINING_MANIFEST=split_manifest.json`. Then sync that manifest intentionally.
It references about 3.5 GiB of processed data. This setting does not affect the
trajectory services, which use `team_00000P_manifest.json` and their frozen
match-disjoint contract. Always use new run names so pilot checkpoints cannot
be mistaken for final results.
