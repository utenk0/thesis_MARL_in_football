# Ubuntu VM training

## Accepted trajectory BC portability check

The accepted BC is the frozen multi-horizon trajectory predictor, not the
older 19-action BC launched by `run_training.sh`.  Do not retrain the accepted
BC merely to move it to a VM.  First reproduce the frozen CPU evaluation from
the portable, project-relative freeze manifest.

From the local machine:

```bash
bash vm/sync_trajectory_bc_to_vm.sh pc12041 thesis_MARL_in_football
```

This transfers the source, three frozen checkpoints, their protected metadata,
and one 15 MB validation chunk.  It does not transfer raw data or the complete
processed dataset.  On the VM, after the methods environment is installed:

```bash
cd /absolute/path/thesis_MARL_in_football
bash vm/run_trajectory_bc_portability.sh
```

The check verifies SHA-256 hashes before loading a checkpoint, validates the
frozen test summary against `single_team_autonomous_trajectory_v1`, and
reproduces ADE, FDE, and every direct-horizon error for seeds 0--2 within
`1e-5` metres.  Its report is written to
`artifacts/vm_runs/trajectory_bc_portability_v1/portability_report.json` and can
be fetched with:

```bash
bash vm/fetch_results.sh pc12041 thesis_MARL_in_football trajectory_bc_portability_v1
```

`run_training.sh` below is retained for the older discrete-action seven-method
experiment.  It is not the launcher for the accepted trajectory BC or for
later methods governed by the trajectory contract.

## Trajectory CQL and CTDE pretests

The shared trajectory image also contains the current CQL and CTDE adapters.
Their training launchers use the same manifest and frozen trajectory contract
as BC and leave the held-out test unopened. On the university workstation,
run either method through its dedicated Compose profile:

```bash
docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml \
  --profile trajectory-cql run --rm trajectory-cql-train

docker compose --env-file docker/server.env \
  -f docker/compose.yaml -f docker/compose.server.yaml \
  --profile trajectory-ctde run --rm trajectory-ctde-train
```

CTDE mounts `artifacts/single_team_multi_horizon_bc_full` read-only and uses
the matching frozen BC seed to initialize its local actor. Set
`CTDE_SEEDS="0 1 2"` only after the validation configuration is fixed. The
default is seed `0`, suitable for the first pretest run.

To transfer the full focal-team trajectory split and CTDE's frozen BC
initializations in one operation, run from the local machine:

```bash
SYNC_PROCESSED_DATASET=final_stage_v1 \
SYNC_SPLIT_MANIFEST=team_00000P_manifest.json \
SYNC_METHOD=ctde \
bash vm/sync_to_vm.sh pc12041 thesis_MARL_in_football
```

Use `SYNC_METHOD=cql` to omit the frozen BC bundle. The default manifest is
`team_00000P_manifest.json`; it is now also the default full-data source for
the seven-method GRF experiment.

For the broader CI/registry/Apptainer strategy and the decision about which
infrastructure to adopt or defer, see
[`../thesis_plan/EXPERIMENT_INFRASTRUCTURE.md`](../thesis_plan/EXPERIMENT_INFRASTRUCTURE.md).

Recommended baseline VM: Ubuntu 22.04 x86_64, 16 vCPUs, 32 GB RAM and 100 GB
disk. An NVIDIA GPU with at least 16 GB VRAM speeds the neural-network updates,
but Google Research Football simulation remains CPU-bound. The launcher detects
CUDA and otherwise uses CPU.

The default VM run uses all chunks from five games containing focal team
`DFL-CLU-00000P`: three training matches (422,774 joint transitions), one
validation match (89,334), and one test match (146,386). Matches remain
disjoint. It trains all seven methods with seeds 0, 1 and 2, 50 offline epochs,
and 250,000 PPO/GAIL agent steps or CQL updates. Test rollouts run for 500 joint
simulator steps.

From the local Mac, create an SSH alias so `rsync` can cross the university login host. Add this to
`~/.ssh/config` (hostnames and username only; never put a password in this file):

```sshconfig
Host uni-math-login
  HostName login.mathematik.uni-marburg.de
  User utenko

Host pc12041
  HostName PC12041
  User utenko
  ProxyJump uni-math-login
```

With the university VPN connected, verify it with `ssh pc12041`. For a quick
seven-method closed-loop GRF smoke/trial, transfer the smaller manifest for the
same selected team and same five match assignments:

```bash
SYNC_PROCESSED_DATASET=final_stage_v1 \
SYNC_SPLIT_MANIFEST=team_00000P_pilot_manifest.json \
SYNC_METHOD=legacy \
bash vm/sync_to_vm.sh pc12041 thesis_MARL_in_football
```

The sync reads the selected manifest and sends only its referenced files (about 121 MiB for this
pilot), not every file in the 3.5 GiB `final_stage_v1` directory. `SYNC_RAW_DATA=1` still
controls whether raw datasets are included.

On the VM:

```bash
cd /absolute/path/thesis_MARL_in_football
bash vm/setup_ubuntu.sh
bash vm/launch_training.sh
bash vm/status.sh
```

`setup_ubuntu.sh` follows GRF's official Linux dependency list and compiles the
vendored engine. It creates `.venv-grf` for legacy Gym/GRF and `.venv-methods`
for Gymnasium and training. The setup finishes with environment, GRF, and test
checks before training can start.

The job writes an incremental manifest after each state change. Re-running
`launch_training.sh` resumes completed seed/method pairs and retries incomplete
ones. A method interrupted midway restarts that method; completed checkpoints
are preserved. Inspect live output with:

```bash
tail -f artifacts/vm_runs/team_00000P_closed_loop_full/console.log
```

Override budgets through environment variables, for example:

```bash
RUN_NAME=vm_trial SEEDS="0" EPOCHS=5 TOTAL_TIMESTEPS=20000 \
  EVAL_GRF_STEPS=100 bash vm/launch_training.sh
```

Fetch results from the local Mac:

```bash
bash vm/fetch_results.sh USER@HOST /absolute/path/thesis_MARL_in_football
```

The default sync sends the complete selected-team manifest and its referenced files and omits the
raw datasets. Set `SYNC_SPLIT_MANIFEST=team_00000P_pilot_manifest.json` for the smaller smoke/trial
transfer. Set `SYNC_RAW_DATA=1` only if the remote machine must rebuild data. The source tree
includes vendored GRF. Sync scripts never delete remote files. Keep at least 50 GB free for a pilot
and preferably 100 GB for environments, processed data, logs, checkpoints, and repeated runs.

For PC12041, prefer the container workflow in [`../docker/README.md`](../docker/README.md). The
direct virtual-environment commands in this file remain useful for an ordinary Ubuntu VM without
Docker.

Official installation reference:
https://github.com/google-research/football#on-your-computer
