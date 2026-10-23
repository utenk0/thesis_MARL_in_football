# Experiment infrastructure decisions

This document evaluates proposed GitHub Actions, Apptainer, configuration,
tracking, and data-versioning infrastructure against the current thesis code.
The goal is reproducible experiments without turning the thesis into a general
machine-learning platform project.

## Plain-language summary

The repository already has the important foundation:

- a GitHub repository;
- a tested Docker image with separate GRF and learning Python environments;
- a common abstract method interface for all seven methods;
- saved per-run configurations, logs, checkpoints, metrics, and manifests;
- scripts to upload to, start on, monitor, and fetch results from an SSH host.

The next priority is therefore the observation/evaluation protocol and a small
local pilot. Cluster automation should be added only after the university
confirms the server interface, scheduler, storage, GPU, registry, and supported
container runtime.

## Decision table

| Proposal | Decision | Reason |
|---|---|---|
| Git repository | Already done | The project is already connected to GitHub. |
| GitHub Actions | Adopt in stages | Begin with tests and configuration validation. Build and publish an OCI image only on tagged revisions or explicit workflow runs, because compiling GRF on every commit is expensive. |
| Build Apptainer images in CI | Defer until the server is known | Apptainer can pull a Docker/OCI image from a registry and convert it to SIF on the Linux host. Building SIF directly in CI is only needed if the cluster requires a prebuilt SIF or cannot access the registry. |
| Container registry | Adopt when deploying | Prefer a university registry if provided; otherwise GHCR or Docker Hub is sufficient. Use immutable commit tags or digests for thesis runs, not only `latest`. |
| Multiple images | Not needed yet | The verified image already contains two isolated virtual environments, solving the GRF `gym==0.21` versus modern Gymnasium dependency conflict. Split images only if the cluster, CPU/GPU variants, size, or security rules require it. |
| Custom SSH CLI | Defer | The `vm/` scripts already upload, launch, monitor, resume, and fetch results. First learn whether the server uses plain SSH or a scheduler such as Slurm. A thin CLI can later wrap the confirmed workflow. |
| Hydra/OmegaConf | Useful after the pilot | Hierarchical YAML configuration would help compose method, environment, dataset, platform, and experiment settings. Do not migrate before the observation and evaluation protocol is stable. |
| Abstract base class | Already done | `FootballMethod` already defines the common method contract, and every method is registered behind it. Strengthen this contract only when a real inconsistency is found. |
| Automatic hyperparameter optimization | Defer | First run small, manually selected experiments and define validation-based selection. Generic sweeps would consume compute before the evaluation protocol is trustworthy. |
| MLflow | Optional later | The current JSON results, CSV comparison, logs, checkpoints, and incremental manifest are sufficient for the pilot. MLflow becomes useful when many runs need searchable curves and comparisons. |
| DVC | Conditional | Current split manifests and SHA-256 hashes already identify the prepared inputs. Add DVC only after remote object storage and dataset-sharing permissions are confirmed. Do not publish licensed raw data accidentally. |
| Streaming/generator data loading | Defer | The current 3,000-frame split fits in memory. Add memory mapping or iterable loading when full-match data makes memory or I/O a measured bottleneck. |
| BMW/AWS compute | Optional fallback | Use only with explicit authorization, an approved data-handling plan, and no dependency of the thesis on private infrastructure. |

## Recommended deployment shape

```text
Git commit
    -> GitHub Actions tests
    -> build one versioned Docker/OCI image
    -> push image to university registry or GHCR
    -> university Linux server pulls it with Apptainer
    -> scheduler or launcher starts the experiment

Server persistent storage
    -> prepared dataset mounted read-only
    -> experiment output directory mounted read-write
    -> checkpoints, logs, resolved config and metrics retained outside image
```

This avoids Docker-in-Docker on the cluster. Docker or GitHub Actions builds
the OCI image once; Apptainer consumes that image on the server. Apptainer can
pull from a `docker://` registry reference and construct a SIF locally. A SIF
can also be pushed and pulled through an OCI registry with `oras://` if the
university specifically chooses that workflow.

The production image should contain code and dependencies, not the raw
datasets, trained checkpoints, or experiment outputs. Those large and mutable
items should live on persistent server storage and be mounted into the
container. The present local smoke image includes the prepared split for
convenience; a later deployment target can remove it once the server storage
layout is known.

## Why one image is currently preferable

The current Docker image has two virtual environments:

```text
.venv-grf       GRF and gym 0.21
.venv-methods   Gymnasium, SB3, imitation, d3rlpy and PyTorch
```

The modern environment starts the GRF worker with the legacy interpreter.
Keeping them in one image preserves this tested communication path. Splitting
them into separate containers would require an additional inter-container
protocol and scheduler coordination without improving the thesis evaluation.

Reasonable future variants are:

- a CPU image for local tests and CPU-only servers;
- a CUDA image if the university confirms a compatible NVIDIA GPU runtime;
- a smaller code/dependency image that mounts the prepared data externally.

## Existing method contract

The suggested abstract base class is already implemented in
`thesis_experiments/methods/base.py`. `FootballMethod` enforces a shared
configuration and `train()` operation, supplies common evaluation/imitation
behaviour, and writes a standard `MethodResult`. The registry maps BC, GAIL,
PPO, CQL, DT, ghosting, and CTDE to that interface.

Fair comparison still depends on more than a common class. Every final run must
also use the same split policy, observation/action definitions, seed protocol,
evaluation horizon, result schema, and validation-based selection rules.

## Configuration plan

The current argparse and environment-variable configuration is adequate for
the smoke test and first pilot. If the number of runs grows, introduce Hydra in
small steps with configuration groups such as:

```text
configs/
├── config.yaml
├── experiment/
│   ├── smoke.yaml
│   ├── pilot.yaml
│   └── final.yaml
├── method/
│   ├── bc.yaml
│   ├── ppo.yaml
│   └── ...
├── environment/
│   ├── local_cpu.yaml
│   └── cluster_gpu.yaml
└── data/
    └── vm_training_3000.yaml
```

Each result directory should retain the fully resolved configuration, Git
commit, image tag/digest, data-manifest hash, package versions, seed, start/end
times, hostname, and device. Secrets, SSH keys, access tokens, and private
storage credentials must never be stored in these YAML files or committed.

A future command could look like this, but it should wrap the scheduler rather
than invent a second job system:

```text
thesis-run experiment=pilot method=ppo platform=cluster_gpu seed=0
```

## Experiment tracking and data handling

For the current stage, keep the existing artifacts:

```text
training_manifest.json
comparison.csv
comparison.json
seed_<n>/methods/<method>/config.json
seed_<n>/methods/<method>/result.json
logs/
checkpoints and rollout arrays
```

Add MLflow only if comparing these files manually becomes difficult. If it is
introduced, the local file-backed mode is sufficient first; a tracking server
is not a prerequisite.

Keep the raw datasets outside Git and container images. The prepared split is
already described by a manifest containing paths, match identities, frame
ranges, and hashes. DVC is valuable only if an approved remote store is
available and the files may legally be copied there. Code and small manifests
remain in Git; data and model binaries remain in controlled storage.

## Phased implementation

### Now: thesis methodology and local pilot

1. Finalize temporal observation features and the evaluation protocol.
2. Run a BC diagnostic and one-seed pilot locally.
3. Keep the working Docker smoke image as reproducibility evidence.
4. Continue using the common method interface and current artifact schema.

### After server details are confirmed

1. Record the hostname/VPN, scheduler, Apptainer version, registry access,
   architecture, CPUs, GPUs, CUDA compatibility, memory, quotas, job limits,
   internet access, and persistent/scratch paths.
2. Decide whether the server can pull the OCI image directly or requires a
   prebuilt SIF.
3. Adapt the existing SSH scripts or add a scheduler job template.
4. Move prepared data and output directories to mounted persistent storage.

### Before the final experiment

1. Add lightweight GitHub Actions tests and an explicit image-publish job.
2. Tag images with the Git commit and record the digest in every run.
3. Introduce hierarchical YAML/Hydra configuration if the experiment matrix
   can no longer be managed clearly with the existing arguments.
4. Add MLflow only if it materially improves run comparison.
5. Run the frozen protocol with multiple seeds and retain all evidence needed
   for the thesis tables and plots.

## References

- [Apptainer: OCI image registries](https://apptainer.org/docs/user/main/registry.html)
- [Apptainer: building containers](https://apptainer.org/docs/user/1.4/build_a_container.html)
- [Python abstract base classes](https://docs.python.org/3/library/abc.html)
- [Hydra configuration composition](https://hydra.cc/docs/intro/)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [DVC data versioning](https://doc.dvc.org/start)
