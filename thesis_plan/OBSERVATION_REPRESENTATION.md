# Observation representation

The player-local observation is now versioned as `local_v2_temporal_61`. It
contains 61 features and is the representation that should be used when the
training data is rebuilt before the next BC run.

## Feature groups

| Indices | Size | Meaning |
|---|---:|---|
| `0:24` | 24 | Existing spatial features: player and ball positions, ball relative position, side, role index, four nearest teammates and four nearest opponents |
| `24:26` | 2 | Controlled player's motion from the previous frame to the current frame |
| `26:28` | 2 | Ball motion from the previous frame to the current frame |
| `28:47` | 19 | One-hot encoding of the previous GRF action |
| `47:57` | 10 | GRF-style sticky state: eight directions, sprint and dribble |
| `57:60` | 3 | Ball possession from the player's perspective: own team, opponent or loose |
| `60` | 1 | Whether this particular player owns the ball |

Motion is represented as displacement per recorded decision step in GRF
coordinates. This is intentional: the training clips and the live GRF worker
both provide consecutive decision frames.

## Causality

For the action predicted at frame `t`, the representation uses:

- positions at `t`;
- motion calculated from `t-1` to `t`;
- action taken at `t-1`;
- sticky state after action `t-1`;
- possession at `t`.

It never uses the target action at `t` or positions from `t+1`. Therefore the
new temporal features do not leak the answer into BC training.

The first frame of every clip has zero motion, idle as its previous action, and
an empty sticky state. Tracking-data possession is conservatively inferred from
the nearest player within two metres. Live GRF evaluation uses GRF's explicit
ball-owner fields.

## Rebuild and validate

Old 24-feature transition files are recognized as `local_v1_spatial_24`, but
they cannot be mixed with the new files. Existing checkpoints trained on 24
features are also not compatible with the new 61-feature observations.

Rebuild the bounded dataset using the same input paths used previously:

```bash
.venv-methods/bin/python -m thesis_experiments.scripts.prepare_training_data \
  --frames 512 \
  --output-dir data/processed/training_stage_v2
```

Then validate every generated transition file, for example:

```bash
.venv-methods/bin/python -m thesis_experiments.scripts.validate_observations \
  data/processed/training_stage_v2/train/*.npz \
  data/processed/training_stage_v2/validation/*.npz \
  data/processed/training_stage_v2/test/*.npz
```

The validator checks finiteness and schema through the dataset loader, then
checks previous-action alignment, sticky-state reconstruction, causal velocity
alignment, and possession encoding. A non-zero exit means the data must not be
used for training.
