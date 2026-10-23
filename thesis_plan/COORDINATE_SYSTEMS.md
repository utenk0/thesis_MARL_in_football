# Stage 5 Coordinate Systems

## Shared GRF world space

All provider positions are converted into one fixed pitch coordinate system:

```text
x ∈ [-1, 1]
y ∈ [-0.42, 0.42]
origin = pitch centre
negative x = left goal
positive x = right goal
```

Positions outside the nominal pitch are clipped to these bounds. This matters
because both datasets include tracked players just outside the field during
substitutions and other stoppages.

## Metrica

Metrica Games 1–3 use normalized coordinates with top-left `(0, 0)`, bottom-
right `(1, 1)`, and centre `(0.5, 0.5)`:

```text
x_grf = clip(2*x - 1, -1, 1)
y_grf = clip(0.42*(2*y - 1), -0.42, 0.42)
```

## Dataset A (BASSEK)

Tracking coordinates use centred metres on a 105 m by 68 m pitch:

```text
x_grf = clip(x_m / 52.5, -1, 1)
y_grf = clip(0.42*y_m / 34, -0.42, 0.42)
```

Dataset A event locations instead use `0..105` by `0..68`; the event converter
subtracts `(52.5, 34)` before applying the tracking formula.

## Attacking direction and mirroring

The serialized frame always remains in shared world coordinates. Per-frame
attacking directions are inferred from team median x positions: the team
occupying the lower-x half attacks positive x. This naturally changes after
teams switch sides.

For policies and GRF right-team controls, a team attacking negative world x
uses a 180-degree rotation:

```text
(x_agent, y_agent) = (-x_world, -y_world)
```

`team_attacking_view` provides this representation without mixing two team-
relative coordinate systems inside the shared global state.
