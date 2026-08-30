"""Scenario initialized from a Metrica audit JSON supplied via environment."""

import json
import os

from . import *


def build_scenario(builder):
  payload = json.loads(os.environ["GRF_METRICA_REPLAY_INITIAL_STATE"])
  builder.config().game_duration = int(payload.get("game_duration", 100))
  builder.config().deterministic = True
  builder.config().offsides = False
  builder.SetBallPosition(*payload["ball"])

  roles = [e_PlayerRole_GK] + [e_PlayerRole_CM] * 10
  builder.SetTeam(Team.e_Left)
  for position, role in zip(payload["home"], roles):
    builder.AddPlayer(float(position[0]), float(position[1]), role)

  builder.SetTeam(Team.e_Right)
  for position, role in zip(payload["away"], roles):
    # Scenario coordinates for the right team are mirrored by GRF.
    builder.AddPlayer(float(-position[0]), float(-position[1]), role)
