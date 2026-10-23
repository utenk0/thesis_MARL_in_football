"""Private JSON-lines GRF process; run with the legacy .venv-grf interpreter."""

import json
import os
import sys
import traceback

import numpy as np

from thesis_experiments.env import ensure_local_football_on_path


def main():
    ensure_local_football_on_path()
    import gfootball.env as football_env

    env = None
    observations = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request["op"] == "close":
                    break
                if request["op"] == "reset":
                    if env is not None:
                        env.close()
                    initial = request["initial"]
                    os.environ["GRF_METRICA_REPLAY_INITIAL_STATE"] = json.dumps(initial)
                    env = football_env.create_environment(
                        env_name="metrica_100_replay", representation="raw", rewards="scoring",
                        render=False, write_goal_dumps=False, write_full_episode_dumps=False,
                        number_of_left_players_agent_controls=11,
                        number_of_right_players_agent_controls=11,
                        other_config_options={"game_engine_random_seed": request["seed"],
                                              "physics_steps_per_frame": 4},
                    )
                    observations = env.reset()
                    rewards, done = np.zeros(22), False
                elif request["op"] == "step":
                    # GRF controller order can differ from stable player-slot order.
                    actions = request["actions"]
                    controller_actions = [actions[int(obs["active"]) + (11 if i >= 11 else 0)]
                                          for i, obs in enumerate(observations)]
                    observations, rewards, done, _ = env.step(controller_actions)
                else:
                    raise ValueError("Unknown worker operation")
                reference = observations[0]
                response = {
                    "players": np.concatenate([reference["left_team"], reference["right_team"]]).tolist(),
                    "ball": np.asarray(reference["ball"])[:2].tolist(),
                    "ball_owned_team": int(reference.get("ball_owned_team", -1)),
                    "ball_owned_player": int(reference.get("ball_owned_player", -1)),
                    "team_rewards": [float(np.mean(rewards[:11])), float(np.mean(rewards[11:]))],
                    "score": list(reference["score"]), "done": bool(done),
                }
            except Exception:
                response = {"error": traceback.format_exc()}
            print("GRF_RESPONSE " + json.dumps(response), flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
