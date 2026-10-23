"""Open an interactive Google Research Football match with keyboard control."""

from __future__ import annotations

import argparse

from thesis_experiments.env import ensure_local_football_on_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level",
        default="11_vs_11_stochastic",
        help="GRF scenario to open (default: 11_vs_11_stochastic).",
    )
    args = parser.parse_args()

    ensure_local_football_on_path()
    from gfootball.env import config, football_env

    environment = football_env.FootballEnv(config.Config({
        "action_set": "full",
        "dump_full_episodes": False,
        "level": args.level,
        "players": ["keyboard:left_players=1"],
        "real_time": True,
        "render": True,
        "write_video": False,
    }))
    environment.render()
    environment.reset()
    print("Interactive GRF is open. Focus the game window and use arrow keys; press Ctrl-C here to quit.")
    try:
        while True:
            _observation, _reward, done, _info = environment.step([])
            if done:
                environment.reset()
    except KeyboardInterrupt:
        pass
    finally:
        environment.close()


if __name__ == "__main__":
    main()
