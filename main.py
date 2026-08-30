import argparse

import gfootball.env as football_env


def run_episode(environment_name: str, steps: int) -> None:
    environment = football_env.create_environment(
        env_name=environment_name,
        representation="simple115v2",
        render=False,
    )

    try:
        observation = environment.reset()
        print(
            f"reset: shape={observation.shape}, dtype={observation.dtype}, "
            f"actions={environment.action_space}"
        )

        total_reward = 0.0
        for step_number in range(1, steps + 1):
            action = environment.action_space.sample()
            observation, reward, done, info = environment.step(action)
            total_reward += float(reward)
            print(
                f"step={step_number} action={action} reward={reward} "
                f"score={info.get('score_reward', 0)} done={done}"
            )
            if done:
                break

        print(f"episode finished: total_reward={total_reward}")
    finally:
        environment.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a headless GRF episode.")
    parser.add_argument(
        "--environment",
        default="academy_empty_goal_close",
        help="GRF scenario name",
    )
    parser.add_argument("--steps", type=int, default=10)
    arguments = parser.parse_args()
    run_episode(arguments.environment, arguments.steps)
