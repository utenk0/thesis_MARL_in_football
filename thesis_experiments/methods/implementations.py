"""Concrete method adapters for the unified football framework."""

from __future__ import annotations

import json

import numpy as np

from thesis_experiments.simulation.gymnasium_replay import FootballReplayEnv, MissingGymnasiumDependency
from thesis_experiments.methods.base import (
    FootballMethod,
    MethodResult,
    OptionalBackendUnavailable,
    action_classification_metrics,
    require_import,
)


def _training_env(config):
    if config.env_backend == "grf":
        from thesis_experiments.simulation.gymnasium_grf import FootballGRFEnv
        from thesis_experiments.simulation.grf_vec_env import SharedPlayerVecEnv
        return SharedPlayerVecEnv(FootballGRFEnv(config.data_paths, grf_python=config.grf_python,
                                                max_episode_steps=config.episode_steps))
    if config.env_backend == "replay":
        from stable_baselines3.common.vec_env import DummyVecEnv
        return DummyVecEnv([lambda: FootballReplayEnv(config.data_paths, random_starts=True,
                                                      max_episode_steps=config.episode_steps)])
    raise ValueError(f"Unknown environment backend: {config.env_backend}")


def _ppo(config, env):
    from stable_baselines3 import PPO
    rollout_size = config.rollout_steps * env.num_envs
    if rollout_size < 2:
        raise ValueError("PPO requires at least two rollout samples")
    batch_size = min(config.batch_size, rollout_size)
    while rollout_size % batch_size or batch_size < 2:
        if batch_size < 2:
            batch_size = rollout_size
            break
        batch_size -= 1
    return PPO("MlpPolicy", env, seed=config.seed, learning_rate=config.learning_rate,
               n_steps=config.rollout_steps, batch_size=batch_size, device=config.device,
               policy_kwargs={"net_arch": [config.hidden_size, config.hidden_size]}, verbose=0)


class BehavioralCloningMethod(FootballMethod):
    name = "bc"
    backend_name = "imitation"

    def train(self) -> MethodResult:
        observation_variant = str(self.config.extra.get("observation_variant", "full"))
        class_weighting = str(self.config.extra.get("class_weighting", "none"))
        local_options_requested = (
            observation_variant != "full" or class_weighting != "none"
        )
        if self.config.backend == "imitation" and local_options_requested:
            raise ValueError(
                "Observation ablations and class-weighted BC require --backend torch."
            )
        if not local_options_requested and self.config.backend in {"auto", "imitation"}:
            try:
                return self.write_result(self._train_with_imitation())
            except (OptionalBackendUnavailable, MissingGymnasiumDependency):
                if self.config.backend == "imitation":
                    raise
        from thesis_experiments.training.bc import train_bc

        checkpoint = self.output_dir / "shared_actor_bc.pt"
        metrics = train_bc(
            self.config.data_paths,
            checkpoint,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            hidden_size=self.config.hidden_size,
            seed=self.config.seed,
            device=self.config.device,
            observation_variant=observation_variant,
            class_weighting=class_weighting,
            max_class_weight=float(self.config.extra.get("max_class_weight", 20.0)),
        )
        explicit_local = self.config.backend == "torch" or local_options_requested
        return self.write_result(MethodResult(
            method=self.name,
            status="trained",
            backend="torch" if explicit_local else "torch-fallback",
            metrics=metrics,
            artifacts={"checkpoint": str(checkpoint)},
            message=(
                "Used the local PyTorch BC trainer."
                if explicit_local
                else "imitation backend was unavailable; used the local PyTorch BC trainer."
            ),
        ))

    def _train_with_imitation(self) -> MethodResult:
        import torch
        torch.manual_seed(self.config.seed)
        require_import("gymnasium", "Install requirements-methods.txt.")
        bc_module = require_import("imitation.algorithms.bc", "Install requirements-methods.txt.")
        types_module = require_import("imitation.data.types", "Install requirements-methods.txt.")
        data = self.load_player_data()
        env = FootballReplayEnv(self.config.data_paths)
        infos = np.asarray([{} for _ in range(len(data.actions))], dtype=object)
        transitions = types_module.Transitions(
            obs=data.observations,
            acts=data.actions,
            infos=infos,
            next_obs=data.next_observations,
            dones=data.dones,
        )
        trainer = bc_module.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            rng=np.random.default_rng(self.config.seed),
            batch_size=min(self.config.batch_size, len(data.actions)),
            optimizer_kwargs={"lr": self.config.learning_rate},
            device=self.config.device,
        )
        trainer.train(n_epochs=self.config.epochs, progress_bar=False)
        policy_path = self.output_dir / "imitation_bc_policy.zip"
        if hasattr(trainer, "save_policy"):
            trainer.save_policy(str(policy_path))
        else:
            trainer.policy.save(str(policy_path))
        predicted, _ = trainer.policy.predict(data.observations, deterministic=True)
        metrics = action_classification_metrics(predicted, data.actions)
        return MethodResult(
            method=self.name,
            status="trained",
            backend="imitation",
            metrics=metrics,
            artifacts={"policy": str(policy_path)},
        )


class CTDEMethod(FootballMethod):
    name = "ctde"
    backend_name = "torch"

    def train(self) -> MethodResult:
        if self.config.bc_checkpoint is None:
            raise ValueError("CTDE requires --bc-checkpoint pointing to a trained BC checkpoint.")
        from thesis_experiments.training.ctde import train_ctde

        checkpoint = self.output_dir / "shared_actor_ctde.pt"
        metrics = train_ctde(
            self.config.data_paths,
            self.config.bc_checkpoint,
            checkpoint,
            critic_epochs=self.config.epochs,
            actor_epochs=int(self.config.extra.get("actor_epochs", 10)),
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            seed=self.config.seed,
        )
        return self.write_result(MethodResult(
            method=self.name,
            status="trained",
            backend=self.backend_name,
            metrics=metrics,
            artifacts={"checkpoint": str(checkpoint)},
        ))


class PPOMethod(FootballMethod):
    name = "ppo"
    backend_name = "stable-baselines3"

    def train(self) -> MethodResult:
        require_import("gymnasium", "Install requirements-methods.txt.")
        sb3 = require_import("stable_baselines3", "Install requirements-methods.txt.")
        env = _training_env(self.config)
        model = _ppo(self.config, env)
        try:
            model.learn(total_timesteps=self.config.total_timesteps)
        finally:
            env.close()
        model_path = self.output_dir / "ppo_replay_policy.zip"
        model.save(str(model_path))
        data = self.load_player_data()
        predicted, _ = model.predict(data.observations, deterministic=True)
        metrics = action_classification_metrics(predicted, data.actions)
        metrics["environment_steps"] = float(model.num_timesteps)
        return self.write_result(MethodResult(
            method=self.name,
            status="trained",
            backend=self.backend_name,
            metrics=metrics,
            artifacts={"policy": str(model_path)},
            message=f"Shared local-observation PPO; environment={self.config.env_backend}.",
        ))


class GAILMethod(FootballMethod):
    name = "gail"
    backend_name = "imitation+stable-baselines3"

    def train(self) -> MethodResult:
        require_import("gymnasium", "Install requirements-methods.txt.")
        sb3 = require_import("stable_baselines3", "Install requirements-methods.txt.")
        vec_env_module = require_import("stable_baselines3.common.vec_env", "Install requirements-methods.txt.")
        gail_module = require_import("imitation.algorithms.adversarial.gail", "Install requirements-methods.txt.")
        reward_nets = require_import("imitation.rewards.reward_nets", "Install requirements-methods.txt.")
        networks = require_import("imitation.util.networks", "Install requirements-methods.txt.")
        types_module = require_import("imitation.data.types", "Install requirements-methods.txt.")

        data = self.load_player_data()
        infos = np.asarray([{} for _ in range(len(data.actions))], dtype=object)
        transitions = types_module.Transitions(
            obs=data.observations,
            acts=data.actions,
            infos=infos,
            next_obs=data.next_observations,
            dones=data.dones,
        )
        venv = _training_env(self.config)
        learner = _ppo(self.config, venv)
        reward_net = reward_nets.BasicRewardNet(
            venv.observation_space,
            venv.action_space,
            normalize_input_layer=networks.RunningNorm,
        )
        demo_batch_size = max(2, min(self.config.batch_size, len(data.actions)))
        trainer = gail_module.GAIL(
            demonstrations=transitions,
            demo_batch_size=demo_batch_size,
            gen_replay_buffer_capacity=max(2048, demo_batch_size * 2),
            n_disc_updates_per_round=int(self.config.extra.get("disc_updates", 4)),
            venv=venv,
            gen_algo=learner,
            reward_net=reward_net,
            allow_variable_horizon=True,
        )
        try:
            rounds = max(1, int(np.ceil(self.config.total_timesteps / learner.n_steps / venv.num_envs)))
            trainer.train(rounds * learner.n_steps * venv.num_envs)
        finally:
            venv.close()
        model_path = self.output_dir / "gail_ppo_generator.zip"
        learner.save(str(model_path))
        predicted, _ = learner.predict(data.observations, deterministic=True)
        metrics = action_classification_metrics(predicted, data.actions)
        metrics["environment_steps"] = float(learner.num_timesteps)
        import torch
        reward_path = self.output_dir / "gail_reward_state.pt"
        torch.save(reward_net.state_dict(), reward_path)
        return self.write_result(MethodResult(
            method=self.name,
            status="trained",
            backend=self.backend_name,
            metrics=metrics,
            artifacts={"policy": str(model_path), "reward_state": str(reward_path)},
            message=f"Adversarial reward and shared PPO generator; environment={self.config.env_backend}.",
        ))


class CQLMethod(FootballMethod):
    name = "cql"
    backend_name = "d3rlpy"

    def train(self) -> MethodResult:
        dataset_module = require_import("d3rlpy.dataset", "Install requirements-methods.txt.")
        algos_module = require_import("d3rlpy.algos", "Install requirements-methods.txt.")
        logging_module = require_import("d3rlpy.logging", "Install requirements-methods.txt.")
        data = self.load_player_data()
        from thesis_experiments.training.cql_data import build_cql_dataset
        d3rlpy = require_import("d3rlpy", "Install requirements-methods.txt.")
        encoders = require_import("d3rlpy.models.encoders", "Install requirements-methods.txt.")
        d3rlpy.seed(self.config.seed)
        dataset = build_cql_dataset(self.config.data_paths)
        algorithm = algos_module.DiscreteCQLConfig(
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            encoder_factory=encoders.VectorEncoderFactory(hidden_units=[self.config.hidden_size] * 2),
        ).create(device=self.config.device)
        log_dir = self.output_dir / "d3rlpy_logs"
        algorithm.fit(
            dataset,
            n_steps=self.config.total_timesteps,
            n_steps_per_epoch=min(1000, self.config.total_timesteps),
            experiment_name="cql",
            with_timestamp=False,
            logger_adapter=logging_module.FileAdapterFactory(root_dir=str(log_dir)),
            show_progress=False,
        )
        model_path = self.output_dir / "cql_policy.d3"
        algorithm.save(str(model_path))
        predicted = algorithm.predict(data.observations)
        metrics = action_classification_metrics(predicted, data.actions)
        return self.write_result(MethodResult(
            method=self.name,
            status="trained",
            backend=self.backend_name,
            metrics=metrics,
            artifacts={"policy": str(model_path), "logs": str(log_dir / "cql")},
        ))


class DecisionTransformerMethod(FootballMethod):
    name = "dt"
    backend_name = "torch"

    def train(self) -> MethodResult:
        from thesis_experiments.training.decision_transformer import train_decision_transformer
        checkpoint = self.output_dir / "decision_transformer.pt"
        metrics = train_decision_transformer(
            self.config.data_paths, checkpoint, epochs=self.config.epochs,
            batch_size=min(64, self.config.batch_size), learning_rate=self.config.learning_rate,
            hidden_size=self.config.hidden_size, seed=self.config.seed,
            context_length=self.config.context_length, device=self.config.device,
        )
        return self.write_result(MethodResult(
            method=self.name, status="trained", backend=self.backend_name,
            metrics=metrics, artifacts={"checkpoint": str(checkpoint)},
            message="Causal return/state/action Transformer, discrete action prediction.",
        ))


class GhostingMethod(FootballMethod):
    name = "ghosting"
    backend_name = "torch"

    def train(self) -> MethodResult:
        from thesis_experiments.training.ghosting import train_ghosting

        checkpoint = self.output_dir / "ghosting_baseline.pt"
        metrics = train_ghosting(
            self.config.data_paths,
            checkpoint,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            hidden_size=self.config.hidden_size,
            seed=self.config.seed,
        )
        return self.write_result(MethodResult(
            method=self.name,
            status="trained",
            backend=self.backend_name,
            metrics=metrics,
            artifacts={"checkpoint": str(checkpoint)},
        ))

    def imitate(self, observations: np.ndarray) -> np.ndarray:
        import torch

        from thesis_experiments.training.ghosting import ghosting_actions, load_ghosting_policy

        checkpoint = self.output_dir / "ghosting_baseline.pt"
        model = load_ghosting_policy(checkpoint)
        with torch.no_grad():
            tensor = torch.from_numpy(np.asarray(observations, dtype=np.float32)).float()
            predicted_xy = model(tensor).numpy()
        return ghosting_actions(np.asarray(observations, dtype=np.float32), predicted_xy)
