"""Train a PPO policy to drive RaceTrackEnv.

    python -m raceline.train --config configs/ppo.yaml

Config-driven and seeded. Logs to TensorBoard, checkpoints the best model by eval reward to
``output_dir/best_model.zip``. The network is intentionally tiny (the observation is a small
sensor vector), which also keeps the ONNX export sub-MB for instant in-browser inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from raceline.envs.racetrack_env import RaceTrackEnv, EnvConfig
from raceline.seeding import seed_everything


def _env_config(env_cfg: dict) -> EnvConfig:
    known = {f for f in EnvConfig().__dict__}
    return EnvConfig(**{k: v for k, v in env_cfg.items() if k in known})


def make_env(env_cfg: dict, seed: int):
    def _thunk():
        env = RaceTrackEnv(_env_config(env_cfg))
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _thunk


def make_vec_env(env_cfg: dict, seed: int, n_envs: int):
    """Build a vectorized env of ``n_envs`` cars practicing in parallel.

    The real speedup for this CPU-bound RL: each env runs the Python car simulator
    independently, so N envs collect ~N x the experience per wall-clock second. We use
    SubprocVecEnv (separate processes, true parallelism across cores) for n_envs > 1, and
    DummyVecEnv (in-process) for a single env. Each env gets a distinct seed so they don't
    all replay the identical trajectory.
    """
    thunks = [make_env(env_cfg, seed + i) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(thunks)
    return SubprocVecEnv(thunks)


def train(config_path: str) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    p = cfg["ppo"]
    n_envs = int(p.get("n_envs", 1))
    train_env = make_vec_env(cfg["env"], seed, n_envs)
    eval_env = make_env(cfg["env"], seed + 1000)()  # single env, different seed than training
    print(f"training with {n_envs} parallel env(s)")

    model = PPO(
        p["policy"],
        train_env,
        seed=seed,
        n_steps=p["n_steps"],
        batch_size=p["batch_size"],
        n_epochs=p["n_epochs"],
        gamma=p["gamma"],
        gae_lambda=p["gae_lambda"],
        learning_rate=p["learning_rate"],
        ent_coef=p["ent_coef"],
        clip_range=p["clip_range"],
        policy_kwargs={"net_arch": p["net_arch"]},
        tensorboard_log=str(out_dir / "tb"),
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir),
        log_path=str(out_dir),
        eval_freq=cfg["eval"]["eval_freq"],
        n_eval_episodes=cfg["eval"]["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=p["total_timesteps"], callback=eval_cb, progress_bar=False)

    final = out_dir / "final_model.zip"
    model.save(str(final))
    best = out_dir / "best_model.zip"
    print(f"\nsaved final -> {final}")
    if best.exists():
        print(f"best (by eval reward) -> {best}")
    return best if best.exists() else final


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PPO on RaceTrackEnv")
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
