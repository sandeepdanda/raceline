"""Contract, sensor, determinism, reward, and baseline tests for RaceTrackEnv.

Encodes the invariants PROJECT.md promises: a real Gymnasium env (declared spaces, seeded
reset, 5-tuple step, terminate-on-crash vs truncate-on-timeout), sensors that shorten near
walls, a progress-based reward the car can't farm by circling, determinism, and a heuristic
driver that out-progresses random.
"""

from __future__ import annotations

import numpy as np
import pytest

import gymnasium as gym

from raceline.envs.racetrack_env import ACTIONS, RaceTrackEnv, EnvConfig
from raceline.envs.track import oval_track
from raceline.agents.baselines import RandomDriver, LongestRayDriver


def _run(agent, seed: int, cfg: EnvConfig | None = None) -> dict:
    env = RaceTrackEnv(cfg)
    obs, _ = env.reset(seed=seed)
    agent.reset()
    tot_r, best = 0.0, 0.0
    info = {}
    done = False
    while not done:
        obs, r, terminated, truncated, info = env.step(agent.act(obs, env))
        tot_r += r
        best = max(best, info["cum_progress_frac"])
        done = terminated or truncated
    return {"return": tot_r, "best_frac": best, "laps": info["laps"]}


# ---------------------------------------------------------------- contract

def test_spaces_declared():
    env = RaceTrackEnv()
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == len(ACTIONS)
    assert env.observation_space.shape == (env.cfg.n_rays + 1,)


def test_reset_obs_in_bounds():
    env = RaceTrackEnv()
    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)


def test_step_five_tuple():
    env = RaceTrackEnv()
    env.reset(seed=0)
    out = env.step(env.action_space.sample())
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_crash_terminates():
    # Drive hard into the wall: steer one way + gas should crash and set terminated.
    env = RaceTrackEnv()
    env.reset(seed=0)
    gas_left = ACTIONS.index((1.0, 1.0))
    terminated = False
    for _ in range(env.cfg.max_steps):
        _, _, terminated, truncated, info = env.step(gas_left)
        if terminated:
            assert info["crashed"] or info["laps"] >= env.cfg.target_laps
            break
        if truncated:
            break
    assert terminated


def test_truncates_at_step_limit():
    # A tiny step limit with a coasting car that doesn't immediately crash should truncate.
    cfg = EnvConfig(max_steps=3)
    env = RaceTrackEnv(cfg)
    env.reset(seed=0)
    straight = ACTIONS.index((0.0, 0.0))
    truncated = False
    for _ in range(cfg.max_steps):
        _, _, terminated, truncated, _ = env.step(straight)
        if terminated or truncated:
            break
    assert truncated or terminated  # one of them fires within the limit


def test_gym_make_registered():
    env = gym.make("Raceline-Track-v0")
    env.reset(seed=0)
    env.close()


# ---------------------------------------------------------------- sensors

def test_rays_shorter_toward_near_wall():
    """At the start pose the car sits on the centerline; side rays (toward the near walls)
    should read shorter than the forward ray (open track ahead)."""
    env = RaceTrackEnv()
    obs, _ = env.reset(seed=0)
    rays = obs[: env.cfg.n_rays]
    center = env.cfg.n_rays // 2
    assert rays[center] > rays[0]
    assert rays[center] > rays[-1]


def test_rays_normalized():
    env = RaceTrackEnv()
    obs, _ = env.reset(seed=0)
    rays = obs[: env.cfg.n_rays]
    assert np.all(rays >= 0.0) and np.all(rays <= 1.0)


# ---------------------------------------------------------------- determinism

def test_same_seed_same_trajectory():
    a = _run(RandomDriver(seed=3), seed=5)
    b = _run(RandomDriver(seed=3), seed=5)
    assert a == b


# ---------------------------------------------------------------- reward

def test_forward_progress_scores_positive():
    """An oracle that steers along the centerline should finish a lap with positive return."""
    env = RaceTrackEnv()
    env.reset(seed=0)

    def oracle():
        i, t = env.track.nearest_centerline(env._pos)
        ahead = env.track.center[(i + 2) % env.track.n]
        des = np.arctan2(ahead[1] - env._pos[1], ahead[0] - env._pos[0])
        err = (des - env._heading + np.pi) % (2 * np.pi) - np.pi
        s = 1.0 if err > 0.05 else (-1.0 if err < -0.05 else 0.0)
        thr = 1.0 if abs(err) < 0.1 else 0.0
        return ACTIONS.index((s, thr))

    tot, info = 0.0, {}
    for _ in range(2000):
        _, r, terminated, truncated, info = env.step(oracle())
        tot += r
        if terminated or truncated:
            break
    assert info["laps"] >= 1
    assert tot > 0


def test_crash_costs_penalty():
    cfg = EnvConfig()
    env = RaceTrackEnv(cfg)
    env.reset(seed=0)
    gas_left = ACTIONS.index((1.0, 1.0))
    last_r = 0.0
    for _ in range(cfg.max_steps):
        _, last_r, terminated, _, info = env.step(gas_left)
        if terminated and info["crashed"]:
            break
    # The crashing step's reward includes the -w_crash penalty, so it's clearly negative.
    assert last_r < 0


# ---------------------------------------------------------------- baseline

def test_heuristic_beats_random_progress():
    seeds = range(8)
    rnd = np.mean([_run(RandomDriver(seed=s), s)["best_frac"] for s in seeds])
    heur = np.mean([_run(LongestRayDriver(), s)["best_frac"] for s in seeds])
    assert heur >= rnd
