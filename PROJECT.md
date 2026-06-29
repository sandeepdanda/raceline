# Raceline - Architecture & Design

The technical spine: the car model, the track, the MDP, reward design (with the
justification each shaping term needs), and the evaluation protocol. Read
[NORTH_STAR.md](NORTH_STAR.md) for the why and [ROADMAP.md](ROADMAP.md) for the order.

## Problem

Drive a car around a closed track as far/fast as possible without hitting a wall, learning
purely from distance sensors and a progress reward. No map of the track is given to the
agent - it only sees how far the walls are along a few rays, plus its own speed. The fun is
that this minimal information is enough to learn cornering.

## The track (`raceline/envs/track.py`)

- A closed loop defined by an ordered list of **centerline waypoints** and a fixed track
  **half-width**. The inner and outer **walls** are the centerline offset by +/- half-width.
- **Progress** is measured as arc-length along the centerline: each step we find the car's
  nearest centerline segment and how far around the loop it is. Completing the loop = a lap.
- **Collision**: the car crashes if its distance from the centerline exceeds the half-width
  (i.e. it crossed a wall).
- **Sensors**: from the car's position, cast `N` rays at fixed angles relative to heading;
  each returns the distance to the nearest wall, clipped to a max range and normalized to
  [0, 1]. Ray-casting is analytic against the wall polylines.

Phase 1 ships one track (a rounded rectangle / simple oval). Extra tracks are Phase 5.

## Car physics

Discrete-time kinematic model (a simplified bicycle model). State: position `(x, y)`,
heading `theta`, speed `v`. Each tick:

- steering action changes `theta` by up to `max_steer` (scaled by speed so it can't pivot
  in place),
- throttle action changes `v` (accelerate / coast / brake), with friction and a speed cap,
- position integrates forward: `x += v*cos(theta)*dt`, `y += v*sin(theta)*dt`.

Deterministic and cheap - thousands of steps per second on CPU.

## MDP formulation

A real `gymnasium.Env` (Gymnasium 1.3 API: 5-tuple `step`, keyword-only seeded `reset`,
`terminated` vs `truncated`).

**Observation** (`Box`, float32): `[ray_1 ... ray_N, speed_norm]` - the N normalized sensor
distances plus normalized speed. Small vector (default N=7 -> 8 dims).

**Action** (`Discrete`): a small grid of (steer in {left, straight, right}) x (throttle in
{brake, coast, gas}) combinations. Discrete keeps training fast and the demo legible; a
continuous variant is a later option.

**Transition:** apply steering + throttle, advance physics, recompute sensors and progress,
check collision.

**`terminated`** (a real MDP terminal - stop bootstrapping): the car crashes into a wall, or
completes the target number of laps. **`truncated`** (outside the MDP - keep bootstrapping):
the per-episode step limit is hit. Splitting these is what lets PPO bootstrap value
estimates correctly at a time-limit cutoff but not at a crash.

## Reward design

Per-step reward:

```
r_t = w_progress * delta_progress_t    # arc-length advanced along the centerline this step
    - w_time                           # small constant per step -> finish faster
    - w_crash * crashed_t              # large one-time penalty on hitting a wall
```

Each term, justified (shaping can silently break the intended behavior):

- **`w_progress * delta_progress`** is the *real objective*: reward is proportional to how
  far around the loop the car got, so going forward fast around the track is what pays. It
  is signed by arc-length *along the loop direction*, so driving backwards earns negative
  progress - the car cannot farm reward by oscillating or circling in place.
- **`-w_time`** (small) breaks ties toward speed: among policies that complete the lap, the
  faster one accumulates less time penalty. Small enough that it never makes crashing
  (which ends progress) look better than driving slowly.
- **`-w_crash * crashed`** encodes the hard constraint. One-time, only on crash; it never
  rewards anything, so it can't distort which non-crash states are preferred - it only makes
  the terminal crash state strongly negative.

Weights live in config.

## Baselines (`raceline/agents/baselines.py`)

- **`RandomDriver`**: uniform random actions. The floor.
- **`LongestRayDriver`**: steer toward whichever sensor ray is longest (most open space),
  constant moderate throttle. A non-learned heuristic that already drives *somewhat* - so a
  PPO win over it is a real result, not "beat random."

## Training (`raceline/train.py`)

SB3 PPO, `MlpPolicy`, yaml config, all RNGs seeded (Python, NumPy, Torch, env). Tiny network
(small observation) - also keeps the ONNX export sub-MB. TensorBoard logging; checkpoint
best-by-eval-reward to `runs/`.

**Parallel environments.** `make_vec_env()` runs `n_envs` car simulators at once -
`SubprocVecEnv` (separate processes, true parallelism across cores) for `n_envs > 1`,
`DummyVecEnv` for one. Each env gets a distinct seed so they don't replay identical
trajectories; the per-env rollout is `n_steps`, so the effective batch is `n_steps * n_envs`
(512 * 8 = 4096 in the shipped config). This is the real speedup for this CPU-bound RL, not
a GPU: 100k steps go from 36.9s (1 env) to 16.1s (8 envs), 2.3x on 12 cores. A GPU does not
help - training is dominated by stepping the Python simulator and the net is tiny, so
transfer overhead beats the small matmul (SB3 recommends CPU for `MlpPolicy`).

## Evaluation protocol (`raceline/eval.py`)

No single-lucky-lap claims.

1. Run the trained car and every baseline across `>= 20` seeds (each a fresh episode with a
   different start RNG).
2. Report **IQM** + **bootstrap 95% CIs** for: laps completed, crash rate, mean progress
   (fraction of a lap), best lap time (steps to complete a lap).
3. A win over the heuristic is reported only when the CIs separate.
4. Output a metrics table + a JSON the HTML report consumes.

Current result (30 held-out seeds, `runs/ppo/eval.json`): the PPO policy laps at **IQM 1.0,
CI [1.0, 1.0], 0% crash, 117-step lap**; `RandomDriver` and `LongestRayDriver` both complete
0 laps and crash on every seed. The CIs separate cleanly, so the RL win is real.

## Serving (`raceline/export_onnx.py` + `web/`)

Export the PPO policy MLP to ONNX. The browser demo loads it with `onnxruntime-web` (WASM)
and runs inference client-side. The same car physics + sensor ray-casting are reimplemented
in JS so the in-browser sim matches the Python env; the page draws the track, the car, and
the live sensor rays on a canvas, with a toggle to swap the untrained vs trained policy.

## Testing (`tests/`)

- Env contract: spaces declared, `reset(seed=...)` reproducible, `step` 5-tuple, `truncated`
  fires at the step limit, `terminated` fires on crash, observation in bounds.
- Sensors: a ray pointing straight at a near wall returns a small distance; open directions
  return ~max range.
- Reward: forward progress scores positive; crashing scores the penalty; circling in place
  does not accumulate positive reward.
- Determinism: same seed -> same trajectory.
- Baseline: `LongestRayDriver` makes more progress than `RandomDriver`.

## Stack

Pinned in `pyproject.toml`, sourced in `research/stack.md`: Gymnasium 1.3, Stable-Baselines3
2.9, PyTorch, ONNX/onnxruntime, TensorBoard. Python 3.11 (torch/SB3 wheels are not on 3.14).
