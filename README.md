# Raceline

**Watch a car teach itself to race.**

A little 2D car starts out knowing nothing - it drives straight into the first wall. A few
hundred practice laps later it's threading corners and stringing together clean, fast laps.
Raceline is a reinforcement-learning agent that learns the *racing line* from nothing but
distance sensors and a reward for making progress, then drives live in your browser.

> The most satisfying kind of RL demo: you literally watch the thing get good. No dataset,
> no labels, no cloud. The car learns purely by crashing, getting a little reward for
> progress, and trying again.

## What it does

The car sees only a handful of distance sensors (rays to the nearest wall) plus its speed,
and picks a steering + throttle action each tick. PPO rewards it for progress around the
track and penalizes crashing. After ~1M steps of practice it laps cleanly, every time - then
the trained brain is exported to ONNX and runs entirely in a browser canvas, with a toggle
to flip between the untrained car (instant crashes) and the trained one (clean laps) on the
same track, sensor rays drawn live.

Four layers, each usable on its own:

- a custom **Gymnasium** env (`RaceTrackEnv`) - a top-down car with ray sensors on a walled track,
- a **PPO** policy (Stable-Baselines3), trained across 8 parallel envs, plus baselines to beat,
- a seed-aware **evaluation harness** - laps, crash rate, lap time with bootstrap CIs (not one lucky run),
- a **zero-backend web demo** - the ONNX policy driving live via `onnxruntime-web`.

## Results

Trained policy vs baselines, 30 held-out seeds (`runs/ppo/eval.json`):

| Driver | Laps (IQM) | 95% CI | Crash rate | Lap time |
|---|---|---|---|---|
| **Raceline (PPO)** | **1.0** | [1.0, 1.0] | **0%** | 117 steps |
| Follow-longest-ray | 0.0 | [0.0, 0.0] | 100% | - |
| Random | 0.0 | [0.0, 0.0] | 100% | - |

The CIs separate cleanly, so the RL win is real - not a cherry-picked lap. Training finishes
1M steps in under 3 minutes on a laptop CPU.

## The story in one screen

| | Untrained (episode 0) | Trained (Raceline) |
|---|---|---|
| First corner | drives straight into the wall | brakes, turns, holds the line |
| A full lap | never finishes | clean laps, repeatably |
| How it knows | random flailing | learned which sensor readings mean "turn now" |
| Training data | none - it learns by trying | none - same, just lots of practice |

## Quickstart

```bash
# Python 3.11 (torch / SB3 wheels are not on 3.14 yet)
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .

# 1. sanity-check the environment (random driver, no training)
python -m raceline.envs.racetrack_env --selftest

# 2. train the PPO driver (8 parallel envs, CPU, ~3 min)
python -m raceline.train --config configs/ppo.yaml

# 3. evaluate the trained car vs baselines, with confidence intervals
python -m raceline.eval --checkpoint runs/ppo/best_model.zip

# 4. export to ONNX for the browser demo
python -m raceline.export_onnx --checkpoint runs/ppo/best_model.zip --out web/policy.onnx

# 5. watch it drive
cd web && python -m http.server 8753   # then open http://localhost:8753
```

## Notebook

Prefer an interactive walkthrough? `notebooks/raceline.ipynb` trains the car, plots the
learning curve, evaluates across seeds, and animates a learned lap inline.

```bash
uv pip install -e ".[notebook]"
jupyter lab notebooks/raceline.ipynb     # or: jupyter notebook
```

## Why no GPU?

This project is CPU-bound on purpose. Training time is dominated by stepping the Python car
simulator, and the policy net is tiny (`[64, 64]`), so a GPU does not help and usually hurts
(transfer overhead beats the small matmul; SB3 recommends CPU for `MlpPolicy`). The real
speedup is **parallel environments** (many cars at once): 100k steps go from 36.9s on 1 env
to 16.1s on 8, a 2.3x win on a 12-core laptop. The notebook has a `DEVICE` toggle
(`"cpu"` / `"mps"` / `"cuda"`) so you can benchmark it yourself.

## Documents

- [NORTH_STAR.md](NORTH_STAR.md) - what it is today, the vision, the next moves.
- [ROADMAP.md](ROADMAP.md) - phased build plan, highest value first.
- [PROJECT.md](PROJECT.md) - the car physics, the MDP, reward design, eval protocol.

## License

MIT. See [LICENSE](LICENSE).
