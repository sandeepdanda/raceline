# Raceline

**Watch a car teach itself to race.**

A little 2D car starts out knowing nothing - it drives straight into the first wall.
A few hundred practice laps later it's threading corners and stringing together clean,
fast laps. Raceline is a reinforcement-learning agent that learns the *racing line* (the
fast path through a track) from nothing but distance sensors and a reward for making
progress, then drives live in your browser.

It is a full product, not a notebook:

- a custom **Gymnasium** environment (`RaceTrackEnv`) - a top-down car with ray sensors on
  a walled track,
- a **PPO** policy trained with Stable-Baselines3, plus simple baselines to beat,
- a seed-aware **evaluation harness** reporting laps completed, crash rate, and lap time
  with confidence intervals (not one lucky run),
- the trained policy exported to **ONNX** and served **fully in the browser** - a track you
  watch the car drive in real time, a toggle between the untrained and trained brain, and
  the sensor rays drawn live. Zero backend.

Trained, it laps at **IQM 1.0, CI [1.0, 1.0], 0% crash** across 30 held-out seeds (a
117-step lap); the random and follow-the-longest-ray baselines complete 0 laps and crash
every time. Training runs on **8 parallel car simulators** (`SubprocVecEnv`) and finishes 1M
steps in under 3 minutes on a laptop CPU - no GPU needed (see below).

> Why this project: it's the most satisfying kind of RL demo - you literally watch the
> thing get good. No dataset, no labels, no cloud. The car learns purely by crashing,
> getting a little reward for progress, and trying again.

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

# 2. train the PPO driver (CPU, minutes)
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

**On GPUs:** this project is CPU-bound on purpose - training time is dominated by stepping
the Python car simulator, and the policy net is tiny (`[64, 64]`), so a GPU does not help
and usually hurts (transfer overhead beats the small matmul; SB3 recommends CPU for
`MlpPolicy`). The real speedup is *parallel environments* (many cars at once), not a GPU.
The notebook has a `DEVICE` toggle (`"cpu"` / `"mps"` / `"cuda"`) and prints wall-clock time
so you can benchmark it yourself.

## Repository layout

```
raceline/
  envs/racetrack_env.py    # the custom Gymnasium env: car physics + ray sensors + track
  envs/track.py            # track geometry (centerline, walls) + sensor ray-casting
  agents/baselines.py      # random + simple "follow the longest ray" drivers to beat
  train.py                 # SB3 PPO training entrypoint (yaml config)
  eval.py                  # seed-aware evaluation: laps, crashes, lap time + bootstrap CIs
  export_onnx.py           # PPO policy -> ONNX for in-browser inference
configs/                   # training + eval configs
web/                       # in-browser demo: canvas track, live car, onnxruntime-web
research/                  # RL tooling stack + similar-projects x-factor analysis (cited)
docs/report/               # before/after HTML report
tests/                     # env contract + sensor + reward + baseline tests
```

## Documents

- [NORTH_STAR.md](NORTH_STAR.md) - what it is today, the vision, the next moves.
- [ROADMAP.md](ROADMAP.md) - phased build plan, highest value first.
- [PROJECT.md](PROJECT.md) - the car physics, the MDP, reward design, eval protocol.

## License

MIT. See [LICENSE](LICENSE).
