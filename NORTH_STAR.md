# Raceline - North Star

Futuristic goals for this project. What it is today, where it's going, and the next
moves. Update this as phases ship.

## What it is today

A car that learns to race, and a browser demo where you watch it happen. A custom
Gymnasium environment simulates a top-down car on a walled track; the car has a handful of
distance sensors (rays) and learns, via PPO, to steer and throttle so it makes progress
around the track without crashing. The trained policy exports to ONNX and drives live in a
browser canvas - toggle between the untrained brain (instant crashes) and the trained one
(clean laps), with the sensor rays drawn so you can see what the car "sees."

Three layers, each usable on its own:

- a pure-Python **car simulator + Gymnasium env** (`raceline/envs/`),
- a **training + evaluation** layer (SB3 PPO, seeded, config-driven, parallel envs, IQM +
  bootstrap CIs),
- a **zero-backend web demo** (ONNX policy, `onnxruntime-web`, runs anywhere static).

Where it stands today: the full pipeline works end-to-end. Training uses 8 parallel car
simulators (`SubprocVecEnv`) and finishes 1M steps in ~2m49s on CPU; the trained policy
evaluates at **laps IQM 1.0, CI [1.0, 1.0], 0% crash, 117-step lap** across 30 held-out
seeds, while both baselines complete 0 laps and crash 100% of the time. There is also a
Jupyter notebook walkthrough (`notebooks/raceline.ipynb`).

## The vision

The most legible RL demo on the internet: anyone, technical or not, drags onto the page,
flips the car from "untrained" to "trained," and instantly *gets it* - the thing taught
itself to drive. No dataset, no labels, no explanation needed. Then the next layers let
you mess with it: draw your own track, change the reward, add a second car to race, watch
the racing line emerge. It stays free and runnable by anyone - the sim is a few hundred
lines of Python and the demo is a single static page.

## Why this project exists

It replaces an old masters-course assignment (tabular Q-learning on gridworlds + a basic
stock-trading env - the textbook "I did the course" signal). The pick here optimizes for
*cool and watchable* over everything: a self-driving-car-on-a-track agent is the canonical
"watch it learn" demo, trains in minutes on a laptop, and the train -> ONNX -> in-browser
pipeline makes it a permanent, clickable artifact. The RL stack research and the
similar-projects x-factor analysis are in `research/`.

## Next level - roadmap (highest value first)

Done: the episode-0-vs-trained toggle, live sensor rays, parallel-env training, and the
seed-aware eval are all shipped. The next moves chase *watchability*. A research pass on the
starred projects in this genre (`research/similar-projects.md`) found the viral pattern is
always the same: many agents on screen at once, a counter that climbs, and one-click
interactivity. The ONNX-in-browser angle is already rare here and is the differentiator -
these build on it, ranked by wow-per-effort.

1. **N ghost cars from one policy.** Spawn 10-100 cars on the same trained policy, diverged
   by jittered spawn + small action noise; they fan out, some crash, the clean ones survive.
   The single most-shared image in the genre, from one policy, zero retraining. (S)
2. **"What the car senses" HUD.** Live ray-distance bars + steer/throttle gauges, so the
   viewer sees *why* the car turns. (S)
3. **Draw-your-own-track.** Sketch a track in the browser and watch the pretrained policy
   attempt it - the biggest genuine differentiator; no famous demo does this. Imperfect
   generalization is itself an honest, interesting thing to show. (M)
4. **Live network-activation overlay.** Draw the PPO MLP and animate it as the car drives -
   the "watch it think" shot (MarI/O). (M)
5. **Save / load / share a run.** Download/upload the brain + a share link encoding track +
   seed - turns viewers into participants. (S)
6. **Racing-line ghost + reward shaping.** Precompute a minimum-curvature line, draw it as a
   ghost, and shape reward toward it so the car laps *fast*, not just finishes. (M)
7. **A harder / second track** as an honest generalization test; **a second car to race**
   (multi-agent) for emergent overtaking. (M)

Full ranked analysis with sources and star counts: `research/similar-projects.md`.

## Constraints that don't change

- No git commit/push - left to the human.
- Every dependency stays free and public. Verify before adding.
- Evaluation is seed-aware always: many seeds, IQM + bootstrap CIs, never a single lucky
  lap. This separates a real result from a cherry-pick.
- Reward shaping is justified in writing: every term must be argued not to break the
  intended behavior (e.g. a progress reward must not let the car farm reward by circling).
- The demo stays backend-free (ONNX in browser) so the live link is permanent and free.
- This is a personal project. Keep it self-contained - no references to any employer,
  service, or proprietary system.
- Ship one roadmap phase at a time, evidence-first, with a before/after report.

## Working agreement (how to build here)

- Custom env is a real `gymnasium.Env`: declared spaces, seeded `reset`, 5-tuple `step`,
  `terminated` (crash / lap done) vs `truncated` (time limit) correct.
- Every performance claim comes with the command that produced it and its raw output.
- Document what failed, not just what worked.
