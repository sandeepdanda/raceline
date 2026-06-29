# Raceline - Roadmap

Prioritized build plan. Each phase builds on the previous. Detailed design is in
[PROJECT.md](PROJECT.md); the RL stack research and the similar-projects x-factor analysis are in `research/`.

## Phase 1: The environment (MVP foundation)

A custom Gymnasium environment: a top-down car with distance sensors on a walled track,
plus baseline drivers to beat.

**What it is:** `RaceTrackEnv(gym.Env)`. Each step the car observes its sensor rays and
speed, picks a steering + throttle action; the simulator advances simple car physics,
checks for wall collisions, measures progress along the track centerline, and returns a
reward. Crash ends the episode (terminated); a step limit truncates it.

**Technical approach:**
- Track: a closed loop defined by a centerline of waypoints with a fixed half-width, giving
  inner and outer walls. Start simple (an oval / rounded rectangle), add corners later.
- Car physics: kinematic bicycle-ish model - position, heading, speed; actions change
  steering angle and throttle; speed has friction/cap. Simple, deterministic, fast.
- Sensors: N rays cast from the car at fixed angles, returning normalized distance to the
  nearest wall. This is the whole observation (plus speed) - small vector, tiny network,
  sub-MB ONNX.
- Reward: `+progress along centerline` per step, `-crash penalty` on hitting a wall,
  small `-time` cost to encourage speed. Justified in PROJECT.md so the car can't farm
  reward by circling in place.

**Baselines:** `RandomDriver` (floor) and `LongestRayDriver` (a heuristic: steer toward
the longest sensor ray, constant throttle) - a non-learned driver that already sort-of
works, so the RL win is meaningful.

**Deliverable:** `python -m raceline.envs.racetrack_env --selftest` runs random + heuristic
drivers and prints progress / crashes. Env passes a contract test.

**Research:** `research/stack.md` (Gymnasium 1.3 API).

---

## Phase 2: Train the car

Train a PPO policy that completes clean laps.

**Technical approach:**
- Stable-Baselines3 PPO, `MlpPolicy`, yaml config, seeded.
- Tiny network (sensors + speed is a small vector) - keeps ONNX export small.
- TensorBoard logging; checkpoint best model by eval reward.

**Deliverable:** `python -m raceline.train` produces `runs/ppo/best_model.zip`.

**Research:** `research/stack.md` (SB3 2.9).

---

## Phase 2.5: Train faster with parallel environments (shipped)

Many cars practicing at once - the real speedup for this CPU-bound RL, not a GPU.

**Technical approach:**
- `make_vec_env()` builds N car simulators, each its own process, with distinct seeds.
- `SubprocVecEnv` for N > 1 (true parallelism across cores), `DummyVecEnv` for N = 1.
- Config-driven: `n_envs: 8`, `n_steps: 512` per env -> effective batch `512 * 8 = 4096`.

**Why not a GPU:** training time is dominated by stepping the Python car simulator and the
net is tiny (`[64, 64]`), so a GPU does not help (transfer overhead beats the small matmul;
SB3 recommends CPU for `MlpPolicy`). Parallel envs collect ~N x the experience per
wall-clock second instead.

**Deliverable + evidence:** 100k steps drop from **36.9s (1 env) to 16.1s (8 envs) = 2.3x**
on a 12-core machine (sub-8x is expected: the policy update is serial and there is IPC
overhead). Full `total_timesteps: 1000000` run finishes in ~2m49s and the trained policy
evaluates at **laps IQM 1.0, CI [1.0, 1.0], 0% crash, 117-step lap** across 30 held-out
seeds, vs both baselines at 0 laps / 100% crash.

---

## Phase 3: Evaluate honestly

Prove it learned to drive, with statistics not vibes.

**Technical approach:**
- Run the trained car and baselines across many seeds (>=20), each a fresh episode.
- Report **IQM** + **bootstrap 95% CIs** for: laps completed, crash rate, mean progress,
  best lap time.
- A win over the heuristic is claimed only when CIs separate.

**Deliverable:** `python -m raceline.eval` prints a metrics table with CIs + writes JSON.

**Research:** seed-aware eval (IQM + bootstrap CIs) is the rigor signal - see `research/stack.md`.

---

## Phase 4: Ship the demo

Export the policy and make it watchable.

**Technical approach:**
- Export the PPO policy network to ONNX (`export_onnx.py`).
- In-browser inference via `onnxruntime-web` (WASM). No backend.
- Canvas: draw the track, the car, and its sensor rays each frame. A **toggle** swaps
  between the untrained (episode-0) and trained brain on the same track. Show speed + lap
  counter.
- Fallback for a same-day link: a Gradio Space.

**Deliverable:** `web/` static demo runs locally and deploys to any static host. Before/
after HTML report in `docs/report/`.

**Research:** `research/stack.md` (ONNX + onnxruntime-web pattern).

---

## Phase 5+: Next level - make it go viral (see NORTH_STAR)

The build is end-to-end complete (env -> train -> eval -> ONNX -> browser). The next phases
chase *watchability*: a real research pass on the starred projects in this genre
(`research/similar-projects.md`) found the viral DNA is always the same three things - many
agents on screen at once, a counter that visibly climbs, and one-click interactivity. Each
phase below is its own vertical slice, ranked by wow-per-effort (S = a day or less, M = a
few days), evidence-first, with a before/after report.

### Phase 5: N ghost cars from one policy (S) - do this first
Spawn 10-100 cars all running the same trained ONNX policy, diverged by jittered spawn
position/heading + small action noise. They fan out, some crash, the clean ones survive -
the single most screenshot-shared image in this genre, manufactured from one policy with
zero retraining. Batch the N forward passes into one ONNX call per frame. Keep it honest:
this is one PPO policy with noise, not evolution.

### Phase 6: "What the car senses" mini-HUD (S)
Render live ray distances as a row of bars + the raw steer/throttle outputs as gauges, so a
viewer understands *why* the car turns. Cheap, high teaching value. Pairs with the sensor
rays already drawn on the canvas.

### Phase 7: Draw-your-own-track (M) - the biggest genuine differentiator
A canvas spline editor feeds a user-drawn track into the existing ray-sensor env and runs
the pretrained policy on it live. No famous demo lets you draw a track and watch a
pretrained policy attempt it. Generalization will be imperfect (trained on one track) -
which is itself an honest, interesting thing to surface: "watch it struggle on a track it
has never seen."

### Phase 8: Live network-activation overlay (M) - the "watch it think" shot
Draw the PPO MLP once (ray inputs -> hidden -> steer/throttle), animate edge color/width by
live activation each frame. The defining premium visual of the genre (MarI/O). Export
hidden-layer outputs as extra ONNX graph outputs, or re-derive the forward pass in JS.

### Phase 9: Save / load / share a run (S) - the growth loop
Download/upload the ONNX policy and a "share this run" link that encodes track + seed in the
URL. Turns viewers into participants.

### Phase 10: Racing-line ghost + reward shaping (M) - lap fast, not just finish
Precompute a minimum-curvature optimal line offline; draw it as a translucent ghost the car
chases, and add distance-to-line as a reward-shaping term so the policy laps fast, not just
completes. Elevates "doesn't crash" to "drives a clean racing line."

### Training-side polish (cheap credibility, optional)
- Second / unseen track as an honest generalization test (good writeup material).
- Beta-distribution continuous action head (bounded, stable steering/throttle).
- Frame-stack / action-repeat so the policy "sees" velocity.
- Reward-curve + lap-completion-over-time diagnostic plots from the SB3 Monitor CSV.

Full ranked analysis with sources: [research/similar-projects.md](research/similar-projects.md).
