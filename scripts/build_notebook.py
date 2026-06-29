"""Generate notebooks/raceline.ipynb programmatically (valid nbformat, no hand-editing).

Run: python scripts/build_notebook.py  ->  writes notebooks/raceline.ipynb
The notebook walks through: setup, the env, train (short demo run), learning curve,
evaluation, and an inline animation of a learned lap. Includes a CPU-vs-MPS device note.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(text.strip("\n"))


cells = []

cells.append(md(r"""
# Raceline - train a car to race, in a notebook

A little 2D car with distance sensors learns to lap a track using PPO. This notebook lets
you train it, watch the learning curve, evaluate it honestly, and animate a learned lap -
all inline.

**A note on GPUs (important):** this project is *CPU-bound on purpose*. Training time is
dominated by stepping the Python car simulator, and the neural net is tiny (`[64,64]`), so a
GPU does not help and often hurts (data-transfer overhead beats the tiny matmul). The real
speedup for this kind of RL is *parallel environments* (many cars at once), not a GPU. The
device toggle below lets you try MPS/CUDA and benchmark it yourself.
"""))

cells.append(md("## 1. Setup\nMake sure you installed the project with notebook extras: `uv pip install -e \".[notebook]\"`."))

cells.append(code(r"""
import sys, time, pathlib
# Make the repo importable when the notebook runs from notebooks/.
ROOT = pathlib.Path.cwd()
if (ROOT / "raceline").exists():
    REPO = ROOT
elif (ROOT.parent / "raceline").exists():
    REPO = ROOT.parent
else:
    raise RuntimeError("run this notebook from the raceline repo (or notebooks/ inside it)")
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import matplotlib.pyplot as plt

print("repo:", REPO)
print("torch:", torch.__version__)
print("CUDA (NVIDIA GPU):", torch.cuda.is_available())
print("MPS  (Apple GPU): ", torch.backends.mps.is_available())
"""))

cells.append(md(r"""
## 2. The environment

`RaceTrackEnv` is a real Gymnasium env: the car sees 7 normalized distance-sensor rays plus
its speed (8 numbers), and picks one of 9 (steer x throttle) actions. Reward pays for
forward progress along the track and penalizes crashing. Let's look at one observation.
"""))

cells.append(code(r"""
from raceline.envs.racetrack_env import RaceTrackEnv, ACTIONS

env = RaceTrackEnv()
obs, info = env.reset(seed=0)
print("observation space:", env.observation_space)
print("action space:", env.action_space, "->", len(ACTIONS), "(steer, throttle) combos")
print("first observation (7 rays + speed):", np.round(obs, 2))
"""))

cells.append(md("Draw the track + the car's sensor rays at the start, so you can see what it 'sees':"))

cells.append(code(r"""
def draw_track(ax, track):
    for poly in (track.outer, track.inner):
        p = np.array(list(poly) + [poly[0]])
        ax.plot(p[:, 0], p[:, 1], color="#6e7681", lw=1.5)

def draw_car(ax, env):
    rays = env.track.cast_rays(env._pos, env._heading, env._ray_angles, env.cfg.max_range)
    for ang, d in zip(env._ray_angles, rays):
        a = env._heading + ang
        ax.plot([env._pos[0], env._pos[0] + np.cos(a) * d],
                [env._pos[1], env._pos[1] + np.sin(a) * d], color="#3fb950", lw=0.8)
    ax.plot(env._pos[0], env._pos[1], "o", color="#2f81f7", ms=8)

fig, ax = plt.subplots(figsize=(7, 4.5))
draw_track(ax, env.track); draw_car(ax, env)
ax.set_aspect("equal"); ax.set_title("Raceline track + the car's distance sensors"); ax.axis("off")
plt.show()
"""))

cells.append(md(r"""
## 3. Train

We train PPO with Stable-Baselines3. This notebook trains for **400k steps** (~3 min on CPU), enough to reliably learn clean
laps; the full project config (`configs/ppo.yaml`) uses 1M steps across 8 parallel envs for
the polished policy (IQM 1.0, 0% crash on held-out seeds).
Drop `TIMESTEPS` to 150k for a quick smoke run, or raise it for a sharper driver.

**Device toggle:** set `DEVICE = "cpu"` (recommended), or `"mps"` / `"cuda"` to experiment.
For this tiny MLP, CPU is normally fastest - the cell prints wall-clock time so you can
compare yourself.
"""))

cells.append(code(r"""
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from raceline.seeding import seed_everything

DEVICE = "cpu"          # try "mps" (Apple GPU) or "cuda" (NVIDIA) and compare the time
TIMESTEPS = 400_000     # ~3 min on CPU and reliably learns to lap; drop to 150k for a quick smoke run

seed_everything(0)
train_env = Monitor(RaceTrackEnv())
train_env.reset(seed=0)

model = PPO(
    "MlpPolicy", train_env, seed=0,
    n_steps=2048, batch_size=256, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
    ent_coef=0.01, clip_range=0.2,
    policy_kwargs={"net_arch": [64, 64]},
    device=DEVICE, verbose=0,
)

t0 = time.time()
model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
print(f"\ntrained {TIMESTEPS} steps on '{DEVICE}' in {time.time() - t0:.1f}s")
"""))

cells.append(md("## 4. Learning curve\nReward per episode over training, read from the Monitor wrapper. It should climb from near-zero toward the track length (~251 = a full lap)."))

cells.append(code(r"""
import numpy as np
rewards = train_env.get_episode_rewards()
if rewards:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rewards, color="#2f81f7", alpha=0.4, lw=1, label="episode reward")
    if len(rewards) >= 10:
        k = 10
        ma = np.convolve(rewards, np.ones(k) / k, mode="valid")
        ax.plot(range(k - 1, len(rewards)), ma, color="#2f81f7", lw=2.5, label="10-ep moving avg")
    ax.axhline(env.track.length, color="#3fb950", ls="--", lw=1, label="≈ one full lap")
    ax.set_xlabel("episode"); ax.set_ylabel("reward"); ax.legend(); ax.set_title("Raceline learning curve")
    plt.show()
else:
    print("no episodes recorded yet - increase TIMESTEPS")
"""))

cells.append(md(r"""
## 5. Evaluate honestly

One lucky lap proves nothing. Run the trained car across many fresh seeds and report the
fraction that complete a clean lap and the crash rate. (The full project also reports IQM +
bootstrap CIs in `raceline/eval.py`.)
"""))

cells.append(code(r"""
def rollout(model, seed):
    e = RaceTrackEnv()
    obs, _ = e.reset(seed=seed)
    info = {}
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = e.step(int(action))
        done = term or trunc
    return info["laps"], info["crashed"], info["steps"]

seeds = range(3000, 3020)
results = [rollout(model, s) for s in seeds]
laps = np.mean([r[0] >= 1 for r in results])
crashes = np.mean([r[1] for r in results])
print(f"over {len(results)} held-out seeds:")
print(f"  completed a full lap: {laps*100:.0f}%")
print(f"  crashed:              {crashes*100:.0f}%")
print("(a short 150k run may not hit 100% - the project config trains to 1M for the polished result)")
"""))

cells.append(md("## 6. Watch a learned lap\nAnimate one episode inline. The car (blue triangle) drives; green lines are its live sensors."))

cells.append(code(r"""
from matplotlib import animation
from IPython.display import HTML

e = RaceTrackEnv()
obs, _ = e.reset(seed=3000)
frames = []
done = False
while not done:
    frames.append((e._pos.copy(), e._heading,
                   e.track.cast_rays(e._pos, e._heading, e._ray_angles, e.cfg.max_range)))
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, info = e.step(int(action))
    done = term or trunc

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.set_aspect("equal"); ax.axis("off")
draw_track(ax, e.track)
car_dot, = ax.plot([], [], "o", color="#2f81f7", ms=9)
ray_lines = [ax.plot([], [], color="#3fb950", lw=0.8)[0] for _ in e._ray_angles]
title = ax.set_title("")

def update(i):
    pos, hd, rays = frames[i]
    car_dot.set_data([pos[0]], [pos[1]])
    for line, ang, d in zip(ray_lines, e._ray_angles, rays):
        a = hd + ang
        line.set_data([pos[0], pos[0] + np.cos(a) * d], [pos[1], pos[1] + np.sin(a) * d])
    title.set_text(f"learned lap - step {i+1}/{len(frames)}")
    return [car_dot, *ray_lines, title]

anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=40, blit=True)
plt.close(fig)
HTML(anim.to_jshtml())
"""))

cells.append(md(r"""
## Next steps

- Bump `TIMESTEPS` to 1M for the polished policy (the project's `configs/ppo.yaml`).
- Want it faster? Use **parallel environments** (`SubprocVecEnv` with 8 envs), not a GPU -
  `python -m raceline.train` already does this (2.3x on 12 cores). See `ROADMAP.md` Phase 2.5.
- Export the trained policy to ONNX (`python -m raceline.export_onnx`) and watch it drive in
  the browser demo under `web/`.
"""))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (raceline)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out = Path("notebooks/raceline.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print(f"wrote {out} ({len(cells)} cells)")
