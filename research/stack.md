# Modern Python Reinforcement-Learning Tooling Stack (2025-2026)

**Date:** 2026-06-29 PST
**Goal:** Pick the best libraries to build a portfolio-grade RL project: trained agent + reproducible training + web demo.
**Scope:** Public-web sources only (PyPI, GitHub, official docs). Researched via WebFetch/WebSearch.
**Confidence:** High on versions/dates/APIs (pulled live from PyPI + GitHub + official docs). Medium only where flagged inline.

---

## TL;DR

Build on the **Farama ecosystem**: Gymnasium 1.3.0 is the standard env API (the `step()` 5-tuple with `terminated`/`truncated` replaces the old `done`). For the agent code, **CleanRL** is the best fit for a portfolio piece, single-file, readable, reproducible, with built-in wandb/TensorBoard logging, optionally backstopped by **Stable-Baselines3** as a trusted baseline. Track experiments with **TensorBoard** (offline default) + **wandb** (free, shareable), config via **Hydra 1.3.3**, and seed all four RNGs. For the web demo, export the policy to **ONNX** and serve it **fully in-browser via onnxruntime-web on GitHub Pages**, zero backend, zero cost, permanent link.

---

## 1. Gymnasium (successor to OpenAI Gym)

**Latest version: `gymnasium 1.3.0`**, released 2026-04-22. (source: https://pypi.org/project/gymnasium/)

### The `step()` API (5-tuple)

`Env.step(action)` returns `(observation, reward, terminated, truncated, info)`. (source: https://gymnasium.farama.org/api/env/)

- **`terminated`** - "Whether the agent reaches the terminal state (as defined under the MDP of the task)." This is a real MDP terminal: value bootstrapping should stop here.
- **`truncated`** - "Whether the truncation condition outside the scope of the MDP is satisfied. Typically, this is a timelimit." The episode was still going, so bootstrapping should continue.

Splitting the old ambiguous `done` into these two is the whole point of the migration: it lets RL algorithms bootstrap value estimates correctly. (source: https://gymnasium.farama.org/introduction/migration_guide/)

### The `reset()` API

```python
Env.reset(*, seed: int | None = None, options: dict | None = None) -> tuple[ObsType, dict]
```
Returns `(observation, info)`. `seed` and `options` are keyword-only. Pass the seed once "right after the environment has been initialized and then never again", there is no standalone `env.seed()` in modern Gymnasium. (source: https://gymnasium.farama.org/api/env/)

### Defining a custom environment

Subclass `gymnasium.Env`, declare `metadata`, define spaces, implement `reset`/`step`/`render`, then register. (source: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)

```python
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, size - 1, shape=(2,), dtype=int)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # ...
        return observation, info

    def step(self, action):
        # ...
        return observation, reward, terminated, False, info  # truncated handled by TimeLimit wrapper

register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)
```

The `id` is `namespace/Name-vN`. Instantiate with `gymnasium.make("gymnasium_env/GridWorld-v0")`.

### Gym -> Gymnasium migration

Gymnasium is a fork of OpenAI Gym v0.26, created by Gym's own maintainers; "OpenAI handed over maintenance a few years ago to an outside team" (the Farama Foundation), and Gymnasium is "where future maintenance will occur going forward." Use `import gymnasium as gym` as a near drop-in replacement. (sources: https://github.com/Farama-Foundation/Gymnasium, https://gymnasium.farama.org/introduction/migration_guide/)

---

## 2. Agent / algorithm libraries

| Library | Latest version | Date | Stars | Philosophy | GPU |
|---|---|---|---|---|---|
| Stable-Baselines3 | 2.9.0 | 2026-06-15 | 13.5k | Stable, batteries-included high-level API (`model.learn()`, save/load) | CUDA via PyTorch |
| CleanRL | v1.0.0 tagged (active on master) | 2022-11 tag | 10k | Single-file, readable; not an importable library | CUDA + JAX/XLA variants |
| RLlib (Ray) | Ray 2.55.1 | 2026-04-22 | (Ray) large | Distributed/scalable, multi-node clusters | Multi-GPU/multi-node |
| Tianshou | 2.0.1 | 2026-04-02 | 10.8k | Modular framework, very broad algorithm coverage | CUDA, multi-GPU |
| SBX (SB3-Jax) | 0.27.0 | 2026-06-15 | - | SB3 ergonomics on JAX for speed; proof-of-concept | JAX/XLA (GPU+TPU) |
| PureJaxRL | none ("No releases published") | - | 1.1k | End-to-end JAX, env-in-JAX, research codebase | JAX/XLA (GPU+TPU) |

Sources: [SB3 PyPI](https://pypi.org/pypi/stable-baselines3/json), [SB3 GitHub](https://github.com/DLR-RM/stable-baselines3); [CleanRL](https://github.com/vwxyzjn/cleanrl); [Ray PyPI](https://pypi.org/project/ray/2.55.1/), [RLlib docs](https://docs.ray.io/en/latest/rllib/index.html); [Tianshou PyPI](https://pypi.org/pypi/tianshou/json), [Tianshou GitHub](https://github.com/thu-ml/tianshou); [SBX PyPI](https://pypi.org/pypi/sbx-rl/json); [PureJaxRL](https://github.com/luchris429/purejaxrl).

Notes on maturity:
- **SB3** describes itself as stable, with development "focused on bug fixes and maintenance"; requires PyTorch >=2.8. Strongest docs (dedicated Read the Docs site, type hints, beginner emphasis). (source: https://github.com/DLR-RM/stable-baselines3)
- **CleanRL**: last *tagged* release is v1.0.0 (2022) but the repo has 843 commits and active master development; treat it as "actively maintained, lightly versioned." It is "high-quality single-file implementation," "not meant to be imported", each algorithm is one standalone script (~340 lines) with TensorBoard, wandb, seeding, and video capture built in. (source: https://github.com/vwxyzjn/cleanrl)
- **SBX** is by Antonin Raffin (also an SB3 maintainer), self-described "proof-of-concept reimplementation". (source: https://pypi.org/pypi/sbx-rl/json)
- **PureJaxRL** is a research codebase (35 commits, no releases); claims up to "1000x faster" by keeping the env inside JAX to avoid CPU-GPU transfer. (source: https://github.com/luchris429/purejaxrl)

### Recommendation: CleanRL (primary), SB3 (baseline)

For a solo dev who wants clean, reproducible, single-file-readable training that looks good on GitHub, use **CleanRL**. It is the only library built around the single-file philosophy: a reviewer opens one `.py` and reads the entire algorithm top to bottom. Reproducibility, TensorBoard/wandb logging, and published benchmarks are already there.

The CleanRL vs SB3 tradeoff:
- **CleanRL** when the *code itself* is the deliverable, readable, educational, self-contained. Cost: no clean import API; you copy/edit files instead of `pip install` and call.
- **SB3** when you want fast, stable *results* and serialization (`model.learn()` then `model.save()`). Cost: the algorithm is abstracted across the library, so it doesn't showcase your understanding of internals.

Strong combined play: build the showcase on CleanRL, use SB3 as a baseline/sanity check to validate your CleanRL results match a trusted reference. That demonstrates both algorithm understanding and engineering judgment. (sources: https://github.com/vwxyzjn/cleanrl, https://github.com/DLR-RM/stable-baselines3)

Skip **RLlib** for solo work, its value is distributed clusters, which is a liability not a selling point for one dev. Reach for **Tianshou** if you want breadth across many algorithms (offline RL, imitation) behind a modular API; **SBX/PureJaxRL** only if raw JAX throughput *is* the project's thesis.

---

## 3. Environment ecosystems

| Ecosystem | Latest | Date | Maintained? | Modern Gymnasium API? |
|---|---|---|---|---|
| PettingZoo (multi-agent) | 1.26.1 | 2026-04-27 | Yes (Farama) | Yes (AEC + Parallel multi-agent) |
| Gymnasium-Robotics | 1.4.2 | 2026-01-02 | Yes (Farama) | Yes (`gymnasium>=1.2.0`) |
| MuJoCo | 3.10.0 | 2026-06-22 | Yes (Google DeepMind) | Via bindings, not direct |
| Atari (ale-py) | 0.12.0 | 2026-05-29 | Yes (Farama) | Yes (Gymnasium-only since 0.9.0) |
| MiniGrid | 3.1.0 | 2026-05-11 | Yes (Farama) | Yes (`gymnasium>=0.28.1`) |
| gym-anytrading | 2.0.0 | 2023-08-27 | Stale since 2023 | Yes |
| FinRL | GH v0.3.8 / PyPI 0.3.7 | 2026-03-20 / 2024-04-12 | Active on GitHub, PyPI stale | Yes (Gymnasium) |

- **PettingZoo**: "Gymnasium for multi-agent reinforcement learning", MIT, Python <3.15,>=3.9. (source: https://pypi.org/pypi/pettingzoo/json)
- **Gymnasium-Robotics**: "RL Robotics environments with Gymnasium API", depends on `gymnasium>=1.2.0` and `mujoco>=2.2.0`; the canonical way to consume MuJoCo robotics tasks (Fetch, Hand, Maze). (source: https://pypi.org/pypi/gymnasium-robotics/json)
- **MuJoCo is fully free and open-source: Apache-2.0**, "maintained by Google DeepMind", "Copyright 2021 DeepMind Technologies Limited." It is the physics engine; reach it via the official `mujoco` Python bindings (modern Gymnasium dropped the old `mujoco-py`). (source: https://github.com/google-deepmind/mujoco) [Medium confidence on the exact open-source date, the README shows a 2021 copyright but no explicit release date.]
- **Atari/ale-py**: GPL-2.0, Farama-owned. ROM headache is gone, since v0.9.0 ROMs ship inside the PyPI package and Gymnasium is the sole backend. Caveat: `import ale_py` before `gymnasium.make(...)` to register Atari envs. (source: https://github.com/Farama-Foundation/Arcade-Learning-Environment/releases)
- **MiniGrid**: "Minimalistic gridworld" envs, MIT, Farama, modern Gymnasium. (source: https://pypi.org/pypi/minigrid/json)
- **gym-anytrading**: uses modern Gymnasium (`import gymnasium as gym`, 5-tuple step, `reset(seed=)`) but quiet since Aug 2023 (~42 commits). Fine for simple single-asset trading; no recent updates. (source: https://github.com/AminHP/gym-anytrading)
- **FinRL**: uses Gymnasium (not legacy gym). Active on GitHub (GH v0.3.8, 2026-03-20, ~3,246 commits) but the PyPI `finrl` is stale (0.3.7, 2024-04-12, no dependency metadata), so install from GitHub `master`, not `pip install finrl`. Now positioned as the "Stage 1.0" educational framework; production users are pointed to FinRL-X / FinRL-Trading. (sources: https://github.com/AI4Finance-Foundation/FinRL, https://pypi.org/pypi/finrl/json)

---

## 4. Experiment tracking + reproducibility

What a hiring manager expects maps onto the Papers With Code "ML Code Completeness Checklist" (the official NeurIPS code-submission guideline): (1) pinned dependencies, (2) a `train.py` that reproduces results, (3) `eval.py`, (4) released checkpoints, (5) a README with a results table and exact reproduction commands. Repos meeting all five correlate with more stars ("median of 196 and mean of 2,664 stars"). (source: https://github.com/paperswithcode/releasing-research-code)

### Weights & Biases (wandb)
Logs metrics (`run.log({...})`), hyperparameters (`config`), and media (videos via `wandb.Video`) to an interactive cloud dashboard. Free tier is $0/mo, "Designed for personal development", 5 GB/mo, up to 5 seats, right for a portfolio. Both CleanRL (`--track --wandb-project-name ...`) and SB3 (`from wandb.integration.sb3 import WandbCallback` with `sync_tensorboard=True`, `monitor_gym=True`) integrate. (sources: https://docs.wandb.ai/guides/track/, https://wandb.ai/site/pricing/, https://stable-baselines3.readthedocs.io/en/master/guide/integrations.html)

### TensorBoard
The zero-cost, offline, no-account default. SB3 wires it in with one constructor arg: `A2C("MlpPolicy", "CartPole-v1", tensorboard_log="./tb/")`, auto-logging `rollout/ep_rew_mean` (the canonical RL learning-curve screenshot), `rollout/ep_len_mean`, `time/fps`, and `train/` losses. PyTorch entry point is `torch.utils.tensorboard.SummaryWriter`. (sources: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html, https://www.tensorflow.org/tensorboard/get_started)

### Hydra (config management)
**Latest: `hydra-core 1.3.3`** (2026-06-11). Composable hierarchical configs, CLI overrides, multirun sweeps (`-m`), and per-run timestamped output dirs that capture the exact composed config that produced each result. Structured Configs (dataclasses) add runtime type checking. Built on OmegaConf. Pin `hydra-core==1.3.3`. (sources: https://pypi.org/project/hydra-core/, https://hydra.cc/docs/intro/)

### Reproducibility practices
Seed all four RNGs from one seed at process start (CleanRL's exact pattern):
```python
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
obs, info = env.reset(seed=seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
# export CUBLAS_WORKSPACE_CONFIG=:4096:8   (CUDA >= 10.2)
```
"Completely reproducible results are not guaranteed across PyTorch releases." Pin deps with `pip freeze > requirements.txt` (or `uv.lock`/`poetry.lock`). CleanRL's `ppo.py` is the de-facto reference. The canonical "why" citation is Henderson et al., "Deep Reinforcement Learning that Matters" (2018). Report multi-seed variance, not a single curve. (sources: https://docs.pytorch.org/docs/stable/notes/randomness.html, https://gymnasium.farama.org/api/env/, https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py, https://arxiv.org/abs/1709.06560)

---

## 5. Serving / deploying the trained policy in a web app

### ONNX export
SB3 documents this on its "Export to ONNX" page. Wrap the policy in a small `nn.Module` (for SAC the actor alone suffices), then `torch.onnx.export`. Requires PyTorch 2.0+ and ONNX Opset 14+. Caveat: the export "returns normalized actions and doesn't include the post-processing" (clipping/unscaling), handle that in your app code; CNN policies need the /255 observation normalization. (source: https://stable-baselines3.readthedocs.io/en/master/guide/export.html)

### Server-side: FastAPI + onnxruntime
Load the `.onnx` once at startup with `onnxruntime.InferenceSession`, call `.run()` per request, wrap in a FastAPI route returning JSON. (sources: https://onnxruntime.ai/docs/get-started/with-python.html, https://fastapi.tiangolo.com/)

### In-browser: onnxruntime-web
"Run and deploy machine learning models in your web application using JavaScript APIs." WASM backend supports all ONNX operators (small MLP policies run fully). Docs: it "It's cheaper... It's safer... It works offline... It's faster." Model + `ort.min.js` are static assets, so the whole demo is static files. TF.js is the alternative if your policy is already a TF/Keras model. (sources: https://onnxruntime.ai/docs/tutorials/web/, https://www.tensorflow.org/js)

### Best lightweight pattern (comparison)
| Option | Hosting | Backend cost | Best when |
|---|---|---|---|
| (a) FastAPI + frontend | Running server | Pay for compute, keep warm, CORS | Model too big for browser, or showing API design |
| (b) onnxruntime-web on GitHub Pages | Static files | **Zero** | Small policy, permanent free demo |
| (c) Gradio/Streamlit on HF Spaces | HF hosts | Free CPU, sleeps when idle | Hosted interactive UI in minutes |

**Recommendation: (b) fully in-browser onnxruntime-web on GitHub Pages.** An RL policy net is tiny (a few MLP layers), so it loads and runs instantly in WASM. Zero backend cost, no cold starts, no credentials, a URL that stays live indefinitely, the cleanest story for a recruiter clicking the link months later. Practical hedge: do (c) on HuggingFace Spaces first (live in an afternoon via Gradio `share=True`), then port to (b) for the permanent link. (sources: https://www.gradio.app/guides/quickstart, https://huggingface.co/docs/hub/spaces-overview)

---

## 6. Recommended pinned dependency set (2026 project)

All versions verified live against PyPI/GitHub on 2026-06-29. CleanRL is cloned, not pip-installed (it is single-file scripts), so it is not a pinned dep, vendor the scripts you use into your repo.

```txt
# requirements.txt  (Python 3.11+; CUDA build of torch as appropriate)

# --- core env API ---
gymnasium==1.3.0

# --- agent / algorithms ---
stable-baselines3==2.9.0      # baseline + serialization; PyTorch >=2.8
# CleanRL: vendor scripts from github.com/vwxyzjn/cleanrl (not a pip dep)

# --- environments (add only what the project uses) ---
ale-py==0.12.0                # Atari (ROMs bundled; import ale_py before gymnasium.make)
minigrid==3.1.0               # gridworld
mujoco==3.10.0                # physics engine (free, Apache-2.0)
gymnasium-robotics==1.4.2     # MuJoCo robotics tasks via Gymnasium
pettingzoo==1.26.1            # multi-agent (if needed)
gym-anytrading==2.0.0         # finance (stale but Gymnasium-compatible)
# FinRL: install from GitHub master, NOT pip (PyPI is stale 0.3.7)

# --- experiment tracking + config ---
wandb                          # pin at build time via `pip index versions wandb`
tensorboard
hydra-core==1.3.3

# --- serving / deploy ---
onnx
onnxruntime                    # server-side inference
fastapi                        # if using a backend
uvicorn                        # ASGI server for FastAPI
gradio                         # fastest hosted demo (HF Spaces)
# onnxruntime-web: JS package (npm / CDN), not a Python dep
```

Sources for each pin: gymnasium https://pypi.org/project/gymnasium/ | stable-baselines3 https://pypi.org/pypi/stable-baselines3/json | ale-py https://github.com/Farama-Foundation/Arcade-Learning-Environment/releases | minigrid https://pypi.org/pypi/minigrid/json | mujoco https://github.com/google-deepmind/mujoco/releases/latest | gymnasium-robotics https://pypi.org/pypi/gymnasium-robotics/json | pettingzoo https://pypi.org/pypi/pettingzoo/json | gym-anytrading https://pypi.org/pypi/gym-anytrading/json | hydra-core https://pypi.org/project/hydra-core/

---

## Unknowns / caveats

- **CleanRL / PureJaxRL last-commit dates**: inferred from commit counts + release dates, not direct timestamps. Run `git log -1` after cloning, or check the GitHub API `pushed_at`, to confirm freshness.
- **MuJoCo open-source date**: README shows a 2021 copyright but no explicit release date; the 2021-2022 timeline is consistent but not directly stated.
- **wandb version pin**: no page surfaced a specific pinned release; resolve with `pip index versions wandb` at build time.
- **ale-py 0.12.0 upload timestamp**: PyPI metadata rendered only through 0.11.2 (2025-07-12); 0.12.0 date taken from the GitHub releases page.
- **PyTorch deterministic env var**: `CUBLAS_WORKSPACE_CONFIG=:4096:8` was removed from the current (2.12) PyTorch docs but the values still apply (verified against pinned v2.3.0 source).
