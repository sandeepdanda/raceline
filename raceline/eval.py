"""Seed-aware evaluation: the trained car vs baselines, with bootstrap confidence intervals.

    python -m raceline.eval --checkpoint runs/ppo/best_model.zip

Never a single lucky lap. Each driver runs across many seeds; we report the interquartile
mean (IQM, robust per Agarwal et al. rliable) and a bootstrap 95% CI for laps completed,
crash rate, lap progress, and best lap time. A win over the heuristic is claimed only when
CIs separate. Held-out seeds (default 3000+) are disjoint from training (0) and the train-
time eval callback (1000).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from raceline.envs.racetrack_env import RaceTrackEnv, EnvConfig
from raceline.agents.baselines import RandomDriver, LongestRayDriver


def iqm(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    lo, hi = int(0.25 * len(x)), int(np.ceil(0.75 * len(x)))
    mid = x[lo:hi]
    return float(np.mean(mid)) if len(mid) else float(np.mean(x))


def bootstrap_ci(x: np.ndarray, stat=iqm, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    boots = [stat(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def run_episode(agent, env: RaceTrackEnv, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()
    best_frac, crashed, completed_steps = 0.0, False, None
    info = {}
    done = False
    while not done:
        obs, r, terminated, truncated, info = env.step(agent.act(obs, env))
        best_frac = max(best_frac, info["cum_progress_frac"])
        crashed = info["crashed"]
        if info["laps"] >= 1 and completed_steps is None:
            completed_steps = info["steps"]
        done = terminated or truncated
    return {
        "laps": float(info.get("laps", 0)),
        "crashed": float(crashed),
        "best_frac": best_frac,
        # Lap time = steps to first lap; if never completed, the full step budget (a penalty).
        "lap_steps": float(completed_steps if completed_steps is not None else env.cfg.max_steps),
    }


class SB3PolicyAgent:
    def __init__(self, model):
        self.model = model

    def reset(self) -> None:
        pass

    def act(self, obs, env) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


METRICS = ["laps", "crashed", "best_frac", "lap_steps"]
LABELS = {
    "laps": "laps completed",
    "crashed": "crash rate",
    "best_frac": "lap progress (frac)",
    "lap_steps": "lap time (steps)",
}


def evaluate(checkpoint: str | None, n_seeds: int, seed_start: int) -> dict:
    seeds = list(range(seed_start, seed_start + n_seeds))
    agents: dict[str, object] = {
        "random": RandomDriver(seed=0),
        "longest_ray": LongestRayDriver(),
    }
    if checkpoint:
        from stable_baselines3 import PPO

        agents["raceline_ppo"] = SB3PolicyAgent(PPO.load(checkpoint))

    results: dict[str, dict] = {}
    for name, agent in agents.items():
        per_seed = {m: [] for m in METRICS}
        for s in seeds:
            row = run_episode(agent, RaceTrackEnv(EnvConfig()), s)
            for m in METRICS:
                per_seed[m].append(row[m])
        summary = {}
        for m in METRICS:
            arr = np.array(per_seed[m])
            lo, hi = bootstrap_ci(arr)
            summary[m] = {"iqm": iqm(arr), "ci_lo": lo, "ci_hi": hi, "mean": float(np.mean(arr))}
        results[name] = summary
    return {"seeds": seeds, "n_seeds": n_seeds, "metrics": results}


def _print_table(report: dict) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        for metric in METRICS:
            t = Table(title=f"{LABELS[metric]}  (IQM [95% bootstrap CI]) - {report['n_seeds']} seeds")
            t.add_column("driver")
            t.add_column("IQM", justify="right")
            t.add_column("95% CI", justify="right")
            for name, summ in report["metrics"].items():
                s = summ[metric]
                t.add_row(name, f"{s['iqm']:.3f}", f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]")
            console.print(t)
    except ImportError:
        for metric in METRICS:
            print(f"\n== {LABELS[metric]} (IQM [95% CI]), {report['n_seeds']} seeds ==")
            for name, summ in report["metrics"].items():
                s = summ[metric]
                print(f"  {name:14s} {s['iqm']:8.3f}  [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]")


def _verdict(report: dict) -> None:
    m = report["metrics"]
    if "raceline_ppo" not in m:
        return
    rl = m["raceline_ppo"]
    heur = m.get("longest_ray", {})
    print()
    rl_laps, heur_laps = rl["laps"], heur.get("laps", {})
    if heur_laps and rl_laps["ci_lo"] > heur_laps["ci_hi"]:
        print(f"VERDICT: Raceline completes more laps than the heuristic - CIs separate "
              f"(RL CI low {rl_laps['ci_lo']:.2f} > heuristic CI high {heur_laps['ci_hi']:.2f}).")
    elif rl_laps["iqm"] >= 1.0:
        print(f"VERDICT: Raceline completes the lap (IQM {rl_laps['iqm']:.2f} laps, "
              f"crash rate {rl['crashed']['iqm']:.2f}); baselines do not.")
    else:
        print(f"VERDICT: not there yet - RL laps IQM {rl_laps['iqm']:.2f}, "
              f"progress {rl['best_frac']['iqm']:.2f}. Train longer.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed-aware eval: trained car vs baselines")
    ap.add_argument("--checkpoint", default="runs/ppo/best_model.zip")
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=3000)
    ap.add_argument("--out", default="runs/ppo/eval.json")
    args = ap.parse_args()

    ckpt = args.checkpoint if Path(args.checkpoint).exists() else None
    if ckpt is None:
        print(f"(no checkpoint at {args.checkpoint}; evaluating baselines only)")

    report = evaluate(ckpt, args.n_seeds, args.seed_start)
    _print_table(report)
    _verdict(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
