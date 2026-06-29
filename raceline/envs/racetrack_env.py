"""RaceTrackEnv - a custom Gymnasium environment where a car learns to race a track.

A top-down car with distance-ray sensors drives a closed walled track. Each step it sees
its sensor rays + speed and picks a (steer, throttle) action; a kinematic car model
advances, we check for a wall crash and measure progress along the track centerline, and
return a reward that pays for forward progress and punishes crashing.

The car is given no map - only how far the walls are along a few rays. Learning to corner
from that minimal signal is the whole point.

Gymnasium 1.3 API: keyword-only seeded ``reset``, 5-tuple ``step``, ``terminated`` (crash
or laps done - a real MDP terminal) vs ``truncated`` (step-limit time-out). The split lets
PPO bootstrap value correctly at a time-out but not at a crash.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .track import Track, oval_track

# Action = (steer, throttle) grid. steer in {-1 left, 0 straight, +1 right};
# throttle in {-1 brake, 0 coast, +1 gas}. 3x3 = 9 discrete actions.
STEER_CHOICES = (-1.0, 0.0, 1.0)
THROTTLE_CHOICES = (-1.0, 0.0, 1.0)
ACTIONS = [(s, t) for s in STEER_CHOICES for t in THROTTLE_CHOICES]


@dataclass
class EnvConfig:
    # --- sensors ---
    n_rays: int = 7
    ray_spread: float = float(np.pi * 0.75)  # total fan angle in front of the car
    max_range: float = 60.0

    # --- car physics ---
    dt: float = 0.2
    max_steer: float = 0.6           # radians/step at full lock (scaled by speed)
    accel: float = 4.0               # speed gained per step at full gas
    brake: float = 9.0               # speed lost per step braking
    drag: float = 0.04               # proportional speed decay per step (no hard stall)
    min_speed: float = 2.0           # car always rolls a little (keeps it controllable)
    max_speed: float = 12.0
    start_speed: float = 6.0

    # --- episode ---
    max_steps: int = 800
    target_laps: int = 1             # terminate after completing this many laps

    # --- reward (see PROJECT.md for per-term justification) ---
    w_progress: float = 1.0          # per unit of arc-length advanced along the loop
    w_time: float = 0.02             # small per-step cost -> finish faster
    w_crash: float = 5.0             # one-time penalty on hitting a wall

    # --- track ---
    half_width: float = 9.0


class RaceTrackEnv(gym.Env):
    """Gymnasium environment: a sensor-driven car learns to lap a track without crashing."""

    metadata = {"render_modes": ["ansi"], "render_fps": 30}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None,
                 track: Track | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.track = track or oval_track(self.cfg.half_width)

        self._ray_angles = np.linspace(
            -self.cfg.ray_spread / 2, self.cfg.ray_spread / 2, self.cfg.n_rays
        )

        self.action_space = spaces.Discrete(len(ACTIONS))
        # Observation: N normalized ray distances + normalized speed.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.cfg.n_rays + 1,), dtype=np.float32
        )

        # State set in reset().
        self._pos = np.zeros(2)
        self._heading = 0.0
        self._speed = 0.0
        self._t = 0
        self._prev_progress = 0.0
        self._laps = 0
        self._cum_progress = 0.0

    # ------------------------------------------------------------------ helpers
    def _start_pose(self) -> tuple[np.ndarray, float]:
        """Place the car on the centerline at waypoint 0, heading along the track."""
        c = self.track.center
        pos = c[0].copy()
        nxt = c[1]
        heading = float(np.arctan2(nxt[1] - pos[1], nxt[0] - pos[0]))
        return pos, heading

    def _observation(self) -> np.ndarray:
        rays = self.track.cast_rays(self._pos, self._heading, self._ray_angles, self.cfg.max_range)
        obs = np.concatenate([rays / self.cfg.max_range, [self._speed / self.cfg.max_speed]])
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._pos, self._heading = self._start_pose()
        # A little seeded jitter in start speed/heading so seeds give distinct episodes.
        self._speed = self.cfg.start_speed
        self._heading += float(self.np_random.uniform(-0.05, 0.05))
        self._t = 0
        self._laps = 0
        self._cum_progress = 0.0
        self._prev_progress = self.track.progress(self._pos)
        return self._observation(), {}

    def step(self, action: int):
        cfg = self.cfg
        steer, throttle = ACTIONS[int(action)]

        # 1. Update speed from throttle, friction, cap.
        if throttle > 0:
            self._speed += cfg.accel * throttle * cfg.dt
        elif throttle < 0:
            self._speed += cfg.brake * throttle * cfg.dt  # throttle negative -> slows
        self._speed *= (1.0 - cfg.drag)                    # proportional drag, never stalls
        self._speed = float(np.clip(self._speed, cfg.min_speed, cfg.max_speed))

        # 2. Steering authority scales mildly with speed but keeps a floor, so a slow car
        #    can still corner (power-steering-like) - a pure speed scaling makes a careful
        #    slow driver unable to turn at all.
        speed_frac = 0.5 + 0.5 * (self._speed / cfg.max_speed)
        self._heading += steer * cfg.max_steer * speed_frac

        # 3. Integrate position.
        self._pos = self._pos + self._speed * cfg.dt * np.array(
            [np.cos(self._heading), np.sin(self._heading)]
        )

        # 4. Progress along the loop (handle wrap past the start/finish line).
        progress = self.track.progress(self._pos)
        delta = progress - self._prev_progress
        if delta < -self.track.length / 2:      # wrapped forward across the seam
            delta += self.track.length
            self._laps += 1
        elif delta > self.track.length / 2:      # wrapped backward
            delta -= self.track.length
        self._prev_progress = progress
        self._cum_progress += delta

        # 5. Collision check.
        crashed = self.track.distance_from_center(self._pos) > cfg.half_width

        # 6. Reward.
        reward = cfg.w_progress * delta - cfg.w_time
        if crashed:
            reward -= cfg.w_crash

        # 7. Termination vs truncation.
        self._t += 1
        terminated = bool(crashed or self._laps >= cfg.target_laps)
        truncated = bool(self._t >= cfg.max_steps)

        info = {
            "crashed": crashed,
            "laps": self._laps,
            "progress": progress,
            "lap_fraction": self.track.lap_fraction(self._pos),
            "cum_progress_frac": self._cum_progress / self.track.length,
            "speed": self._speed,
            "steps": self._t,
        }
        obs = (
            self._observation()
            if not (terminated or truncated)
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "ansi":
            return None
        return (
            f"t={self._t:4d} pos=({self._pos[0]:6.1f},{self._pos[1]:6.1f}) "
            f"hd={self._heading:5.2f} v={self._speed:4.1f} "
            f"lap%={self.track.lap_fraction(self._pos):.2f} laps={self._laps}"
        )


if "Raceline-Track-v0" not in gym.registry:
    gym.register(id="Raceline-Track-v0", entry_point="raceline.envs.racetrack_env:RaceTrackEnv")


def _selftest() -> None:
    """Run random + heuristic drivers over a few episodes and print progress / crashes."""
    from raceline.agents.baselines import LongestRayDriver, RandomDriver

    def run(agent, seed: int) -> dict:
        env = RaceTrackEnv()
        obs, _ = env.reset(seed=seed)
        agent.reset()
        tot_r, steps = 0.0, 0
        best_frac, crashed = 0.0, False
        done = False
        while not done:
            obs, r, terminated, truncated, info = env.step(agent.act(obs, env))
            tot_r += r
            steps += 1
            best_frac = max(best_frac, info["cum_progress_frac"])
            crashed = info["crashed"]
            done = terminated or truncated
        return {"return": tot_r, "steps": steps, "best_lap_frac": best_frac,
                "laps": info["laps"], "crashed": crashed}

    print("RaceTrackEnv self-test (mean over seeds 0-4):")
    for name, agent in [("random      ", RandomDriver()), ("longest_ray ", LongestRayDriver())]:
        rows = [run(agent, s) for s in range(5)]
        mean = lambda k: np.mean([r[k] for r in rows])
        print(
            f"  {name}  return={mean('return'):7.2f}  best_lap_frac={mean('best_lap_frac'):.2f}  "
            f"laps={mean('laps'):.1f}  crash_rate={np.mean([r['crashed'] for r in rows]):.2f}"
        )
    print("(longest_ray should make more lap progress than random)")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("usage: python -m raceline.envs.racetrack_env --selftest")
