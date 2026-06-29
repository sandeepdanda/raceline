"""Non-learned drivers to compare the RL policy against.

RandomDriver is the floor. LongestRayDriver is a real heuristic that already drives
*somewhat* - it steers toward the most open direction - so a PPO win over it is meaningful,
not just "beat random". Both expose ``reset()`` and ``act(obs, env) -> action_index``.
"""

from __future__ import annotations

import numpy as np

from raceline.envs.racetrack_env import ACTIONS, STEER_CHOICES, THROTTLE_CHOICES


def _action_index(steer: float, throttle: float) -> int:
    return ACTIONS.index((steer, throttle))


class RandomDriver:
    """Uniform random actions. The floor every learned policy must clear."""

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs, env) -> int:
        return int(self._rng.integers(0, env.action_space.n))


class LongestRayDriver:
    """Steer toward the longest sensor ray (most open space), modest constant throttle.

    The observation is ``[ray_0 .. ray_{N-1}, speed]`` with rays ordered left->right. Pick
    the ray with the most clearance and steer that way; gas when fairly straight, coast
    when turning hard. A simple reactive driver that laps slowly but rarely random-crashes.
    """

    def reset(self) -> None:
        pass

    def act(self, obs, env) -> int:
        n = env.cfg.n_rays
        rays = np.asarray(obs[:n])
        center = n // 2

        # Rays are ordered by angle from negative (right side) to positive (left side);
        # steer +1 turns left (CCW), steer -1 turns right. Aim at the most open ray.
        best = int(np.argmax(rays))
        if best > center:
            steer = STEER_CHOICES[2]      # open space to the left -> steer left (+1)
        elif best < center:
            steer = STEER_CHOICES[0]      # open space to the right -> steer right (-1)
        else:
            steer = STEER_CHOICES[1]

        # Coast (no gas): stay near min speed so corners are takeable. A slow, careful
        # driver - the floor a learned policy should clearly beat on speed and progress.
        return _action_index(steer, THROTTLE_CHOICES[1])
