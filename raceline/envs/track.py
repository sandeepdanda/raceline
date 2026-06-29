"""Track geometry and sensor ray-casting for Raceline.

A track is a closed loop: an ordered list of centerline waypoints plus a fixed half-width.
The inner and outer walls are the centerline offset by +/- half_width. We provide:

- progress(pos): how far around the loop the car is, as arc-length (and a lap fraction),
- distance_from_center(pos): for collision (crash if > half_width),
- cast_rays(pos, heading, angles, max_range): sensor distances to the nearest wall.

Kept dependency-light (numpy only) and analytic, so thousands of steps/sec on CPU.
"""

from __future__ import annotations

import numpy as np


def _resample_closed(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polyline to n roughly-equal-arc-length points (smooths the loop)."""
    pts = np.asarray(points, dtype=np.float64)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 2))
    for i, t in enumerate(targets):
        j = np.searchsorted(cum, t, side="right") - 1
        j = min(max(j, 0), len(seg) - 1)
        frac = 0.0 if seg[j] == 0 else (t - cum[j]) / seg[j]
        out[i] = closed[j] + frac * (closed[j + 1] - closed[j])
    return out


class Track:
    """A closed-loop track with a centerline and walls offset by +/- half_width."""

    def __init__(self, centerline: np.ndarray, half_width: float):
        self.center = _resample_closed(centerline, max(64, len(centerline)))
        self.half_width = float(half_width)
        self.n = len(self.center)

        # Per-segment vectors + cumulative arc length around the loop (for progress).
        nxt = np.roll(self.center, -1, axis=0)
        self._seg = nxt - self.center
        self._seg_len = np.linalg.norm(self._seg, axis=1)
        self.length = float(self._seg_len.sum())
        self._cum = np.concatenate([[0.0], np.cumsum(self._seg_len)])[:-1]

        # Inner/outer walls as closed polylines, via per-vertex normals.
        normals = self._vertex_normals()
        self.outer = self.center + normals * self.half_width
        self.inner = self.center - normals * self.half_width
        self._wall_segments = self._build_wall_segments()

    def _vertex_normals(self) -> np.ndarray:
        prv = np.roll(self.center, 1, axis=0)
        nxt = np.roll(self.center, -1, axis=0)
        tang = nxt - prv
        tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
        # Left normal (rotate tangent +90 deg).
        return np.stack([-tang[:, 1], tang[:, 0]], axis=1)

    def _build_wall_segments(self) -> np.ndarray:
        """All wall segments as (A, B) pairs, shape (M, 2, 2), for ray intersection."""
        def loop_segs(poly):
            a = poly
            b = np.roll(poly, -1, axis=0)
            return np.stack([a, b], axis=1)

        return np.vstack([loop_segs(self.outer), loop_segs(self.inner)])

    # ------------------------------------------------------------------ queries
    def nearest_centerline(self, pos: np.ndarray) -> tuple[int, float]:
        """Index of the nearest centerline segment and the projection fraction along it."""
        p = np.asarray(pos, dtype=np.float64)
        ap = p - self.center
        seg = self._seg
        seg_len2 = (seg * seg).sum(axis=1) + 1e-12
        t = np.clip((ap * seg).sum(axis=1) / seg_len2, 0.0, 1.0)
        proj = self.center + seg * t[:, None]
        d2 = ((proj - p) ** 2).sum(axis=1)
        i = int(np.argmin(d2))
        return i, float(t[i])

    def distance_from_center(self, pos: np.ndarray) -> float:
        i, t = self.nearest_centerline(pos)
        proj = self.center[i] + self._seg[i] * t
        return float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - proj))

    def progress(self, pos: np.ndarray) -> float:
        """Arc-length around the loop at the car's projection (0 .. track length)."""
        i, t = self.nearest_centerline(pos)
        return float(self._cum[i] + t * self._seg_len[i])

    def lap_fraction(self, pos: np.ndarray) -> float:
        return self.progress(pos) / self.length

    def cast_rays(self, pos: np.ndarray, heading: float, angles: np.ndarray,
                  max_range: float) -> np.ndarray:
        """Distance from pos to the nearest wall along each ray (heading + angle), clipped
        to max_range. Vectorized ray-vs-segment intersection over all wall segments."""
        p = np.asarray(pos, dtype=np.float64)
        segs = self._wall_segments
        A = segs[:, 0, :]
        B = segs[:, 1, :]
        AB = B - A
        out = np.empty(len(angles))

        for k, ang in enumerate(angles):
            d = np.array([np.cos(heading + ang), np.sin(heading + ang)])
            # Solve p + t*d = A + u*AB  ->  use 2x2 cross-product form.
            denom = AB[:, 0] * (-d[1]) - AB[:, 1] * (-d[0])
            mask = np.abs(denom) > 1e-12
            ap = p - A
            t = np.full(len(segs), np.inf)
            u = np.full(len(segs), -1.0)
            t[mask] = (AB[mask, 0] * (-ap[mask, 1]) - AB[mask, 1] * (-ap[mask, 0])) / denom[mask]
            u[mask] = (ap[mask, 0] * (-d[1]) - ap[mask, 1] * (-d[0])) / denom[mask]
            hit = mask & (t > 1e-6) & (u >= 0.0) & (u <= 1.0)
            dist = t[hit].min() if np.any(hit) else max_range
            out[k] = min(dist, max_range)
        return out


def oval_track(half_width: float = 7.0) -> Track:
    """A rounded-rectangle loop - the Phase 1 default track."""
    cx, cy, rx, ry = 0.0, 0.0, 45.0, 28.0
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    # Superellipse-ish rounded rectangle.
    x = cx + rx * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** 0.6
    y = cy + ry * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** 0.6
    return Track(np.stack([x, y], axis=1), half_width)
