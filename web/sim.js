// Raceline in-browser simulator: a JS port of raceline/envs/racetrack_env.py + track.py,
// so the ONNX policy sees exactly the observation it was trained on. Dependency-free
// (only onnxruntime-web, loaded in index.html).

export const STEER = [-1.0, 0.0, 1.0];
export const THROTTLE = [-1.0, 0.0, 1.0];
// Action order must match Python: nested loop steer (outer) x throttle (inner).
export const ACTIONS = [];
for (const s of STEER) for (const t of THROTTLE) ACTIONS.push([s, t]);

export const CFG = {
  nRays: 7,
  raySpread: Math.PI * 0.75,
  maxRange: 60.0,
  dt: 0.2,
  maxSteer: 0.6,
  accel: 4.0,
  brake: 9.0,
  drag: 0.04,
  minSpeed: 2.0,
  maxSpeed: 12.0,
  startSpeed: 6.0,
  maxSteps: 800,
  halfWidth: 9.0,
};

// --- track geometry (mirrors track.oval_track + Track) ---
function resampleClosed(points, n) {
  const closed = points.concat([points[0]]);
  const seg = [], cum = [0];
  for (let i = 0; i < closed.length - 1; i++) {
    const d = Math.hypot(closed[i + 1][0] - closed[i][0], closed[i + 1][1] - closed[i][1]);
    seg.push(d);
    cum.push(cum[i] + d);
  }
  const total = cum[cum.length - 1];
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = (total * i) / n;
    let j = 0;
    while (j < seg.length - 1 && cum[j + 1] < t) j++;
    const frac = seg[j] === 0 ? 0 : (t - cum[j]) / seg[j];
    out.push([
      closed[j][0] + frac * (closed[j + 1][0] - closed[j][0]),
      closed[j][1] + frac * (closed[j + 1][1] - closed[j][1]),
    ]);
  }
  return out;
}

export class Track {
  constructor(centerline, halfWidth) {
    this.center = resampleClosed(centerline, Math.max(64, centerline.length));
    this.halfWidth = halfWidth;
    this.n = this.center.length;

    this.seg = [];
    this.segLen = [];
    this.cum = [];
    let acc = 0;
    for (let i = 0; i < this.n; i++) {
      const nx = this.center[(i + 1) % this.n];
      const s = [nx[0] - this.center[i][0], nx[1] - this.center[i][1]];
      this.seg.push(s);
      const l = Math.hypot(s[0], s[1]);
      this.segLen.push(l);
      this.cum.push(acc);
      acc += l;
    }
    this.length = acc;

    // vertex normals (left normal of the tangent) -> inner/outer walls
    this.outer = [];
    this.inner = [];
    for (let i = 0; i < this.n; i++) {
      const p = this.center[(i - 1 + this.n) % this.n];
      const q = this.center[(i + 1) % this.n];
      let tx = q[0] - p[0], ty = q[1] - p[1];
      const tl = Math.hypot(tx, ty) + 1e-9;
      tx /= tl; ty /= tl;
      const nx = -ty, ny = tx;
      this.outer.push([this.center[i][0] + nx * halfWidth, this.center[i][1] + ny * halfWidth]);
      this.inner.push([this.center[i][0] - nx * halfWidth, this.center[i][1] - ny * halfWidth]);
    }
    this.wallSegs = [];
    for (const poly of [this.outer, this.inner]) {
      for (let i = 0; i < poly.length; i++) {
        this.wallSegs.push([poly[i], poly[(i + 1) % poly.length]]);
      }
    }
  }

  nearestCenterline(pos) {
    let bi = 0, bt = 0, bd = Infinity;
    for (let i = 0; i < this.n; i++) {
      const s = this.seg[i];
      const ax = pos[0] - this.center[i][0], ay = pos[1] - this.center[i][1];
      const sl2 = s[0] * s[0] + s[1] * s[1] + 1e-12;
      let t = (ax * s[0] + ay * s[1]) / sl2;
      t = Math.max(0, Math.min(1, t));
      const px = this.center[i][0] + s[0] * t, py = this.center[i][1] + s[1] * t;
      const d = (px - pos[0]) ** 2 + (py - pos[1]) ** 2;
      if (d < bd) { bd = d; bi = i; bt = t; }
    }
    return [bi, bt];
  }

  distanceFromCenter(pos) {
    const [i, t] = this.nearestCenterline(pos);
    const px = this.center[i][0] + this.seg[i][0] * t;
    const py = this.center[i][1] + this.seg[i][1] * t;
    return Math.hypot(pos[0] - px, pos[1] - py);
  }

  progress(pos) {
    const [i, t] = this.nearestCenterline(pos);
    return this.cum[i] + t * this.segLen[i];
  }

  castRays(pos, heading, angles, maxRange) {
    const out = new Float64Array(angles.length);
    for (let k = 0; k < angles.length; k++) {
      const dx = Math.cos(heading + angles[k]), dy = Math.sin(heading + angles[k]);
      let best = maxRange;
      for (const [A, B] of this.wallSegs) {
        const abx = B[0] - A[0], aby = B[1] - A[1];
        const denom = abx * -dy - aby * -dx;
        if (Math.abs(denom) < 1e-12) continue;
        const apx = pos[0] - A[0], apy = pos[1] - A[1];
        const t = (abx * -apy - aby * -apx) / denom;
        const u = (apx * -dy - apy * -dx) / denom;
        if (t > 1e-6 && u >= 0 && u <= 1 && t < best) best = t;
      }
      out[k] = best;
    }
    return out;
  }
}

export function ovalTrack(halfWidth = CFG.halfWidth) {
  const pts = [];
  const rx = 45, ry = 28;
  for (let i = 0; i < 40; i++) {
    const th = (2 * Math.PI * i) / 40;
    const c = Math.cos(th), s = Math.sin(th);
    pts.push([
      rx * Math.sign(c) * Math.abs(c) ** 0.6,
      ry * Math.sign(s) * Math.abs(s) ** 0.6,
    ]);
  }
  return new Track(pts, halfWidth);
}

// --- the car / environment ---
export class Car {
  constructor(track, cfg = CFG) {
    this.track = track;
    this.cfg = cfg;
    this.rayAngles = [];
    for (let i = 0; i < cfg.nRays; i++) {
      this.rayAngles.push(-cfg.raySpread / 2 + (cfg.raySpread * i) / (cfg.nRays - 1));
    }
    this.reset();
  }

  reset() {
    const c = this.track.center;
    this.pos = [c[0][0], c[0][1]];
    this.heading = Math.atan2(c[1][1] - c[0][1], c[1][0] - c[0][0]);
    this.speed = this.cfg.startSpeed;
    this.t = 0;
    this.laps = 0;
    this.prevProgress = this.track.progress(this.pos);
    this.cumProgress = 0;
    this.crashed = false;
    this.lastRays = this.castRays();
  }

  castRays() {
    return this.track.castRays(this.pos, this.heading, this.rayAngles, this.cfg.maxRange);
  }

  observation() {
    const rays = this.castRays();
    this.lastRays = rays;
    const obs = new Float32Array(this.cfg.nRays + 1);
    for (let i = 0; i < this.cfg.nRays; i++) obs[i] = Math.min(1, Math.max(0, rays[i] / this.cfg.maxRange));
    obs[this.cfg.nRays] = Math.min(1, Math.max(0, this.speed / this.cfg.maxSpeed));
    return obs;
  }

  step(actionIdx) {
    const cfg = this.cfg;
    const [steer, throttle] = ACTIONS[actionIdx];

    if (throttle > 0) this.speed += cfg.accel * throttle * cfg.dt;
    else if (throttle < 0) this.speed += cfg.brake * throttle * cfg.dt;
    this.speed *= 1 - cfg.drag;
    this.speed = Math.min(cfg.maxSpeed, Math.max(cfg.minSpeed, this.speed));

    const speedFrac = 0.5 + 0.5 * (this.speed / cfg.maxSpeed);
    this.heading += steer * cfg.maxSteer * speedFrac;

    this.pos = [
      this.pos[0] + this.speed * cfg.dt * Math.cos(this.heading),
      this.pos[1] + this.speed * cfg.dt * Math.sin(this.heading),
    ];

    const progress = this.track.progress(this.pos);
    let delta = progress - this.prevProgress;
    if (delta < -this.track.length / 2) { delta += this.track.length; this.laps += 1; }
    else if (delta > this.track.length / 2) { delta -= this.track.length; }
    this.prevProgress = progress;
    this.cumProgress += delta;

    this.crashed = this.track.distanceFromCenter(this.pos) > cfg.halfWidth;
    this.t += 1;
    const done = this.crashed || this.laps >= 1 || this.t >= cfg.maxSteps;
    return { crashed: this.crashed, laps: this.laps, lapFraction: progress / this.track.length, speed: this.speed, done };
  }
}

// RL controller backed by the ONNX policy (argmax of logits in JS).
export class OnnxController {
  constructor(session) { this.session = session; }
  async act(car) {
    const obs = car.observation();
    const tensor = new ort.Tensor("float32", obs, [1, obs.length]);
    const out = await this.session.run({ observation: tensor });
    const logits = out.action_logits.data;
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
}

// "Untrained" controller: random actions, to mimic the episode-0 brain.
export class RandomController {
  constructor(seed = 1) { this._a = seed >>> 0; }
  _rand() { this._a = (this._a + 0x6d2b79f5) | 0; let t = Math.imul(this._a ^ (this._a >>> 15), 1 | this._a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }
  async act(car) { return Math.floor(this._rand() * ACTIONS.length); }
}
