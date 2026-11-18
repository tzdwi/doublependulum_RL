// Port of DoublePendulum/physics/doublependulum.py and the decision step of
// DoublePendulum/envs/doublependulumworld.py. Constants come from models.json,
// written by export.py, so the two cannot drift apart silently.

const TWO_PI = 2 * Math.PI;
const LOW = -Math.PI / 2;
const HIGH = 3 * Math.PI / 2;

function solveSymmetric3x3(a, b, c, d, e, f, r0, r1, r2) {
  const c00 = d * f - e * e;
  const c01 = c * e - b * f;
  const c02 = b * e - c * d;
  const c11 = a * f - c * c;
  const c12 = b * c - a * e;
  const c22 = a * d - b * b;
  const det = a * c00 + b * c01 + c * c02;
  return [
    (c00 * r0 + c01 * r1 + c02 * r2) / det,
    (c01 * r0 + c11 * r1 + c12 * r2) / det,
    (c02 * r0 + c12 * r1 + c22 * r2) / det,
  ];
}

function derivative(x, force, p) {
  const [d11, d22, d33, d12, d13, d23, c12, c13, c23, g2, g3] = p;
  const [, t1, t2, , w1, w2] = x;
  const cos1 = Math.cos(t1), cos2 = Math.cos(t2);
  const sin1 = Math.sin(t1), sin2 = Math.sin(t2);
  const cos12 = Math.cos(t1 - t2), sin12 = Math.sin(t1 - t2);
  const [a0, a1, a2] = solveSymmetric3x3(
    d11, d12 * cos1, d13 * cos2, d22, d23 * cos12, d33,
    force - c12 * sin1 * w1 ** 2 - c13 * sin2 * w2 ** 2,
    -c23 * sin12 * w2 ** 2 - g2 * sin1,
    c23 * sin12 * w1 ** 2 - g3 * sin2,
  );
  return [x[3], w1, w2, a0, a1, a2];
}

const axpy = (x, h, k) => x.map((v, i) => v + h * k[i]);

export function rk4Step(x, force, dt, p) {
  const k1 = derivative(x, force, p);
  const k2 = derivative(axpy(x, 0.5 * dt, k1), force, p);
  const k3 = derivative(axpy(x, 0.5 * dt, k2), force, p);
  const k4 = derivative(axpy(x, dt, k3), force, p);
  return x.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

function wrap(x) {
  for (const i of [1, 2]) {
    const t = x[i];
    if (Number.isFinite(t) && (t < LOW || t > HIGH)) {
      const m = (t - LOW) % TWO_PI;
      x[i] = LOW + (m < 0 ? m + TWO_PI : m);
    }
  }
}

export const uprightError = (t) => Math.atan2(Math.sin(t - Math.PI), Math.cos(t - Math.PI));

export class Plant {
  constructor(c) {
    this.c = c;
    this.reset();
  }

  // Same as reset(options={"balanced": True}): perfectly upright and at rest.
  reset(x = [0, Math.PI, Math.PI, 0, 0, 0]) {
    this.x = x.slice();
    wrap(this.x);
    this.force = 0;
    this.successFrames = 0;
    this.fallen = false;
  }

  velocities() {
    const { l1, l2 } = this.c;
    const [, t1, t2, vm, w1, w2] = this.x;
    const vx1 = vm + l1 * Math.cos(t1) * w1;
    const vy1 = l1 * Math.sin(t1) * w1;
    return { vm, vx1, vy1, vx2: vx1 + l2 * Math.cos(t2) * w2, vy2: vy1 + l2 * Math.sin(t2) * w2 };
  }

  positions() {
    const { l1, l2 } = this.c;
    const [xm, t1, t2] = this.x;
    const x1 = xm + l1 * Math.sin(t1), y1 = -l1 * Math.cos(t1);
    return { xm, x1, y1, x2: x1 + l2 * Math.sin(t2), y2: y1 - l2 * Math.cos(t2) };
  }

  isUp() {
    const lim = this.c.fall_angle;
    return Math.abs(uprightError(this.x[1])) <= lim && Math.abs(uprightError(this.x[2])) <= lim;
  }

  physicsStep() {
    const { dt, params, theta_tol, still_speed, rail } = this.c;
    this.x = rk4Step(this.x, this.force, dt, params);
    wrap(this.x);
    // Stillness is judged before the rail bounce, as in _physics_step.
    const v = this.velocities();
    const up = Math.abs(uprightError(this.x[1])) < theta_tol && Math.abs(uprightError(this.x[2])) < theta_tol;
    const still = Math.abs(v.vm) <= still_speed && Math.hypot(v.vx1, v.vy1) <= still_speed
      && Math.hypot(v.vx2, v.vy2) <= still_speed;
    this.successFrames = up && still ? this.successFrames + 1 : 0;
    if (Math.abs(this.x[0]) > rail) {
      this.x[3] = -this.x[3];
      this.x[0] = Math.min(Math.max(this.x[0], -rail), rail);
    }
    if (!this.isUp()) this.fallen = true;
  }

  // One decision: hold the clipped force for action_repeat physics steps, stopping on a fall.
  step(force) {
    const { max_F, action_repeat } = this.c;
    this.force = Math.min(Math.max(force, -max_F), max_F);
    for (let i = 0; i < action_repeat && !this.fallen; i++) this.physicsStep();
    return this.observe();
  }

  // An external kick, like _shove: one decision of force, then the stillness counter restarts.
  kick(force) {
    const obs = this.step(force);
    this.force = 0;
    this.successFrames = 0;
    return obs;
  }

  settled() {
    return this.successFrames * this.c.dt >= this.c.success_time;
  }

  observe() {
    const { obs_scales: s, obs_clip: clip } = this.c;
    const [xm, t1, t2, vm, w1, w2] = this.x;
    const v = this.velocities();
    const raw = [xm, uprightError(t1), uprightError(t2), vm, w1, w2, v.vx1, v.vx2];
    const lim = Math.fround(clip);
    return Float32Array.from(raw, (r, i) => Math.min(Math.max(Math.fround(r / s[i]), -lim), lim));
  }
}
