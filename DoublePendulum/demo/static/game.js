// Browser game: a trained policy balances the pendulum and the player kicks it until it falls.
// The policies run with onnxruntime-web (global `ort`) and the plant runs in plant.js.
import { Plant } from "./plant.js";

const KICK_COOLDOWN = 1.0; // seconds between kicks
const $ = (id) => document.getElementById(id);
const canvas = $("view"), ctx = canvas.getContext("2d");
const ui = { model: $("model"), start: $("start"), left: $("left"), right: $("right"), size: $("size"),
  sizeValue: $("size-value"), duration: $("duration"), durationValue: $("duration-value"), status: $("status"), time: $("time"), kicks: $("kicks"), biggest: $("biggest"),
  bar: $("force-bar"), settled: $("settled") };

ort.env.wasm.numThreads = 1;
const { plant: constants, models } = await (await fetch("models/models.json")).json();
const decision = constants.dt * constants.action_repeat;
const sessions = {};
const plant = new Plant(constants);
let game = null;

for (const m of models) ui.model.add(new Option(m.label, m.id));
ui.duration.addEventListener("input", () => { ui.durationValue.textContent = (ui.duration.value * decision).toFixed(2); });
ui.size.max = constants.max_F;
ui.size.addEventListener("input", () => { ui.sizeValue.textContent = ui.size.value; });
draw({ t: 0, kicks: 0, biggest: 0, force: 0, cooldown: 1 });

async function session(id) {
  const file = models.find((m) => m.id === id).file;
  sessions[id] ??= ort.InferenceSession.create(file, { executionProviders: ["wasm"] });
  return sessions[id];
}

async function policyForce(s, obs) {
  const out = await s.run({ obs: new ort.Tensor("float32", obs, [1, 8]) });
  return out.force.data[0];
}

ui.start.addEventListener("click", async () => {
  ui.start.disabled = true;
  ui.status.textContent = "Loading model...";
  const s = await session(ui.model.value);
  plant.reset();
  game = { s, obs: plant.observe(), t: 0, acc: 0, last: performance.now(), kick: null, left: 0, cooldown: 0,
    kicks: 0, biggest: 0, force: 0, busy: false };
  ui.status.textContent = "Kick it with the buttons or the arrow keys.";
  ui.left.disabled = ui.right.disabled = false;
  requestAnimationFrame(frame);
});

function kick(sign) {
  if (!game || game.cooldown > 0 || game.kick !== null) return;
  game.kick = sign * Number(ui.size.value);
  game.left = Number(ui.duration.value);
  game.kicks++;
  game.biggest = Math.max(game.biggest, Math.abs(game.kick));
}
ui.left.addEventListener("click", () => kick(-1));
ui.right.addEventListener("click", () => kick(1));
addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    e.preventDefault();
    kick(e.key === "ArrowLeft" ? -1 : 1);
  }
});

// Fixed 20 Hz decisions on a real-time accumulator; a slow frame catches up by at most a few decisions.
async function frame(now) {
  if (!game) return;
  const g = game;
  g.acc = Math.min(g.acc + (now - g.last) / 1000, 4 * decision);
  g.last = now;
  if (!g.busy) {
    g.busy = true;
    while (g.acc >= decision && !plant.fallen) {
      g.acc -= decision;
      if (g.kick !== null) {
        // The kick replaces the policy's force for its duration. Training shoves last one decision.
        g.force = g.kick;
        g.obs = plant.kick(g.kick);
        if (--g.left === 0) {
          g.kick = null;
          g.cooldown = KICK_COOLDOWN;
        }
      } else {
        g.force = await policyForce(g.s, g.obs);
        g.obs = plant.step(g.force);
      }
      g.t += decision;
      g.cooldown = Math.max(0, g.cooldown - decision);
    }
    g.busy = false;
  }
  draw(g);
  if (plant.fallen) return end(g);
  requestAnimationFrame(frame);
}

function end(g) {
  ui.status.textContent = `It fell after ${g.t.toFixed(1)} s and ${g.kicks} kicks. Press Start to play again.`;
  ui.left.disabled = ui.right.disabled = true;
  ui.start.disabled = false;
  game = null;
}

function draw(g) {
  // Same framing as the pygame renderer: the whole rail plus both poles fits the view.
  const W = canvas.width, H = canvas.height, rail = constants.rail;
  const scale = W / (2 * (constants.l1 + constants.l2 + 0.2 + rail));
  const px = (x) => W / 2 + x * scale, py = (y) => H / 2 - y * scale;
  const p = plant.positions();
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(px(-rail), py(0)); ctx.lineTo(px(rail), py(0));
  ctx.moveTo(px(p.xm), py(0)); ctx.lineTo(px(p.x1), py(p.y1)); ctx.lineTo(px(p.x2), py(p.y2));
  ctx.stroke();
  const cart = 0.05 * rail * scale, ball = 0.025 * rail * scale;
  ctx.fillStyle = "#000";
  ctx.fillRect(px(p.xm) - cart, py(0) - cart, 2 * cart, 2 * cart);
  ctx.fillStyle = "#cc0";
  for (const [x, y] of [[p.x1, p.y1], [p.x2, p.y2]]) {
    ctx.beginPath(); ctx.arc(px(x), py(y), ball, 0, 2 * Math.PI); ctx.fill();
  }

  ui.time.textContent = g.t.toFixed(1);
  ui.kicks.textContent = g.kicks;
  ui.biggest.textContent = g.biggest.toFixed(1);
  const frac = g.force / constants.max_F;
  ui.bar.style.left = `${50 + Math.min(frac, 0) * 50}%`;
  ui.bar.style.width = `${Math.abs(frac) * 50}%`;
  ui.settled.hidden = !plant.settled();
  ui.left.disabled = ui.right.disabled = g.cooldown > 0 || g.kick != null;
}
