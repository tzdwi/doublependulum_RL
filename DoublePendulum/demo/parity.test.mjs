// Checks the browser game against Python, using fixtures.json from export.py.
// Run: npm test (from DoublePendulum/demo)
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import ort from "onnxruntime-node";
import { Plant } from "./static/plant.js";

const here = new URL(".", import.meta.url);
const fixtures = JSON.parse(readFileSync(new URL("fixtures.json", here)));
const { models, plant: constants } = JSON.parse(readFileSync(new URL("static/models/models.json", here)));
const maxAbs = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

test("models.json and fixtures.json come from the same export", () => {
  assert.deepEqual(constants, fixtures.plant);
});

test("one decision of physics matches Python", () => {
  const plant = new Plant(constants);
  let worstState = 0, worstObs = 0;
  for (const row of fixtures.physics) {
    plant.reset(row.x);
    plant.successFrames = row.success_frames;
    const obs = plant.step(row.force);
    worstState = Math.max(worstState, maxAbs(plant.x, row.next));
    worstObs = Math.max(worstObs, maxAbs(Array.from(obs), row.obs));
    assert.equal(plant.successFrames, row.next_success_frames);
    assert.equal(plant.fallen, row.fell);
  }
  assert.ok(worstState < 1e-9, `state differs by ${worstState}`);
  assert.ok(worstObs < 1e-6, `observation differs by ${worstObs}`);
});

for (const model of models) {
  test(`${model.id}: ONNX forces and held-out outcomes match Python`, async () => {
    const session = await ort.InferenceSession.create(new URL(`static/${model.file}`, here).pathname);
    const act = async (obs) => (await session.run({ obs: new ort.Tensor("float32", obs, [obs.length / 8, 8]) })).force.data;

    const { obs, force } = fixtures.policies[model.id];
    const got = await act(Float32Array.from(obs.flat()));
    const worst = maxAbs(Array.from(got), force);
    assert.ok(worst < 1e-3, `force differs by ${worst} N`);

    // Each held-out seed, run to the same time limit as training: same fall and success outcome.
    const plant = new Plant(constants);
    for (const row of fixtures.closed_loop[model.id]) {
      plant.reset(row.start);
      let o = plant.observe(), n = 0;
      while (n < 1200 && !plant.fallen) {
        o = plant.step((await act(o))[0]);
        n++;
      }
      assert.equal(plant.fallen, row.fell, `seed ${row.seed} fell`);
      assert.equal(!plant.fallen && plant.settled(), row.success, `seed ${row.seed} success`);
    }
  });
}
