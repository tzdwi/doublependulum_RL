"""Export the trained policies for the browser game.

For each learner whose final weights (from learner.dump()) exist, this script:
  * loads them with learner.load(best=False),
  * scores the deterministic policy on held-out seeds 100-119 in the Python environment,
  * exports that deterministic action (8 observation values -> 1 force) to ONNX,
  * checks the ONNX model against torch.
It then writes static/models/models.json, with the plant constants the JavaScript port reads,
and fixtures.json, which parity.test.mjs uses to compare the JavaScript plant and the ONNX
models with Python.

Run from the repo root:
    python -m DoublePendulum.demo.export [--out DIR]
--out also copies the finished static bundle to DIR.
"""

import os

# Export on the CPU, so the policies trace the same way the browser runs them.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from DoublePendulum.nn.nn_common import CHECKPOINT_VERSION, HOLDOUT_SEEDS, _policy_kwargs, init_env

DEMO_DIR = Path(__file__).parent
STATIC_DIR = DEMO_DIR / "static"
MODEL_DIR = STATIC_DIR / "models"
FIXTURES = DEMO_DIR / "fixtures.json"


def _learners():
    from DoublePendulum.nn import bro, ddpg, sac, sac_flowg, sac_flowt

    # id, label, learner class, final policy pickle
    return [
        ("ddpg", "Deep Deterministic Policy Gradient", ddpg.DDPG_Learner, ddpg.DDPG_policy_pickle),
        ("sac", "Soft Actor-Critic", sac.SAC_Learner, sac.SAC_policy_pickle),
        ("sac_flowt", "SAC Flow (transformer)", sac_flowt.SAC_FLOWT_Learner, sac_flowt.SAC_T_velocity_pickle),
        ("sac_flowg", "SAC Flow (gated)", sac_flowg.SAC_FLOWG_Learner, sac_flowg.SAC_G_velocity_pickle),
        ("bro", "Bigger, Regularized, Optimistic", bro.BRO_Learner, bro.BRO_pessimistic_pickle),
    ]


class DeterministicPolicy(torch.nn.Module):
    """The learner's deterministic action, the same call the held-out checks score."""

    def __init__(self, learner):
        super().__init__()
        self.learner = learner
        self.kwargs = _policy_kwargs(learner)
        for name, value in vars(learner).items():
            if isinstance(value, torch.nn.Module):
                self.add_module(name, value)

    def forward(self, obs):
        action = self.learner.select_action(obs, **self.kwargs)
        if isinstance(action, tuple):
            action = action[0]
        return action.reshape(-1, 1)


def _batch_forces(policy, observations):
    with torch.no_grad():
        return policy(torch.tensor(np.asarray(observations, dtype=np.float32))).reshape(-1).numpy()


def _held_out(policy):
    """Deterministic rollouts on the held-out seeds. Returns per-seed start states and outcomes."""
    envs = [init_env() for _ in HOLDOUT_SEEDS]
    limit = int(envs[0].spec.max_episode_steps)
    observations, starts = [], []
    for env, seed in zip(envs, HOLDOUT_SEEDS):
        obs, _ = env.reset(seed=seed)
        observations.append(obs)
        starts.append(env.unwrapped.pendulum.x.tolist())
    alive = [True] * len(envs)
    decisions = [0] * len(envs)
    seen = []
    for _ in range(limit):
        chosen = [i for i in range(len(envs)) if alive[i]]
        if not chosen:
            break
        forces = _batch_forces(policy, [observations[i] for i in chosen])
        for i, force in zip(chosen, forces):
            seen.append(observations[i])
            obs, _, terminated, truncated, _ = envs[i].step(float(force))
            decisions[i] += 1
            observations[i] = obs
            if terminated or truncated:
                alive[i] = False
    rows = []
    for env, seed, start, count in zip(envs, HOLDOUT_SEEDS, starts, decisions):
        plant = env.unwrapped
        fell = not plant.healthy
        rows.append({
            "seed": seed,
            "start": start,
            "decisions": count,
            "fell": fell,
            "success": (not fell) and plant.success_frames * plant.dt >= plant.SUCCESS_TIME,
            "final": plant.pendulum.x.tolist(),
        })
    return rows, seen


def _plant_constants():
    plant = init_env().unwrapped
    return {
        "params": plant.pendulum._params.tolist(),
        "l1": plant.l1,
        "l2": plant.l2,
        "dt": plant.dt,
        "action_repeat": plant.action_repeat,
        "max_F": plant.max_F,
        "rail": plant.size,
        "fall_angle": 0.8,
        "theta_tol": plant.theta_tol,
        "still_speed": plant._still_speed,
        "success_time": plant.SUCCESS_TIME,
        "obs_scales": [plant.size, 0.5, 0.5, 2.0, 0.5, 0.5, 1.0, 1.0],
        "obs_clip": float(plant.observation_space.high[0]),
        "checkpoint_version": CHECKPOINT_VERSION,
    }


def _physics_fixtures(states, rng, count=400):
    """One decision from varied states and forces, including a few runs into the rail."""
    env = init_env()
    plant = env.unwrapped
    rows = []
    picks = rng.choice(len(states), size=count, replace=len(states) < count)
    for n, index in enumerate(picks):
        x = np.array(states[index], dtype=float)
        x[1:3] += rng.uniform(-0.3, 0.3, size=2)
        x[3:6] += rng.uniform(-0.5, 0.5, size=3)
        if n % 20 == 0:
            x[0] = np.sign(rng.uniform(-1, 1)) * (plant.size - 0.05)
            x[3] = np.sign(x[0]) * rng.uniform(1.0, 5.0)
        env.reset()
        plant.pendulum.x = x.copy()
        plant.pendulum._validate_state_arr()
        plant.success_frames = int(rng.integers(0, 600))
        plant.state = plant._get_state()
        before = (plant.pendulum.x.tolist(), plant.success_frames)
        force = float(rng.uniform(-plant.max_F, plant.max_F))
        obs, _, terminated, _, _ = plant.step(force)
        rows.append({
            "x": before[0],
            "success_frames": before[1],
            "force": force,
            "next": plant.pendulum.x.tolist(),
            "next_success_frames": plant.success_frames,
            "obs": [float(v) for v in obs],
            "fell": bool(terminated),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="also copy the static bundle here")
    args = parser.parse_args()

    import onnxruntime as ort

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    models, policy_fixtures, closed_loop, states = [], {}, {}, []
    for model_id, label, learner_cls, pickle in _learners():
        if not Path(pickle).is_file():
            print(f"{model_id}: no final weights at {pickle}, skipped")
            continue
        torch.manual_seed(0)
        learner = learner_cls(buffer_length=1000)
        learner.load(best=False)
        policy = DeterministicPolicy(learner).eval()

        rows, seen = _held_out(policy)
        score = sum(row["success"] for row in rows)
        print(f"{model_id}: held-out {score}/{len(rows)}, falls {sum(row['fell'] for row in rows)}")
        if not states:
            states = [row["start"] for row in rows]
        closed_loop[model_id] = rows

        path = MODEL_DIR / f"{model_id}.onnx"
        example = torch.zeros(1, 8)
        torch.onnx.export(
            policy, (example,), str(path), input_names=["obs"], output_names=["force"],
            dynamic_axes={"obs": {0: "batch"}, "force": {0: "batch"}}, dynamo=False,
        )

        picks = rng.choice(len(seen), size=min(500, len(seen)), replace=False)
        observations = np.asarray([seen[i] for i in picks], dtype=np.float32)
        observations[: len(observations) // 5] = rng.uniform(-5, 5, size=(len(observations) // 5, 8))
        expected = _batch_forces(policy, observations)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = session.run(None, {"obs": observations})[0].reshape(-1)
        print(f"{model_id}: onnx vs torch max |dF| {float(np.max(np.abs(got - expected))):.2e} N, "
              f"{path.stat().st_size / 1024:.0f} KB")
        policy_fixtures[model_id] = {"obs": observations.tolist(), "force": expected.tolist()}
        models.append({"id": model_id, "label": label, "file": f"models/{model_id}.onnx",
                       "held_out": f"{score}/{len(rows)}"})

    if not models:
        raise SystemExit("no final weights found; run a train script to completion first")

    with open(MODEL_DIR / "models.json", "w") as f:
        json.dump({"plant": _plant_constants(), "models": models}, f, indent=1)
    with open(FIXTURES, "w") as f:
        json.dump({
            "plant": _plant_constants(),
            "physics": _physics_fixtures(states, rng),
            "policies": policy_fixtures,
            "closed_loop": closed_loop,
        }, f)
    print(f"wrote {MODEL_DIR / 'models.json'} and {FIXTURES}")

    if args.out is not None:
        shutil.copytree(STATIC_DIR, args.out, dirs_exist_ok=True)
        print(f"copied the bundle to {args.out}")


if __name__ == "__main__":
    main()
