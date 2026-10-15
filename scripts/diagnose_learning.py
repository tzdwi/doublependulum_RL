import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import scipy.linalg
from DoublePendulum.nn.nn_common import init_env

TERMS = [
    "balance_reward",
    "control_penalty",
    "offtrack_penalty",
]


def rollout(env, seed, policy):
    env.reset(seed=seed)
    totals = {key: 0.0 for key in TERMS}
    reward_sum = 0.0
    decisions = 0
    peak_force = 0.0
    info = {}
    done = False
    while not done:
        force = float(policy(env))
        peak_force = max(peak_force, abs(force))
        _, reward, terminated, truncated, info = env.step(force)
        reward_sum += reward
        for key in TERMS:
            totals[key] += info[key]
        decisions += 1
        done = terminated or truncated
    physics_steps = env.unwrapped.pendulum.t / env.unwrapped.dt
    return physics_steps, decisions, bool(info["success"]), reward_sum, totals, peak_force


def wrapped_error(state):
    error = np.array(state, dtype=float)
    error[1] = np.arctan2(np.sin(state[1] - np.pi), np.cos(state[1] - np.pi))
    error[2] = np.arctan2(np.sin(state[2] - np.pi), np.cos(state[2] - np.pi))
    return error


def one_decision(plant, state, force):
    pendulum = plant.pendulum
    saved = pendulum.x.copy(), pendulum.t
    pendulum.x = np.array(state, dtype=float)
    pendulum.t = 0.0
    pendulum._make_system()
    for _ in range(plant.action_repeat):
        pendulum.step(force)
    result = pendulum.x.copy()
    pendulum.x, pendulum.t = saved
    pendulum._make_system()
    return result


def discrete_lqr(plant):
    upright = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0])
    state_step = 1e-4
    force_step = 1e-2
    columns = []
    for index in range(6):
        step = np.zeros(6)
        step[index] = state_step
        ahead = one_decision(plant, upright + step, 0.0)
        behind = one_decision(plant, upright - step, 0.0)
        columns.append((ahead - behind) / (2 * state_step))
    ahead = one_decision(plant, upright, force_step)
    behind = one_decision(plant, upright, -force_step)
    dynamics = np.column_stack(columns)
    control = ((ahead - behind) / (2 * force_step)).reshape(6, 1)
    cost = np.diag([0.01, 10.0, 10.0, 0.1, 1.0, 1.0])
    effort = np.array([[10.0]])
    value = scipy.linalg.solve_discrete_are(dynamics, control, cost, effort)
    gain = np.linalg.solve(effort + control.T @ value @ control, control.T @ value @ dynamics)
    return gain.reshape(-1)


def lqr_force(env, gain):
    error = wrapped_error(env.unwrapped.pendulum.x)
    force = -gain @ error
    return float(np.clip(force, -env.unwrapped.max_F, env.unwrapped.max_F))


def collect_near_upright(env, gain, angle=0.1):
    saved = []
    for seed in range(20):
        env.reset(seed=seed)
        done = False
        while not done:
            error = wrapped_error(env.unwrapped.pendulum.x)
            if abs(error[1]) < angle and abs(error[2]) < angle:
                saved.append(env.unwrapped.pendulum.x.copy())
            _, _, terminated, truncated, _ = env.step(lqr_force(env, gain))
            done = terminated or truncated
    return saved


def spread(states, count):
    if len(states) <= count:
        return list(states)
    tilt = np.array([
        max(abs(wrapped_error(state)[1]), abs(wrapped_error(state)[2]))
        for state in states
    ])
    order = np.argsort(tilt)
    picks = np.unique(np.round(np.linspace(0, len(order) - 1, count)).astype(int))
    return [states[order[index]] for index in picks]


def choose_states(saved, count=10):
    same, mixed = [], []
    for state in saved:
        error = wrapped_error(state)
        (mixed if error[1] * error[2] < 0 else same).append(state)
    return spread(same, count) + spread(mixed, count), len(same), len(mixed)


def place_state(env, state):
    env.reset()
    pendulum = env.unwrapped.pendulum
    pendulum.x = np.array(state, dtype=float)
    pendulum.t = 0.0
    pendulum._validate_state_arr()
    pendulum._make_system()
    plant = env.unwrapped
    plant.success_frames = 0
    plant._agent_location = 0.0
    plant.state = plant._get_state()


def episode_from(env, state, first_force, gain):
    place_state(env, state)
    reward_sum = 0.0
    first = True
    done = False
    success = False
    while not done:
        force = first_force if first else lqr_force(env, gain)
        first = False
        _, reward, terminated, truncated, info = env.step(force)
        reward_sum += float(reward)
        success = bool(info["success"])
        done = terminated or truncated
    return reward_sum, success


def tilted_states():
    states = []
    for degrees in (2.0, 6.0, 12.0):
        tilt = np.deg2rad(degrees)
        for first, second in ((tilt, tilt), (-tilt, -tilt), (tilt, -tilt), (-tilt, tilt)):
            states.append(np.array([0.0, np.pi + first, np.pi + second, 0.0, 0.0, 0.0]))
    return states


def rank_actions(env, gain, states, label):
    print(f"reward ranking on {len(states)} {label}")
    if not states:
        return
    error = wrapped_error(states[0])
    sample_force = float(np.clip(-gain @ error, -env.unwrapped.max_F, env.unwrapped.max_F))
    repeat = episode_from(env, states[0], sample_force, gain)
    repeat_gap = abs(repeat[0] - episode_from(env, states[0], sample_force, gain)[0])
    print(f"repeat gap on the first state {repeat_gap:.6f}")
    wins_opposite = 0
    wins_zero = 0
    gaps_opposite = []
    gaps_zero = []
    successes = {"LQR": 0, "zero": 0, "opposite": 0}
    for index, state in enumerate(states):
        error = wrapped_error(state)
        force = float(np.clip(-gain @ error, -env.unwrapped.max_F, env.unwrapped.max_F))
        results = {
            name: episode_from(env, state, action, gain)
            for name, action in (("LQR", force), ("zero", 0.0), ("opposite", -force))
        }
        for name in successes:
            successes[name] += results[name][1]
        gap_opposite = results["LQR"][0] - results["opposite"][0]
        gap_zero = results["LQR"][0] - results["zero"][0]
        gaps_opposite.append(gap_opposite)
        gaps_zero.append(gap_zero)
        wins_opposite += gap_opposite > repeat_gap
        wins_zero += gap_zero > repeat_gap
        sign = "mixed" if error[1] * error[2] < 0 else "same"
        marks = " ".join(
            f"{name} {results[name][0]:+.2f} {'S' if results[name][1] else 'F'}"
            for name in ("LQR", "zero", "opposite")
        )
        print(
            f"state {index:2d} {sign}: e1={error[1]:+.4f} e2={error[2]:+.4f} "
            f"F={force:+.2f} | {marks}"
        )
    count = len(states)
    print(
        f"LQR beats opposite on {wins_opposite}/{count}, "
        f"median gap {np.median(gaps_opposite):.2f}"
    )
    print(
        f"LQR beats zero on {wins_zero}/{count}, "
        f"median gap {np.median(gaps_zero):.2f}"
    )
    print(
        "successes "
        + " ".join(f"{name} {successes[name]}/{count}" for name in ("LQR", "zero", "opposite"))
    )
    print(f"pass {wins_opposite >= int(np.ceil(0.9 * count))}")


def force_grid(env, gain):
    tilt = np.deg2rad(2.0)
    state = np.array([0.0, np.pi + tilt, np.pi - tilt, 0.0, 0.0, 0.0])
    print("force grid on a 2 degree opposite lean, then LQR")
    for force in (-25.0, -10.0, 0.0, 10.0, 25.0):
        place_state(env, state)
        observation, first_reward, terminated, truncated, first_info = env.step(force)
        error = wrapped_error(env.unwrapped.pendulum.x)
        raw = env.unwrapped.pendulum.x
        names = ("x", "e1", "e2", "vx", "w1", "w2", "F")
        obs_text = " ".join(f"{name}={float(value):+.4f}" for name, value in zip(names, observation))
        clipped = [name for name, value in zip(names, observation) if abs(float(value)) >= 1.0 - 1e-6]
        rest = 0.0
        success = bool(first_info["success"])
        done = terminated or truncated
        while not done:
            _, reward, terminated, truncated, info = env.step(lqr_force(env, gain))
            rest += float(reward)
            success = bool(info["success"])
            done = terminated or truncated
        total = float(first_reward) + rest
        terms = " ".join(f"{key}={first_info[key]:.4f}" for key in TERMS)
        print(
            f"F={force:+6.1f} first {float(first_reward):+.4f} rest {rest:+.2f} "
            f"total {total:+.2f} {'S' if success else 'F'} | "
            f"e1={np.rad2deg(error[1]):+.3f} deg e2={np.rad2deg(error[2]):+.3f} deg | {terms}"
        )
        print(
            f"         obs {obs_text} | clip {','.join(clipped) if clipped else 'none'} | "
            f"vx={raw[3]:+.4f} m/s w1={raw[4]:+.4f} rad/s w2={raw[5]:+.4f} rad/s"
        )


def report(label, rows):
    print(label)
    successes = 0
    for seed, (physics_steps, decisions, success, reward_sum, totals, peak_force) in enumerate(rows):
        successes += success
        terms = " ".join(f"{key}={totals[key]:.2f}" for key in TERMS)
        print(
            f"seed {seed:2d}: {physics_steps:6.0f} phys steps, {decisions:4d} decisions, "
            f"success {success}, peak |F| {peak_force:5.2f}, return {reward_sum:.2f} | {terms}"
        )
    lengths = [row[0] for row in rows]
    print(
        f"successes {successes}/20 | phys steps "
        f"min {min(lengths):.0f} median {np.median(lengths):.0f} max {max(lengths):.0f}"
    )


def main():
    env = init_env()
    plant = env.unwrapped
    print(
        f"mm={plant.mm} m1={plant.m1} m2={plant.m2} l1={plant.l1} l2={plant.l2} "
        f"dt={plant.dt} action_repeat={plant.action_repeat} max_F={plant.max_F} "
        f"theta_tol={plant.theta_tol:.4f} max_episode_steps={env.spec.max_episode_steps}"
    )
    report("zero force", [rollout(env, seed, lambda env: 0.0) for seed in range(20)])
    gain = discrete_lqr(plant)
    print("LQR gain " + " ".join(f"{value:.3f}" for value in gain))
    report("LQR", [rollout(env, seed, lambda env, gain=gain: lqr_force(env, gain)) for seed in range(20)])
    states, same_count, mixed_count = choose_states(collect_near_upright(env, gain))
    print(f"near-upright pool {same_count} same-sign, {mixed_count} mixed-sign; using {len(states)}")
    rank_actions(env, gain, states, "near-upright states")
    rank_actions(env, gain, tilted_states(), "hand-set tilts")
    force_grid(env, gain)


if __name__ == "__main__":
    main()
