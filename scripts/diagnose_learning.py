import copy
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import scipy.linalg
import torch
from torch import nn, optim

from DoublePendulum.nn.nn_common import Transition, device, init_env
from DoublePendulum.nn.sac import SAC_Learner, SAC_POLICY_DP, SAC_Q_DP

TERMS = [
    "balance_reward",
    "tilt_rate",
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
    for _ in range(plant.action_repeat):
        pendulum.step(force)
    result = pendulum.x.copy()
    pendulum.x, pendulum.t = saved
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


def critic_fit(env, gain):
    tilt = np.deg2rad(2.0)
    physical = [
        np.array([0.0, np.pi + tilt, np.pi - tilt, 0.0, 0.0, 0.0]),
        np.array([0.0, np.pi - tilt, np.pi + tilt, 0.0, 0.0, 0.0]),
    ]
    forces = (-25.0, -10.0, 0.0, 10.0, 25.0)
    rows = []
    for state in physical:
        for force in forces:
            place_state(env, state)
            observation = env.unwrapped._get_obs().copy()
            reward_sum, success = episode_from(env, state, force, gain)
            rows.append((observation, force, reward_sum, success))
    torch.manual_seed(0)
    net = SAC_Q_DP(7).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=3e-4)
    observations = torch.tensor(np.stack([row[0] for row in rows]), dtype=torch.float32, device=device)
    scale = float(env.unwrapped.max_F)
    actions = torch.tensor([[row[1] / scale] for row in rows], dtype=torch.float32, device=device)
    targets = torch.tensor([[row[2]] for row in rows], dtype=torch.float32, device=device)
    loss_fn = nn.MSELoss()
    for _ in range(20000):
        loss = loss_fn(net(torch.cat((observations, actions), dim=1)), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        rmse = float(torch.sqrt(loss_fn(net(torch.cat((observations, actions), dim=1)), targets)))
    print(f"critic regression on {len(rows)} returns, RMSE {rmse:.2f}")
    passed = True
    for index, state in enumerate(physical):
        subset = rows[index * len(forces):(index + 1) * len(forces)]
        observation = torch.tensor(subset[0][0], dtype=torch.float32, device=device).unsqueeze(0)
        predicted = []
        for force, row in zip(forces, subset):
            action = torch.tensor([[force / scale]], dtype=torch.float32, device=device)
            with torch.no_grad():
                value = float(net(torch.cat((observation, action), dim=1)))
            predicted.append(value)
            print(f"state {index} F={force:+6.1f} return {row[2]:8.2f} predicted {value:8.2f}")
        action = torch.zeros(1, 1, device=device, requires_grad=True)
        net(torch.cat((observation, action), dim=1)).backward()
        slope = float(action.grad[0, 0] / scale)
        corrective = 25.0 if index == 0 else -25.0
        agrees = slope * corrective > 0 and predicted[0 if corrective < 0 else -1] > predicted[-1 if corrective < 0 else 0]
        passed = passed and agrees
        print(f"state {index} slope {slope:+.3f} per newton, corrective {corrective:+.0f} N, agrees {agrees}")
    print(f"pass {passed}")


def trajectory(env, state, first_force, gain):
    place_state(env, state)
    rows = []
    reward_sum = 0.0
    first = True
    done = False
    while not done:
        observation = env.unwrapped._get_obs().copy()
        force = first_force if first else lqr_force(env, gain)
        first = False
        next_observation, reward, terminated, truncated, _ = env.step(force)
        reward_sum += float(reward)
        state_t = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        action_t = torch.tensor([[force]], dtype=torch.float32, device=device)
        reward_t = torch.tensor([float(reward)], dtype=torch.float32, device=device)
        next_state = None if terminated else torch.tensor(next_observation, dtype=torch.float32, device=device).unsqueeze(0)
        rows.append(Transition(state_t, action_t, next_state, reward_t))
        done = terminated or truncated
    return rows, reward_sum


def critic_td(env, gain):
    tilt = np.deg2rad(2.0)
    physical = [
        np.array([0.0, np.pi + tilt, np.pi - tilt, 0.0, 0.0, 0.0]),
        np.array([0.0, np.pi - tilt, np.pi + tilt, 0.0, 0.0, 0.0]),
    ]
    forces = (-25.0, -10.0, 0.0, 10.0, 25.0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    learner = SAC_Learner()
    returns = []
    for state in physical:
        for force in forces:
            rows, reward_sum = trajectory(env, state, force, gain)
            for row in rows:
                learner.memory.push(*row)
            place_state(env, state)
            returns.append((env.unwrapped._get_obs().copy(), force, reward_sum))
    print(f"SAC critic updates on {len(learner.memory)} frozen transitions")
    for _ in range(10000):
        learner.optimize_model(do_policy_step=False)
        for net, target in zip(
            (learner.Q_net_1, learner.Q_net_2),
            (learner.Q_target_1, learner.Q_target_2),
        ):
            online = net.state_dict()
            frozen = target.state_dict()
            mixed = {
                key: (frozen[key] * learner.TAU + online[key] * (1.0 - learner.TAU)).detach()
                for key in online
            }
            target.load_state_dict(mixed)
    passed = True
    for index in range(len(physical)):
        subset = returns[index * len(forces):(index + 1) * len(forces)]
        observation = torch.tensor(subset[0][0], dtype=torch.float32, device=device).unsqueeze(0)
        predicted = []
        for force, row in zip(forces, subset):
            action = torch.tensor([[force / learner.scale]], dtype=torch.float32, device=device)
            with torch.no_grad():
                scores = torch.cat([
                    net(torch.cat((observation, action), dim=1))
                    for net in (learner.Q_net_1, learner.Q_net_2)
                ], dim=1)
                predicted.append(float(torch.min(scores)))
            print(f"state {index} F={force:+6.1f} return {row[2]:8.2f} Q {predicted[-1]:8.2f}")
        low = torch.tensor([[-1.0 / learner.scale]], dtype=torch.float32, device=device)
        high = torch.tensor([[1.0 / learner.scale]], dtype=torch.float32, device=device)
        with torch.no_grad():
            ends = []
            for action in (low, high):
                scores = torch.cat([
                    net(torch.cat((observation, action), dim=1))
                    for net in (learner.Q_net_1, learner.Q_net_2)
                ], dim=1)
                ends.append(float(torch.min(scores)))
        slope = (ends[1] - ends[0]) / 2.0
        corrective = 25.0 if index == 0 else -25.0
        agrees = slope * corrective > 0 and predicted[0 if corrective < 0 else -1] > predicted[-1 if corrective < 0 else 0]
        passed = passed and agrees
        print(f"state {index} slope {slope:+.3f} per newton, corrective {corrective:+.0f} N, agrees {agrees}")
    print(f"pass {passed}")


def stored_action_update(learner, states, actions, reward_span, boot_states, boot_actions, boot_mask, gamma_powers, batch_index, frozen_bootstrap=None):
    chosen = torch.tensor(batch_index, device=device)
    q_in = torch.cat((states[chosen], actions[chosen] / learner.scale), dim=-1)
    with torch.no_grad():
        discounted = (reward_span[chosen] * gamma_powers).sum(dim=-1)
        if frozen_bootstrap is None:
            target_in = torch.cat((boot_states[chosen], boot_actions[chosen] / learner.scale), dim=-1)
            scores = torch.cat(
                [net(target_in) for net in (learner.Q_target_1, learner.Q_target_2)],
                dim=-1,
            )
            bootstrap = boot_mask[chosen] * gamma_powers[-1] * learner.GAMMA * torch.min(scores, dim=-1).values
        else:
            bootstrap = frozen_bootstrap[chosen]
    targets = discounted + bootstrap
    loss = learner.criterion(learner.Q_net_1(q_in).squeeze(-1), targets)
    loss = loss + learner.criterion(learner.Q_net_2(q_in).squeeze(-1), targets)
    learner.Q_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(learner.Q_net_1.parameters(), 100)
    torch.nn.utils.clip_grad_value_(learner.Q_net_2.parameters(), 100)
    learner.Q_optimizer.step()
    for net, target in zip(
        (learner.Q_net_1, learner.Q_net_2),
        (learner.Q_target_1, learner.Q_target_2),
    ):
        target.load_state_dict(net.state_dict())


def critic_td_stored(env, gain, opening_only=False):
    tilt = np.deg2rad(2.0)
    physical = [
        np.array([0.0, np.pi + tilt, np.pi - tilt, 0.0, 0.0, 0.0]),
        np.array([0.0, np.pi - tilt, np.pi + tilt, 0.0, 0.0, 0.0]),
    ]
    forces = (-25.0, -10.0, 0.0, 10.0, 25.0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    learner = SAC_Learner()
    horizon = 30
    returns = []
    saved = {}
    state_rows = []
    action_rows = []
    reward_rows = []
    boot_state_rows = []
    boot_action_rows = []
    boot_flags = []
    first_decisions = []
    for state_index, state in enumerate(physical):
        for force in forces:
            rows, reward_sum = trajectory(env, state, force, gain)
            first_decisions.append(len(state_rows))
            discounted = 0.0
            weight = 1.0
            for row in rows:
                discounted += weight * float(row.reward)
                weight *= learner.GAMMA
            rewards = torch.tensor(
                [float(row.reward) for row in rows], dtype=torch.float32, device=device
            )
            for index, row in enumerate(rows):
                span = torch.zeros(horizon, dtype=torch.float32, device=device)
                take = min(horizon, len(rows) - index)
                span[:take] = rewards[index:index + take]
                reward_rows.append(span)
                state_rows.append(row.state)
                action_rows.append(row.action)
                ahead = index + horizon
                if ahead < len(rows):
                    boot_state_rows.append(rows[ahead].state)
                    boot_action_rows.append(rows[ahead].action)
                    boot_flags.append(1.0)
                else:
                    boot_state_rows.append(torch.zeros_like(row.state))
                    boot_action_rows.append(torch.zeros_like(row.action))
                    boot_flags.append(0.0)
            place_state(env, state)
            returns.append((env.unwrapped._get_obs().copy(), force, reward_sum, discounted))
            if state_index == 0:
                saved[force] = rows
    states = torch.cat(state_rows)
    actions = torch.cat(action_rows)
    reward_span = torch.stack(reward_rows)
    boot_states = torch.cat(boot_state_rows)
    boot_actions = torch.cat(boot_action_rows)
    boot_mask = torch.tensor(boot_flags, dtype=torch.float32, device=device)
    gamma_powers = learner.GAMMA ** torch.arange(horizon, device=device)
    print(
        f"stored-action critic updates on {states.shape[0]} frozen transitions, "
        f"{horizon}-decision return, target copied each update"
    )

    def snapshot(step):
        values = []
        for state_index in range(len(physical)):
            subset = returns[state_index * len(forces):(state_index + 1) * len(forces)]
            observation = torch.tensor(subset[0][0], dtype=torch.float32, device=device).unsqueeze(0)
            for force in (forces[0], forces[-1]):
                action = torch.tensor([[force / learner.scale]], dtype=torch.float32, device=device)
                with torch.no_grad():
                    scores = torch.cat([
                        net(torch.cat((observation, action), dim=1))
                        for net in (learner.Q_net_1, learner.Q_net_2)
                    ], dim=1)
                    values.append(float(torch.min(scores)))
        print(
            f"after {step:6d} updates "
            f"state0 Q(-25)={values[0]:6.1f} Q(+25)={values[1]:6.1f} "
            f"state1 Q(-25)={values[2]:6.1f} Q(+25)={values[3]:6.1f}"
        )

    updates = 200000
    for step in range(1, updates + 1):
        stored_action_update(
            learner, states, actions, reward_span, boot_states, boot_actions, boot_mask, gamma_powers,
            random.sample(range(states.shape[0]), learner.BATCH_SIZE),
        )
        if step % 20000 == 0:
            snapshot(step)
    passed = True
    for index in range(len(physical)):
        subset = returns[index * len(forces):(index + 1) * len(forces)]
        observation = torch.tensor(subset[0][0], dtype=torch.float32, device=device).unsqueeze(0)
        predicted = []
        for force, row in zip(forces, subset):
            action = torch.tensor([[force / learner.scale]], dtype=torch.float32, device=device)
            with torch.no_grad():
                scores = torch.cat([
                    net(torch.cat((observation, action), dim=1))
                    for net in (learner.Q_net_1, learner.Q_net_2)
                ], dim=1)
                predicted.append(float(torch.min(scores)))
            print(
                f"state {index} F={force:+6.1f} return {row[2]:8.2f} "
                f"discounted {row[3]:8.2f} Q {predicted[-1]:8.2f}"
            )
        low = torch.tensor([[-1.0 / learner.scale]], dtype=torch.float32, device=device)
        high = torch.tensor([[1.0 / learner.scale]], dtype=torch.float32, device=device)
        with torch.no_grad():
            ends = []
            for action in (low, high):
                scores = torch.cat([
                    net(torch.cat((observation, action), dim=1))
                    for net in (learner.Q_net_1, learner.Q_net_2)
                ], dim=1)
                ends.append(float(torch.min(scores)))
        slope = (ends[1] - ends[0]) / 2.0
        corrective = 25.0 if index == 0 else -25.0
        agrees = slope * corrective > 0 and predicted[0 if corrective < 0 else -1] > predicted[-1 if corrective < 0 else 0]
        passed = passed and agrees
        print(f"state {index} slope {slope:+.3f} per newton, corrective {corrective:+.0f} N, agrees {agrees}")
    print(f"pass {passed}")

    def value_at(row):
        with torch.no_grad():
            scores = torch.cat([
                net(torch.cat((row.state, row.action / learner.scale), dim=1))
                for net in (learner.Q_net_1, learner.Q_net_2)
            ], dim=1)
        return float(torch.min(scores))

    def remaining(rows, start):
        total = 0.0
        weight = 1.0
        for row in rows[start:]:
            total += weight * float(row.reward)
            weight *= learner.GAMMA
        return total

    print("decision 30 on state 0")
    for force in forces:
        rows = saved[force]
        reward_sum = 0.0
        weight = 1.0
        for row in rows[:horizon]:
            reward_sum += weight * float(row.reward)
            weight *= learner.GAMMA
        q30 = value_at(rows[horizon])
        opening = reward_sum + weight * q30
        print(
            f"F={force:+6.1f} remaining {remaining(rows, horizon):8.2f} "
            f"Q {q30:8.2f} opening target {opening:8.2f}"
        )

    with torch.no_grad():
        target_in = torch.cat((boot_states, boot_actions / learner.scale), dim=-1)
        scores = torch.cat(
            [net(target_in) for net in (learner.Q_target_1, learner.Q_target_2)],
            dim=-1,
        )
        bootstrap = boot_mask * gamma_powers[-1] * learner.GAMMA * torch.min(scores, dim=-1).values
        fixed_targets = ((reward_span * gamma_powers).sum(dim=-1) + bootstrap).detach()
    torch.manual_seed(0)
    fresh = SAC_Q_DP(7).to(device)
    fresh_optimizer = optim.AdamW(fresh.parameters(), lr=3e-4)
    q_in = torch.cat((states, actions / learner.scale), dim=-1)
    loss_fn = nn.MSELoss()
    fixed = fixed_targets.unsqueeze(1)
    for _ in range(20000):
        loss = loss_fn(fresh(q_in), fixed)
        fresh_optimizer.zero_grad()
        loss.backward()
        fresh_optimizer.step()
    with torch.no_grad():
        rmse = float(torch.sqrt(loss_fn(fresh(q_in), fixed)))
    print(f"fresh critic regression on {states.shape[0]} frozen targets, RMSE {rmse:.2f}")
    for index in range(len(physical)):
        subset = returns[index * len(forces):(index + 1) * len(forces)]
        observation = torch.tensor(subset[0][0], dtype=torch.float32, device=device).unsqueeze(0)
        predicted = []
        for offset, (force, row) in enumerate(zip(forces, subset)):
            action = torch.tensor([[force / learner.scale]], dtype=torch.float32, device=device)
            choice = first_decisions[index * len(forces) + offset]
            with torch.no_grad():
                value = float(fresh(torch.cat((observation, action), dim=1)))
            predicted.append(value)
            print(
                f"state {index} F={force:+6.1f} discounted {row[3]:8.2f} "
                f"target {float(fixed_targets[choice]):8.2f} predicted {value:8.2f}"
            )
        action = torch.zeros(1, 1, device=device, requires_grad=True)
        fresh(torch.cat((observation, action), dim=1)).backward()
        slope = float(action.grad[0, 0] / learner.scale)
        corrective = 25.0 if index == 0 else -25.0
        agrees = slope * corrective > 0 and predicted[0 if corrective < 0 else -1] > predicted[-1 if corrective < 0 else 0]
        print(f"state {index} slope {slope:+.3f} per newton, corrective {corrective:+.0f} N, agrees {agrees}")

    rest = [i for i in range(states.shape[0]) if i not in set(first_decisions)]
    half = learner.BATCH_SIZE // 2
    trained = [
        copy.deepcopy(learner.Q_net_1.state_dict()),
        copy.deepcopy(learner.Q_net_2.state_dict()),
        copy.deepcopy(learner.Q_target_1.state_dict()),
        copy.deepcopy(learner.Q_target_2.state_dict()),
        copy.deepcopy(learner.Q_optimizer.state_dict()),
    ]

    def restore_trained():
        learner.Q_net_1.load_state_dict(trained[0])
        learner.Q_net_2.load_state_dict(trained[1])
        learner.Q_target_1.load_state_dict(trained[2])
        learner.Q_target_2.load_state_dict(trained[3])
        learner.Q_optimizer.load_state_dict(trained[4])

    def opening_table(tag):
        print(tag)
        for index in range(len(physical)):
            subset = returns[index * len(forces):(index + 1) * len(forces)]
            observation = torch.tensor(subset[0][0], dtype=torch.float32, device=device).unsqueeze(0)
            predicted = []
            for offset, (force, row) in enumerate(zip(forces, subset)):
                action = torch.tensor([[force / learner.scale]], dtype=torch.float32, device=device)
                choice = first_decisions[index * len(forces) + offset]
                with torch.no_grad():
                    scores = torch.cat([
                        net(torch.cat((observation, action), dim=1))
                        for net in (learner.Q_net_1, learner.Q_net_2)
                    ], dim=1)
                    predicted.append(float(torch.min(scores)))
                print(
                    f"state {index} F={force:+6.1f} discounted {row[3]:8.2f} "
                    f"target {float(fixed_targets[choice]):8.2f} Q {predicted[-1]:8.2f}"
                )
            low = torch.tensor([[-1.0 / learner.scale]], dtype=torch.float32, device=device)
            high = torch.tensor([[1.0 / learner.scale]], dtype=torch.float32, device=device)
            with torch.no_grad():
                ends = []
                for action in (low, high):
                    scores = torch.cat([
                        net(torch.cat((observation, action), dim=1))
                        for net in (learner.Q_net_1, learner.Q_net_2)
                    ], dim=1)
                    ends.append(float(torch.min(scores)))
            slope = (ends[1] - ends[0]) / 2.0
            corrective = 25.0 if index == 0 else -25.0
            agrees = slope * corrective > 0 and predicted[0 if corrective < 0 else -1] > predicted[-1 if corrective < 0 else 0]
            print(f"state {index} slope {slope:+.3f} per newton, corrective {corrective:+.0f} N, agrees {agrees}")

    print(
        f"updates on the trained critic, frozen targets, "
        f"batches of the {len(first_decisions)} opening decisions only"
    )
    for step in range(1, 20001):
        stored_action_update(
            learner, states, actions, reward_span, boot_states, boot_actions, boot_mask, gamma_powers,
            random.choices(first_decisions, k=learner.BATCH_SIZE), bootstrap,
        )
        if step % 5000 == 0:
            opening_table(f"after {step} opening-only updates")
    restore_trained()
    if opening_only:
        return

    print(
        f"updates on the trained critic, frozen targets, "
        f"batches half of {len(first_decisions)} opening decisions and half of the rest"
    )
    for step in range(1, 20001):
        batch = (
            random.choices(first_decisions, k=half)
            + random.sample(rest, learner.BATCH_SIZE - half)
        )
        stored_action_update(
            learner, states, actions, reward_span, boot_states, boot_actions, boot_mask, gamma_powers,
            batch, bootstrap,
        )
        if step % 5000 == 0:
            opening_table(f"after {step} half-opening updates")
    if opening_only:
        return

    def value_at(row):
        with torch.no_grad():
            scores = torch.cat([
                net(torch.cat((row.state, row.action / learner.scale), dim=1))
                for net in (learner.Q_net_1, learner.Q_net_2)
            ], dim=1)
        return float(torch.min(scores))

    def remaining(rows, start):
        total = 0.0
        weight = 1.0
        for row in rows[start:]:
            total += weight * float(row.reward)
            weight *= learner.GAMMA
        return total

    marks = [0, 10, 25, 50, 100, 200, 400, 600]
    for force, label in ((-25.0, "falling -25 N"), (25.0, "balancing +25 N")):
        rows = saved[force]
        print(f"along {label}, {len(rows)} decisions")
        indices = [i for i in marks if i < len(rows)]
        if not indices or indices[-1] != len(rows) - 1:
            indices.append(len(rows) - 1)
        for i in indices:
            print(f"decision {i:4d} remaining {remaining(rows, i):7.2f} Q {value_at(rows[i]):7.2f}")

    rows = saved[25.0]
    print("one-step targets on balancing +25 N")
    for i in (0, 10, 25, 50):
        row = rows[i]
        reward = float(row.reward)
        successor = value_at(rows[i + 1])
        target = reward + learner.GAMMA * successor
        print(
            f"decision {i:4d} Q {value_at(row):7.2f} "
            f"reward {reward:7.4f} next Q {successor:7.2f} target {target:7.2f}"
        )

    plant = env.unwrapped
    print("cart position, angular velocity, and Q, decisions 0 through 30")
    for force, label in ((-25.0, "falling -25 N"), (25.0, "balancing +25 N")):
        rows = saved[force]
        print(label)
        for i in range(31):
            state = rows[i].state[0]
            x = float(state[0])
            w1 = float(state[4])
            w2 = float(state[5])
            print(
                f"decision {i:4d} Q {value_at(rows[i]):7.2f} "
                f"x obs {x:+7.4f} ({x * plant.size:+7.3f} m) "
                f"w1 obs {w1:+7.4f} ({w1 * 180:+7.2f} deg/s) "
                f"w2 obs {w2:+7.4f} ({w2 * 180:+7.2f} deg/s)"
            )

    names = ("x", "e1", "e2", "vx", "w1", "w2", "F")
    fall = saved[-25.0][16]
    balance = saved[25.0][16]
    print("channel swap at decision 16: balance state, one fall channel, balance action")
    print(
        f"balance Q {value_at(balance):7.2f} action {float(balance.action):+7.2f}  "
        f"fall Q {value_at(fall):7.2f} action {float(fall.action):+7.2f}"
    )
    for channel, name in enumerate(names):
        state = balance.state.clone()
        state[0, channel] = fall.state[0, channel]
        swapped = Transition(state, balance.action, None, balance.reward)
        print(
            f"{name:2s} balance {float(balance.state[0, channel]):+8.4f} "
            f"fall {float(fall.state[0, channel]):+8.4f} Q {value_at(swapped):7.2f}"
        )
    print("channel groups at decision 16: balance state, fall channels, balance action")
    for group in (("e1", "vx", "w1"), ("e1", "e2")):
        state = balance.state.clone()
        for name in group:
            channel = names.index(name)
            state[0, channel] = fall.state[0, channel]
        swapped = Transition(state, balance.action, None, balance.reward)
        print(f"{'+'.join(group):10s} Q {value_at(swapped):7.2f}")

    frozen_bootstrap = None

    def target_of(index):
        chosen = torch.tensor([index], device=device)
        with torch.no_grad():
            discounted = (reward_span[chosen] * gamma_powers).sum(dim=-1)
            if frozen_bootstrap is None:
                target_in = torch.cat((boot_states[chosen], boot_actions[chosen] / learner.scale), dim=-1)
                scores = torch.cat(
                    [net(target_in) for net in (learner.Q_target_1, learner.Q_target_2)],
                    dim=-1,
                )
                bootstrap = boot_mask[chosen] * gamma_powers[-1] * learner.GAMMA * torch.min(scores, dim=-1).values
            else:
                bootstrap = frozen_bootstrap[chosen]
        return float(discounted + bootstrap)

    def force_table(tag):
        print(tag)
        for index in range(len(physical)):
            subset = returns[index * len(forces):(index + 1) * len(forces)]
            observation = torch.tensor(subset[0][0], dtype=torch.float32, device=device).unsqueeze(0)
            predicted = []
            for offset, (force, row) in enumerate(zip(forces, subset)):
                action = torch.tensor([[force / learner.scale]], dtype=torch.float32, device=device)
                with torch.no_grad():
                    scores = torch.cat([
                        net(torch.cat((observation, action), dim=1))
                        for net in (learner.Q_net_1, learner.Q_net_2)
                    ], dim=1)
                    predicted.append(float(torch.min(scores)))
                target = target_of(first_decisions[index * len(forces) + offset])
                print(
                    f"state {index} F={force:+6.1f} return {row[2]:8.2f} "
                    f"discounted {row[3]:8.2f} target {target:8.2f} Q {predicted[-1]:8.2f}"
                )
            low = torch.tensor([[-1.0 / learner.scale]], dtype=torch.float32, device=device)
            high = torch.tensor([[1.0 / learner.scale]], dtype=torch.float32, device=device)
            with torch.no_grad():
                ends = []
                for action in (low, high):
                    scores = torch.cat([
                        net(torch.cat((observation, action), dim=1))
                        for net in (learner.Q_net_1, learner.Q_net_2)
                    ], dim=1)
                    ends.append(float(torch.min(scores)))
            slope = (ends[1] - ends[0]) / 2.0
            corrective = 25.0 if index == 0 else -25.0
            agrees = slope * corrective > 0 and predicted[0 if corrective < 0 else -1] > predicted[-1 if corrective < 0 else 0]
            print(f"state {index} slope {slope:+.3f} per newton, corrective {corrective:+.0f} N, agrees {agrees}")

    with torch.no_grad():
        target_in = torch.cat((boot_states, boot_actions / learner.scale), dim=-1)
        scores = torch.cat(
            [net(target_in) for net in (learner.Q_target_1, learner.Q_target_2)],
            dim=-1,
        )
        frozen_bootstrap = (
            boot_mask * gamma_powers[-1] * learner.GAMMA * torch.min(scores, dim=-1).values
        ).detach()
    print(f"updates on the {len(first_decisions)} first decisions only, decision-30 bootstrap frozen")
    for step in range(1, 5001):
        stored_action_update(
            learner, states, actions, reward_span, boot_states, boot_actions, boot_mask, gamma_powers,
            random.choices(first_decisions, k=learner.BATCH_SIZE),
            frozen_bootstrap,
        )
        if step % 1000 == 0:
            force_table(f"after {step} first-decision updates")

    def pack():
        return (
            copy.deepcopy(learner.Q_net_1.state_dict()),
            copy.deepcopy(learner.Q_net_2.state_dict()),
            copy.deepcopy(learner.Q_target_1.state_dict()),
            copy.deepcopy(learner.Q_target_2.state_dict()),
            copy.deepcopy(learner.Q_optimizer.state_dict()),
        )

    def unpack(saved):
        learner.Q_net_1.load_state_dict(saved[0])
        learner.Q_net_2.load_state_dict(saved[1])
        learner.Q_target_1.load_state_dict(saved[2])
        learner.Q_target_2.load_state_dict(saved[3])
        learner.Q_optimizer.load_state_dict(saved[4])

    ends = first_decisions[1:] + [states.shape[0]]
    early = []
    late = []
    for start, end in zip(first_decisions, ends):
        early.extend(range(start + 1, min(start + 30, end)))
        late.extend(range(start + 30, end))
    fitted = pack()
    for label, pool in (("decisions 1 through 29", early), ("decision 30 onward", late)):
        unpack(fitted)
        print(f"updates on {len(pool)} transitions, {label}, decision-30 bootstrap frozen")
        for _ in range(5000):
            stored_action_update(
                learner, states, actions, reward_span, boot_states, boot_actions, boot_mask, gamma_powers,
                random.sample(pool, learner.BATCH_SIZE),
                frozen_bootstrap,
            )
        force_table(f"after 5000 updates on {label}")
    unpack(fitted)

    print(f"updates on all {states.shape[0]} transitions, decision-30 bootstrap frozen")
    for step in range(1, 20001):
        stored_action_update(
            learner, states, actions, reward_span, boot_states, boot_actions, boot_mask, gamma_powers,
            random.sample(range(states.shape[0]), learner.BATCH_SIZE),
            frozen_bootstrap,
        )
        if step % 5000 == 0:
            force_table(f"after {step} full-replay updates")


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


def _transition(state, force, observation, reward, terminated):
    if torch.is_tensor(state):
        state_t = state
    else:
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    action_t = torch.tensor([[float(force)]], dtype=torch.float32, device=device)
    reward_t = torch.tensor([float(reward)], dtype=torch.float32, device=device)
    next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    return Transition(state_t, action_t, next_state, reward_t)


def _min_q(learner, observation, force):
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    action = torch.tensor([[float(force) / learner.scale]], dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = torch.cat([
            net(torch.cat((state, action), dim=1))
            for net in (learner.Q_net_1, learner.Q_net_2)
        ], dim=1)
    return float(torch.min(scores))


def _polyak(learner):
    for net, target in zip(
        (learner.Q_net_1, learner.Q_net_2),
        (learner.Q_target_1, learner.Q_target_2),
    ):
        online = net.state_dict()
        frozen = target.state_dict()
        mixed = {
            key: (frozen[key] * learner.TAU + online[key] * (1.0 - learner.TAU)).detach()
            for key in online
        }
        target.load_state_dict(mixed)


def _next_actions(rows):
    nxt = [None] * len(rows)
    for index in range(len(rows) - 1):
        if rows[index].next_state is not None:
            nxt[index] = rows[index + 1].action
    return nxt


def _stored_next_update(learner, rows, next_actions):
    batch = random.sample(range(len(rows)), learner.BATCH_SIZE)
    state_batch = torch.cat([rows[index].state for index in batch])
    action_batch = torch.cat([rows[index].action for index in batch])
    reward_batch = torch.cat([rows[index].reward for index in batch])
    q_in = torch.cat((state_batch, action_batch / learner.scale), dim=-1)
    q1 = learner.Q_net_1(q_in).squeeze(-1)
    q2 = learner.Q_net_2(q_in).squeeze(-1)
    with torch.no_grad():
        bootstrap = torch.zeros(learner.BATCH_SIZE, device=device)
        positions = [pos for pos, index in enumerate(batch) if next_actions[index] is not None]
        if positions:
            chosen = [batch[pos] for pos in positions]
            nxt_state = torch.cat([rows[index].next_state for index in chosen])
            nxt_action = torch.cat([next_actions[index] for index in chosen])
            target_in = torch.cat((nxt_state, nxt_action / learner.scale), dim=-1)
            scores = torch.cat([
                net(target_in) for net in (learner.Q_target_1, learner.Q_target_2)
            ], dim=-1)
            bootstrap[positions] = learner.GAMMA * torch.min(scores, dim=-1).values
        targets = reward_batch + bootstrap
    loss = learner.criterion(q1, targets) + learner.criterion(q2, targets)
    learner.Q_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(learner.Q_net_1.parameters(), 100)
    torch.nn.utils.clip_grad_value_(learner.Q_net_2.parameters(), 100)
    learner.Q_optimizer.step()


def warmup_mixtures(env, gain, stored_backup=False):
    # 2,000 warmup actions, then 20,000 steps from the frozen initial policy.
    # Critic updates match the pre-actor phase: one normal SAC update per step, no actor.
    pre_actor = 22000
    updates = 20000
    demo_seeds = range(200, 220)
    near = 0.1
    tilt = np.deg2rad(2.0)
    mirrors = [
        np.array([0.0, np.pi + tilt, np.pi - tilt, 0.0, 0.0, 0.0]),
        np.array([0.0, np.pi - tilt, np.pi + tilt, 0.0, 0.0, 0.0]),
    ]
    forces = (-25.0, -10.0, 0.0, 10.0, 25.0)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    collector = SAC_Learner()
    behavior = []
    behavior_next = []
    episode = 0
    episode_rows = []
    observation, _ = env.reset(seed=300)
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    while len(behavior) < pre_actor:
        action = collector.select_action(state)
        collector.steps_done += 1
        observation, reward, terminated, truncated, _ = env.step(float(action))
        row = _transition(state, float(action), observation, reward, terminated)
        behavior.append(row)
        episode_rows.append(row)
        if terminated or truncated:
            behavior_next.extend(_next_actions(episode_rows))
            episode_rows = []
            episode += 1
            observation, _ = env.reset(seed=300 + episode)
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            state = row.next_state
    if episode_rows:
        behavior_next.extend(_next_actions(episode_rows))
    print(f"behavior transitions {len(behavior)} over {episode + 1} episodes, seeds 300 onward")

    probe = init_env()
    lqr_rows = []
    inverse_rows = []
    lqr_next = []
    inverse_next = []
    far_rows = []
    pair_rows = []
    opposite_rows = []
    for seed in demo_seeds:
        for rows, nxt, sign in (
            (lqr_rows, lqr_next, 1.0),
            (inverse_rows, inverse_next, -1.0),
        ):
            observation, _ = env.reset(seed=seed)
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            episode_rows = []
            done = False
            while not done:
                physical = env.unwrapped.pendulum.x.copy()
                force = sign * lqr_force(env, gain)
                observation, reward, terminated, truncated, _ = env.step(force)
                row = _transition(state, force, observation, reward, terminated)
                rows.append(row)
                episode_rows.append(row)
                error = wrapped_error(physical)
                if sign > 0 and abs(error[1]) < near and abs(error[2]) < near:
                    place_state(probe, physical)
                    opp_obs, opp_reward, opp_terminated, _, _ = probe.step(-force)
                    opposite = _transition(state, -force, opp_obs, opp_reward, opp_terminated)
                    opposite_rows.append(opposite)
                    pair_rows.append(row)
                    pair_rows.append(opposite)
                elif sign > 0:
                    far_rows.append(row)
                done = terminated or truncated
                state = None if done else row.next_state
            nxt.extend(_next_actions(episode_rows))
    other_rows = behavior + inverse_rows + far_rows
    print(
        f"LQR transitions {len(lqr_rows)}, inverse transitions {len(inverse_rows)}, "
        f"near-upright pairs {len(pair_rows)}, other {len(other_rows)}"
    )

    def five_force(learner):
        for index, physical in enumerate(mirrors):
            place_state(env, physical)
            observation = env.unwrapped._clip_obs(env.unwrapped._get_obs())
            predicted = [_min_q(learner, observation, force) for force in forces]
            for force, value in zip(forces, predicted):
                print(f"state {index} F={force:+6.1f} Q {value:8.2f}")
            slope = (predicted[-1] - predicted[0]) / 50.0
            corrective = 25.0 if index == 0 else -25.0
            agrees = slope * corrective > 0 and predicted[-1 if corrective > 0 else 0] > predicted[0 if corrective > 0 else -1]
            print(f"state {index} slope {slope:+.3f} per newton, corrective {corrective:+.0f} N, agrees {agrees}")

    def held_out(learner):
        gaps = []
        first_gaps = []
        forces_used = []
        successes = 0
        for seed in range(100, 120):
            observation, _ = env.reset(seed=seed)
            done = False
            first = True
            info = {}
            while not done:
                force = lqr_force(env, gain)
                gap = _min_q(learner, observation, force) - _min_q(learner, observation, -force)
                gaps.append(gap)
                forces_used.append(abs(force))
                if first:
                    first_gaps.append(gap)
                    first = False
                observation, _, terminated, truncated, info = env.step(force)
                done = terminated or truncated
            successes += bool(info.get("success"))
        prefer = sum(gap > 0 for gap in gaps)
        first_prefer = sum(gap > 0 for gap in first_gaps)
        print(
            f"held-out LQR states {len(gaps)}, prefer corrective {prefer}/{len(gaps)}, "
            f"median gap {np.median(gaps):.2f}, median |F| {np.median(forces_used):.2f}"
        )
        print(
            f"held-out first decisions {len(first_gaps)}, prefer corrective {first_prefer}/{len(first_gaps)}, "
            f"median gap {np.median(first_gaps):.2f}, successes {successes}/20"
        )

    def run(label, rows, sample, update=None, next_actions=None):
        print(f"mixture {label}: {len(rows)} transitions, {updates} critic updates")
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        learner = SAC_Learner()
        for row in rows:
            learner.memory.push(*row)
        if sample is not None:
            learner.memory.sample = sample
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        for step in range(1, updates + 1):
            if update is None:
                learner.optimize_model(do_policy_step=False)
            else:
                update(learner, rows, next_actions)
            _polyak(learner)
            if step % 5000 == 0:
                place_state(env, mirrors[0])
                observation = env.unwrapped._clip_obs(env.unwrapped._get_obs())
                print(
                    f"after {step} updates state0 "
                    f"Q(-25)={_min_q(learner, observation, -25.0):.1f} "
                    f"Q(+25)={_min_q(learner, observation, 25.0):.1f}"
                )
        five_force(learner)
        held_out(learner)

    if stored_backup:
        rows = behavior + lqr_rows + inverse_rows
        nxt = behavior_next + lqr_next + inverse_next
        linked = sum(action is not None for action in nxt)
        print(f"stored next action on {linked}/{len(rows)} transitions")
        run("episodes, stored next action", rows, None, update=_stored_next_update, next_actions=nxt)
        return

    run("random", behavior, None)
    run("episodes", behavior + lqr_rows + inverse_rows, None)

    def paired_sample(batch_size):
        half = batch_size // 2
        return random.sample(pair_rows, half) + random.sample(other_rows, batch_size - half)

    run("paired", behavior + lqr_rows + inverse_rows + opposite_rows, paired_sample)


def clone_lqr(env, gain):
    scale = float(env.action_space.high[0])

    def trace(seeds):
        observations, forces, first = [], [], []
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            done = False
            opening = True
            while not done:
                force = lqr_force(env, gain)
                observations.append(np.asarray(observation, dtype=np.float64))
                forces.append(force)
                first.append(opening)
                opening = False
                observation, _, terminated, truncated, _ = env.step(force)
                done = terminated or truncated
        return np.vstack(observations), np.asarray(forces), np.asarray(first)

    train_obs, train_force, train_first = trace(range(20))
    design = np.column_stack((train_obs, np.ones(len(train_obs))))
    solution, _, _, _ = np.linalg.lstsq(design, train_force, rcond=None)
    linear_w, linear_b = solution[:-1], float(solution[-1])
    linear_fit = train_obs @ linear_w + linear_b
    print(
        f"train |F| median {np.median(np.abs(train_force)):.4f}, "
        f"first-decision median {np.median(np.abs(train_force[train_first])):.4f}, "
        f"previous-force weight {linear_w[6]:.4f}"
    )
    print("linear weights " + " ".join(f"{weight:.4f}" for weight in linear_w) + f" bias {linear_b:.4f}")
    print(f"linear train RMSE {np.sqrt(np.mean((linear_fit - train_force) ** 2)):.4f}")

    torch.manual_seed(0)
    policy = SAC_POLICY_DP(train_obs.shape[1]).to(device)
    logstd_weight = policy.layer3.weight.data[1].clone()
    logstd_bias = policy.layer3.bias.data[1].clone()
    optimizer = optim.AdamW(policy.parameters(), lr=3e-4, amsgrad=True)
    obs_t = torch.tensor(train_obs, dtype=torch.float32, device=device)
    force_t = torch.tensor(train_force, dtype=torch.float32, device=device)
    for _ in range(20000):
        batch = torch.randint(0, obs_t.shape[0], (128,), device=device)
        predicted = scale * torch.tanh(policy(obs_t[batch])[:, 0])
        loss = torch.nn.functional.mse_loss(predicted, force_t[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            policy.layer3.weight[1].copy_(logstd_weight)
            policy.layer3.bias[1].copy_(logstd_bias)
    with torch.no_grad():
        neural_fit = scale * torch.tanh(policy(obs_t)[:, 0])
    neural_fit = neural_fit.cpu().numpy()
    print(f"neural train RMSE {np.sqrt(np.mean((neural_fit - train_force) ** 2)):.4f}")

    torch.manual_seed(0)
    tanh_w = torch.zeros(obs_t.shape[1], device=device, requires_grad=True)
    tanh_b = torch.zeros((), device=device, requires_grad=True)
    optimizer = optim.AdamW((tanh_w, tanh_b), lr=3e-4, amsgrad=True)
    for _ in range(20000):
        batch = torch.randint(0, obs_t.shape[0], (128,), device=device)
        predicted = scale * torch.tanh(obs_t[batch] @ tanh_w + tanh_b)
        loss = torch.nn.functional.mse_loss(predicted, force_t[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        tanh_fit = (scale * torch.tanh(obs_t @ tanh_w + tanh_b)).cpu().numpy()
    print(f"tanh linear train RMSE {np.sqrt(np.mean((tanh_fit - train_force) ** 2)):.4f}")

    torch.manual_seed(0)
    plain_w = torch.zeros(obs_t.shape[1], device=device, requires_grad=True)
    plain_b = torch.zeros((), device=device, requires_grad=True)
    optimizer = optim.AdamW((plain_w, plain_b), lr=3e-4, amsgrad=True, weight_decay=0.0)
    for _ in range(20000):
        batch = torch.randint(0, obs_t.shape[0], (128,), device=device)
        predicted = scale * torch.tanh(obs_t[batch] @ plain_w + plain_b)
        loss = torch.nn.functional.mse_loss(predicted, force_t[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        plain_fit = (scale * torch.tanh(obs_t @ plain_w + plain_b)).cpu().numpy()
    print(f"tanh linear, no decay train RMSE {np.sqrt(np.mean((plain_fit - train_force) ** 2)):.4f}")

    starts = np.flatnonzero(train_first)
    early = np.zeros(len(train_force), dtype=bool)
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else len(train_force)
        early[start:min(start + 30, stop)] = True
    early_index = torch.tensor(np.flatnonzero(early), device=device)
    torch.manual_seed(0)
    early_w = torch.zeros(obs_t.shape[1], device=device, requires_grad=True)
    early_b = torch.zeros((), device=device, requires_grad=True)
    optimizer = optim.AdamW((early_w, early_b), lr=3e-4, amsgrad=True, weight_decay=0.0)
    for _ in range(20000):
        batch = early_index[torch.randint(0, early_index.shape[0], (128,), device=device)]
        predicted = scale * torch.tanh(obs_t[batch] @ early_w + early_b)
        loss = torch.nn.functional.mse_loss(predicted, force_t[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        early_fit = (scale * torch.tanh(obs_t @ early_w + early_b)).cpu().numpy()
    print(
        f"tanh linear, first 30 of {int(early.sum())} decisions, "
        f"train RMSE {np.sqrt(np.mean((early_fit - train_force) ** 2)):.4f}"
    )

    ratio = np.clip(train_force / scale, -1.0 + 1e-6, 1.0 - 1e-6)
    mean_t = torch.tensor(np.arctanh(ratio), dtype=torch.float32, device=device)
    torch.manual_seed(0)
    atanh_policy = SAC_POLICY_DP(train_obs.shape[1]).to(device)
    atanh_logstd_weight = atanh_policy.layer3.weight.data[1].clone()
    atanh_logstd_bias = atanh_policy.layer3.bias.data[1].clone()
    optimizer = optim.AdamW(atanh_policy.parameters(), lr=3e-4, amsgrad=True)
    for _ in range(20000):
        batch = torch.randint(0, obs_t.shape[0], (128,), device=device)
        loss = torch.nn.functional.mse_loss(atanh_policy(obs_t[batch])[:, 0], mean_t[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            atanh_policy.layer3.weight[1].copy_(atanh_logstd_weight)
            atanh_policy.layer3.bias[1].copy_(atanh_logstd_bias)
    with torch.no_grad():
        atanh_fit = (scale * torch.tanh(atanh_policy(obs_t)[:, 0])).cpu().numpy()
    print(f"atanh actor train RMSE {np.sqrt(np.mean((atanh_fit - train_force) ** 2)):.4f}")

    def linear_force(observation):
        return float(np.clip(observation @ linear_w + linear_b, -scale, scale))

    def neural_force(observation):
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return float(scale * torch.tanh(policy(state)[0, 0]))

    def tanh_linear_force(observation):
        state = torch.tensor(observation, dtype=torch.float32, device=device)
        with torch.no_grad():
            return float(scale * torch.tanh(state @ tanh_w + tanh_b))

    def plain_tanh_force(observation):
        state = torch.tensor(observation, dtype=torch.float32, device=device)
        with torch.no_grad():
            return float(scale * torch.tanh(state @ plain_w + plain_b))

    def early_tanh_force(observation):
        state = torch.tensor(observation, dtype=torch.float32, device=device)
        with torch.no_grad():
            return float(scale * torch.tanh(state @ early_w + early_b))

    def atanh_force(observation):
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return float(scale * torch.tanh(atanh_policy(state)[0, 0]))

    def summarize(label, seeds, policy_fn):
        rows = [rollout(env, seed, policy_fn) for seed in seeds]
        successes = sum(row[2] for row in rows)
        lengths = [row[1] for row in rows]
        print(
            f"{label} seeds {seeds[0]}-{seeds[-1]}: "
            f"success {successes}/20, median decisions {np.median(lengths):.0f}"
        )

    def from_obs(force_fn):
        def policy_fn(env):
            observation = env.unwrapped._clip_obs(env.unwrapped._get_obs())
            return force_fn(observation)
        return policy_fn

    policies = (
        ("linear", from_obs(linear_force)),
        ("neural", from_obs(neural_force)),
        ("tanh linear", from_obs(tanh_linear_force)),
        ("tanh linear, no decay", from_obs(plain_tanh_force)),
        ("tanh linear, first 30", from_obs(early_tanh_force)),
        ("atanh actor", from_obs(atanh_force)),
        ("zero", lambda env: 0.0),
        ("LQR", lambda env, gain=gain: lqr_force(env, gain)),
    )
    for name, policy_fn in policies:
        summarize(name, range(20), policy_fn)
        summarize(name, range(100, 120), policy_fn)

    valid_obs, valid_force, valid_first = trace(range(100, 120))

    def force_report(name, force_fn):
        predicted = np.array([force_fn(observation) for observation in valid_obs])
        for tag, mask in (
            ("initial", valid_first),
            ("|F|>0.01", np.abs(valid_force) > 0.01),
        ):
            error = np.abs(predicted[mask] - valid_force[mask])
            signs = np.sign(predicted[mask]) == np.sign(valid_force[mask])
            print(
                f"{name} {tag}: n {int(mask.sum())}, sign {int(signs.sum())}/{int(mask.sum())}, "
                f"median |error| {np.median(error):.4f}"
            )

    force_report("linear", linear_force)
    force_report("neural", neural_force)
    force_report("tanh linear", tanh_linear_force)
    force_report("tanh linear, no decay", plain_tanh_force)
    force_report("tanh linear, first 30", early_tanh_force)
    force_report("atanh actor", atanh_force)


def adam_linear_budget(env, gain):
    scale = float(env.action_space.high[0])
    observations, forces, first = [], [], []
    for seed in range(20):
        observation, _ = env.reset(seed=seed)
        done = False
        opening = True
        while not done:
            force = lqr_force(env, gain)
            observations.append(np.asarray(observation, dtype=np.float64))
            forces.append(force)
            first.append(opening)
            opening = False
            observation, _, terminated, truncated, _ = env.step(force)
            done = terminated or truncated
    train_obs = np.vstack(observations)
    train_force = np.asarray(forces)
    train_first = np.asarray(first)
    design = np.column_stack((train_obs, np.ones(len(train_obs))))
    solution, _, _, _ = np.linalg.lstsq(design, train_force, rcond=None)
    weights, bias = solution[:-1], float(solution[-1])
    largest = float(np.max(np.abs(solution)))
    singular = np.linalg.svd(design, compute_uv=False)
    design32 = torch.tensor(design, dtype=torch.float32)
    force32 = torch.tensor(train_force, dtype=torch.float32)
    singular32 = torch.linalg.svdvals(design32)
    try:
        solved = torch.linalg.solve(design32.T @ design32, design32.T @ force32)
        residual = design32 @ solved - force32
        float32_rmse = float(torch.sqrt(torch.mean(residual ** 2)))
    except RuntimeError as error:
        solved = None
        float32_rmse = None
        print(f"float32 normal-equation solve failed: {error}")
    print("least-squares weights " + " ".join(f"{weight:.4f}" for weight in weights) + f" bias {bias:.4f}")
    print(
        f"largest |weight| {largest:.4f}, condition {singular[0] / singular[-1]:.3e}, "
        f"float32 condition {float(singular32[0] / singular32[-1]):.3e}"
    )
    print("column std " + " ".join(f"{value:.6f}" for value in train_obs.std(axis=0)))
    if float32_rmse is not None:
        print(f"float32 least-squares RMSE {float32_rmse:.6f}")
        print("float32 weights " + " ".join(f"{float(value):.4f}" for value in solved))

    obs_t = torch.tensor(train_obs, dtype=torch.float32, device=device)
    force_t = torch.tensor(train_force, dtype=torch.float32, device=device)
    lr = 3e-4
    budgets = (
        ("20k", 20000),
        ("2W", int(np.ceil(2.0 * largest / lr))),
    )

    def fit(steps):
        torch.manual_seed(0)
        weight = torch.zeros(obs_t.shape[1], device=device, requires_grad=True)
        offset = torch.zeros((), device=device, requires_grad=True)
        optimizer = optim.AdamW(
            (weight, offset), lr=lr, amsgrad=True, weight_decay=0.0, capturable=device.type == "cuda"
        )
        index = torch.empty(128, dtype=torch.long, device=device)

        def one_step():
            chosen = obs_t[index]
            loss = torch.nn.functional.mse_loss(chosen @ weight + offset, force_t[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        def draw():
            with torch.no_grad():
                index.copy_(torch.randint(0, obs_t.shape[0], (128,), device=device))

        if device.type != "cuda":
            for step in range(1, steps + 1):
                draw()
                one_step()
                if step % 1000000 == 0:
                    print(f"after {step} steps max |weight| {float(weight.detach().abs().max()):.4f}")
            return weight.detach(), offset.detach()

        for _ in range(3):
            draw()
            one_step()
        graph = torch.cuda.CUDAGraph()
        draw()
        with torch.cuda.graph(graph):
            one_step()
        completed = 4
        for step in range(completed + 1, steps + 1):
            draw()
            graph.replay()
            if step % 1000000 == 0:
                print(f"after {step} steps max |weight| {float(weight.detach().abs().max()):.4f}")
        return weight.detach(), offset.detach()

    def evaluate(label, weight, offset):
        learned = np.concatenate((weight.cpu().numpy(), [float(offset)]))
        gap = float(np.max(np.abs(learned - solution)))
        print(
            f"{label} final max |weight| {float(np.max(np.abs(learned))):.4f}, "
            f"max gap to least squares {gap:.4f}"
        )
        print(label + " weights " + " ".join(f"{value:.4f}" for value in learned))

        def force_fn(observation):
            return float(np.clip(observation @ learned[:-1] + learned[-1], -scale, scale))

        def policy_fn(env):
            return force_fn(env.unwrapped._clip_obs(env.unwrapped._get_obs()))

        rows = [rollout(env, seed, policy_fn) for seed in range(100, 120)]
        successes = sum(row[2] for row in rows)
        lengths = [row[1] for row in rows]
        print(
            f"{label} seeds 100-119: success {successes}/20, "
            f"median decisions {np.median(lengths):.0f}"
        )
        valid_obs, valid_force, valid_first = [], [], []
        for seed in range(100, 120):
            observation, _ = env.reset(seed=seed)
            done = False
            opening = True
            while not done:
                force = lqr_force(env, gain)
                valid_obs.append(np.asarray(observation, dtype=np.float64))
                valid_force.append(force)
                valid_first.append(opening)
                opening = False
                observation, _, terminated, truncated, _ = env.step(force)
                done = terminated or truncated
        valid_obs = np.vstack(valid_obs)
        valid_force = np.asarray(valid_force)
        valid_first = np.asarray(valid_first)
        predicted = valid_obs @ learned[:-1] + learned[-1]
        signed = predicted[valid_first] - valid_force[valid_first]
        print(
            f"{label} initial signed error {np.median(signed):.4f}, "
            f"median |predicted| {np.median(np.abs(predicted[valid_first])):.4f}"
        )
        mask = np.abs(valid_force) > 0.01
        print(f"{label} |F|>0.01 median |error| {np.median(np.abs(predicted[mask] - valid_force[mask])):.4f}")

    for label, steps in budgets:
        print(f"Adam unsquashed linear, lr {lr:.1e}, {steps} steps")
        evaluate(label, *fit(steps))


def adam_normalized(env, gain):
    scale = float(env.action_space.high[0])
    observations, forces = [], []
    for seed in range(20):
        observation, _ = env.reset(seed=seed)
        done = False
        while not done:
            force = lqr_force(env, gain)
            observations.append(np.asarray(observation, dtype=np.float64))
            forces.append(force)
            observation, _, terminated, truncated, _ = env.step(force)
            done = terminated or truncated
    train_obs = np.vstack(observations)
    train_force = np.asarray(forces)
    center = train_obs.mean(axis=0)
    spread = train_obs.std(axis=0)
    spread = np.where(spread < 1e-8, 1.0, spread)
    normalized = (train_obs - center) / spread
    design = np.column_stack((normalized, np.ones(len(normalized))))
    solution, _, _, _ = np.linalg.lstsq(design, train_force, rcond=None)
    singular = np.linalg.svd(design, compute_uv=False)
    print(
        "normalized least-squares weights "
        + " ".join(f"{weight:.4f}" for weight in solution)
    )
    print(
        f"largest |weight| {np.max(np.abs(solution)):.4f}, "
        f"condition {singular[0] / singular[-1]:.3e}"
    )

    obs_t = torch.tensor(normalized, dtype=torch.float32, device=device)
    force_t = torch.tensor(train_force, dtype=torch.float32, device=device)
    torch.manual_seed(0)
    weight = torch.zeros(obs_t.shape[1], device=device, requires_grad=True)
    offset = torch.zeros((), device=device, requires_grad=True)
    optimizer = optim.AdamW((weight, offset), lr=3e-4, amsgrad=True, weight_decay=0.0)
    for _ in range(20000):
        batch = torch.randint(0, obs_t.shape[0], (128,), device=device)
        loss = torch.nn.functional.mse_loss(obs_t[batch] @ weight + offset, force_t[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    learned = np.concatenate((weight.detach().cpu().numpy(), [float(offset.detach())]))
    print(
        f"normalized Adam max |weight| {np.max(np.abs(learned)):.4f}, "
        f"max gap to least squares {np.max(np.abs(learned - solution)):.4f}"
    )
    print("normalized Adam weights " + " ".join(f"{value:.4f}" for value in learned))

    def force_fn(observation):
        scaled = (np.asarray(observation, dtype=np.float64) - center) / spread
        return float(np.clip(scaled @ learned[:-1] + learned[-1], -scale, scale))

    def policy_fn(env):
        return force_fn(env.unwrapped._clip_obs(env.unwrapped._get_obs()))

    rows = [rollout(env, seed, policy_fn) for seed in range(100, 120)]
    lengths = [row[1] for row in rows]
    print(
        f"normalized Adam seeds 100-119: success {sum(row[2] for row in rows)}/20, "
        f"median decisions {np.median(lengths):.0f}"
    )
    valid_obs, valid_force, valid_first = [], [], []
    for seed in range(100, 120):
        observation, _ = env.reset(seed=seed)
        done = False
        opening = True
        while not done:
            force = lqr_force(env, gain)
            valid_obs.append(np.asarray(observation, dtype=np.float64))
            valid_force.append(force)
            valid_first.append(opening)
            opening = False
            observation, _, terminated, truncated, _ = env.step(force)
            done = terminated or truncated
    valid_obs = np.vstack(valid_obs)
    valid_force = np.asarray(valid_force)
    valid_first = np.asarray(valid_first)
    predicted = np.array([force_fn(observation) for observation in valid_obs])
    signed = predicted[valid_first] - valid_force[valid_first]
    active = np.abs(valid_force) > 0.01
    print(
        f"normalized Adam initial signed error {np.median(signed):.4f}, "
        f"median |predicted| {np.median(np.abs(predicted[valid_first])):.4f}"
    )
    print(
        f"normalized Adam |F|>0.01 median |error| "
        f"{np.median(np.abs(predicted[active] - valid_force[active])):.4f}"
    )


def adam_normalized_full(env, gain):
    scale = float(env.action_space.high[0])
    observations, forces, first = [], [], []
    for seed in range(20):
        observation, _ = env.reset(seed=seed)
        done = False
        opening = True
        while not done:
            force = lqr_force(env, gain)
            observations.append(np.asarray(observation, dtype=np.float64))
            forces.append(force)
            first.append(opening)
            opening = False
            observation, _, terminated, truncated, _ = env.step(force)
            done = terminated or truncated
    train_obs = np.vstack(observations)
    train_force = np.asarray(forces)
    train_first = np.asarray(first)
    center = train_obs.mean(axis=0)
    spread = np.where(train_obs.std(axis=0) < 1e-8, 1.0, train_obs.std(axis=0))
    normalized = (train_obs - center) / spread
    design = np.column_stack((normalized, np.ones(len(normalized))))
    solution, _, _, _ = np.linalg.lstsq(design, train_force, rcond=None)
    angle = int(np.argmax(np.abs(solution[:-1])))
    print(
        "normalized least-squares weights "
        + " ".join(f"{weight:.4f}" for weight in solution)
        + f", angle coordinate {angle}"
    )

    obs_t = torch.tensor(normalized, dtype=torch.float32, device=device)
    force_t = torch.tensor(train_force, dtype=torch.float32, device=device)
    torch.manual_seed(0)
    weight = torch.zeros(obs_t.shape[1], device=device, requires_grad=True)
    offset = torch.zeros((), device=device, requires_grad=True)
    optimizer = optim.AdamW((weight, offset), lr=3e-4, amsgrad=True, weight_decay=0.0)
    beta1, beta2 = optimizer.defaults["betas"]
    eps = optimizer.defaults["eps"]
    checkpoints = {0, 100, 1000, 5000, 10000, 20000}

    valid_obs, valid_force, valid_first = [], [], []
    for seed in range(100, 120):
        observation, _ = env.reset(seed=seed)
        done = False
        opening = True
        while not done:
            force = lqr_force(env, gain)
            valid_obs.append(np.asarray(observation, dtype=np.float64))
            valid_force.append(force)
            valid_first.append(opening)
            opening = False
            observation, _, terminated, truncated, _ = env.step(force)
            done = terminated or truncated
    valid_obs = np.vstack(valid_obs)
    valid_force = np.asarray(valid_force)
    valid_first = np.asarray(valid_first)
    valid_scaled = (valid_obs - center) / spread

    def report(step):
        learned = np.concatenate((weight.detach().cpu().numpy(), [float(offset.detach())]))
        gap = float(np.max(np.abs(learned - solution)))
        with torch.no_grad():
            train_pred = (obs_t @ weight + offset).detach().cpu().numpy()
        full_rmse = float(np.sqrt(np.mean((train_pred - train_force) ** 2)))
        first_rmse = float(np.sqrt(np.mean((train_pred[train_first] - train_force[train_first]) ** 2)))
        predicted = valid_scaled @ learned[:-1] + learned[-1]
        opening_error = np.abs(predicted[valid_first] - valid_force[valid_first])

        def force_fn(observation):
            scaled = (np.asarray(observation, dtype=np.float64) - center) / spread
            return float(np.clip(scaled @ learned[:-1] + learned[-1], -scale, scale))

        def policy_fn(env):
            return force_fn(env.unwrapped._clip_obs(env.unwrapped._get_obs()))

        rows = [rollout(env, seed, policy_fn) for seed in range(100, 120)]
        lengths = [row[1] for row in rows]
        successes = [bool(row[2]) for row in rows]
        state = optimizer.state.get(weight)
        if state and "max_exp_avg_sq" in state and step:
            moment = state["max_exp_avg_sq"][angle]
            average = state["exp_avg"][angle]
            correction = (1 - beta2 ** step) ** 0.5
            denom = moment.sqrt() / correction + eps
            update = abs(float((optimizer.defaults["lr"] / (1 - beta1 ** step)) * average / denom))
            angle_line = (
                f"angle weight {float(weight.detach()[angle]):.4f}, "
                f"sqrt(v_max) {float(moment.sqrt()):.4e}, update {update:.4e}"
            )
        else:
            angle_line = "angle weight 0, sqrt(v_max) n/a, update n/a"
        print(
            f"after {step} full-batch updates gap {gap:.4f}, "
            f"RMSE {full_rmse:.4f}, first RMSE {first_rmse:.4f}, "
            f"initial |error| {np.median(opening_error):.4f}, "
            f"success {sum(successes)}/20, median decisions {np.median(lengths):.0f}, {angle_line}"
        )
        return opening_error, successes

    report(0)
    final = None
    for step in range(1, 20001):
        loss = torch.nn.functional.mse_loss(obs_t @ weight + offset, force_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in checkpoints:
            final = report(step)
    opening_error, successes = final
    order = np.argsort(-opening_error)
    print("validation openings, largest absolute error first")
    for index in order:
        print(
            f"seed {100 + index} |F| {abs(valid_force[valid_first][index]):.4f}, "
            f"|error| {opening_error[index]:.4f}, success {successes[index]}"
        )
    failed = np.flatnonzero(~np.asarray(successes))
    worst = set(order[:len(failed)].tolist())
    print(f"failures are the largest opening errors {set(failed.tolist()) == worst}")


def actor_normalized_full(env, gain):
    scale = float(env.action_space.high[0])
    observations, forces = [], []
    for seed in range(20):
        observation, _ = env.reset(seed=seed)
        done = False
        while not done:
            force = lqr_force(env, gain)
            observations.append(np.asarray(observation, dtype=np.float64))
            forces.append(force)
            observation, _, terminated, truncated, _ = env.step(force)
            done = terminated or truncated
    train_obs = np.vstack(observations)
    train_force = np.asarray(forces)
    center = train_obs.mean(axis=0)
    spread = np.where(train_obs.std(axis=0) < 1e-8, 1.0, train_obs.std(axis=0))
    normalized = (train_obs - center) / spread
    design = np.column_stack((normalized, np.ones(len(normalized))))
    solution, _, _, _ = np.linalg.lstsq(design, train_force, rcond=None)
    print(
        "normalized least-squares angle weights "
        f"{solution[1]:.4f} {solution[2]:.4f}"
    )

    obs_t = torch.tensor(normalized, dtype=torch.float32, device=device)
    force_t = torch.tensor(train_force, dtype=torch.float32, device=device)
    torch.manual_seed(0)
    policy = SAC_POLICY_DP(obs_t.shape[1]).to(device)
    logstd_weight = policy.layer3.weight.data[1].clone()
    logstd_bias = policy.layer3.bias.data[1].clone()
    optimizer = optim.AdamW(policy.parameters(), lr=3e-4, amsgrad=True, weight_decay=0.0)

    valid_obs, valid_force, valid_first = [], [], []
    for seed in range(100, 120):
        observation, _ = env.reset(seed=seed)
        done = False
        opening = True
        while not done:
            force = lqr_force(env, gain)
            valid_obs.append(np.asarray(observation, dtype=np.float64))
            valid_force.append(force)
            valid_first.append(opening)
            opening = False
            observation, _, terminated, truncated, _ = env.step(force)
            done = terminated or truncated
    valid_obs = np.vstack(valid_obs)
    valid_force = np.asarray(valid_force)
    valid_first = np.asarray(valid_first)
    valid_t = torch.tensor((valid_obs - center) / spread, dtype=torch.float32, device=device)
    place_state(env, np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0]))
    upright = env.unwrapped._clip_obs(env.unwrapped._get_obs())
    upright_z = (np.asarray(upright, dtype=np.float64) - center) / spread

    def report(step):
        with torch.no_grad():
            means = policy(valid_t)[:, 0]
            predicted = (scale * torch.tanh(means)).cpu().numpy()
            opening_means = means[torch.tensor(valid_first, device=device)]
        opening_error = np.abs(predicted[valid_first] - valid_force[valid_first])
        active = np.abs(valid_force) > 0.01
        probe = torch.tensor(upright_z, dtype=torch.float32, device=device).unsqueeze(0)
        probe.requires_grad_(True)
        force = scale * torch.tanh(policy(probe)[0, 0])
        jacobian = torch.autograd.grad(force, probe)[0][0].detach().cpu().numpy()

        def policy_fn(env):
            observation = env.unwrapped._clip_obs(env.unwrapped._get_obs())
            scaled = (np.asarray(observation, dtype=np.float64) - center) / spread
            state = torch.tensor(scaled, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                return float(scale * torch.tanh(policy(state)[0, 0]))

        rows = [rollout(env, seed, policy_fn) for seed in range(100, 120)]
        lengths = [row[1] for row in rows]
        print(
            f"after {step} actor updates success {sum(row[2] for row in rows)}/20, "
            f"median decisions {np.median(lengths):.0f}, min decisions {np.min(lengths):.0f}, "
            f"opening |error| median {np.median(opening_error):.4f} max {np.max(opening_error):.4f}, "
            f"opening |F| predicted {np.median(np.abs(predicted[valid_first])):.4f} "
            f"LQR {np.median(np.abs(valid_force[valid_first])):.4f}, "
            f"|F|>0.01 median |error| {np.median(np.abs(predicted[active] - valid_force[active])):.4f}, "
            f"max |pre-tanh| {float(opening_means.abs().max()):.4f}"
        )
        print(
            "upright Jacobian "
            + " ".join(f"{value:.4f}" for value in jacobian)
            + f" angle slopes {jacobian[1]:.4f} {jacobian[2]:.4f}"
        )

    report(0)
    checkpoints = {1000, 5000, 10000, 20000}
    for step in range(1, 20001):
        loss = torch.nn.functional.mse_loss(scale * torch.tanh(policy(obs_t)[:, 0]), force_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            policy.layer3.weight[1].copy_(logstd_weight)
            policy.layer3.bias[1].copy_(logstd_bias)
        if step in checkpoints:
            report(step)


def return_critic_actor(env, gain):
    scale = float(env.action_space.high[0])
    gamma = 0.99

    def discounted(rewards):
        values = np.zeros(len(rewards))
        running = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            running = rewards[index] + gamma * running
            values[index] = running
        return values

    lqr_obs, lqr_forces, lqr_returns = [], [], []
    for seed in range(20):
        observation, _ = env.reset(seed=seed)
        episode_obs, episode_forces, rewards = [], [], []
        done = False
        while not done:
            episode_obs.append(np.asarray(observation, dtype=np.float64))
            force = lqr_force(env, gain)
            episode_forces.append(force)
            observation, reward, terminated, truncated, _ = env.step(force)
            rewards.append(float(reward))
            done = terminated or truncated
        lqr_obs.extend(episode_obs)
        lqr_forces.extend(episode_forces)
        lqr_returns.extend(discounted(rewards))
    lqr_obs = np.vstack(lqr_obs)
    lqr_forces = np.asarray(lqr_forces)
    lqr_returns = np.asarray(lqr_returns)
    center = lqr_obs.mean(axis=0)
    spread = np.where(lqr_obs.std(axis=0) < 1e-8, 1.0, lqr_obs.std(axis=0))

    zero_obs, zero_returns = [], []
    for seed in range(20):
        observation, _ = env.reset(seed=seed)
        zero_obs.append(np.asarray(observation, dtype=np.float64))
        rewards = []
        done = False
        while not done:
            observation, reward, terminated, truncated, _ = env.step(0.0)
            rewards.append(float(reward))
            done = terminated or truncated
        zero_returns.append(discounted(rewards)[0])
    zero_obs = np.vstack(zero_obs)
    zero_returns = np.asarray(zero_returns)
    print(
        f"LQR return median {np.median(lqr_returns):.2f}, "
        f"zero-opening return median {np.median(zero_returns):.2f}"
    )

    def standardize(observation):
        return (np.asarray(observation, dtype=np.float64) - center) / spread

    lqr_z = torch.tensor(standardize(lqr_obs), dtype=torch.float32, device=device)
    zero_z = torch.tensor(standardize(zero_obs), dtype=torch.float32, device=device)
    lqr_action = torch.tensor(lqr_forces / scale, dtype=torch.float32, device=device).unsqueeze(-1)
    zero_action = torch.zeros(len(zero_obs), 1, dtype=torch.float32, device=device)
    lqr_target = torch.tensor(lqr_returns, dtype=torch.float32, device=device)
    zero_target = torch.tensor(zero_returns, dtype=torch.float32, device=device)
    lqr_in = torch.cat((lqr_z, lqr_action), dim=-1)
    zero_in = torch.cat((zero_z, zero_action), dim=-1)

    torch.manual_seed(0)
    critics = [SAC_Q_DP(lqr_z.shape[1]).to(device) for _ in range(2)]
    critic_optimizer = optim.AdamW(
        [parameter for critic in critics for parameter in critic.parameters()],
        lr=3e-4, amsgrad=True, weight_decay=0.0,
    )
    for _ in range(20000):
        predictions = [critic(lqr_in).squeeze(-1) for critic in critics]
        zero_predictions = [critic(zero_in).squeeze(-1) for critic in critics]
        loss = sum(
            torch.nn.functional.mse_loss(prediction, lqr_target)
            + torch.nn.functional.mse_loss(zero_prediction, zero_target)
            for prediction, zero_prediction in zip(predictions, zero_predictions)
        )
        critic_optimizer.zero_grad()
        loss.backward()
        critic_optimizer.step()
    with torch.no_grad():
        for critic in critics:
            lqr_error = critic(lqr_in).squeeze(-1) - lqr_target
            zero_error = critic(zero_in).squeeze(-1) - zero_target
            print(
                f"critic RMSE LQR {float(torch.sqrt(torch.mean(lqr_error ** 2))):.3f}, "
                f"zero openings {float(torch.sqrt(torch.mean(zero_error ** 2))):.3f}"
            )
    for critic in critics:
        for parameter in critic.parameters():
            parameter.requires_grad_(False)

    valid_obs, valid_force, valid_first = [], [], []
    for seed in range(100, 120):
        observation, _ = env.reset(seed=seed)
        done = False
        opening = True
        while not done:
            force = lqr_force(env, gain)
            valid_obs.append(np.asarray(observation, dtype=np.float64))
            valid_force.append(force)
            valid_first.append(opening)
            opening = False
            observation, _, terminated, truncated, _ = env.step(force)
            done = terminated or truncated
    valid_obs = np.vstack(valid_obs)
    valid_force = np.asarray(valid_force)
    valid_first = np.asarray(valid_first)
    opening_obs = valid_obs[valid_first]
    opening_force = valid_force[valid_first]
    opening_z = torch.tensor(standardize(opening_obs), dtype=torch.float32, device=device)

    def min_q(states, force):
        action = torch.tensor(
            np.asarray(force, dtype=np.float64) / scale,
            dtype=torch.float32, device=device,
        ).reshape(-1, 1)
        scores = torch.cat([critic(torch.cat((states, action), dim=-1)) for critic in critics], dim=-1)
        return torch.min(scores, dim=-1).values

    with torch.no_grad():
        lqr_value = min_q(opening_z, opening_force)
        zero_value = min_q(opening_z, np.zeros(len(opening_force)))
        gap = lqr_value - zero_value
    print(
        f"validation resets prefer LQR {int((gap > 0).sum())}/{len(gap)}, "
        f"median gap {float(torch.median(gap)):.3f}"
    )

    torch.manual_seed(0)
    policy = SAC_POLICY_DP(lqr_z.shape[1]).to(device)
    logstd_weight = policy.layer3.weight.data[1].clone()
    logstd_bias = policy.layer3.bias.data[1].clone()
    actor_optimizer = optim.AdamW(policy.parameters(), lr=3e-4, amsgrad=True, weight_decay=0.0)
    valid_t = torch.tensor(standardize(valid_obs), dtype=torch.float32, device=device)
    place_state(env, np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0]))
    upright = standardize(env.unwrapped._clip_obs(env.unwrapped._get_obs()))

    def report(step):
        with torch.no_grad():
            means = policy(valid_t)[:, 0]
            predicted = (scale * torch.tanh(means)).cpu().numpy()
        opening_error = np.abs(predicted[valid_first] - valid_force[valid_first])
        probe = torch.tensor(upright, dtype=torch.float32, device=device).unsqueeze(0)
        probe.requires_grad_(True)
        force = scale * torch.tanh(policy(probe)[0, 0])
        jacobian = torch.autograd.grad(force, probe)[0][0].detach().cpu().numpy()

        def policy_fn(env):
            state = torch.tensor(
                standardize(env.unwrapped._clip_obs(env.unwrapped._get_obs())),
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                return float(scale * torch.tanh(policy(state)[0, 0]))

        rows = [rollout(env, seed, policy_fn) for seed in range(100, 120)]
        lengths = [row[1] for row in rows]
        print(
            f"after {step} actor updates success {sum(row[2] for row in rows)}/20, "
            f"median decisions {np.median(lengths):.0f}, min decisions {np.min(lengths):.0f}, "
            f"opening |error| median {np.median(opening_error):.4f}, "
            f"opening |F| predicted {np.median(np.abs(predicted[valid_first])):.4f} "
            f"LQR {np.median(np.abs(valid_force[valid_first])):.4f}, "
            f"max |pre-tanh| {float(means[torch.tensor(valid_first, device=device)].abs().max()):.4f}"
        )
        print(
            "upright Jacobian "
            + " ".join(f"{value:.4f}" for value in jacobian)
            + f" angle slopes {jacobian[1]:.4f} {jacobian[2]:.4f}"
        )

    report(0)
    for step in range(1, 20001):
        action = torch.tanh(policy(lqr_z)[:, 0]).unsqueeze(-1)
        scores = torch.cat([
            critic(torch.cat((lqr_z, action), dim=-1)) for critic in critics
        ], dim=-1)
        loss = -torch.min(scores, dim=-1).values.mean()
        actor_optimizer.zero_grad()
        loss.backward()
        actor_optimizer.step()
        with torch.no_grad():
            policy.layer3.weight[1].copy_(logstd_weight)
            policy.layer3.bias[1].copy_(logstd_bias)
        if step in {1000, 20000}:
            report(step)


def _limit_blas():
    import ctypes
    from pathlib import Path
    import scipy
    folder = Path(scipy.__file__).resolve().parent.parent / "scipy.libs"
    for path in folder.glob("libscipy_openblas*.so"):
        ctypes.CDLL(str(path)).scipy_openblas_set_num_threads(1)
    for path in Path(sys.prefix).glob("lib/libopenblas*.so"):
        setter = getattr(ctypes.CDLL(str(path)), "openblas_set_num_threads", None)
        if setter is not None:
            setter(1)


def _discounted_continuation(env, physics, first_force, gain, gamma=0.99):
    place_state(env, physics)
    total = 0.0
    discount = 1.0
    first = True
    done = False
    while not done:
        force = float(first_force) if first else lqr_force(env, gain)
        first = False
        _, reward, terminated, truncated, _ = env.step(force)
        total += discount * float(reward)
        discount *= gamma
        done = terminated or truncated
    return total


def _decision_indices(count):
    head = list(range(min(15, count)))
    if count <= 15:
        return head
    extra = np.unique(np.round(np.linspace(15, count - 1, 15)).astype(int)).tolist()
    return head + extra


def _paired_labels(seed, gain):
    _limit_blas()
    env = init_env()
    observation, _ = env.reset(seed=int(seed))
    observations, physics, forces = [], [], []
    done = False
    while not done:
        observations.append(np.asarray(observation, dtype=np.float64))
        physics.append(env.unwrapped.pendulum.x.copy())
        force = lqr_force(env, gain)
        forces.append(force)
        observation, _, terminated, truncated, _ = env.step(force)
        done = terminated or truncated
    grid = np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0])
    selected_obs, selected_actions, selected_returns = [], [], []
    for index in _decision_indices(len(observations)):
        actions = np.concatenate([grid, [forces[index]]])
        returns = [
            _discounted_continuation(env, physics[index], action, gain) for action in actions
        ]
        selected_obs.append(observations[index])
        selected_actions.append(actions)
        selected_returns.append(returns)
    return (
        np.vstack(observations),
        np.asarray(forces),
        np.vstack(selected_obs),
        np.vstack(selected_actions),
        np.vstack(selected_returns),
    )


def _opening_labels(seed, gain):
    _limit_blas()
    env = init_env()
    observation, _ = env.reset(seed=int(seed))
    physics = env.unwrapped.pendulum.x.copy()
    actions = np.concatenate([
        np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]),
        [lqr_force(env, gain)],
    ])
    returns = [
        _discounted_continuation(env, physics, action, gain) for action in actions
    ]
    return np.asarray(observation, dtype=np.float64), actions, np.asarray(returns)


def _fork_map(func, argument_sets, workers=16, message="labeled seed"):
    import multiprocessing as mp
    import time
    import __main__
    setattr(__main__, func.__name__, func)
    saved_module = func.__module__
    func.__module__ = "__main__"
    outputs = {}
    try:
        with mp.get_context("fork").Pool(workers) as pool:
            pending = {
                pool.apply_async(func, arguments): arguments[0]
                for arguments in argument_sets
            }
            while pending:
                finished = [handle for handle in pending if handle.ready()]
                if not finished:
                    time.sleep(1.0)
                    continue
                for handle in finished:
                    seed = pending.pop(handle)
                    outputs[seed] = handle.get()
                    print(f"{message} {seed}", flush=True)
    finally:
        func.__module__ = saved_module
    return [outputs[arguments[0]] for arguments in argument_sets]


def paired_return_actor(env, gain):
    scale = float(env.action_space.high[0])
    train_jobs = [(seed, gain) for seed in range(20)]
    train_rows = _fork_map(_paired_labels, train_jobs)
    all_obs = np.vstack([row[0] for row in train_rows])
    all_force = np.concatenate([row[1] for row in train_rows])
    selected_obs = np.vstack([row[2] for row in train_rows])
    selected_actions = np.vstack([row[3] for row in train_rows])
    selected_returns = np.vstack([row[4] for row in train_rows])
    center = all_obs.mean(axis=0)
    spread = np.where(all_obs.std(axis=0) < 1e-8, 1.0, all_obs.std(axis=0))
    normalized = (all_obs - center) / spread
    design = np.column_stack((normalized, np.ones(len(normalized))))
    solution, _, _, _ = np.linalg.lstsq(design, all_force, rcond=None)
    print(
        f"labeled states {len(selected_obs)}, actions {selected_actions.shape[1]}, "
        f"normalized least-squares angle weights {solution[1]:.4f} {solution[2]:.4f}"
    )
    print(
        "training return range median "
        f"{np.median(selected_returns.max(axis=1) - selected_returns.min(axis=1)):.2f}"
    )

    def standardize(observation):
        return (np.asarray(observation, dtype=np.float64) - center) / spread

    paired_obs = np.repeat(selected_obs, selected_actions.shape[1], axis=0)
    paired_force = selected_actions.reshape(-1)
    paired_target = selected_returns.reshape(-1)
    paired_in = torch.tensor(
        np.column_stack((standardize(paired_obs), paired_force / scale)),
        dtype=torch.float32, device=device,
    )
    target = torch.tensor(paired_target, dtype=torch.float32, device=device)
    torch.manual_seed(0)
    critics = [SAC_Q_DP(selected_obs.shape[1]).to(device) for _ in range(2)]
    critic_optimizer = optim.AdamW(
        [parameter for critic in critics for parameter in critic.parameters()],
        lr=3e-4, amsgrad=True, weight_decay=0.0,
    )
    for _ in range(20000):
        loss = sum(
            torch.nn.functional.mse_loss(critic(paired_in).squeeze(-1), target)
            for critic in critics
        )
        critic_optimizer.zero_grad()
        loss.backward()
        critic_optimizer.step()
    with torch.no_grad():
        for critic in critics:
            error = critic(paired_in).squeeze(-1) - target
            print(f"critic RMSE {float(torch.sqrt(torch.mean(error ** 2))):.3f}")
    for critic in critics:
        for parameter in critic.parameters():
            parameter.requires_grad_(False)

    def min_values(observations, actions):
        width = actions.shape[1]
        repeated = np.repeat(standardize(observations), width, axis=0)
        states = torch.tensor(repeated, dtype=torch.float32, device=device)
        action = torch.tensor(
            actions.reshape(-1) / scale, dtype=torch.float32, device=device,
        ).unsqueeze(-1)
        with torch.no_grad():
            scores = torch.cat([
                critic(torch.cat((states, action), dim=-1)) for critic in critics
            ], dim=-1)
            values = torch.min(scores, dim=-1).values.reshape(len(observations), width)
        return values.cpu().numpy()

    def summarize(label, actions, returns, values):
        oracle = np.argmax(returns, axis=1)
        critic_choice = np.argmax(values, axis=1)
        rows = np.arange(len(actions))
        oracle_force = actions[rows, oracle]
        critic_force = actions[rows, critic_choice]
        interior = lambda force: np.abs(np.abs(force) - 15.0) >= 1e-6
        print(
            f"{label} critic matches oracle {int(np.sum(oracle == critic_choice))}/{len(actions)}, "
            f"median |force gap| {np.median(np.abs(critic_force - oracle_force)):.4f}, "
            f"interior critic {int(np.sum(interior(critic_force)))}/{len(actions)}, "
            f"oracle {int(np.sum(interior(oracle_force)))}/{len(actions)}, "
            f"oracle picks LQR {int(np.sum(np.isclose(oracle_force, actions[:, -1])))}/{len(actions)}"
        )

    summarize(
        "training",
        selected_actions,
        selected_returns,
        min_values(selected_obs, selected_actions),
    )
    valid_jobs = [(seed, gain) for seed in range(100, 120)]
    valid_rows = _fork_map(_opening_labels, valid_jobs)
    opening_obs = np.vstack([row[0] for row in valid_rows])
    opening_actions = np.vstack([row[1] for row in valid_rows])
    opening_returns = np.vstack([row[2] for row in valid_rows])
    summarize(
        "validation resets",
        opening_actions,
        opening_returns,
        min_values(opening_obs, opening_actions),
    )

    torch.manual_seed(0)
    policy = SAC_POLICY_DP(selected_obs.shape[1]).to(device)
    logstd_weight = policy.layer3.weight.data[1].clone()
    logstd_bias = policy.layer3.bias.data[1].clone()
    actor_optimizer = optim.AdamW(policy.parameters(), lr=3e-4, amsgrad=True, weight_decay=0.0)
    actor_states = torch.tensor(standardize(selected_obs), dtype=torch.float32, device=device)
    valid_obs, valid_force, valid_first = [], [], []
    for seed in range(100, 120):
        observation, _ = env.reset(seed=seed)
        done = False
        opening = True
        while not done:
            force = lqr_force(env, gain)
            valid_obs.append(np.asarray(observation, dtype=np.float64))
            valid_force.append(force)
            valid_first.append(opening)
            opening = False
            observation, _, terminated, truncated, _ = env.step(force)
            done = terminated or truncated
    valid_obs = np.vstack(valid_obs)
    valid_force = np.asarray(valid_force)
    valid_first = np.asarray(valid_first)
    valid_t = torch.tensor(standardize(valid_obs), dtype=torch.float32, device=device)
    place_state(env, np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0]))
    upright = standardize(env.unwrapped._clip_obs(env.unwrapped._get_obs()))

    def report(step):
        with torch.no_grad():
            means = policy(valid_t)[:, 0]
            predicted = (scale * torch.tanh(means)).cpu().numpy()
        opening_error = np.abs(predicted[valid_first] - valid_force[valid_first])
        probe = torch.tensor(upright, dtype=torch.float32, device=device).unsqueeze(0)
        probe.requires_grad_(True)
        force = scale * torch.tanh(policy(probe)[0, 0])
        jacobian = torch.autograd.grad(force, probe)[0][0].detach().cpu().numpy()

        def policy_fn(env):
            state = torch.tensor(
                standardize(env.unwrapped._clip_obs(env.unwrapped._get_obs())),
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                return float(scale * torch.tanh(policy(state)[0, 0]))

        rows = [rollout(env, seed, policy_fn) for seed in range(100, 120)]
        lengths = [row[1] for row in rows]
        print(
            f"after {step} actor updates success {sum(row[2] for row in rows)}/20, "
            f"median decisions {np.median(lengths):.0f}, min decisions {np.min(lengths):.0f}, "
            f"opening |error| median {np.median(opening_error):.4f} max {np.max(opening_error):.4f}, "
            f"opening |F| predicted {np.median(np.abs(predicted[valid_first])):.4f} "
            f"LQR {np.median(np.abs(valid_force[valid_first])):.4f}, "
            f"max |pre-tanh| {float(means[torch.tensor(valid_first, device=device)].abs().max()):.4f}"
        )
        print(
            "upright Jacobian "
            + " ".join(f"{value:.4f}" for value in jacobian)
            + f" angle slopes {jacobian[1]:.4f} {jacobian[2]:.4f}"
        )

    report(0)
    for step in range(1, 20001):
        action = torch.tanh(policy(actor_states)[:, 0]).unsqueeze(-1)
        scores = torch.cat([
            critic(torch.cat((actor_states, action), dim=-1)) for critic in critics
        ], dim=-1)
        loss = -torch.min(scores, dim=-1).values.mean()
        actor_optimizer.zero_grad()
        loss.backward()
        actor_optimizer.step()
        with torch.no_grad():
            policy.layer3.weight[1].copy_(logstd_weight)
            policy.layer3.bias[1].copy_(logstd_bias)
        if step in {1000, 20000}:
            report(step)


def oracle_grid_actor(env, gain):
    scale = float(env.action_space.high[0])
    train_rows = _fork_map(_paired_labels, [(seed, gain) for seed in range(20)])
    all_obs = np.vstack([row[0] for row in train_rows])
    all_force = np.concatenate([row[1] for row in train_rows])
    selected_obs = np.vstack([row[2] for row in train_rows])
    selected_actions = np.vstack([row[3] for row in train_rows])
    selected_returns = np.vstack([row[4] for row in train_rows])
    center = all_obs.mean(axis=0)
    spread = np.where(all_obs.std(axis=0) < 1e-8, 1.0, all_obs.std(axis=0))
    normalized = (all_obs - center) / spread
    design = np.column_stack((normalized, np.ones(len(normalized))))
    solution, _, _, _ = np.linalg.lstsq(design, all_force, rcond=None)
    oracle = np.argmax(selected_returns, axis=1)
    row_index = np.arange(len(selected_obs))
    oracle_force = selected_actions[row_index, oracle]
    at_rail = np.abs(np.abs(oracle_force) - 15.0) < 1e-6
    print(
        f"labeled states {len(selected_obs)}, "
        f"oracle at ±15 {int(at_rail.sum())}/{len(oracle_force)}, "
        f"oracle picks LQR {int(np.isclose(oracle_force, selected_actions[:, -1]).sum())}/{len(oracle_force)}, "
        f"least-squares angle weights {solution[1]:.4f} {solution[2]:.4f}"
    )
    valid_rows = _fork_map(_opening_labels, [(seed, gain) for seed in range(100, 120)])
    opening_obs = np.vstack([row[0] for row in valid_rows])
    opening_actions = np.vstack([row[1] for row in valid_rows])
    opening_returns = np.vstack([row[2] for row in valid_rows])
    opening_oracle = np.argmax(opening_returns, axis=1)
    opening_best = opening_actions[np.arange(len(opening_obs)), opening_oracle]
    opening_lqr = opening_actions[:, -1]

    def standardize(observation):
        return (np.asarray(observation, dtype=np.float64) - center) / spread

    def snap(predicted, candidates):
        return np.argmin(np.abs(candidates - predicted[:, None]), axis=1)

    obs_t = torch.tensor(standardize(selected_obs), dtype=torch.float32, device=device)
    force_t = torch.tensor(oracle_force, dtype=torch.float32, device=device)
    opening_t = torch.tensor(standardize(opening_obs), dtype=torch.float32, device=device)
    train_z = standardize(selected_obs)
    torch.manual_seed(0)
    policy = SAC_POLICY_DP(obs_t.shape[1]).to(device)
    logstd_weight = policy.layer3.weight.data[1].clone()
    logstd_bias = policy.layer3.bias.data[1].clone()
    optimizer = optim.AdamW(policy.parameters(), lr=3e-4, amsgrad=True, weight_decay=0.0)
    place_state(env, np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0]))
    upright = standardize(env.unwrapped._clip_obs(env.unwrapped._get_obs()))

    def cloud_distance(episodes):
        if not episodes:
            return None
        sample = np.vstack(episodes)
        distance = np.linalg.norm(
            standardize(sample)[:, None, :] - train_z[None, :, :], axis=2
        ).min(axis=1)
        return float(np.median(distance))

    def report(step):
        with torch.no_grad():
            train_means = policy(obs_t)[:, 0]
            train_predicted = (scale * torch.tanh(train_means)).cpu().numpy()
            opening_means = policy(opening_t)[:, 0]
            opening_predicted = (scale * torch.tanh(opening_means)).cpu().numpy()
        train_error = train_predicted - oracle_force
        train_agree = int(np.sum(snap(train_predicted, selected_actions) == oracle))
        opening_agree = int(np.sum(snap(opening_predicted, opening_actions) == opening_oracle))
        near_rail = int(np.sum(np.abs(np.abs(opening_predicted) - 15.0) <= 1.0))
        probe = torch.tensor(upright, dtype=torch.float32, device=device).unsqueeze(0)
        probe.requires_grad_(True)
        probe_force = scale * torch.tanh(policy(probe)[0, 0])
        jacobian = torch.autograd.grad(probe_force, probe)[0][0].detach().cpu().numpy()
        episodes = []
        for seed in range(100, 120):
            observation, _ = env.reset(seed=seed)
            seen = []
            done = False
            info = {}
            while not done:
                seen.append(np.asarray(observation, dtype=np.float64))
                state = torch.tensor(
                    standardize(observation), dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    force = float(scale * torch.tanh(policy(state)[0, 0]))
                observation, _, terminated, truncated, info = env.step(force)
                done = terminated or truncated
            episodes.append((np.vstack(seen), bool(info["success"])))
        lengths = [len(seen) for seen, _ in episodes]
        successes = [seen for seen, success in episodes if success]
        failures = [seen for seen, success in episodes if not success]
        success_distance = cloud_distance(successes)
        failure_distance = cloud_distance(failures)
        print(
            f"after {step} actor updates success {len(successes)}/20, "
            f"median decisions {np.median(lengths):.0f}, min decisions {np.min(lengths):.0f}, "
            f"train RMSE {float(np.sqrt(np.mean(train_error ** 2))):.4f}, "
            f"train grid agreement {train_agree}/{len(oracle_force)}"
        )
        print(
            f"opening |error| to oracle median {np.median(np.abs(opening_predicted - opening_best)):.4f}, "
            f"opening grid agreement {opening_agree}/20, "
            f"opening |error| to LQR median {np.median(np.abs(opening_predicted - opening_lqr)):.4f}, "
            f"opening |F| {np.median(np.abs(opening_predicted)):.4f}, "
            f"within 1 N of ±15 {near_rail}/20, "
            f"max |pre-tanh| {float(opening_means.abs().max()):.4f}"
        )
        print(
            "upright Jacobian "
            + " ".join(f"{value:.4f}" for value in jacobian)
            + f" angle slopes {jacobian[1]:.4f} {jacobian[2]:.4f}"
        )
        print(
            "nearest training state median distance "
            f"success {success_distance} fail {failure_distance}"
        )

    report(0)
    for step in range(1, 20001):
        loss = torch.nn.functional.mse_loss(scale * torch.tanh(policy(obs_t)[:, 0]), force_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            policy.layer3.weight[1].copy_(logstd_weight)
            policy.layer3.bias[1].copy_(logstd_bias)
        if step in {1000, 20000}:
            report(step)


def _reseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _scale_transition(row, center, spread):
    state = (row.state.detach().cpu().numpy() - center) / spread
    state_t = torch.tensor(state, dtype=torch.float32, device=device)
    if row.next_state is None:
        next_state = None
    else:
        nxt = (row.next_state.detach().cpu().numpy() - center) / spread
        next_state = torch.tensor(nxt, dtype=torch.float32, device=device)
    return Transition(state_t, row.action, next_state, row.reward)


def _critic_until_actor(rows):
    _reseed(0)
    learner = SAC_Learner()
    updates = 0
    for row in rows:
        learner.memory.push(*row)
        if len(learner.memory) > learner.start_steps:
            learner.optimize_model(do_policy_step=False)
            _polyak(learner)
            updates += 1
    for net in (learner.Q_net_1, learner.Q_net_2):
        for parameter in net.parameters():
            parameter.requires_grad_(False)
    return learner, updates


def _actor_only(learner):
    import math
    transitions = learner.memory.sample(learner.BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    state_batch = torch.cat(batch.state)
    means, log_stds = learner.policy_net(state_batch).chunk(2, dim=-1)
    log_stds = log_stds.clamp(-20, 2)
    dist = torch.distributions.Normal(means, torch.exp(log_stds))
    u = dist.rsample()
    action = learner.scale * torch.tanh(u)
    logprobs = (
        dist.log_prob(u)
        - torch.log(1.0 - torch.pow(torch.tanh(u), 2.0) + 1e-12)
        - math.log(learner.scale)
    ).squeeze(-1)
    q_in = torch.cat((state_batch, action / learner.scale), dim=-1)
    scores = torch.cat([
        net(q_in) for net in (learner.Q_net_1, learner.Q_net_2)
    ], dim=-1)
    selected = torch.min(scores, dim=-1).values
    loss = -(selected - learner.log_alpha.exp().detach() * logprobs).sum() / learner.BATCH_SIZE
    learner.policy_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(learner.policy_net.parameters(), 100)
    learner.policy_optimizer.step()
    alpha_loss = -(learner.log_alpha * (logprobs.detach() + learner.target_entropy)).mean()
    learner.alpha_optimizer.zero_grad()
    alpha_loss.backward()
    learner.alpha_optimizer.step()


def actor_update_scale(env, gain):
    _reseed(0)
    collector = SAC_Learner()
    rows = []
    episode = 0
    observation, _ = env.reset(seed=episode)
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    while len(rows) < 22000:
        action = collector.select_action(state)
        collector.steps_done += 1
        observation, reward, terminated, truncated, _ = env.step(float(action))
        rows.append(_transition(state, float(action), observation, reward, terminated))
        if terminated or truncated:
            episode += 1
            observation, _ = env.reset(seed=episode)
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            state = rows[-1].next_state
    raw = np.vstack([row.state.detach().cpu().numpy() for row in rows])
    forces = np.vstack([row.action.detach().cpu().numpy() for row in rows]).reshape(-1)
    center = raw.mean(axis=0)
    spread = np.where(raw.std(axis=0) < 1e-8, 1.0, raw.std(axis=0))
    print(
        f"replay {len(rows)} transitions, {episode + 1} episodes, "
        f"median |F| {np.median(np.abs(forces)):.4f}"
    )
    print("replay column stds " + " ".join(f"{value:.6f}" for value in raw.std(axis=0)))

    def run_arm(label, arm_rows, arm_center, arm_spread):
        learner, updates = _critic_until_actor(arm_rows)
        print(f"{label} critic updates {updates}")

        def transform(observation):
            return (np.asarray(observation, dtype=np.float64) - arm_center) / arm_spread

        def report(step):
            openings = []
            gradients = []
            toward = 0
            for seed in range(100, 120):
                observation, _ = env.reset(seed=seed)
                lqr = lqr_force(env, gain)
                state = torch.tensor(transform(observation), dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    pre_tanh = learner.policy_net(state)[0, 0]
                    predicted = float(learner.scale * torch.tanh(pre_tanh))
                action = torch.tensor(
                    [[predicted / learner.scale]], dtype=torch.float32, device=device, requires_grad=True,
                )
                scores = torch.cat([
                    net(torch.cat((state, action), dim=-1))
                    for net in (learner.Q_net_1, learner.Q_net_2)
                ], dim=-1)
                gradient = torch.autograd.grad(torch.min(scores), action)[0]
                force_gradient = float(gradient) / learner.scale
                openings.append((predicted, pre_tanh.detach(), lqr))
                gradients.append(force_gradient)
                toward += force_gradient * (lqr - predicted) > 0
            predicted = np.array([row[0] for row in openings])
            pre_tanh = torch.stack([row[1] for row in openings])
            lqr = np.array([row[2] for row in openings])
            place_state(env, np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0]))
            upright = transform(env.unwrapped._clip_obs(env.unwrapped._get_obs()))
            probe = torch.tensor(upright, dtype=torch.float32, device=device).unsqueeze(0)
            probe.requires_grad_(True)
            force = learner.scale * torch.tanh(learner.policy_net(probe)[0, 0])
            jacobian = torch.autograd.grad(force, probe)[0][0].detach().cpu().numpy() / arm_spread

            def policy_fn(env):
                observed = env.unwrapped._clip_obs(env.unwrapped._get_obs())
                state = torch.tensor(transform(observed), dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    return float(learner.scale * torch.tanh(learner.policy_net(state)[0, 0]))

            rolled = [rollout(env, seed, policy_fn) for seed in range(100, 120)]
            lengths = [row[1] for row in rolled]
            print(
                f"{label} after {step} actor updates success {sum(row[2] for row in rolled)}/20, "
                f"median decisions {np.median(lengths):.0f}, min decisions {np.min(lengths):.0f}, "
                f"opening |F| {np.median(np.abs(predicted)):.4f} LQR {np.median(np.abs(lqr)):.4f}, "
                f"max |pre-tanh| {float(pre_tanh.abs().max()):.4f}, "
                f"alpha {float(learner.log_alpha.detach().exp()):.4f}"
            )
            print(
                f"{label} Q force gradient median {np.median(gradients):.4f}, "
                f"median magnitude {np.median(np.abs(gradients)):.4f}, toward LQR {toward}/20"
            )
            print(
                f"{label} upright raw angle slopes {jacobian[1]:.4f} {jacobian[2]:.4f}"
            )

        report(0)
        for step in range(1, 20001):
            _actor_only(learner)
            if step in {1000, 20000}:
                report(step)

    run_arm("raw", rows, np.zeros(raw.shape[1]), np.ones(raw.shape[1]))
    run_arm(
        "standardized",
        [_scale_transition(row, center, spread) for row in rows],
        center,
        spread,
    )


def warmup_q_coverage(env, gain):
    _reseed(0)
    collector = SAC_Learner()
    rows = []
    physics = []
    episode = 0
    observation, _ = env.reset(seed=episode)
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    while len(rows) < 22000:
        physics.append(env.unwrapped.pendulum.x.copy())
        action = collector.select_action(state)
        collector.steps_done += 1
        observation, reward, terminated, truncated, _ = env.step(float(action))
        rows.append(_transition(state, float(action), observation, reward, terminated))
        if terminated or truncated:
            episode += 1
            observation, _ = env.reset(seed=episode)
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            state = rows[-1].next_state
    learner, updates = _critic_until_actor(rows)
    print(f"replay {len(rows)} transitions, {episode + 1} episodes, critic updates {updates}")
    errors = np.vstack([wrapped_error(state) for state in physics])
    tilt = np.maximum(np.abs(errors[:, 1]), np.abs(errors[:, 2]))
    near = tilt < 0.1
    reset_band = tilt < 0.01
    print(
        f"both poles within 0.1 rad {int(near.sum())}/{len(rows)}, "
        f"within 0.01 rad {int(reset_band.sum())}/{len(rows)}"
    )
    scale = float(env.action_space.high[0])
    limit = float(env.unwrapped.max_F)
    replay_force = np.clip(-errors @ gain, -limit, limit)
    replay_obs = np.vstack([row.state.detach().cpu().numpy() for row in rows])

    def values(observations, forces):
        state_t = torch.tensor(observations, dtype=torch.float32, device=device)
        action_t = torch.tensor(
            np.asarray(forces, dtype=np.float64) / scale, dtype=torch.float32, device=device,
        ).reshape(-1, 1)
        with torch.no_grad():
            scores = torch.cat([
                net(torch.cat((state_t, action_t), dim=-1))
                for net in (learner.Q_net_1, learner.Q_net_2)
            ], dim=-1)
        return torch.min(scores, dim=-1).values.cpu().numpy()

    def report(label, observations, forces):
        if len(observations) == 0:
            print(f"{label} 0 states")
            return
        q_lqr = values(observations, forces)
        q_zero = values(observations, np.zeros(len(observations)))
        q_neg = values(observations, np.full(len(observations), -25.0))
        q_pos = values(observations, np.full(len(observations), 25.0))
        gap = q_lqr - q_zero
        print(
            f"{label} {len(observations)} states, "
            f"median |F_LQR| {np.median(np.abs(forces)):.4f}, "
            f"prefer LQR {int(np.sum(gap > 0))}/{len(observations)}, "
            f"median gap {np.median(gap):.3f}, "
            f"median Q(-25) {np.median(q_neg):.3f}, "
            f"Q(0) {np.median(q_zero):.3f}, Q(+25) {np.median(q_pos):.3f}"
        )

    hold_obs, hold_force = [], []
    for seed in range(100, 120):
        observation, _ = env.reset(seed=seed)
        hold_obs.append(np.asarray(observation, dtype=np.float64))
        hold_force.append(lqr_force(env, gain))
    report("held-out resets", np.vstack(hold_obs), np.asarray(hold_force))
    report("near upright", replay_obs[near], replay_force[near])
    report("reset noise", replay_obs[reset_band], replay_force[reset_band])
    report("fallen", replay_obs[~near], replay_force[~near])


def _preserved_rng(function):
    saved = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    )
    try:
        return function()
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])
        if saved[3] is not None:
            torch.cuda.set_rng_state(saved[3])


def _warmup_scale(env, seed):
    _reseed(seed)
    learner = SAC_Learner()
    observations = []
    episode = 0
    observation, _ = env.reset(seed=episode)
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    while learner.steps_done < learner.start_steps:
        action = learner.select_action(state)
        next_observation, _, terminated, truncated, _ = env.step(float(action))
        learner.steps_done += 1
        observations.append(np.asarray(observation, dtype=np.float64))
        if terminated or truncated:
            episode += 1
            next_observation, _ = env.reset(seed=episode)
        observation = next_observation
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    raw = np.vstack(observations)
    center = raw.mean(axis=0)
    spread = np.where(raw.std(axis=0) < 1e-8, 1.0, raw.std(axis=0))
    return center, spread


def sac_prefix_ablation(env, gain):
    import math
    probe = init_env()
    checkpoints = {2000, 4000, 8000, 12000, 22000, 24000}
    angle_per_obs = 4.0 / np.pi

    def run_arm(seed, label, holdback, center, spread):
        _reseed(seed)
        learner = SAC_Learner()
        tilts = []
        episode = 0
        step_in_episode = 0
        observation, _ = env.reset(seed=episode)
        state = torch.tensor(
            (np.asarray(observation, dtype=np.float64) - center) / spread,
            dtype=torch.float32, device=device,
        ).unsqueeze(0)

        def transform(observed):
            return (np.asarray(observed, dtype=np.float64) - center) / spread

        def report(step):
            def measure():
                observations, forces = [], []
                for reset_seed in range(100, 120):
                    observed, _ = probe.reset(seed=reset_seed)
                    observations.append(transform(observed))
                    forces.append(lqr_force(probe, gain))
                observations = np.vstack(observations)
                forces = np.asarray(forces)
                state_t = torch.tensor(observations, dtype=torch.float32, device=device)
                with torch.no_grad():
                    pre_tanh = learner.policy_net(state_t)[:, 0]
                    predicted = (learner.scale * torch.tanh(pre_tanh)).cpu().numpy()
                    log_std = learner.policy_net(state_t)[:, 1].clamp(-20, 2)
                    dist = torch.distributions.Normal(pre_tanh, log_std.exp())
                    sampled = dist.sample()
                    log_prob = (
                        dist.log_prob(sampled)
                        - torch.log(1.0 - torch.tanh(sampled).pow(2) + 1e-12)
                        - math.log(learner.scale)
                    )
                    entropy = float((-log_prob).mean())

                def values(chosen):
                    action_t = torch.tensor(
                        np.asarray(chosen, dtype=np.float64) / learner.scale,
                        dtype=torch.float32, device=device,
                    ).reshape(-1, 1)
                    with torch.no_grad():
                        scores = torch.cat([
                            net(torch.cat((state_t, action_t), dim=-1))
                            for net in (learner.Q_net_1, learner.Q_net_2)
                        ], dim=-1)
                    return torch.min(scores, dim=-1).values.cpu().numpy()

                gap = values(forces) - values(np.zeros(len(forces)))
                action_t = torch.tensor(
                    predicted / learner.scale, dtype=torch.float32, device=device,
                ).reshape(-1, 1)
                action_t.requires_grad_(True)
                scores = torch.cat([
                    net(torch.cat((state_t, action_t), dim=-1))
                    for net in (learner.Q_net_1, learner.Q_net_2)
                ], dim=-1)
                gradient = (
                    torch.autograd.grad(torch.min(scores, dim=-1).values.sum(), action_t)[0]
                    .detach().cpu().numpy().reshape(-1) / learner.scale
                )
                place_state(probe, np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0]))
                upright = torch.tensor(
                    transform(probe.unwrapped._clip_obs(probe.unwrapped._get_obs())),
                    dtype=torch.float32, device=device,
                ).unsqueeze(0)
                upright.requires_grad_(True)
                upright_force = learner.scale * torch.tanh(learner.policy_net(upright)[0, 0])
                jacobian = torch.autograd.grad(upright_force, upright)[0][0].detach().cpu().numpy()
                slopes = jacobian / spread * angle_per_obs

                def policy_fn(env):
                    observed = env.unwrapped._clip_obs(env.unwrapped._get_obs())
                    current = torch.tensor(transform(observed), dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        return float(learner.scale * torch.tanh(learner.policy_net(current)[0, 0]))

                rolled = [rollout(probe, reset_seed, policy_fn) for reset_seed in range(100, 120)]
                lengths = [row[1] for row in rolled]
                tilt = np.asarray(tilts)
                print(
                    f"seed {seed} {label} step {step} success {sum(row[2] for row in rolled)}/20, "
                    f"median decisions {np.median(lengths):.0f}, min decisions {np.min(lengths):.0f}, "
                    f"opening |F| median {np.median(np.abs(predicted)):.4f} "
                    f"max {np.max(np.abs(predicted)):.4f}, "
                    f"|F|>15 {int(np.sum(np.abs(predicted) > 15))}/20, "
                    f"angle slopes {slopes[1]:.4f} {slopes[2]:.4f}",
                    flush=True,
                )
                print(
                    f"seed {seed} {label} step {step} "
                    f"Q gap prefer LQR {int(np.sum(gap > 0))}/20 median {np.median(gap):.3f}, "
                    f"gradient toward LQR {int(np.sum(gradient * (forces - predicted) > 0))}/20 "
                    f"median magnitude {np.median(np.abs(gradient)):.4f}, "
                    f"replay within 0.01 {int(np.sum(tilt < 0.01))}/{len(tilt)} "
                    f"within 0.1 {int(np.sum(tilt < 0.1))}/{len(tilt)}, "
                    f"alpha {float(learner.log_alpha.detach().exp()):.4f}, entropy {entropy:.4f}",
                    flush=True,
                )

            _preserved_rng(measure)

        while learner.steps_done < 24000:
            error = wrapped_error(env.unwrapped.pendulum.x)
            tilts.append(max(abs(error[1]), abs(error[2])))
            action = learner.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(float(action))
            learner.steps_done += 1
            reward_t = torch.tensor([float(reward)], device=device)
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    transform(observation), dtype=torch.float32, device=device,
                ).unsqueeze(0)
            learner.memory.push(state, action, next_state, reward_t)
            if len(learner.memory) > learner.start_steps:
                actor_ready = learner.steps_done > learner.start_steps + holdback
                learner.optimize_model(
                    do_policy_step=actor_ready and (step_in_episode % learner.policy_delay == 0)
                )
                _polyak(learner)
            if learner.steps_done in checkpoints:
                report(learner.steps_done)
            if terminated or truncated:
                episode += 1
                observation, _ = env.reset(seed=episode)
                state = torch.tensor(transform(observation), dtype=torch.float32, device=device).unsqueeze(0)
                step_in_episode = 0
            else:
                state = next_state
                step_in_episode += 1

    for seed in (0, 1, 2):
        center, spread = _warmup_scale(env, seed)
        print(
            f"seed {seed} warmup scale stds " + " ".join(f"{value:.6f}" for value in spread),
            flush=True,
        )
        run_arm(seed, "raw delayed", 20000, np.zeros(7), np.ones(7))
        run_arm(seed, "raw immediate", 0, np.zeros(7), np.ones(7))
        run_arm(seed, "standardized delayed", 20000, center, spread)
        run_arm(seed, "standardized immediate", 0, center, spread)


def reset_action_contrast(env):
    _reseed(0)
    collector = SAC_Learner()
    forces = []
    tilts = []
    episode = 0
    observation, _ = env.reset(seed=episode)
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    while len(forces) < 22000:
        error = wrapped_error(env.unwrapped.pendulum.x)
        tilts.append(max(abs(error[1]), abs(error[2])))
        action = collector.select_action(state)
        forces.append(float(action))
        collector.steps_done += 1
        observation, _, terminated, truncated, _ = env.step(float(action))
        if terminated or truncated:
            episode += 1
            observation, _ = env.reset(seed=episode)
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    forces = np.asarray(forces)
    near = np.asarray(tilts) < 0.01
    print(f"replay {len(forces)} transitions, {episode + 1} episodes")

    def report(label, chosen):
        if len(chosen) == 0:
            print(f"{label} 0 states")
            return
        print(
            f"{label} {len(chosen)} states, median force {np.median(chosen):.4f}, "
            f"10th {np.percentile(chosen, 10):.4f}, 90th {np.percentile(chosen, 90):.4f}, "
            f"below -15 {int(np.sum(chosen < -15))}/{len(chosen)}, "
            f"below -5 {int(np.sum(chosen < -5))}/{len(chosen)}, "
            f"above 5 {int(np.sum(chosen > 5))}/{len(chosen)}, "
            f"above 15 {int(np.sum(chosen > 15))}/{len(chosen)}"
        )

    report("warmup reset band", forces[:2000][near[:2000]])
    report("holdback reset band", forces[2000:][near[2000:]])


def _force_from_physics(physics, gain, limit):
    error = wrapped_error(physics)
    return float(np.clip(-error @ gain, -limit, limit))


def _bellman_labels(seed, gain):
    _limit_blas()
    env = init_env()
    limit = float(env.unwrapped.max_F)
    gamma = 0.99
    observation, _ = env.reset(seed=int(seed))
    observations, physics, forces = [], [], []
    done = False
    while not done:
        observations.append(np.asarray(observation, dtype=np.float64))
        physics.append(env.unwrapped.pendulum.x.copy())
        force = lqr_force(env, gain)
        forces.append(force)
        observation, _, terminated, truncated, _ = env.step(force)
        done = terminated or truncated
    grid = np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0])
    selected_obs, selected_actions = [], []
    rewards, terminals, truncated_flags = [], [], []
    monte_carlo, continuation = [], []
    for index in _decision_indices(len(observations)):
        actions = np.concatenate([grid, [forces[index]]])
        row_reward, row_terminal, row_truncated = [], [], []
        row_return, row_value = [], []
        for action in actions:
            place_state(env, physics[index])
            _, reward, terminated, truncated, _ = env.step(float(action))
            successor = env.unwrapped.pendulum.x.copy()
            if terminated:
                value = 0.0
            else:
                value = _discounted_continuation(
                    env, successor, _force_from_physics(successor, gain, limit), gain, gamma,
                )
            row_reward.append(float(reward))
            row_terminal.append(bool(terminated))
            row_truncated.append(bool(truncated))
            row_value.append(value)
            row_return.append(_discounted_continuation(env, physics[index], float(action), gain, gamma))
        selected_obs.append(observations[index])
        selected_actions.append(actions)
        rewards.append(row_reward)
        terminals.append(row_terminal)
        truncated_flags.append(row_truncated)
        monte_carlo.append(row_return)
        continuation.append(row_value)
    return (
        np.vstack(selected_obs),
        np.vstack(selected_actions),
        np.vstack(rewards),
        np.vstack(terminals),
        np.vstack(truncated_flags),
        np.vstack(monte_carlo),
        np.vstack(continuation),
    )


def bellman_equivalence(env, gain):
    rows = _fork_map(_bellman_labels, [(seed, gain) for seed in range(20)])
    observations = np.vstack([row[0] for row in rows])
    actions = np.vstack([row[1] for row in rows])
    rewards = np.vstack([row[2] for row in rows])
    terminals = np.vstack([row[3] for row in rows]).astype(bool)
    truncated = np.vstack([row[4] for row in rows]).astype(bool)
    monte_carlo = np.vstack([row[5] for row in rows])
    continuation = np.vstack([row[6] for row in rows])
    gamma = 0.99
    bellman = rewards + gamma * np.where(terminals, 0.0, continuation)
    gap = monte_carlo - bellman
    print(
        f"labeled states {len(observations)}, actions {actions.shape[1]}, "
        f"terminals {int(terminals.sum())}, truncated {int(truncated.sum())}"
    )

    def difference(label, chosen):
        if not np.any(chosen):
            print(f"{label} 0 samples")
            return
        values = gap[chosen]
        print(
            f"{label} max |G-Y| {np.max(np.abs(values)):.6f}, "
            f"RMSE {np.sqrt(np.mean(values ** 2)):.6f}"
        )

    difference("all", np.ones(gap.shape, dtype=bool))
    difference("terminal", terminals)
    difference("nonterminal", ~terminals)

    scale = float(env.action_space.high[0])
    paired_obs = np.repeat(observations, actions.shape[1], axis=0)
    paired_force = actions.reshape(-1)
    monte_carlo_t = torch.tensor(monte_carlo.reshape(-1), dtype=torch.float32, device=device)
    reward_t = torch.tensor(rewards.reshape(-1), dtype=torch.float32, device=device)
    value_t = torch.tensor(continuation.reshape(-1), dtype=torch.float32, device=device)
    terminal_t = torch.tensor(terminals.reshape(-1), dtype=torch.bool, device=device)
    bootstrap = torch.zeros_like(reward_t)
    alive = ~terminal_t
    bootstrap[alive] = gamma * value_t[alive]
    assembled = reward_t + bootstrap
    inputs = torch.tensor(
        np.column_stack((paired_obs, paired_force / scale)), dtype=torch.float32, device=device,
    )
    print(
        f"assembled versus G max |difference| {float((assembled - monte_carlo_t).abs().max()):.6f}"
    )

    def new_critics():
        torch.manual_seed(0)
        first = SAC_Q_DP(observations.shape[1]).to(device)
        second = SAC_Q_DP(observations.shape[1]).to(device)
        return first, second

    monte_carlo_nets = new_critics()
    bellman_nets = new_critics()
    for source, destination in zip(monte_carlo_nets, bellman_nets):
        destination.load_state_dict(source.state_dict())

    def prediction(nets):
        return [net(inputs).squeeze(-1) for net in nets]

    def loss_on(nets, target):
        return sum(torch.nn.functional.mse_loss(value, target) for value in prediction(nets))

    monte_carlo_loss = loss_on(monte_carlo_nets, monte_carlo_t)
    bellman_loss = loss_on(bellman_nets, assembled)
    monte_carlo_loss.backward()
    bellman_loss.backward()

    def flat_grad(nets):
        return torch.cat([parameter.grad.detach().reshape(-1) for net in nets for parameter in net.parameters()])

    monte_carlo_grad = flat_grad(monte_carlo_nets)
    bellman_grad = flat_grad(bellman_nets)
    cosine = torch.nn.functional.cosine_similarity(monte_carlo_grad, bellman_grad, dim=0)
    monte_carlo_norm = float(monte_carlo_grad.norm())
    bellman_norm = float(bellman_grad.norm())
    print(
        f"initial loss Monte Carlo {float(monte_carlo_loss.detach()):.6f}, "
        f"Bellman {float(bellman_loss.detach()):.6f}, gradient cosine {float(cosine):.6f}, "
        f"relative norm difference {abs(monte_carlo_norm - bellman_norm) / max(monte_carlo_norm, bellman_norm):.6e}"
    )
    parameter_gap = 0.0
    for left, right in zip(monte_carlo_nets, bellman_nets):
        for left_parameter, right_parameter in zip(left.parameters(), right.parameters()):
            parameter_gap = max(
                parameter_gap,
                float((left_parameter.grad - right_parameter.grad).abs().max()),
            )
    print(f"max per-parameter gradient difference {parameter_gap:.6e}")
    for nets in (monte_carlo_nets, bellman_nets):
        for net in nets:
            net.zero_grad()
    monte_carlo_optimizer = optim.AdamW(
        [parameter for net in monte_carlo_nets for parameter in net.parameters()],
        lr=3e-4, amsgrad=True,
    )
    bellman_optimizer = optim.AdamW(
        [parameter for net in bellman_nets for parameter in net.parameters()],
        lr=3e-4, amsgrad=True,
    )
    valid_rows = _fork_map(_opening_labels, [(seed, gain) for seed in range(100, 120)])
    valid_obs = np.vstack([row[0] for row in valid_rows])
    valid_actions = np.vstack([row[1] for row in valid_rows])
    valid_returns = np.vstack([row[2] for row in valid_rows])
    valid_inputs = torch.tensor(
        np.column_stack((
            np.repeat(valid_obs, valid_actions.shape[1], axis=0),
            valid_actions.reshape(-1) / scale,
        )),
        dtype=torch.float32, device=device,
    )

    def scores(nets, batch):
        with torch.no_grad():
            values = torch.stack([net(batch).squeeze(-1) for net in nets])
        return values.mean(dim=0).cpu().numpy()

    def centered_rmse(predicted, target):
        predicted = predicted.reshape(-1, actions.shape[1])
        target = target.reshape(-1, actions.shape[1])
        predicted = predicted - predicted.mean(axis=1, keepdims=True)
        target = target - target.mean(axis=1, keepdims=True)
        return float(np.sqrt(np.mean((predicted - target) ** 2)))

    def report(step):
        print(f"after {step} updates max parameter difference {max_parameter_difference():.6e}")
        for label, nets, target in (
            ("Monte Carlo", monte_carlo_nets, monte_carlo),
            ("Bellman", bellman_nets, bellman),
        ):
            predicted = scores(nets, inputs).reshape(monte_carlo.shape)
            oracle = np.argmax(monte_carlo, axis=1)
            choice = np.argmax(predicted, axis=1)
            gap_error = np.abs(
                (predicted[:, 4] - predicted[:, 2]) - (monte_carlo[:, 4] - monte_carlo[:, 2])
            )
            valid_predicted = scores(nets, valid_inputs).reshape(valid_returns.shape)
            valid_oracle = np.argmax(valid_returns, axis=1)
            valid_choice = np.argmax(valid_predicted, axis=1)
            valid_force = valid_actions[np.arange(len(valid_actions)), valid_choice]
            print(
                f"{label} RMSE {np.sqrt(np.mean((predicted - target) ** 2)):.4f}, "
                f"centered RMSE {centered_rmse(predicted, monte_carlo):.4f}, "
                f"train argmax {int(np.sum(choice == oracle))}/{len(oracle)}, "
                f"median |+5 minus -5 error| {np.median(gap_error):.4f}, "
                f"validation argmax {int(np.sum(valid_choice == valid_oracle))}/{len(valid_oracle)}, "
                f"validation boundary {int(np.sum(np.abs(np.abs(valid_force) - 15.0) < 1e-6))}/{len(valid_oracle)}"
            )

    def max_parameter_difference():
        difference = 0.0
        for left, right in zip(monte_carlo_nets, bellman_nets):
            for left_parameter, right_parameter in zip(left.parameters(), right.parameters()):
                difference = max(difference, float((left_parameter - right_parameter).abs().max()))
        return difference

    for step in range(1, 20001):
        for nets, target, optimizer in (
            (monte_carlo_nets, monte_carlo_t, monte_carlo_optimizer),
            (bellman_nets, assembled, bellman_optimizer),
        ):
            loss = loss_on(nets, target)
            optimizer.zero_grad()
            loss.backward()
            for net in nets:
                torch.nn.utils.clip_grad_value_(net.parameters(), 100)
            optimizer.step()
        if step in {1, 100, 1000, 20000}:
            report(step)


def _behavior_chunk(index, weights, scale, probes):
    _limit_blas()
    torch.set_num_threads(1)
    env = init_env()
    policy = SAC_POLICY_DP(7)
    policy.load_state_dict(weights)
    policy.eval()
    gamma = 0.99
    rollouts = 24

    def prepare(physics, previous_force):
        place_state(env, physics)
        env.unwrapped._agent_location = float(previous_force)
        env.unwrapped.state = env.unwrapped._get_state()

    def sample_force():
        observed = env.unwrapped._clip_obs(env.unwrapped._get_obs())
        state = torch.tensor(np.asarray(observed), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = policy(state).chunk(2, dim=-1)
            log_std = log_std.clamp(-20, 2)
            dist = torch.distributions.Normal(mean, log_std.exp())
            sampled = dist.sample()
            return float(torch.clamp(scale * torch.tanh(sampled), -scale, scale))

    def one_return(physics, previous_force, first_force, seed):
        torch.manual_seed(seed)
        prepare(physics, previous_force)
        total = 0.0
        discount = 1.0
        done = False
        force = first_force
        while not done:
            if force is None:
                force = sample_force()
            _, reward, terminated, truncated, _ = env.step(float(force))
            total += discount * float(reward)
            discount *= gamma
            done = terminated or truncated
            force = None
        return total

    results = []
    for probe in probes:
        physics, previous_force, forces, seed = probe
        if forces is None:
            results.append(np.asarray([
                one_return(physics, previous_force, None, seed + rollout)
                for rollout in range(rollouts)
            ]))
        else:
            results.append(np.asarray([
                [one_return(physics, previous_force, float(force), seed + force_index * rollouts + rollout)
                 for rollout in range(rollouts)]
                for force_index, force in enumerate(forces)
            ]))
    return results


def behavior_backup_audit(env, gain):
    import math
    _reseed(0)
    learner = SAC_Learner()
    records = []
    episode = 0
    observation, _ = env.reset(seed=episode)
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    while learner.steps_done < 22000:
        physics = env.unwrapped.pendulum.x.copy()
        error = wrapped_error(physics)
        action = learner.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(float(action))
        learner.steps_done += 1
        reward_t = torch.tensor([float(reward)], device=device)
        next_state = None if terminated else torch.tensor(
            observation, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        learner.memory.push(state, action, next_state, reward_t)
        records.append({
            "physics": physics,
            "next_physics": env.unwrapped.pendulum.x.copy(),
            "observation": np.asarray(state.detach().cpu().numpy(), dtype=np.float64).reshape(-1),
            "next_observation": None if terminated else np.asarray(observation, dtype=np.float64),
            "action": float(action),
            "reward": float(reward),
            "terminated": bool(terminated),
            "tilt": max(abs(error[1]), abs(error[2])),
            "episode": episode,
        })
        if len(learner.memory) > learner.start_steps:
            learner.optimize_model(do_policy_step=False)
            _polyak(learner)
        if terminated or truncated:
            episode += 1
            if 100 <= episode < 120:
                episode = 120
            observation, _ = env.reset(seed=episode)
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            state = next_state
    episodes = {record["episode"] for record in records}
    print(
        f"replay {len(records)} transitions, {len(episodes)} episodes, "
        f"held-out seeds present {len(episodes & set(range(100, 120)))}",
        flush=True,
    )
    scale = learner.scale
    weights = {key: value.detach().cpu() for key, value in learner.policy_net.state_dict().items()}
    grid = np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0])
    limit = float(env.unwrapped.max_F)
    rng = np.random.default_rng(0)
    tilts = np.asarray([record["tilt"] for record in records])
    inside = rng.choice(np.flatnonzero(tilts < 0.01), 100, replace=False)
    outside = rng.choice(np.flatnonzero(tilts >= 0.01), 100, replace=False)
    probes = []
    for seed in range(100, 120):
        observation, _ = env.reset(seed=seed)
        probes.append((
            env.unwrapped.pendulum.x.copy(),
            0.0,
            np.concatenate([grid, [_force_from_physics(env.unwrapped.pendulum.x, gain, limit)]]),
            10_000 + seed,
            np.asarray(observation, dtype=np.float64),
        ))
    for index in np.concatenate([inside, outside]):
        record = records[int(index)]
        probes.append((
            record["physics"],
            float(record["observation"][6] * limit),
            np.concatenate([grid, [_force_from_physics(record["physics"], gain, limit)]]),
            20_000 + int(index),
            record["observation"],
        ))
    backup_inside = rng.choice(np.flatnonzero(tilts < 0.01), 128, replace=False)
    backup_outside = rng.choice(np.flatnonzero(tilts >= 0.01), 128, replace=False)
    backup_index = np.concatenate([backup_inside, backup_outside])
    for offset, index in enumerate(backup_index):
        record = records[int(index)]
        if record["terminated"]:
            continue
        probes.append((
            record["next_physics"],
            record["action"],
            None,
            30_000 + offset,
            None,
        ))
    chunks = np.array_split(np.arange(len(probes)), 16)
    jobs = []
    for chunk_index, chunk in enumerate(chunks):
        jobs.append((
            chunk_index,
            weights,
            scale,
            [(probes[int(index)][0], probes[int(index)][1], probes[int(index)][2], probes[int(index)][3])
             for index in chunk],
        ))
    rolled = _fork_map(_behavior_chunk, jobs, message="rollout chunk")
    returns = [value for chunk in rolled for value in chunk]
    probe_count = 20 + len(inside) + len(outside)

    def summarize(label, chosen, observations):
        ranges, errors, separations, resolvable, live_match, monte_carlo_boundary, live_boundary = [], [], [], 0, 0, 0, 0
        centered_error, gap_error, best_force, lqr_force = [], [], [], []
        for local, observation in zip(chosen, observations):
            samples = returns[local]
            means = samples.mean(axis=1)
            errors_row = samples.std(axis=1, ddof=1) / np.sqrt(samples.shape[1])
            best = int(np.argmax(means))
            spread = float(means.max() - means.min())
            threshold = 3.0 * np.sqrt(errors_row[best] ** 2 + errors_row[np.argmin(means)] ** 2)
            ranked = np.argsort(means)
            separation = float(means[ranked[-1]] - means[ranked[-2]])
            separation_error = np.sqrt(errors_row[ranked[-1]] ** 2 + errors_row[ranked[-2]] ** 2)
            ranges.append(spread)
            errors.append(float(np.median(errors_row)))
            separations.append(separation / separation_error if separation_error else np.inf)
            resolvable += spread > threshold
            values = _paired_q(learner, np.repeat(observation[None, :], len(means), axis=0), probes[local][2])
            centered_error.append(np.mean(((values - values.mean()) - (means - means.mean())) ** 2))
            gap_error.append(abs((values.max() - values.min()) - spread))
            live_match += int(np.argmax(values) == best)
            forces = probes[local][2]
            live_boundary += abs(abs(forces[np.argmax(values)]) - 15.0) < 1e-6
            monte_carlo_boundary += abs(abs(forces[best]) - 15.0) < 1e-6
            best_force.append(abs(forces[best]))
            lqr_force.append(abs(forces[-1]))
        print(
            f"{label} {len(chosen)} states, median Monte Carlo range {np.median(ranges):.3f}, "
            f"median standard error {np.median(errors):.3f}, "
            f"resolvable {resolvable}/{len(chosen)}, "
            f"live argmax {live_match}/{len(chosen)}, "
            f"centered RMSE {np.sqrt(np.mean(centered_error)):.3f}, "
            f"median |range error| {np.median(gap_error):.3f}, "
            f"live boundary {live_boundary}/{len(chosen)}, "
            f"Monte Carlo boundary {monte_carlo_boundary}/{len(chosen)}, "
            f"median best |F| {np.median(best_force):.3f}, median |F_LQR| {np.median(lqr_force):.3f}, "
            f"median best-second / standard error {np.median(separations):.3f}",
            flush=True,
        )

    summarize("validation", range(20), [probe[4] for probe in probes[:20]])
    summarize("reset band", range(20, 120), [probe[4] for probe in probes[20:120]])
    summarize("outside band", range(120, probe_count), [probe[4] for probe in probes[120:probe_count]])

    live_targets, reward_targets, entropy_terms, monte_carlo_means, monte_carlo_errors = [], [], [], [], []
    successor_cursor = probe_count
    for index in backup_index:
        record = records[int(index)]
        reward = record["reward"]
        if record["terminated"]:
            live_targets.append(reward)
            reward_targets.append(reward)
            entropy_terms.append(0.0)
            monte_carlo_means.append(0.0)
            monte_carlo_errors.append(0.0)
            continue
        samples = returns[successor_cursor]
        successor_cursor += 1
        monte_carlo_mean = float(samples.mean())
        monte_carlo_error = float(samples.std(ddof=1) / np.sqrt(len(samples)))
        next_state = torch.tensor(record["next_observation"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = learner.policy_net(next_state).chunk(2, dim=-1)
            log_std = log_std.clamp(-20, 2)
            dist = torch.distributions.Normal(mean, log_std.exp())
            sampled = dist.sample()
            next_action = torch.clamp(scale * torch.tanh(sampled), -scale, scale)
            log_prob = (
                dist.log_prob(sampled)
                - torch.log(1.0 - torch.tanh(sampled).pow(2) + 1e-12)
                - math.log(scale)
            ).squeeze()
            target_input = torch.cat((next_state, next_action / scale), dim=-1)
            scores = torch.cat([
                net(target_input) for net in (learner.Q_target_1, learner.Q_target_2)
            ], dim=-1)
            bootstrap = learner.GAMMA * (
                torch.min(scores, dim=-1).values - learner.log_alpha.exp() * log_prob
            )
        live_targets.append(reward + float(bootstrap))
        reward_targets.append(reward + learner.GAMMA * monte_carlo_mean)
        entropy_terms.append(float(
            learner.GAMMA * learner.log_alpha.detach().exp() * log_prob.abs()
        ))
        monte_carlo_means.append(monte_carlo_mean)
        monte_carlo_errors.append(monte_carlo_error)
    live_targets = np.asarray(live_targets)
    reward_targets = np.asarray(reward_targets)
    entropy_terms = np.asarray(entropy_terms)
    monte_carlo_errors = np.asarray(monte_carlo_errors)
    raw_gap = np.abs(live_targets - reward_targets)
    corrected_gap = np.abs(live_targets - (reward_targets - np.sign(live_targets) * 0.0) )
    # The one-step entropy allowance is gamma*alpha*|log pi|; future entropy remains in the soft target.
    allowance = entropy_terms + 3.0 * monte_carlo_errors
    print(
        f"backup {len(live_targets)} transitions, "
        f"median |live target - reward backup| {np.median(raw_gap):.3f}, "
        f"median gamma*alpha*|log pi| {np.median(entropy_terms):.3f}, "
        f"median Monte Carlo standard error {np.median(monte_carlo_errors):.3f}, "
        f"beyond one-step entropy allowance {int(np.sum(raw_gap > allowance))}/{len(live_targets)}",
        flush=True,
    )
    del corrected_gap

    def rows_for(locals_):
        observations, forces, targets = [], [], []
        for local in locals_:
            for force, target in zip(probes[local][2], returns[local].mean(axis=1)):
                observations.append(probes[local][4])
                forces.append(force / scale)
                targets.append(target)
        return (
            np.column_stack((np.vstack(observations), np.asarray(forces))),
            np.asarray(targets, dtype=np.float64),
        )

    train_input, train_target = rows_for(range(20, probe_count))
    valid_input, valid_target = rows_for(range(20))
    train_target = train_target.reshape(-1, 8)
    valid_target = valid_target.reshape(-1, 8)
    center = train_input.mean(axis=0)
    spread = np.maximum(train_input.std(axis=0), 1e-8)
    target_scale = max(float(train_target.std()), 1e-8)
    train_standard = (train_input - center) / spread
    valid_standard = (valid_input - center) / spread

    def report_fit(label, train_pred, valid_pred):
        def centered(pred, truth):
            delta = (pred - pred.mean(axis=1, keepdims=True)) - (truth - truth.mean(axis=1, keepdims=True))
            return float(np.sqrt(np.mean(delta ** 2)))

        def matches(pred, truth):
            return int(np.sum(np.argmax(pred, axis=1) == np.argmax(truth, axis=1)))

        range_error = np.abs(
            (valid_pred.max(axis=1) - valid_pred.min(axis=1))
            - (valid_target.max(axis=1) - valid_target.min(axis=1))
        )
        print(
            f"{label} train RMSE {np.sqrt(np.mean((train_pred - train_target) ** 2)):.3f}, "
            f"train centered RMSE {centered(train_pred, train_target):.3f}, "
            f"train argmax {matches(train_pred, train_target)}/{len(train_target)}, "
            f"validation argmax {matches(valid_pred, valid_target)}/{len(valid_target)}, "
            f"validation centered RMSE {centered(valid_pred, valid_target):.3f}, "
            f"median validation |range error| {np.median(range_error):.3f}",
            flush=True,
        )

    def train_twin(features, targets, standardized):
        feature_t = torch.tensor(features, dtype=torch.float32, device=device)
        target_t = torch.tensor(targets.reshape(-1) / (target_scale if standardized else 1.0), dtype=torch.float32, device=device)
        torch.manual_seed(0)
        nets = [SAC_Q_DP(7).to(device) for _ in range(2)]
        optimizer = optim.AdamW(
            [parameter for net in nets for parameter in net.parameters()],
            lr=3e-4, amsgrad=True,
        )
        for update in range(20000):
            loss = sum(torch.nn.functional.mse_loss(net(feature_t).squeeze(-1), target_t) for net in nets)
            optimizer.zero_grad()
            loss.backward()
            if update in (0, 100, 1000, 5000, 19999):
                gradients = [parameter.grad.detach() for net in nets for parameter in net.parameters()]
                flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
                changed = int(torch.count_nonzero(flat.abs() > 100))
                print(
                    f"{'standardized' if standardized else 'raw'} network before step {update} "
                    f"loss {float(loss.detach()):.6f}, gradient norm {float(torch.linalg.vector_norm(flat)):.6f}, "
                    f"clipped elements {changed}/{flat.numel()}",
                    flush=True,
                )
            for net in nets:
                torch.nn.utils.clip_grad_value_(net.parameters(), 100)
            optimizer.step()
        with torch.no_grad():
            final_loss = sum(torch.nn.functional.mse_loss(net(feature_t).squeeze(-1), target_t) for net in nets)
        print(
            f"{'standardized' if standardized else 'raw'} network step 20000 "
            f"loss {float(final_loss):.6f}",
            flush=True,
        )

        def predict(features_):
            values = torch.tensor(features_, dtype=torch.float32, device=device)
            with torch.no_grad():
                heads = torch.stack([net(values).squeeze(-1) for net in nets])
            scale_back = target_scale if standardized else 1.0
            return (heads.mean(dim=0).cpu().numpy() * scale_back).reshape(-1, 8), (
                heads.min(dim=0).values.cpu().numpy() * scale_back
            ).reshape(-1, 8)

        return predict(features), predict(valid_standard if standardized else valid_input)

    raw_train, raw_valid = train_twin(train_input, train_target, standardized=False)
    report_fit("raw network mean head", raw_train[0], raw_valid[0])
    report_fit("raw network min head", raw_train[1], raw_valid[1])
    standard_train, standard_valid = train_twin(train_standard, train_target, standardized=True)
    report_fit("standardized network mean head", standard_train[0], standard_valid[0])
    report_fit("standardized network min head", standard_train[1], standard_valid[1])
    design = np.column_stack((train_standard, np.ones(len(train_standard))))
    solution, _, _, _ = np.linalg.lstsq(design, train_target.reshape(-1), rcond=None)
    valid_design = np.column_stack((valid_standard, np.ones(len(valid_standard))))
    report_fit(
        "standardized linear",
        (design @ solution).reshape(-1, 8),
        (valid_design @ solution).reshape(-1, 8),
    )


def _paired_q(learner, observations, forces):
    state = torch.tensor(observations, dtype=torch.float32, device=device)
    action = torch.tensor(
        np.asarray(forces, dtype=np.float64) / learner.scale,
        dtype=torch.float32, device=device,
    ).reshape(-1, 1)
    with torch.no_grad():
        scores = torch.cat([
            net(torch.cat((state, action), dim=-1))
            for net in (learner.Q_net_1, learner.Q_net_2)
        ], dim=-1)
    return torch.min(scores, dim=-1).values.cpu().numpy()


def scaled_sac(env, gain, freeze_alpha=False):
    observations = []
    for seed in range(20):
        observation, _ = env.reset(seed=seed)
        done = False
        while not done:
            observations.append(np.asarray(observation, dtype=np.float64))
            observation, _, terminated, truncated, _ = env.step(lqr_force(env, gain))
            done = terminated or truncated
    columns = np.vstack(observations)
    center = torch.tensor(columns.mean(axis=0), dtype=torch.float32, device=device)
    spread = torch.tensor(np.maximum(columns.std(axis=0), 1e-8), dtype=torch.float32, device=device)
    print(
        "observation center " + " ".join(f"{value:.6f}" for value in center.tolist()),
        flush=True,
    )
    print(
        "observation spread " + " ".join(f"{value:.6f}" for value in spread.tolist()),
        flush=True,
    )

    def build():
        return SAC_Learner(
            obs_center=center, obs_spread=spread, reward_scale=100.0, freeze_alpha=freeze_alpha,
        )

    label = "fixed-alpha SAC" if freeze_alpha else "scaled SAC"
    library_outcome(label, build, trace=True, gain=gain)


def _polyak(learner):
    for net, target in ((learner.Q_net_1, learner.Q_target_1), (learner.Q_net_2, learner.Q_target_2)):
        net_state = net.state_dict()
        target_state = target.state_dict()
        for key in net_state:
            target_state[key] = target_state[key] * learner.TAU + net_state[key] * (1 - learner.TAU)
        target.load_state_dict(target_state)


def _module_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _load_module(module, state):
    module.load_state_dict({key: value.to(device) for key, value in state.items()})


def _capture_learner(learner):
    count = len(learner.memory)
    states = torch.empty(count, 7)
    actions = torch.empty(count, 1)
    nexts = torch.empty(count, 7)
    rewards = torch.empty(count)
    terminal = torch.empty(count, dtype=torch.bool)
    for index, row in enumerate(learner.memory.memory):
        states[index] = row.state.detach().reshape(-1).cpu()
        actions[index] = row.action.detach().reshape(-1).cpu()
        rewards[index] = row.reward.detach().reshape(-1).cpu().squeeze(0)
        terminal[index] = row.next_state is None
        if row.next_state is not None:
            nexts[index] = row.next_state.detach().reshape(-1).cpu()
    return {
        "states": states,
        "actions": actions,
        "nexts": nexts,
        "rewards": rewards,
        "terminal": terminal,
        "policy": _module_state(learner.policy_net),
        "q1": _module_state(learner.Q_net_1),
        "q2": _module_state(learner.Q_net_2),
        "target1": _module_state(learner.Q_target_1),
        "target2": _module_state(learner.Q_target_2),
        "q_opt": copy.deepcopy(learner.Q_optimizer.state_dict()),
        "policy_opt": copy.deepcopy(learner.policy_optimizer.state_dict()),
        "log_alpha": learner.log_alpha.detach().cpu().clone(),
        "steps_done": learner.steps_done,
        "center": learner.obs_center.detach().clone(),
        "spread": learner.obs_spread.detach().clone(),
        "rng": (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        ),
    }


def _restore_rng(saved):
    random.setstate(saved[0])
    np.random.set_state(saved[1])
    torch.set_rng_state(saved[2])
    if saved[3] is not None:
        torch.cuda.set_rng_state(saved[3])


class _StackMemory:
    def __init__(self, captured):
        self.states = captured["states"].to(device)
        self.actions = captured["actions"].to(device)
        self.nexts = captured["nexts"].to(device)
        self.rewards = captured["rewards"].to(device)
        self.terminal = captured["terminal"]

    def __len__(self):
        return self.states.shape[0]

    def sample(self, batch_size):
        picks = random.sample(range(len(self)), batch_size)
        batch = []
        for index in picks:
            nxt = None if bool(self.terminal[index]) else self.nexts[index:index + 1]
            batch.append(Transition(
                self.states[index:index + 1], self.actions[index:index + 1], nxt, self.rewards[index:index + 1],
            ))
        return batch


def _branch_learner(captured, replay=None):
    source = captured if replay is None else replay
    learner = SAC_Learner(
        obs_center=captured["center"], obs_spread=captured["spread"],
        reward_scale=100.0, freeze_alpha=True,
    )
    _load_module(learner.policy_net, captured["policy"])
    _load_module(learner.Q_net_1, captured["q1"])
    _load_module(learner.Q_net_2, captured["q2"])
    _load_module(learner.Q_target_1, captured["target1"])
    _load_module(learner.Q_target_2, captured["target2"])
    learner.Q_optimizer.load_state_dict(captured["q_opt"])
    learner.policy_optimizer.load_state_dict(captured["policy_opt"])
    learner.log_alpha = captured["log_alpha"].to(device).requires_grad_(True)
    learner.steps_done = captured["steps_done"]
    learner.memory = _StackMemory(source)
    return learner


def _frozen_report(name, update, learner, gain, rows, starts):
    def measure():
        batch = Transition(*zip(*rows))
        mask = torch.tensor([row is not None for row in batch.next_state], device=device)
        state_batch = learner._condition(torch.cat(batch.state))
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward) / learner.reward_scale
        with torch.no_grad():
            joined = torch.cat((state_batch, action_batch / learner.scale), dim=-1)
            q1 = learner.Q_net_1(joined).squeeze(-1)
            q2 = learner.Q_net_2(joined).squeeze(-1)
            bootstrap = torch.zeros(len(rows), device=device)
            if mask.any():
                nxt = learner._condition(torch.cat([row for row in batch.next_state if row is not None]))
                mean, log_std = learner.policy_net(nxt).chunk(2, dim=-1)
                log_std = log_std.clamp(-20, 2)
                dist = torch.distributions.Normal(mean, log_std.exp())
                raw = dist.sample()
                force = learner.scale * torch.tanh(raw)
                logprob = (
                    dist.log_prob(raw)
                    - torch.log(1 - torch.tanh(raw) ** 2 + 1e-12)
                    - np.log(learner.scale)
                ).squeeze(-1)
                target_in = torch.cat((nxt, force / learner.scale), dim=-1)
                q_next = torch.minimum(learner.Q_target_1(target_in), learner.Q_target_2(target_in)).squeeze(-1)
                bootstrap[mask] = learner.GAMMA * (q_next - learner.log_alpha.exp() * logprob)
            targets = reward_batch + bootstrap
            loss = torch.mean((q1 - targets) ** 2) + torch.mean((q2 - targets) ** 2)
        means, stds, slopes, saturated = [], [], [], 0
        for state in starts:
            with torch.no_grad():
                action = learner.select_action(state, do_deterministic=True)
                mean, log_std = learner.policy_net(learner._condition(state)).chunk(2, dim=-1)
            means.append(abs(float(mean)))
            stds.append(float(log_std.exp()))
            saturated += int(abs(float(action)) > 0.99 * learner.scale)
            force = action.detach().reshape(1, 1).requires_grad_(True)
            critic_in = torch.cat((learner._condition(state), force / learner.scale), dim=-1)
            value = torch.minimum(learner.Q_net_1(critic_in), learner.Q_net_2(critic_in)).sum()
            slopes.append(float(torch.autograd.grad(value, force)[0].detach()))
        low, mid, high = torch.quantile(targets, torch.tensor([0.1, 0.5, 0.9], device=device))
        print(
            f"{name} update {update} q1 {torch.median(q1):.3f} q2 {torch.median(q2):.3f} "
            f"|q1-q2| {torch.median(torch.abs(q1 - q2)):.3f} "
            f"target {float(low):.3f}/{float(mid):.3f}/{float(high):.3f} loss {float(loss):.3f} "
            f"dQ/dF {np.median(slopes):.6f} pre-tanh |mean| {np.median(means):.3f} "
            f"pre-tanh std {np.median(stds):.3f} saturated {saturated}/20",
            flush=True,
        )

    _preserved_rng(measure)


def _frozen_updates(name, learner, gain, rows, starts, updates):
    held_step = learner.Q_optimizer.step
    if name == "actor":
        learner.Q_optimizer.step = lambda *args, **kwargs: None
    for update in range(updates + 1):
        if update % 100 == 0:
            _frozen_report(name, update, learner, gain, rows, starts)
        if update == updates:
            break
        learner.optimize_model(do_policy_step=name != "critic")
        if name != "actor":
            _polyak(learner)
    learner.Q_optimizer.step = held_step


def frozen_replay_split(env, gain):
    from DoublePendulum.nn.nn_common import EarlyStop

    observations = []
    for seed in range(20):
        observation, _ = env.reset(seed=seed)
        done = False
        while not done:
            observations.append(np.asarray(observation, dtype=np.float64))
            observation, _, terminated, truncated, _ = env.step(lqr_force(env, gain))
            done = terminated or truncated
    columns = np.vstack(observations)
    center = torch.tensor(columns.mean(axis=0), dtype=torch.float32, device=device)
    spread = torch.tensor(np.maximum(columns.std(axis=0), 1e-8), dtype=torch.float32, device=device)
    print("observation center " + " ".join(f"{value:.6f}" for value in center.tolist()), flush=True)
    print("observation spread " + " ".join(f"{value:.6f}" for value in spread.tolist()), flush=True)

    original_check = EarlyStop.check
    captured = {}

    def remembering_check(self, durations, successes, actor_started):
        result = original_check(self, durations, successes, actor_started)
        counted = len(durations)
        if counted and counted % 100 == 0:
            _preserved_rng(lambda: _scaled_audit(
                "fixed-alpha SAC", remembering_check.learner, gain, counted, successes, durations,
            ))
            score = None if library_outcome.best is None else library_outcome.best["score"]
            if "states" not in captured and score is not None and score[0] == 20:
                captured.update(_capture_learner(remembering_check.learner))
                captured["episode"] = counted
                return f"snapshot at episode {counted}"
            if "states" not in captured and counted >= 4000:
                return "no 20/20 holdout by episode 4000"
        return result

    EarlyStop.check = remembering_check
    try:
        library_outcome.best = None
        _reseed(0)
        learner = SAC_Learner(obs_center=center, obs_spread=spread, reward_scale=100.0, freeze_alpha=True)
        learner.dump_best = lambda *args, **kwargs: None
        remembering_check.learner = learner
        original_reset = learner.env.reset

        def reset(*args, **kwargs):
            seed = kwargs.get("seed")
            if seed is not None and seed >= 100:
                kwargs["seed"] = seed + 20
            return original_reset(*args, **kwargs)

        learner.env.reset = reset
        learner.train(num_episodes=4000)
    finally:
        EarlyStop.check = original_check
    if "states" not in captured:
        print("frozen replay split found no 20/20 snapshot", flush=True)
        return
    print(
        f"frozen replay snapshot episode {captured['episode']} transitions {len(captured['states'])}",
        flush=True,
    )
    probe = init_env()
    starts = []
    for seed in range(100, 120):
        observation, _ = probe.reset(seed=seed)
        starts.append(torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0))
    for name in ("critic", "actor", "normal"):
        _restore_rng(captured["rng"])
        branch = _branch_learner(captured)
        fixed_rows = _preserved_rng(lambda: branch.memory.sample(256))
        _restore_rng(captured["rng"])
        _frozen_updates(name, branch, gain, fixed_rows, starts, 3000)


def _replay_facts(name, captured):
    rewards = captured["rewards"].numpy().reshape(-1)
    angles = np.max(np.abs(captured["states"].numpy()[:, 1:3]) * (np.pi / 4.0), axis=1)
    speeds = np.max(np.abs(captured["states"].numpy()[:, 3:6]), axis=1)
    reward_low, reward_mid, reward_high = np.quantile(rewards, [0.1, 0.5, 0.9])
    print(
        f"{name} replay episode {captured['episode']} transitions {len(rewards)} "
        f"terminated {float(captured['terminal'].float().mean()):.3f} "
        f"reward {reward_low:.3f}/{reward_mid:.3f}/{reward_high:.3f} "
        f"tilt p50 {np.median(angles):.4f} p90 {np.quantile(angles, 0.9):.4f} "
        f"within 0.01 rad {float(np.mean(angles < 0.01)):.3f} "
        f"clipped speed p50 {np.median(speeds):.3f} p90 {np.quantile(speeds, 0.9):.3f}",
        flush=True,
    )


def replay_swap(env, gain):
    from DoublePendulum.nn.nn_common import EarlyStop

    observations = []
    for seed in range(20):
        observation, _ = env.reset(seed=seed)
        done = False
        while not done:
            observations.append(np.asarray(observation, dtype=np.float64))
            observation, _, terminated, truncated, _ = env.step(lqr_force(env, gain))
            done = terminated or truncated
    columns = np.vstack(observations)
    center = torch.tensor(columns.mean(axis=0), dtype=torch.float32, device=device)
    spread = torch.tensor(np.maximum(columns.std(axis=0), 1e-8), dtype=torch.float32, device=device)
    print("observation center " + " ".join(f"{value:.6f}" for value in center.tolist()), flush=True)
    print("observation spread " + " ".join(f"{value:.6f}" for value in spread.tolist()), flush=True)

    original_check = EarlyStop.check
    shots = {}

    def remembering_check(self, durations, successes, actor_started):
        result = original_check(self, durations, successes, actor_started)
        counted = len(durations)
        if counted and counted % 100 == 0:
            audit = _preserved_rng(lambda: _scaled_audit(
                "fixed-alpha SAC", remembering_check.learner, gain, counted, successes, durations,
            ))
            if "solved" not in shots and audit["successes"] == 20:
                shots["solved"] = _capture_learner(remembering_check.learner)
                shots["solved"]["episode"] = counted
                print(f"snapshot solved at episode {counted}", flush=True)
            elif (
                "swaying" not in shots and "solved" in shots and audit["successes"] == 0
                and audit["length"] >= 600 and audit["upright"] >= 0.99 and audit["speed"] >= 2
            ):
                shots["swaying"] = _capture_learner(remembering_check.learner)
                shots["swaying"]["episode"] = counted
                print(f"snapshot swaying at episode {counted}", flush=True)
            elif (
                "saturated" not in shots and "swaying" in shots
                and audit["opening"] >= 24 and audit["length"] <= 80
            ):
                shots["saturated"] = _capture_learner(remembering_check.learner)
                shots["saturated"]["episode"] = counted
                print(f"snapshot saturated at episode {counted}", flush=True)
                return "snapshots complete"
            if "saturated" not in shots and counted >= 8000:
                return "snapshot cap at episode 8000"
        return result

    EarlyStop.check = remembering_check
    try:
        library_outcome.best = None
        _reseed(0)
        learner = SAC_Learner(obs_center=center, obs_spread=spread, reward_scale=100.0, freeze_alpha=True)
        learner.dump_best = lambda *args, **kwargs: None
        remembering_check.learner = learner
        original_reset = learner.env.reset

        def reset(*args, **kwargs):
            seed = kwargs.get("seed")
            if seed is not None and seed >= 100:
                kwargs["seed"] = seed + 20
            return original_reset(*args, **kwargs)

        learner.env.reset = reset
        learner.train(num_episodes=8000)
        del learner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        EarlyStop.check = original_check
    names = [name for name in ("solved", "swaying", "saturated") if name in shots]
    if len(names) < 2:
        print(f"replay swap saved {names}", flush=True)
        return
    probe = init_env()
    starts = []
    for seed in range(100, 120):
        observation, _ = probe.reset(seed=seed)
        starts.append(torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0))
    for name in names:
        _replay_facts(name, shots[name])
    for learner_name in names:
        for replay_name in names:
            label = f"{learner_name} learner + {replay_name} replay"
            _restore_rng(shots[learner_name]["rng"])
            branch = _branch_learner(shots[learner_name], shots[replay_name])
            fixed_rows = _preserved_rng(lambda: branch.memory.sample(256))
            _restore_rng(shots[learner_name]["rng"])
            _frozen_updates(label, branch, gain, fixed_rows, starts, 3000)
            del branch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _live_learner(captured, maxlen):
    from collections import deque
    from DoublePendulum.nn.nn_common import ReplayMemory

    learner = _branch_learner(captured)
    stack = learner.memory
    rows = deque(maxlen=maxlen)
    for index in range(len(stack)):
        nxt = None if bool(stack.terminal[index]) else stack.nexts[index:index + 1].clone()
        rows.append(Transition(
            stack.states[index:index + 1].clone(),
            stack.actions[index:index + 1].clone(),
            nxt,
            stack.rewards[index:index + 1].clone(),
        ))
    replay = ReplayMemory(maxlen)
    replay.memory = rows
    learner.memory = replay
    return learner


def _audited_train(learner, gain, label, num_episodes, start_episode, on_audit):
    from DoublePendulum.nn.nn_common import EarlyStop

    original_check = EarlyStop.check

    def remembering_check(self, durations, successes, actor_started):
        result = original_check(self, durations, successes, actor_started)
        counted = len(durations)
        if counted and counted % 100 == 0 and counted > start_episode:
            audit = _preserved_rng(lambda: _scaled_audit(
                label, learner, gain, counted, successes, durations,
            ))
            stop = on_audit(counted, audit, learner)
            if stop:
                return stop
        return result

    EarlyStop.check = remembering_check
    try:
        learner.train(num_episodes=num_episodes, start_episode=start_episode)
    finally:
        EarlyStop.check = original_check


def _continue_live(captured, gain, label, maxlen, num_episodes):
    print(f"continuing {label} from episode {captured['episode']} with buffer {maxlen}", flush=True)
    _restore_rng(captured["rng"])
    learner = _live_learner(captured, maxlen)
    learner.dump_best = lambda *args, **kwargs: None
    original_reset = learner.env.reset

    def reset(*args, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None and seed >= 100:
            kwargs["seed"] = seed + 20
        return original_reset(*args, **kwargs)

    learner.env.reset = reset
    start = captured["episode"]
    learner.episode_durations = [1] * start
    learner.final_state_balanced = [0] * start
    learner.final_state_truncated = [0] * start
    learner.cumulative_rewards = [0.0] * start
    _audited_train(learner, gain, label, num_episodes, start, lambda *args: None)
    del learner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def failure_tests(env, gain):
    from DoublePendulum.nn.nn_common import EarlyStop

    observations = []
    for seed in range(20):
        observation, _ = env.reset(seed=seed)
        done = False
        while not done:
            observations.append(np.asarray(observation, dtype=np.float64))
            observation, _, terminated, truncated, _ = env.step(lqr_force(env, gain))
            done = terminated or truncated
    columns = np.vstack(observations)
    center = torch.tensor(columns.mean(axis=0), dtype=torch.float32, device=device)
    spread = torch.tensor(np.maximum(columns.std(axis=0), 1e-8), dtype=torch.float32, device=device)
    print("observation center " + " ".join(f"{value:.6f}" for value in center.tolist()), flush=True)
    print("observation spread " + " ".join(f"{value:.6f}" for value in spread.tolist()), flush=True)

    shots = {}
    held = {}

    def on_audit(counted, audit, learner):
        if "fifo" not in shots and counted < 4500 and audit["opening"] >= 24 and audit["length"] <= 80:
            print(f"capturing fifo snapshot at episode {counted}", flush=True)
            shots["fifo"] = _capture_learner(learner)
            shots["fifo"]["episode"] = counted
            print(f"snapshot fifo at episode {counted} transitions {len(shots['fifo']['states'])}", flush=True)
        if "collapse" not in shots and counted >= 5200 and audit["length"] <= 80:
            if "states" not in held:
                return None
            print(f"capturing collapse replay at episode {counted}", flush=True)
            shots["before"] = dict(held)
            shots["collapse"] = _capture_learner(learner)
            shots["collapse"]["episode"] = counted
            print(
                f"snapshot collapse at episode {counted} from learner episode {shots['before']['episode']}",
                flush=True,
            )
            return None
        if "collapse" not in shots and counted >= 4800:
            print(f"capturing pre-collapse learner at episode {counted}", flush=True)
            held.clear()
            held.update(_capture_learner(learner))
            held["episode"] = counted
        return None

    original_check = EarlyStop.check
    try:
        library_outcome.best = None
        _reseed(0)
        learner = SAC_Learner(obs_center=center, obs_spread=spread, reward_scale=100.0, freeze_alpha=True)
        learner.dump_best = lambda *args, **kwargs: None
        original_reset = learner.env.reset

        def reset(*args, **kwargs):
            seed = kwargs.get("seed")
            if seed is not None and seed >= 100:
                kwargs["seed"] = seed + 20
            return original_reset(*args, **kwargs)

        learner.env.reset = reset
        _audited_train(learner, gain, "buffer 1M", 8000, 0, on_audit)
        del learner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        EarlyStop.check = original_check

    if "before" in shots and "collapse" in shots:
        _replay_facts("before", shots["before"])
        _replay_facts("collapse", shots["collapse"])
        probe = init_env()
        starts = []
        for seed in range(100, 120):
            observation, _ = probe.reset(seed=seed)
            starts.append(torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0))
        label = "pre-collapse learner + collapse replay"
        _restore_rng(shots["before"]["rng"])
        branch = _branch_learner(shots["before"], shots["collapse"])
        fixed_rows = _preserved_rng(lambda: branch.memory.sample(256))
        _restore_rng(shots["before"]["rng"])
        _frozen_updates(label, branch, gain, fixed_rows, starts, 3000)
        del branch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("failure tests did not capture the late collapse", flush=True)
    if "fifo" in shots:
        _continue_live(shots["fifo"], gain, "buffer 2M", 2_000_000, 8000)
    else:
        print("failure tests did not capture the episode-3500 snapshot", flush=True)


def library_outcome(name, builder=None, trace=False, gain=None):
    from DoublePendulum.nn.bro import BRO_Learner
    from DoublePendulum.nn.crossQ import CrossQ_Learner
    from DoublePendulum.nn.ddpg import DDPG_Learner
    from DoublePendulum.nn.nn_common import EarlyStop

    original_check = EarlyStop.check

    def remembering_check(self, durations, successes, actor_started):
        result = original_check(self, durations, successes, actor_started)
        if result:
            EarlyStop.last_result = result
        counted = len(durations)
        interval = 100 if trace else 500
        if counted and counted % interval == 0:
            if trace:
                _preserved_rng(lambda: _scaled_audit(
                    name, library_outcome.learner, gain, counted, successes, durations,
                ))
            else:
                holdout = _holdout(library_outcome.learner)
                print(
                    f"{name} episode {counted} steps {library_outcome.learner.steps_done} "
                    f"training successes {int(sum(successes))}/{counted} "
                    f"recent median length {np.median(durations[-100:]):.0f} "
                    f"holdout {holdout}",
                    flush=True,
                )
        return result

    EarlyStop.check = remembering_check
    EarlyStop.last_result = None
    library_outcome.best = None
    _reseed(0)
    learners = {
        "SAC": SAC_Learner,
        "DDPG": DDPG_Learner,
        "CrossQ": CrossQ_Learner,
        "BRO": BRO_Learner,
    }
    learner = builder() if builder else learners[name]()
    learner.dump_best = lambda *args, **kwargs: None
    library_outcome.learner = learner
    original_reset = learner.env.reset

    def reset(*args, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None and seed >= 100:
            kwargs["seed"] = seed + 20
        return original_reset(*args, **kwargs)

    learner.env.reset = reset
    episodes = 2000 if name == "BRO" else 20000
    learner.train(num_episodes=episodes)
    successes = learner.final_state_balanced
    print(
        f"{name} finished episodes {len(learner.episode_durations)} steps {learner.steps_done} "
        f"stop {EarlyStop.last_result} training successes {int(sum(successes))}/{len(successes)} "
        f"holdout {_holdout(learner)}",
        flush=True,
    )
    best = library_outcome.best
    if best is not None:
        current = {key: value.detach().cpu().clone() for key, value in learner.policy_net.state_dict().items()}
        learner.policy_net.load_state_dict(best["policy"])
        print(
            f"{name} best policy episode {best['episode']} holdout {_holdout(learner)}",
            flush=True,
        )
        learner.policy_net.load_state_dict(current)


def _scaled_audit(name, learner, gain, counted, successes, durations):
    probe = init_env()
    limit = float(probe.unwrapped.max_F)
    tolerance = float(probe.unwrapped.theta_tol)
    still_limit = 0.001 * np.mean([probe.unwrapped.l1, probe.unwrapped.l2]) / probe.unwrapped.dt
    episode_successes, lengths, openings = [], [], []
    stochastic_successes, stochastic_lengths = [], []
    pre_means, pre_stds, force_stds = [], [], []
    upright_fractions, upright_speeds, holds = [], [], []
    values, slopes, toward = [], [], 0
    for seed in range(100, 120):
        observation, _ = probe.reset(seed=seed)
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = learner.select_action(state, do_deterministic=True)
            mean, log_std = learner.policy_net(learner._condition(state)).chunk(2, dim=-1)
            log_std = log_std.clamp(-20, 2)
            dist = torch.distributions.Normal(mean, log_std.exp())
            sampled = dist.sample((32,))
            sampled_force = torch.clamp(
                learner.scale * torch.tanh(sampled), -learner.scale, learner.scale,
            )
        pre_means.append(abs(float(mean)))
        pre_stds.append(float(log_std.exp()))
        force_stds.append(float(sampled_force.std(unbiased=False)))
        force = action.detach().reshape(1, 1).requires_grad_(True)
        critic_in = torch.cat((learner._condition(state), force / learner.scale), dim=-1)
        value = torch.minimum(learner.Q_net_1(critic_in), learner.Q_net_2(critic_in)).sum()
        slope = torch.autograd.grad(value, force)[0]
        lqr = _force_from_physics(probe.unwrapped.pendulum.x, gain, limit)
        values.append(float(value.detach()))
        slopes.append(float(slope.detach()))
        toward += int(float(slope.detach()) * (lqr - float(force.detach())) > 0)
        done = False
        length = 0
        opening = abs(float(action))
        info = {}
        upright_steps, speeds_when_upright, hold = 0, [], 0.0
        while not done:
            observation, _, terminated, truncated, info = probe.step(float(action))
            length += 1
            done = terminated or truncated
            physical = wrapped_error(probe.unwrapped.pendulum.x)
            tilt = float(np.max(np.abs(physical[1:3])))
            speed = float(np.max(np.abs(physical[3:6])))
            if tilt < tolerance:
                upright_steps += 1
                speeds_when_upright.append(speed / still_limit)
            hold = max(hold, probe.unwrapped.success_frames * probe.unwrapped.dt)
            if not done:
                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = learner.select_action(state, do_deterministic=True)
        episode_successes.append(bool(info.get("success", False)))
        lengths.append(length)
        openings.append(opening)
        upright_fractions.append(upright_steps / length)
        upright_speeds.append(float(np.median(speeds_when_upright)) if speeds_when_upright else np.nan)
        holds.append(hold)
        observation, _ = probe.reset(seed=seed)
        done = False
        length = 0
        info = {}
        while not done:
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = learner.select_action(state, do_deterministic=False)
            observation, _, terminated, truncated, info = probe.step(float(action))
            length += 1
            done = terminated or truncated
        stochastic_successes.append(bool(info.get("success", False)))
        stochastic_lengths.append(length)
    rows = list(learner.memory.memory)
    stride = max(1, len(rows) // 4096)
    tilts = []
    for row in rows[::stride][:4096]:
        observed = np.asarray(row.state.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        tilts.append(np.max(np.abs(observed[1:3] * (np.pi / 4.0))))
    tilts = np.asarray(tilts)
    holdout_successes = int(sum(episode_successes))
    median_length = float(np.median(lengths))
    score = (holdout_successes, median_length)
    if library_outcome.best is None or score > library_outcome.best["score"]:
        library_outcome.best = {
            "episode": counted,
            "score": score,
            "policy": {key: value.detach().cpu().clone() for key, value in learner.policy_net.state_dict().items()},
        }
        print(f"{name} kept policy at episode {counted}", flush=True)
    narrow = int(np.sum(tilts < 0.01))
    wide = int(np.sum(tilts < 0.1))
    print(
        f"{name} episode {counted} steps {learner.steps_done} "
        f"training successes {int(sum(successes))}/{counted} "
        f"recent median length {np.median(durations[-100:]):.0f} "
        f"alpha {float(learner.log_alpha.detach().exp()):.6f} "
        f"holdout {holdout_successes}/20 median length {median_length:.0f} "
        f"stochastic {int(sum(stochastic_successes))}/20 median length {np.median(stochastic_lengths):.0f} "
        f"opening |F| {np.median(openings):.3f} "
        f"pre-tanh |mean| {np.median(pre_means):.3f} pre-tanh std {np.median(pre_stds):.3f} "
        f"force std {np.median(force_stds):.3f} "
        f"upright fraction {np.median(upright_fractions):.3f} "
        f"upright speed/limit {np.nanmedian(upright_speeds):.3f} "
        f"longest hold {np.median(holds):.2f}s "
        f"median Q {np.median(values):.3f} median dQ/dF {np.median(slopes):.6f} "
        f"toward LQR {toward}/20 "
        f"replay within 0.01 rad {narrow}/{len(tilts)} within 0.1 rad {wide}/{len(tilts)}",
        flush=True,
    )
    return {
        "successes": holdout_successes,
        "length": median_length,
        "upright": float(np.median(upright_fractions)),
        "speed": float(np.nanmedian(upright_speeds)),
        "opening": float(np.median(openings)),
    }


def _holdout(learner):
    probe = init_env()
    successes = []
    lengths = []
    openings = []
    for seed in range(100, 120):
        observation, _ = probe.reset(seed=seed)
        done = False
        length = 0
        opening = None
        info = {}
        while not done:
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            action = learner.select_action(state, do_deterministic=True)
            force = float(action)
            if opening is None:
                opening = abs(force)
            observation, _, terminated, truncated, info = probe.step(force)
            length += 1
            done = terminated or truncated
        successes.append(bool(info.get("success", False)))
        lengths.append(length)
        openings.append(opening)
    return (
        f"{int(sum(successes))}/20 median length {np.median(lengths):.0f} "
        f"opening |F| {np.median(openings):.3f}"
    )


def return_teacher(env, gain):
    import math
    _reseed(0)
    learner = SAC_Learner()
    records = []
    episode = 0
    observation, _ = env.reset(seed=episode)
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    while learner.steps_done < 22000:
        physics = env.unwrapped.pendulum.x.copy()
        error = wrapped_error(physics)
        action = learner.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(float(action))
        learner.steps_done += 1
        records.append({
            "physics": physics,
            "observation": np.asarray(state.detach().cpu().numpy(), dtype=np.float64).reshape(-1),
            "tilt": max(abs(error[1]), abs(error[2])),
        })
        if terminated or truncated:
            episode += 1
            if 100 <= episode < 120:
                episode = 120
            observation, _ = env.reset(seed=episode)
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    weights = {key: value.detach().cpu() for key, value in learner.policy_net.state_dict().items()}
    scale = learner.scale
    grid = np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0])
    limit = float(env.unwrapped.max_F)
    rng = np.random.default_rng(0)
    tilts = np.asarray([record["tilt"] for record in records])
    chosen = np.concatenate([
        rng.choice(np.flatnonzero(tilts < 0.01), 100, replace=False),
        rng.choice(np.flatnonzero(tilts >= 0.01), 100, replace=False),
    ])
    probes = []
    observations = []
    lqr_forces = []
    for seed in range(100, 120):
        observation, _ = env.reset(seed=seed)
        physics = env.unwrapped.pendulum.x.copy()
        lqr = _force_from_physics(physics, gain, limit)
        probes.append((physics, 0.0, np.concatenate([grid, [lqr]]), 10_000 + seed))
        observations.append(np.asarray(observation, dtype=np.float64))
        lqr_forces.append(lqr)
    for index in chosen:
        record = records[int(index)]
        lqr = _force_from_physics(record["physics"], gain, limit)
        probes.append((
            record["physics"],
            float(record["observation"][6] * limit),
            np.concatenate([grid, [lqr]]),
            20_000 + int(index),
        ))
        observations.append(record["observation"])
        lqr_forces.append(lqr)
    jobs = []
    chunks = np.array_split(np.arange(len(probes)), 16)
    for chunk_index, chunk in enumerate(chunks):
        jobs.append((
            chunk_index, weights, scale,
            [probes[int(index)] for index in chunk],
        ))
    rolled = _fork_map(_behavior_chunk, jobs, message="teacher chunk")
    samples = np.stack([value for chunk in rolled for value in chunk])
    observations = np.vstack(observations)
    lqr_forces = np.asarray(lqr_forces)
    means = samples.mean(axis=2)
    errors = samples.std(axis=2, ddof=1) / np.sqrt(samples.shape[2])

    def target_force(local):
        centered = means[local] - means[local].max()
        temperature = max(float(np.median(errors[local])), 1e-6)
        weights_local = np.exp(centered / temperature)
        weights_local /= weights_local.sum()
        return float(weights_local @ probes[local][2])

    targets = np.asarray([target_force(local) for local in range(len(probes))])
    train_slice = slice(20, None)
    valid_slice = slice(0, 20)
    print(
        f"teacher targets median |F| train {np.median(np.abs(targets[train_slice])):.3f} "
        f"validation {np.median(np.abs(targets[valid_slice])):.3f} "
        f"validation |F_LQR| {np.median(np.abs(lqr_forces[valid_slice])):.3f} "
        f"within 2 N of LQR {int(np.sum(np.abs(targets[valid_slice] - lqr_forces[valid_slice]) < 2))}/20",
        flush=True,
    )

    def fit_actor(label, desired):
        columns = observations[train_slice]
        center = columns.mean(axis=0)
        spread = np.maximum(columns.std(axis=0), 1e-8)
        _reseed(0)
        actor = SAC_POLICY_DP(7).to(device)
        optimizer = optim.AdamW(actor.parameters(), lr=3e-4, amsgrad=True, weight_decay=0)
        desired_t = torch.tensor(desired, dtype=torch.float32, device=device)
        feature_t = torch.tensor((columns - center) / spread, dtype=torch.float32, device=device)
        for _ in range(20000):
            mean, log_std = actor(feature_t).chunk(2, dim=-1)
            loss = torch.nn.functional.mse_loss(scale * torch.tanh(mean.squeeze(-1)), desired_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(actor.parameters(), 100)
            optimizer.step()
        _closed_loop(label, actor, center, spread, scale)

    fit_actor("softmax return", targets[train_slice])
    fit_actor("maximum return", np.asarray([
        probes[local][2][int(np.argmax(means[local]))] for local in range(20, len(probes))
    ]))

    train_actions = np.concatenate([probes[local][2] for local in range(20, len(probes))])
    train_obs = np.repeat(observations[train_slice], 8, axis=0)
    train_input = np.column_stack((train_obs, train_actions / scale))
    train_target = means[train_slice].reshape(-1)
    valid_actions = np.concatenate([probes[local][2] for local in range(20)])
    valid_obs = np.repeat(observations[valid_slice], 8, axis=0)
    valid_input = np.column_stack((valid_obs, valid_actions / scale))
    center = train_input.mean(axis=0)
    spread = np.maximum(train_input.std(axis=0), 1e-8)
    target_scale = max(float(train_target.std()), 1e-8)
    feature_t = torch.tensor((train_input - center) / spread, dtype=torch.float32, device=device)
    target_t = torch.tensor(train_target / target_scale, dtype=torch.float32, device=device)
    torch.manual_seed(0)
    nets = [SAC_Q_DP(7).to(device) for _ in range(2)]
    optimizer = optim.AdamW(
        [parameter for net in nets for parameter in net.parameters()],
        lr=3e-4, amsgrad=True,
    )
    previous = None
    loss = None
    for update in range(200000):
        loss = sum(torch.nn.functional.mse_loss(net(feature_t).squeeze(-1), target_t) for net in nets)
        if update in (20000, 50000, 100000) or (update and update % 20000 == 0):
            _pair_report(
                f"critic step {update}", nets, center, spread, target_scale,
                valid_input, means, errors, train=False,
            )
            _pair_report(
                f"critic step {update} train", nets, center, spread, target_scale,
                train_input, means[train_slice], errors[train_slice], train=True,
            )
        if update and update % 20000 == 0 and previous is not None:
            if float(loss.detach()) > 0.99 * previous:
                print(f"critic plateau at step {update} loss {float(loss.detach()):.6f}", flush=True)
                break
        if update % 20000 == 0:
            previous = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        for net in nets:
            torch.nn.utils.clip_grad_value_(net.parameters(), 100)
        optimizer.step()
    _pair_report(
        f"critic final {update + 1}", nets, center, spread, target_scale,
        valid_input, means, errors, train=False,
    )
    for net in nets:
        net.requires_grad_(False)
    _ascend_actor(nets, observations[train_slice], center, spread, target_scale, scale)


def _pair_report(label, nets, center, spread, target_scale, features, means, errors, train):
    values = torch.tensor((features - center) / spread, dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = torch.stack([net(values).squeeze(-1) for net in nets]).mean(dim=0)
    prediction = (prediction.cpu().numpy() * target_scale).reshape(-1, 8)
    truth = means[:20] if not train else means
    error_rows = errors[:20] if not train else errors
    if train:
        truth = means
        error_rows = errors
    correct = weighted_correct = weight = 0.0
    gap_errors = []
    pairs = 0
    for row in range(len(prediction)):
        for left in range(8):
            for right in range(left + 1, 8):
                gap = float(truth[row, left] - truth[row, right])
                threshold = 3.0 * np.sqrt(error_rows[row, left] ** 2 + error_rows[row, right] ** 2)
                if abs(gap) <= threshold:
                    continue
                pairs += 1
                predicted = float(prediction[row, left] - prediction[row, right])
                hit = float(predicted * gap > 0)
                correct += hit
                weighted_correct += hit * abs(gap)
                weight += abs(gap)
                gap_errors.append(abs(predicted - gap))
    accuracy = correct / pairs if pairs else float("nan")
    weighted = weighted_correct / weight if weight else float("nan")
    median_error = float(np.median(gap_errors)) if gap_errors else float("nan")
    print(
        f"{label} resolved pairs {pairs} directional {accuracy:.3f} "
        f"weighted {weighted:.3f} median |gap error| {median_error:.3f}",
        flush=True,
    )


def _closed_loop(label, actor, center, spread, scale):
    probe = init_env()
    successes = []
    lengths = []
    openings = []
    for seed in range(100, 120):
        observation, _ = probe.reset(seed=seed)
        done = False
        length = 0
        opening = None
        info = {}
        while not done:
            feature = np.asarray(observation, dtype=np.float64)
            if center is not None:
                feature = (feature - center) / spread
            value = torch.tensor(feature, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean = actor(value).chunk(2, dim=-1)[0]
                force = float(torch.clamp(scale * torch.tanh(mean), -scale, scale))
            if opening is None:
                opening = abs(force)
            observation, _, terminated, truncated, info = probe.step(force)
            length += 1
            done = terminated or truncated
        successes.append(bool(info.get("success", False)))
        lengths.append(length)
        openings.append(opening)
    print(
        f"{label} holdout {int(sum(successes))}/20 median length {np.median(lengths):.0f} "
        f"opening |F| {np.median(openings):.3f}",
        flush=True,
    )


def _ascend_actor(nets, observations, center, spread, target_scale, scale):
    import math
    _reseed(0)
    actor = SAC_POLICY_DP(7).to(device)
    optimizer = optim.AdamW(actor.parameters(), lr=3e-4, amsgrad=True)
    states = torch.tensor(observations, dtype=torch.float32, device=device)
    alpha = 0.01
    for _ in range(20000):
        mean, log_std = actor(states).chunk(2, dim=-1)
        log_std = log_std.clamp(-20, 2)
        dist = torch.distributions.Normal(mean, log_std.exp())
        sampled = dist.rsample()
        action = torch.clamp(scale * torch.tanh(sampled), -scale, scale)
        log_prob = (
            dist.log_prob(sampled)
            - torch.log(1.0 - torch.tanh(sampled).pow(2) + 1e-12)
            - math.log(scale)
        ).squeeze(-1)
        standardized = (
            torch.cat((states, action / scale), dim=-1)
            - torch.tensor(center, dtype=torch.float32, device=device)
        ) / torch.tensor(spread, dtype=torch.float32, device=device)
        scores = torch.cat([net(standardized) for net in nets], dim=-1)
        selected = torch.min(scores, dim=-1).values * target_scale
        loss = -(selected - alpha * log_prob).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(actor.parameters(), 100)
        optimizer.step()
    _closed_loop("critic ascent", actor, None, None, scale)


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
    critic_fit(env, gain)
    critic_td(env, gain)
    critic_td_stored(env, gain)


if __name__ == "__main__":
    main()
