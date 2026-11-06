"""Audit one learner after the stillness reward was added.

Run it once per terminal, so each learner has its own Cursor terminal:

    python scripts/diagnose_stillness.py ddpg
    python scripts/diagnose_stillness.py sac
    python scripts/diagnose_stillness.py crossq

An optional second argument is the episode count (default 8000).

DDPG and CrossQ use the settings in their train scripts. SAC does too: entropy
weight 0.0001, frozen, reward divided by 100, replay of 2M. A held-out check on
seeds 100-119 runs every 100 episodes and prints one short line. Nothing is
written to the checkpoint files. The DDPG run also runs a probe every 500
episodes and saves its networks to /tmp/stillness_ddpg_ep<N>.pt.

With reward divided by 100, a perfect controller's value is at most 2.0 (LQR
scores about 1.97), so a critic value far above 2 is an overestimate.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

import diagnose_learning as diag

LEARNERS = ("ddpg", "sac", "crossq")


def short_line(line):
    """Shorten an audit line to the numbers worth watching, or return None for noise."""
    if "kept policy" in line:
        return None
    head = re.match(r"(.+?) episode (\d+) steps \d+ training successes (\d+)/\d+", line)
    if not head:
        return line

    def find(pattern):
        found = re.search(pattern, line)
        return found.group(1) if found else "?"

    return (
        f"{head.group(1)} ep {head.group(2)} | train {head.group(3)} | "
        f"held-out {find(r'holdout (\d+/20)')} len {find(r'holdout \d+/20 median length (\d+)')} "
        f"stoch {find(r'stochastic (\d+/20)')} | longest hold {find(r'longest hold ([\d.]+)s')}s "
        f"| speed/limit {find(r'upright speed/limit ([\d.]+)')} | opening |F| {find(r'opening \|F\| ([\d.]+)')} "
        f"| Q {find(r'median Q (-?[\d.]+)')} | toward LQR {find(r'toward LQR (\d+/20)')}"
    )


class ShortLines:
    """Print each audit line in its short form."""

    def __init__(self, stream):
        self.stream = stream
        self.pending = ""

    def write(self, text):
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            shortened = short_line(line)
            if shortened is not None:
                self.stream.write(shortened + "\n")
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def ddpg_probe(counted, learner, gain):
    """Every 500 episodes: how saturated the actor and replay are, and how the critic scores forces."""
    if counted % 500:
        return
    scale = float(learner.policy_net.scale)
    ceiling = 2.0 / learner.reward_scale / (1.0 - learner.GAMMA)
    env = diag.init_env()
    observations, lqr_forces = [], []
    for seed in range(100, 120):
        observation, _ = env.reset(seed=seed)
        for _ in range(200):
            observations.append(np.asarray(observation, dtype=np.float32))
            lqr_forces.append(diag.lqr_force(env, gain))
            state = torch.tensor(observation, dtype=torch.float32, device=diag.device).unsqueeze(0)
            with torch.no_grad():
                force = float(learner.select_action(state, do_deterministic=True))
            observation, _, terminated, truncated, _ = env.step(force)
            if terminated or truncated:
                break
    states = learner._condition(torch.tensor(np.stack(observations), device=diag.device))
    lqr = torch.tensor(lqr_forces, dtype=torch.float32, device=diag.device)
    with torch.no_grad():
        chosen = learner.policy_net(states).squeeze(1)

        def value(force):
            return learner.Q_net(torch.cat((states, force.reshape(-1, 1) / scale), dim=1)).squeeze(1)

        q_policy, q_lqr, q_zero = value(chosen), value(lqr), value(torch.zeros_like(lqr))
    recent = [row.action for row in list(learner.memory.memory)[-100000:]]
    actions = torch.cat(recent).abs()
    print(
        f"  probe ep {counted}: held-out policy |F| median {float(chosen.abs().median()):.1f} N, "
        f"at the rail {float((chosen.abs() >= 0.96 * scale).float().mean()):.0%}, "
        f"LQR median |F| {float(lqr.abs().median()):.2f} N | "
        f"replay (last {len(recent)}) at the rail {float((actions >= 0.96 * scale).float().mean()):.0%}, "
        f"under 5 N {float((actions < 5.0).float().mean()):.0%} | "
        f"Q at policy {float(q_policy.median()):.2f}, at LQR {float(q_lqr.median()):.2f}, at 0 N "
        f"{float(q_zero.median()):.2f} (ceiling {ceiling:.1f}); critic prefers LQR over policy "
        f"{float((q_lqr > q_policy).float().mean()):.0%}",
        flush=True,
    )
    torch.save(
        {"policy": learner.policy_net.state_dict(), "critic": learner.Q_net.state_dict()},
        f"/tmp/stillness_ddpg_ep{counted}.pt",
    )


def run_one(name, episodes):
    sys.stdout = ShortLines(sys.stdout)
    env = diag.init_env()
    gain = diag.discrete_lqr(env.unwrapped)
    print(f"{name}: value ceiling for this reward is 2.0 (LQR scores about 1.97)", flush=True)
    if name == "ddpg":
        diag.ddpg_scaled_run(
            env, gain, label="stillness ddpg", num_episodes=episodes,
            on_audit=lambda counted, audit, learner: ddpg_probe(counted, learner, gain),
        )
    elif name == "sac":
        diag.speed_penalty_run(
            env, gain, alpha=0.0001, freeze_alpha=True,
            num_episodes=episodes, run_label="stillness sac",
        )
    else:
        diag.crossq_scaled_run(env, gain, label="stillness crossq", num_episodes=episodes)


def main():
    args = sys.argv[1:]
    if not args or args[0] not in LEARNERS:
        raise SystemExit("usage: diagnose_stillness.py ddpg|sac|crossq [episodes]")
    episodes = int(args[1]) if len(args) > 1 else 8000
    run_one(args[0], episodes)


if __name__ == "__main__":
    main()
