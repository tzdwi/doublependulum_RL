"""Audit one learner with the current reward, observation, and stop rule.

Run it once per terminal, so each learner has its own Cursor terminal:

    python scripts/diagnose_stillness.py ddpg
    python scripts/diagnose_stillness.py sac
    python scripts/diagnose_stillness.py crossq

An optional second argument is the episode count (default 8000).

DDPG and CrossQ use their training defaults. SAC learns its temperature,
starting at 0.01 with a target near 2.11 nats, and keeps a replay of 2M.
A held-out check on seeds 100-119 runs every 100 episodes and prints one short
line. Three perfect checks in a row stop the run. Nothing is written to the
checkpoint files. The DDPG run also probes every 500 episodes and saves its
networks to /tmp/stillness_ddpg_ep<N>.pt.

Rewards are divided by 1/(1 - 0.99) inside each learner, so a perfect
controller's value is 1. A critic value far above 1 is an overestimate. Soft
Q can sit a little above 1 because of the entropy bonus.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

import diagnose_learning as diag

LEARNERS = ("ddpg", "ddpg-twin", "ddpg-td3", "sac", "crossq")

# Diagnostic variants only. Plain "ddpg" stays the paper algorithm.
DDPG_VARIANTS = {
    "ddpg": {},
    "ddpg-twin": {"twin_critic": True},
    "ddpg-td3": {"twin_critic": True, "target_noise": 0.2, "target_noise_clip": 0.5},
}


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
        f"| Q {find(r'median Q (-?[\d.]+)')} | streak {find(r'streak (\d+)')} "
        f"| toward LQR {find(r'toward LQR (\d+/20)')}"
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


def ddpg_probe(name, counted, learner, gain):
    """Every 500 episodes: how saturated the actor and replay are, and how the critic scores forces."""
    if counted % 500:
        return
    scale = float(learner.policy_net.scale)
    ceiling = 1.0 / learner.reward_scale / (1.0 - learner.GAMMA)
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
    take = min(100000, len(learner.memory))
    end = learner.memory.position
    start = (end - take) % learner.memory.capacity
    if start < end:
        recent = learner.memory.actions[start:end]
    else:
        recent = torch.cat((learner.memory.actions[start:], learner.memory.actions[:end]), dim=0)
    actions = recent.abs().reshape(-1)
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
    snapshot = {"policy": learner.policy_net.state_dict(), "critic": learner.Q_net.state_dict()}
    if learner.twin_critic:
        snapshot["critic_2"] = learner.Q_net_2.state_dict()
    torch.save(snapshot, f"/tmp/stillness_{name.replace('-', '_')}_ep{counted}.pt")


def run_one(name, episodes):
    sys.stdout = ShortLines(sys.stdout)
    env = diag.init_env()
    gain = diag.discrete_lqr(env.unwrapped)
    print(f"{name}: value ceiling for this reward is 1.0", flush=True)
    if name in DDPG_VARIANTS:
        diag.ddpg_scaled_run(
            env, gain, label=f"stillness {name}", num_episodes=episodes,
            on_audit=lambda counted, audit, learner: ddpg_probe(name, counted, learner, gain),
            learner_kwargs=DDPG_VARIANTS[name],
        )
    elif name == "sac":
        diag.speed_penalty_run(
            env, gain, freeze_alpha=False,
            num_episodes=episodes, run_label="stillness sac",
        )
    else:
        diag.crossq_scaled_run(env, gain, label="stillness crossq", num_episodes=episodes)


def main():
    args = sys.argv[1:]
    if not args or args[0] not in LEARNERS:
        raise SystemExit("usage: diagnose_stillness.py " + "|".join(LEARNERS) + " [episodes]")
    episodes = int(args[1]) if len(args) > 1 else 8000
    run_one(args[0], episodes)


if __name__ == "__main__":
    main()
