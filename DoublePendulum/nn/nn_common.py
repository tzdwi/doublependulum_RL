"""
Some common tools across the RL algorithms I want to play with
"""

import gymnasium as gym
import imageio
import torch
import random
from collections import namedtuple, deque
import os
import numpy as np
from pathlib import Path

from .. import max_seconds

# Distribution checks synchronize the GPU on every sample. The policy already
# clamps its standard deviation, so skip the checks during training.
torch.distributions.Distribution.set_default_validate_args(False)

this_dir = str(os.path.dirname(os.path.realpath(__file__)))
plot_dir = str(Path(__file__).parent.parent.parent)+"/scripts/figures"
vid_dir = str(Path(__file__).parent.parent.parent)+"/scripts/videos"
HOLDOUT_SEEDS = list(range(100, 120))

# Bump this whenever the reward, observation, or timing changes. Checkpoints saved under another stamp will not load.
CHECKPOINT_VERSION = "reward-v3 obs-v2 timing-dt0.01x5 scale-1/(1-gamma)"


def save_checkpoint(state_dict, path):
    torch.save({"version": CHECKPOINT_VERSION, "state": state_dict}, path)


def load_checkpoint(path, weights_only=True, map_location=None):
    blob = torch.load(path, weights_only=True, map_location=map_location or device)
    found = blob.get("version") if isinstance(blob, dict) else None
    if found != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"{path} was saved as {found!r}, expected {CHECKPOINT_VERSION!r}; "
            "it predates the current reward or observation and will not load"
        )
    return blob["state"]

def prepare_parallel(learner, num_envs, **kwargs):
    learner.num_envs = int(num_envs)
    learner._env_kwargs = kwargs


def init_vector_env(num_envs, **kwargs):
    from functools import partial
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode
    from DoublePendulum.envs import make_env

    factories = [partial(make_env, dict(kwargs)) for _ in range(num_envs)]
    return AsyncVectorEnv(
        factories,
        shared_memory=True,
        copy=True,
        context="spawn",
        autoreset_mode=AutoresetMode.DISABLED,
    )


def _as_float(action):
    item = getattr(action, "item", None)
    return float(item() if item else action)


def _shift_training_seed(seed):
    # Seeds at or above 100 move up by 20, so 100-119 stay out of training.
    if seed is not None and seed >= 100:
        return seed + 20
    return seed


def _push_transition(memory, state, action, next_state, reward, env_index):
    if "env_index" in memory.push.__code__.co_varnames:
        memory.push(state, action, next_state, reward, env_index=env_index)
    else:
        memory.push(state, action, next_state, reward)


def _end_parallel_episode(memory, success, env_index):
    end_episode = getattr(memory, "end_episode", None)
    if end_episode is None:
        return
    if "env_index" in end_episode.__code__.co_varnames:
        end_episode(bool(success), env_index=env_index)
    else:
        end_episode(bool(success))


def run_parallel(learner, num_episodes, start_episode, choose_actions, on_step, actor_started, progress=False):
    """Step num_envs environments, then update once for each transition, in env order."""
    from tqdm import tqdm

    vec = init_vector_env(learner.num_envs, **learner._env_kwargs)
    learner.vec = vec
    n = learner.num_envs
    budget = num_episodes - start_episode
    bar = tqdm(total=budget) if progress else None
    try:
        episode_index = [start_episode + i for i in range(n)]
        next_index = start_episode + n
        seeds = [_shift_training_seed(index) for index in episode_index]
        observations, _ = vec.reset(seed=seeds, options={"training": True})
        states = [
            torch.tensor(observations[i], dtype=torch.float32, device=device).unsqueeze(0)
            for i in range(n)
        ]
        step_index = [0 for _ in range(n)]
        returns = [0.0 for _ in range(n)]
        early = EarlyStop()
        new_episodes = 0
        stop = None
        for index in episode_index:
            _maybe_record(learner, index)
        while new_episodes < budget and stop is None:
            actions = choose_actions(states)
            action_batch = np.array([[_as_float(action)] for action in actions], dtype=np.float32)
            observations, rewards, terminated, truncated, infos = vec.step(action_batch)
            finished = []
            for i in range(n):
                reward = float(rewards[i])
                done = bool(terminated[i] or truncated[i])
                reward_tensor = torch.tensor([reward], device=device)
                if terminated[i]:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observations[i], dtype=torch.float32, device=device
                    ).unsqueeze(0)
                _push_transition(
                    learner.memory, states[i], actions[i], next_state, reward_tensor, i
                )
                learner.steps_done += 1
                returns[i] += reward
                on_step(step_index[i], early)
                if done:
                    info = {
                        key: value[i]
                        for key, value in infos.items()
                        if not key.startswith("_") and not isinstance(value, dict)
                    }
                    _end_parallel_episode(learner.memory, info["success"], i)
                    learner.episode_durations.append(step_index[i] + 1)
                    learner.cumulative_rewards.append(returns[i])
                    learner.final_state_balanced.append(1 if info["success"] else 0)
                    learner.final_state_truncated.append(1 if truncated[i] else 0)
                    stop = early.check(
                        learner.episode_durations,
                        learner.final_state_balanced,
                        actor_started(),
                        holdout_streak=getattr(learner, "holdout_streak", 0),
                    )
                    new_episodes += 1
                    if bar is not None:
                        bar.update(1)
                    finished.append(i)
                else:
                    states[i] = next_state
                    step_index[i] += 1
            if finished and new_episodes < budget and stop is None:
                mask = np.zeros(n, dtype=bool)
                reset_seeds = [None for _ in range(n)]
                for i in finished:
                    mask[i] = True
                    episode_index[i] = next_index
                    reset_seeds[i] = _shift_training_seed(next_index)
                    _maybe_record(learner, next_index)
                    next_index += 1
                    step_index[i] = 0
                    returns[i] = 0.0
                reset_obs, _ = vec.reset(
                    seed=reset_seeds, options={"reset_mask": mask, "training": True}
                )
                for i in finished:
                    states[i] = torch.tensor(
                        reset_obs[i], dtype=torch.float32, device=device
                    ).unsqueeze(0)
        if stop:
            print(stop)
        _record_final(learner)
    finally:
        if bar is not None:
            bar.close()
        vec.close()
        learner.vec = None


def _policy_kwargs(learner):
    kwargs = {"do_deterministic": True}
    # BRO trains the optimistic mean. The video shows that mean, without action noise.
    if getattr(learner, "video_prefix", "") == "bro":
        kwargs["do_optimistic"] = True
    return kwargs


def _policy_force(learner, state):
    return _policy_forces(learner, [state])[0]


def _policy_forces(learner, states):
    """One deterministic force per state, from a single policy call."""
    action = learner.select_action(torch.cat(states, dim=0), **_policy_kwargs(learner))
    if isinstance(action, tuple):
        action = action[0]
    return [float(action.reshape(-1)[i]) for i in range(len(states))]


def _frame_steps(repeat, frames):
    """Physics-step indexes that get a video frame. One frame is the last step of the decision."""
    frames = max(1, min(int(frames), repeat))
    return {min(repeat - 1, int(round((k + 1) * repeat / frames)) - 1) for k in range(frames)}


def _video_env_kwargs(env):
    plant = env.unwrapped
    return dict(
        size=plant.size,
        max_F=plant.max_F,
        max_ang_vel=plant.max_ang_vel,
        mm=plant.mm,
        m1=plant.m1,
        l1=plant.l1,
        m2=plant.m2,
        l2=plant.l2,
        dt=plant.dt,
        theta_tol=plant.theta_tol,
        action_repeat=plant.action_repeat,
    )


def _video_envs(learner, count):
    envs = getattr(learner, "_video_envs", None)
    if envs is None or len(envs) < count:
        kwargs = _video_env_kwargs(learner.env)
        envs = [init_env(**kwargs) for _ in range(count)]
        learner._video_envs = envs
    return envs[:count]


def _as_state(observation):
    return torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


def _critic_value(learner, state, force):
    """The critic's value of this force at this state, in learner units (a perfect controller is 1)."""
    with torch.no_grad():
        action = torch.tensor([[force]], dtype=torch.float32, device=device)
        q_in = torch.cat((learner._condition(state), action / float(learner.env.action_space.high[0])), dim=-1)
        if hasattr(learner, "calc_Qs"):
            return float(torch.stack(learner.calc_Qs(q_in)).mean())
        if hasattr(learner, "Q_net"):
            return float(learner.Q_net(q_in))
        return float(torch.min(learner.Q_net_1(q_in), learner.Q_net_2(q_in)))


def _write_decision(writer, learner, envs, states, active, canvas, overlay, forces=None, frame_steps=None):
    """Hold one force and write the requested frames of that decision."""
    chosen = [i for i in range(len(envs)) if active[i]]
    if not chosen:
        return
    held = _policy_forces(learner, [states[i] for i in chosen])
    running = []
    for i, force in zip(chosen, held):
        plant = envs[i].unwrapped
        plant.hold_force(force)
        if forces is not None:
            forces.append(force)
        running.append((i, plant, []))
    repeat = running[0][1].action_repeat
    if frame_steps is None:
        frame_steps = set(range(repeat))
    shots = []
    for step in range(repeat):
        shown = []
        for i, plant, infos in running:
            if not active[i]:
                continue
            infos.append(plant._physics_step())
            shown.append(plant)
            if not overlay and step in frame_steps:
                shots.append(np.array(plant.pendulum.x, copy=True))
            if infos[-1]["failure_penalty"]:
                active[i] = False
        if not shown:
            break
        if overlay and step in frame_steps:
            writer.append_data(canvas.render_many(shown))
        if all(not active[i] for i, _, _ in running):
            break
    if overlay:
        for i, plant, _ in running:
            if active[i]:
                states[i] = _as_state(plant._clip_obs(plant._get_obs()))
        return
    _, plant, infos = running[0]
    observation, reward, terminated, _ = plant.decision_from_infos(infos)
    for shot in shots:
        plant.pendulum.x[:] = shot
        plant.state = plant._get_state()
        writer.append_data(plant.render(log_reward=True, reward=reward))
    if not terminated and active[0]:
        states[0] = _as_state(observation)


def record_policy_video(learner, seeds, episode_number):
    """Roll deterministic held-out seeds and save one frame per decision."""
    canvas = learner.log_env.unwrapped
    overlay = len(seeds) > 1
    if overlay:
        envs = _video_envs(learner, len(seeds))
        name = f"{learner.video_prefix}_ep{episode_number}.mp4"
    else:
        envs = [learner.log_env]
        name = f"{learner.video_prefix}_final.mp4"
    states = []
    for env, seed in zip(envs, seeds):
        observation, _ = env.reset(seed=int(seed))
        states.append(_as_state(observation))
    opening = _policy_forces(learner, states)
    start_values = [_critic_value(learner, state, force) for state, force in zip(states, opening)]
    forces = []
    active = [True] * len(envs)
    limit = int(envs[0].spec.max_episode_steps)
    os.makedirs(vid_dir, exist_ok=True)
    path = os.path.join(vid_dir, name)
    # One frame per decision plays back in real time at 20 fps.
    # frames_per_decision can keep more frames, up to one per physics step.
    repeat = canvas.action_repeat
    frames = int(getattr(learner, "frames_per_decision", 1))
    frame_steps = _frame_steps(repeat, frames)
    writer = imageio.get_writer(path, fps=int(round(len(frame_steps) / (canvas.dt * repeat))))
    try:
        decisions = 0
        while any(active) and decisions < limit:
            _write_decision(writer, learner, envs, states, active, canvas, overlay, forces, frame_steps)
            decisions += 1
    finally:
        writer.close()
    plants = [env.unwrapped for env in envs]
    score = sum(
        1 for i, plant in enumerate(plants)
        if active[i] and plant.success_frames * plant.dt >= plant.SUCCESS_TIME
    )
    stats = {
        "q": float(np.median(start_values)),
        "saturated": float(np.mean(np.abs(forces) > 24.0)) if forces else 0.0,
    }
    return score, stats


def _maybe_record(learner, episode_index):
    if learner.log_env and episode_index > 0 and episode_index % 250 == 0:
        score, stats = record_policy_video(learner, HOLDOUT_SEEDS, episode_index)
        full = len(HOLDOUT_SEEDS)
        learner.holdout_streak = (getattr(learner, "holdout_streak", 0) + 1) if score == full else 0
        print(
            f"held-out {score}/{full} at episode {episode_index}, perfect checks in a row {learner.holdout_streak}; "
            f"critic {stats['q']:.3f} (ceiling 1.000); actions past 24 N {stats['saturated']:.3f}"
        )
        if score > getattr(learner, "holdout_best", -1):
            learner.holdout_best = score
            learner.dump_best(episode_index, f"{score}/{full}")


def _record_final(learner):
    if learner.log_env:
        record_policy_video(learner, [HOLDOUT_SEEDS[0]], None)


def init_env(size=50, max_F=25.0, max_ang_vel=4*np.pi, mm=5.0, m1=0.5, l1=10.0, m2=0.5, l2=10.0, dt=0.01, theta_tol=np.pi/10, action_repeat=5, render_mode=None):
    max_episode_steps = int(max_seconds / (dt * action_repeat))
    env = gym.make("DoublePendulum/DoublePendulum-v0",
                   max_episode_steps=max_episode_steps,
                   size=size,
                   max_F=max_F,
                   max_ang_vel=max_ang_vel,
                   mm=mm,
                   m1=m1,
                   l1=l1,
                   m2=m2,
                   l2=l2,
                   dt=dt,
                   theta_tol=theta_tol,
                   action_repeat=action_repeat,
                   render_mode=render_mode
                  )
    return env

def polyak_update(online, target, tau):
    with torch.no_grad():
        torch._foreach_lerp_(list(target.parameters()), list(online.parameters()), 1 - tau)

def warmup_action(action_space):
    scale = float(action_space.high[0])
    action = np.random.normal(0.0, 0.2 * scale)
    return float(np.clip(action, action_space.low[0], action_space.high[0]))


def uniform_warmup_action(action_space):
    return float(np.random.uniform(action_space.low[0], action_space.high[0]))

class EarlyStop:
    """Stop once the deterministic policy scores every held-out seed on several checks in a row."""

    def __init__(self, checks_needed=3):
        self.checks_needed = checks_needed

    def check(self, durations, successes, actor_started, holdout_streak=0):
        # durations and successes come from the training episodes and no longer decide the stop.
        if actor_started and holdout_streak >= self.checks_needed:
            full = len(HOLDOUT_SEEDS)
            return f"solved: {holdout_streak} held-out checks in a row at {full}/{full}"
        return None

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# define transitions from state to state as a function of action
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

TransitionSequence = namedtuple('TransitionSequence',
                        ('state', 'action', 'reward'))

# A sampled batch is already joined: state and next_state are (batch, obs),
# action is (batch, act), reward is (batch,), and non_final marks rows with a next state.
ReplayBatch = namedtuple("ReplayBatch", ("state", "action", "next_state", "reward", "non_final"))


class ReplayMemory(object):
    """Ring buffer of transitions in one tensor block."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.position = 0
        self.size = 0

    def push(self, state, action, next_state, reward):
        if not hasattr(self, "states"):
            obs = int(state.reshape(-1).numel())
            act = int(action.reshape(-1).numel())
            store = state.device
            self.states = torch.empty(self.capacity, obs, device=store)
            self.actions = torch.empty(self.capacity, act, device=store)
            self.rewards = torch.empty(self.capacity, device=store)
            self.next_states = torch.empty(self.capacity, obs, device=store)
            self.non_final = torch.empty(self.capacity, dtype=torch.bool, device=store)
        index = self.position
        self.states[index] = state.detach().reshape(-1)
        self.actions[index] = action.detach().reshape(-1)
        self.rewards[index] = reward.detach().reshape(-1)[0]
        if next_state is None:
            self.non_final[index] = False
        else:
            self.non_final[index] = True
            self.next_states[index] = next_state.detach().reshape(-1)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        picks = random.sample(range(self.size), batch_size)
        index = torch.tensor(picks, device=self.states.device)
        return ReplayBatch(
            self.states[index],
            self.actions[index],
            self.next_states[index],
            self.rewards[index],
            self.non_final[index],
        )

    def __len__(self):
        return self.size

class ExperienceMemory(object):
    def __init__(self, capacity, sequence_length=80, sequence_overlap=40):
            self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a sequence of transitions"""
        self.memory.append([TransitionSequence(*a) for a in args])

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
