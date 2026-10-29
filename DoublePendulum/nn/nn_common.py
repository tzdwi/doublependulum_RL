"""
Some common tools across the RL algorithms I want to play with
"""

import gymnasium as gym
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
        observations, _ = vec.reset(seed=seeds)
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
            _maybe_record(learner, index, num_episodes)
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
                    )
                    if early.improved or (stop or "").startswith("solved"):
                        learner.dump_best(len(learner.episode_durations), early.best)
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
                    _maybe_record(learner, next_index, num_episodes)
                    next_index += 1
                    step_index[i] = 0
                    returns[i] = 0.0
                reset_obs, _ = vec.reset(
                    seed=reset_seeds, options={"reset_mask": mask}
                )
                for i in finished:
                    states[i] = torch.tensor(
                        reset_obs[i], dtype=torch.float32, device=device
                    ).unsqueeze(0)
        if stop:
            print(stop)
    finally:
        if bar is not None:
            bar.close()
        vec.close()
        learner.vec = None


def _maybe_record(learner, episode_index, num_episodes):
    if learner.log_env and episode_index > 0 and (
        (np.log2(episode_index) % 2 == 0) or episode_index == num_episodes - 1
    ):
        learner.run_sim(episode_index)


def init_env(size=50, max_F=25.0, max_ang_vel=4*np.pi, mm=5.0, m1=0.5, l1=10.0, m2=0.5, l2=10.0, dt=0.03, theta_tol=np.pi/10, action_repeat=3, render_mode=None):
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

class EarlyStop:
    """Stop training once the task is solved, or once episode length stops improving."""

    def __init__(self, window=100, solved_count=95, patience=15000, min_gain=2):
        self.window = window
        self.solved_count = solved_count
        self.patience = patience
        self.min_gain = min_gain
        self.best = None
        self.since = 0
        self.hold = 0
        self.improved = False

    def reset(self):
        # Keep the all-time best so a post-reset dip cannot replace its checkpoint.
        # Restart the stall timer, and ignore windows that still contain pre-reset episodes.
        self.since = 0
        self.hold = self.window
        self.improved = False

    def check(self, durations, successes, actor_started):
        self.improved = False
        if len(durations) < self.window or not actor_started:
            return None
        if self.hold:
            self.hold -= 1
            return None
        recent = durations[-self.window:]
        recent_len = float(np.mean(recent))
        recent_succ = int(sum(successes[-self.window:]))
        if recent_succ >= self.solved_count:
            return f"solved: {recent_succ}/{self.window} of the last episodes succeeded"
        if self.best is None or recent_len > self.best + self.min_gain:
            self.best = recent_len
            self.since = 0
            self.improved = True
        else:
            self.since += 1
            if self.since >= self.patience:
                return (f"stalled: mean length {recent_len:.0f} has not beaten {self.best:.0f} "
                        f"by {self.min_gain} for {self.patience} episodes")
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

# hold transitions in memory
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

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
