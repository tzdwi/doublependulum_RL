from DoublePendulum.envs.doublependulumworld import DoublePendulumEnv


def make_env(env_kwargs):
    """Build one pendulum env. Kept here so worker processes do not import torch."""
    import gymnasium as gym
    from DoublePendulum import max_seconds

    kwargs = dict(env_kwargs)
    dt = kwargs.get("dt", 0.03)
    action_repeat = kwargs.get("action_repeat", 3)
    max_episode_steps = kwargs.pop(
        "max_episode_steps", int(max_seconds / (dt * action_repeat))
    )
    return gym.make(
        "DoublePendulum/DoublePendulum-v0",
        max_episode_steps=max_episode_steps,
        **kwargs,
    )
