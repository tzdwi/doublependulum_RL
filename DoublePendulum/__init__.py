from gymnasium.envs.registration import register

max_seconds = 60 #1 minute

register(
    id="DoublePendulum/DoublePendulum-v0",
    entry_point="DoublePendulum.envs:DoublePendulumEnv",
	max_episode_steps=int(max_seconds / 0.03)
)
