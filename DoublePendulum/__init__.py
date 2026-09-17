from gymnasium.envs.registration import register

max_seconds = 300 #5 minutes

register(
    id="DoublePendulum/DoublePendulum-v0",
    entry_point="DoublePendulum.envs:DoublePendulumEnv",
	max_episode_steps=int(300 / 0.03)
)
