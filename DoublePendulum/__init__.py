from gymnasium.envs.registration import register

register(
    id="DoublePendulum/GridWorld-v0",
    entry_point="DoublePendulum.envs:GridWorldEnv",
)
