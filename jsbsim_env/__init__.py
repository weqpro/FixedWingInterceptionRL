from gymnasium.envs.registration import register

# Truncation is handled inside the env (InterceptConfig.max_episode_steps).
register(
    id="jsbsim_env/Intercept-v0",
    entry_point="jsbsim_env.envs:InterceptEnv",
)
