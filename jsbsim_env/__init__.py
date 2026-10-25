import gymnasium as gym
from jsbsim_env.envs.intercept import InterceptConfig, InterceptEnv

gym.register(
    id="jsbsim_env/Intercept-v0",
    entry_point="jsbsim_env.envs.intercept:InterceptEnv",
)

__all__ = ["InterceptConfig", "InterceptEnv"]
