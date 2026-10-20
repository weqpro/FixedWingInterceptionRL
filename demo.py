import jsbsim_env  # noqa: F401
from jsbsim_env.envs import InterceptEnv

env = InterceptEnv(render_mode="human")
obs, _ = env.reset(seed=0)
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
print(info["end_reason"], info["distance_m"])
env.close()
