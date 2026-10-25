"""Training script for PPO on InterceptEnv using Stable-Baselines3 with Observation Scaling & Reward Shaping."""

from pathlib import Path
import math
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import jsbsim_env  # noqa: F401
from jsbsim_env.envs import InterceptEnv


class PPOTrainingWrapper(gym.Wrapper):
    """
    A Gymnasium wrapper to scale observations (Step 1) and shape rewards (Step 3) 
    solely during training, preventing baseline and evaluation breakage.
    """

    def __init__(self, env):
        super().__init__(env)
        # We manually apply specific scale adjustments so observations reside roughly in [-1, 1]
        self.scales = np.array([
            1.0,   # roll rad (approx [-1, 1])
            1.0,   # pitch rad (approx [-1, 1])
            1.0,   # sin_psi
            1.0,   # cos_psi
            1.0,   # p rad/s
            1.0,   # q rad/s
            1.0,   # r rad/s
            0.02,  # airspeed m/s (~50 m/s -> ~1.0)
            0.001, # height AGL m (approx [0, 1])
            0.05,  # vertical speed m/s
            0.001, # target forward m (up to 2000m -> ~2.0)
            0.001, # target right m (up to 500m -> ~0.5)
            0.001  # target down m
        ], dtype=np.float32)

    def _scale_obs(self, obs: np.ndarray) -> np.ndarray:
        return obs * self.scales

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._scale_obs(obs), info

    def step(self, action):
        # Action is expected in [-1.0, 1.0]
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Scale observation (Step 1)
        scaled_obs = self._scale_obs(obs)

        # Shape the Reward Function (Step 3)
        # Base reward already has progress_scale * (distance closed this step)
        shaped_reward = float(reward)

        # Extract relative offsets to target
        fwd = float(obs[10])   # target forward
        right = float(obs[11]) # target right
        down = float(obs[12])  # target down
        
        # Pointing Alignment Reward (Step 3)
        # High score when target is directly ahead (maximizes fwd / dist)
        dist = math.sqrt(fwd**2 + right**2 + down**2)
        if dist > 1e-3:
            alignment = fwd / dist  # Range from -1.0 (behind) to 1.0 (dead ahead)
            # Significant directional guidance
            shaped_reward += 0.08 * alignment
            
            # Slightly penalize being offset to encourage clean pointing
            offset_penalty = math.hypot(right, down) / dist
            shaped_reward -= 0.02 * offset_penalty

        # Minimize control surface chatter
        shaped_reward -= 0.005 * float(np.sum(np.abs(action)))

        return scaled_obs, shaped_reward, terminated, truncated, info


def train():
    models_dir = Path("models")
    logs_dir = Path("tensorboard_logs")
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # Instantiate, wrap for Monitor statistics, and add the feature scale/reward shaping wrapper
    raw_env = InterceptEnv()
    wrapped_env = PPOTrainingWrapper(raw_env)
    env = Monitor(wrapped_env)

    # Checkpoint callback to save intermediate models
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(models_dir),
        name_prefix="ppo_intercept_chkpt",
    )

    print("--- Initializing PPO Agent with Scaled Features & Shaped Rewards ---")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        tensorboard_log=str(logs_dir),
    )

    # Increased training steps (Step 2)
    total_timesteps = 1_200_000
    print(f"--- Starting Training for {total_timesteps} steps ---")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    final_model_path = models_dir / "ppo_intercept"
    model.save(final_model_path)
    print(f"\n--- Training Complete! Model saved to {final_model_path}.zip ---")

    env.close()


if __name__ == "__main__":
    train()