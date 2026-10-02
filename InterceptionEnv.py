import gymnasium as gym
from gymnasium import spaces
import numpy as np
import gym_jsbsim

class InterceptionEnv(gym.Env):
    """
    Custom Gymnasium environment wrapper for Fixed-Wing Aircraft Target Interception using JSBSim.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # 1. Initialize the underlying JSBSim environment (e.g., using a standard aircraft config)
        self.jsbsim_env = gym.make('JSBSim-HeadingControlTask-F15-Shaping.EXTRA-FG-v0')
        
        # 2. Define Action Space: continuous values for [aileron, elevator, rudder, throttle]
        # Normalized typically between -1.0 and 1.0
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # 3. Define Observation Space: aircraft telemetry + relative target state
        # (e.g., velocities, orientation angles, distance & relative angles to the moving target)
        # Let's assume an observation vector of size 10 (placeholder shape)
        obs_low = -np.inf * np.ones(10, dtype=np.float32)
        obs_high = np.inf * np.ones(10, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        """Resets the environment and the moving target position at the start of an episode."""
        super().reset(seed=seed)
        
        # Reset the JSBSim simulation
        raw_obs = self.jsbsim_env.reset()
        
        # TODO: Initialize your procedural moving target generator here (linear/sinusoidal path)
        self.target_position = np.array([1000.0, 1000.0, 500.0]) # Example initial target coordinates
        
        # Combine aircraft state and target relative state into your observation vector
        observation = self._get_observation(raw_obs)
        info = {}
        
        return observation, info

    def step(self, action):
        """Executes one simulation step given the agent's control actions."""
        # 1. Apply action to JSBSim
        raw_obs, base_reward, terminated, truncated, info = self.jsbsim_env.step(action)
        
        # 2. Update moving target position based on its trajectory function
        # self._update_target_position()
        
        # 3. Calculate custom Reward Shaping for interception
        reward = self._compute_reward(raw_obs)
        
        # 4. Check early termination rules (e.g., ground collision, critical stall angles)
        terminated = self._check_failure_conditions(raw_obs)
        
        observation = self._get_observation(raw_obs)
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self, raw_obs):
        """Extracts and formats telemetry and relative target metrics into the observation array."""
        # Merge aircraft state from JSBSim with relative distance/vector to the target
        return np.zeros(10, dtype=np.float32) # Placeholder

    def _compute_reward(self, raw_obs, action, prev_action):
        """
        computes reward based on interception progress and safety of flying
        """
        reward = 0.0
        # 1. Компонент відстані до цілі (Distance Shaping)
        # Рахуємо поточну відстань між літаком і ціллю
        current_distance = np.linalg.norm(self.target_position - self.current_plane_position)
        
        # Якщо ми маємо попередню відстань, заохочуємо її зменшення (прогрес)
        if hasattr(self, 'previous_distance'):
            distance_delta = self.previous_distance - current_distance
            reward += distance_delta * 10.0 # Надаємо вагу наближенню до цілі
            
        # Бонус за успішне перехоплення (якщо підійшли ближче ніж на 50 метрів)
        INTERCEPTION_THRESHOLD = 50.0
        if current_distance < INTERCEPTION_THRESHOLD:
            reward += 200.0 # Великий бонус за перемогу в епізоді
            
        # 2. Штраф за різкість керування (Control Smoothness / Jerk Penalty)
        # Змушує агента керувати плавно, без хаотичних ривків рулями
        action_smoothness_penalty = np.sum(np.square(action - prev_action))
        reward -= 0.1 * action_smoothness_penalty
        
        # 3. Штраф за небезпечну поведінку (Safety / Ground Crash Penalty)
        # Наприклад, якщо висота падає нижче критичного рівня
        altitude = raw_obs[2] # Припустимо, висота на 3-му індексі
        if altitude < 50.0:
            reward -= 50.0
            
        # Зберігаємо поточну відстань для наступного кроку
        self.previous_distance = current_distance
        
        return reward

    def _check_failure_conditions(self, raw_obs):
        """Checks if the episode should terminate early (ground crash, extreme pitch/roll, stall)."""
        # TODO: Implement safety boundaries
        return False

    def close(self):
        self.jsbsim_env.close()