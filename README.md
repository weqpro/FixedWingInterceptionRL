# Fixed-Wing Interception RL

**Online adaptation of an RL policy for intercepting a target with a fixed-wing aircraft**

A reproducible reinforcement learning pipeline built on the [JSBSim](https://github.com/JSBSim-Team/jsbsim) flight dynamics engine. A Cessna 172 has to fly into a sphere around a target before time runs out or it hits the ground. The long-term goal is a policy that keeps adapting *during flight* to conditions it never saw in training, such as wind or model errors.

> Semester project for the *Artificial Intelligence* course, Faculty of Applied Sciences, Ukrainian Catholic University (2026).

---

## Motivation

Intercepting a target with a fixed-wing aircraft is a hard continuous-control problem. The dynamics are nonlinear and underactuated, the control axes are aerodynamically coupled, and the aircraft can stall. Most RL approaches train a policy offline and then freeze it, so performance drops as soon as real flight conditions differ from the training ones.

We take the *Learning on the Fly* framework \[1\], originally developed for quadrotors, and adapt it to:

- **a different platform:** a fixed-wing aircraft instead of a quadrotor;
- **a different task:** intercepting a target instead of tracking a trajectory;
- **a different simulator:** the aerodynamic engine JSBSim.

## Project status

The project is at the **pre-midterm stage**. The current goal is a working simulation environment, the first baselines and an initial benchmark.

| Component | Status |
| --- | --- |
| JSBSim setup and aircraft model check | Done |
| Interception environment (Gymnasium API, spaces, reward) | Done |
| Evaluation metrics (SR, TTI, miss distance, smoothness) | Done |
| Interactive 3D visualization (static and animated) | Done |
| Trivial baselines (zero and random policies) | Done |
| Moving target generator | In progress |
| Pure Pursuit baseline | Done |
| Offline PPO baseline | Done |
| First benchmark | In progress |

> **Note:** the target is currently **stationary**. Maneuvering targets, wind disturbances and online adaptation come after the midterm.

---

## Installation

Requires **Python 3.11+**.

```bash
git clone <REPO_URL>
cd FixedWingInterceptionRL

pip install -e .            # core dependencies
pip install -e ".[dev]"     # + pytest, ruff, pre-commit
```

Core dependencies: `gymnasium`, `jsbsim`, `numpy`, `plotly`, `tensorboard`, `stable-baselines3`. The JSBSim Python package ships with its aircraft models, so no extra downloads are needed.

## Quick start

### Run the demo

```bash
python demo_p.py
```

The demo does two things:

1. Evaluates every enabled policy on 20 fixed scenarios and prints the metrics.
2. Flies one episode and saves two interactive 3D views, `flight.html` (static trajectory) and `flight_live.html` (real-time animation). Both open in your browser automatically.

To choose which policy is shown in 3D, change `DEMO_POLICY` at the top of `demo_p.py`.

If `models/ppo_intercept.zip` exists, the demo automatically evaluates the PPO policy alongside the baselines and uses it for the 3D flight.

### Compare baselines

```bash
python -m jsbsim_env.evaluate_p
```

### Train PPO

```bash
python train_ppo.py
```

The model is saved to `models/ppo_intercept.zip`. Intermediate checkpoints are saved every 100 000 steps under `models/`. Training progress is logged to `tensorboard_logs/`:

```bash
tensorboard --logdir tensorboard_logs
```

### Use the environment directly

Importing `jsbsim_env` registers the environment with Gymnasium:

```python
import gymnasium as gym
import jsbsim_env  # registers "jsbsim_env/Intercept-v0"

env = gym.make("jsbsim_env/Intercept-v0")
obs, info = env.reset(seed=42)

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(info["end_reason"] or "timeout", f"{info['distance_m']:.0f} m to target")
env.close()
```

---

## Environment

### Scenario

Each episode starts with a **Cessna 172** (`c172x`) in trimmed level flight at 3000 ft and 100 kts, heading east. The start point is over Lviv (49.8°N, 24.0°E). The target spawns at a random offset from the start position:

| Direction | Range |
| --- | --- |
| Forward | 800 to 2000 m |
| Right | −500 to 500 m |
| Down | −100 to 100 m |

The episode is a **success** once the aircraft gets within **50 m** of the target. This is checked on every simulator step, so even a fast fly-through counts.

A fresh JSBSim instance is created and trimmed for every episode. This keeps state from leaking between episodes and makes resets deterministic for a given seed.

### Timing

- Physics runs at **120 Hz**.
- The agent acts every 12 simulator steps, i.e. at **10 Hz** (0.1 s per action).
- An episode is truncated after **600 agent steps**, which is 60 s of flight.

### Action space

`Box(-1, 1, shape=(2,))`

| Index | Control | Meaning |
| --- | --- | --- |
| 0 | Aileron | Offset from the trimmed aileron position |
| 1 | Elevator | Elevator command on top of the pitch trim |

A zero action keeps the aircraft exactly in its trimmed state. Rudder and throttle are held at their trim values. Out-of-range actions are clipped.

### Observation space

`Box(-inf, inf, shape=(13,))`, float32, SI units.

| Index | Quantity | Unit |
| --- | --- | --- |
| 0 | Roll | rad |
| 1 | Pitch | rad |
| 2–3 | sin / cos of heading | — |
| 4–6 | Roll, pitch, yaw rates (p, q, r) | rad/s |
| 7 | True airspeed | m/s |
| 8 | Height above ground | m |
| 9 | Vertical speed (+ = climbing) | m/s |
| 10–12 | Target offset: forward, right, down | m |

Heading is encoded as sin/cos to avoid the jump from 2π to 0. The target offset is expressed in the aircraft's **heading frame**, so the policy does not depend on the absolute world heading.

### Reward

```
reward = 0.01 × (metres closed toward the target this step)
       + 10   on success
       − 10   on crash, numerical error or simulator failure
```

The dense progress term rewards getting closer, and the terminal terms make the outcome of the episode matter.

### Episode termination

| `end_reason` | Condition | Type |
| --- | --- | --- |
| `success` | Within 50 m of the target | terminated |
| `crash` | Below 15 m above ground | terminated |
| `numerical_error` | Non-finite state | terminated |
| `sim_failure` | JSBSim stopped running | terminated |
| `None` | 600 agent steps reached | truncated |

### Info dictionary

| Key | Description |
| --- | --- |
| `distance_m` | Current distance to the target |
| `end_reason` | `EndReason`, or `None` while running or on truncation |
| `is_success` | `True` if the target was reached |
| `sim_time_s` | Simulation time |
| `aircraft_ned_m` | Aircraft position (north, east, down) relative to the origin |
| `target_ned_m` | Target position (north, east, down) relative to the origin |

### Configuration

All parameters live in the frozen dataclass `InterceptConfig`, which validates them on creation:

```python
from jsbsim_env.envs import InterceptConfig, InterceptEnv

config = InterceptConfig(
    success_radius_m=30.0,
    target_forward_m=(1000.0, 3000.0),
    max_episode_steps=900,
)
env = InterceptEnv(config)
# or: gym.make("jsbsim_env/Intercept-v0", config=config)
```

You can also place the target at a fixed offset for a single episode:

```python
obs, info = env.reset(options={"target_offset_m": (600.0, 50.0, 0.0)})  # forward, right, down
```

---

## Evaluation

`evaluate_policy` runs one episode per seed. The same seeds produce the same scenarios, so every policy is tested under identical conditions. A policy is any function `obs -> action`:

```python
from jsbsim_env.evaluate_p import evaluate_policy

def my_policy(obs):
    return [0.0, -0.1]  # aileron, elevator

metrics = evaluate_policy(my_policy, seeds=range(20))
print(metrics["success_rate"])
```

### Metrics

| Metric | Definition | Better |
| --- | --- | --- |
| **Success Rate (SR)** | Fraction of episodes that reached the target | ↑ |
| **Time-to-Intercept (TTI)** | Mean simulation time to reach the target, over successful episodes only | ↓ |
| **Miss Distance** | Mean closest approach to the target, over failed episodes only | ↓ |
| **Control Smoothness** | Mean \|aₜ − aₜ₋₁\| across both control channels; penalizes high-frequency control surface oscillation | ↓ |

## Baselines

### Pure Pursuit

A geometric guidance law that steers the aircraft toward the target at all times by minimising the Line-of-Sight (LOS) angle error. Both channels use proportional-derivative (PD) control.

**Lateral channel (bank-to-turn)**

The bearing to the target in the aircraft's heading frame is `atan2(right, forward)`. A desired roll angle is computed by scaling the bearing and clamping to ±35°. A PD controller then drives the aileron:

```
desired_roll = clip(k_roll × bearing,  ±max_bank)
aileron      = kp_roll × (desired_roll − roll) − kd_roll × p
```

**Longitudinal channel**

The elevation angle to the target is `atan2(−down, horizontal_distance)`. A PD controller drives the elevator. Note that in JSBSim a **negative** elevator command raises the nose:

```
desired_pitch = atan2(−down, horizontal)
elevator      = −( kp_pitch × (desired_pitch − pitch) − kd_pitch × q )
```

**Tuned gains**

| Parameter | Value | Role |
| --- | --- | --- |
| `k_roll` | 1.0 | LOS bearing → desired roll |
| `max_bank_deg` | 35° | Maximum bank angle |
| `kp_roll` | 2.0 | Roll proportional gain |
| `kd_roll` | 0.5 | Roll derivative gain (damps via roll rate p) |
| `kp_pitch` | 3.0 | Pitch proportional gain |
| `kd_pitch` | 0.5 | Pitch derivative gain (damps via pitch rate q) |

### PPO

Offline Proximal Policy Optimisation using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

**Architecture:** default SB3 `MlpPolicy` — two hidden layers of 64 units each with `tanh` activations and a shared actor-critic backbone.

**Observation scaling**

A `PPOTrainingWrapper` scales raw observations before they reach the policy, bringing most inputs into roughly `[−1, 1]`:

| Indices | Quantity | Scale |
| --- | --- | --- |
| 0–6 | roll, pitch, sin/cos ψ, p, q, r | ×1.0 |
| 7 | airspeed | ×0.02 |
| 8 | height AGL | ×0.001 |
| 9 | vertical speed | ×0.05 |
| 10–12 | target forward / right / down | ×0.001 |

The same scales are applied at inference time in `make_ppo_policy`, so the loaded model always sees identically scaled inputs.

**Reward shaping** (training only, applied on top of the base environment reward)

| Term | Formula | Weight |
| --- | --- | --- |
| Alignment bonus | `forward / distance` | +0.08 |
| Offset penalty | `hypot(right, down) / distance` | −0.02 |
| Control penalty | `Σ\|action\|` | −0.005 |

The alignment bonus rewards pointing directly at the target; the offset penalty discourages lateral/vertical misalignment; the control penalty suppresses chattery surface inputs.

**Hyperparameters**

| Parameter | Value |
| --- | --- |
| Learning rate | 3 × 10⁻⁴ |
| n\_steps | 2 048 |
| Batch size | 64 |
| n\_epochs | 10 |
| γ | 0.99 |
| GAE λ | 0.95 |
| Clip range | 0.2 |
| Entropy coef | 0.005 |
| Total timesteps | 1 200 000 |

Checkpoints are saved every 100 000 steps to `models/`. See the **Quick start** section for the training command.

## Visualization

`jsbsim_env/visualize.py` is a small standalone Plotly library. It runs no simulation and works with any trajectories you pass in:

- **`plot_paths`** draws a static 3D scene. Trajectories are lines with a marker at the start, and points such as targets are drawn as markers.
- **`animate_paths`** plays the flight back in real time, with play (1× and 0.25×), pause and a time slider.

Both functions accept NED or ENU coordinates, display East / North / Up in metres, and write a standalone HTML file.

> Plotly is loaded from a CDN, so viewing the HTML files requires an internet connection.

\<!-- TODO: add a screenshot or GIF of flight\_live.html -->

---

## Project structure

```
FixedWingInterceptionRL/
├── jsbsim_env/
│   ├── __init__.py          # registers jsbsim_env/Intercept-v0
│   ├── envs/
│   │   ├── __init__.py
│   │   └── intercept.py     # InterceptEnv, InterceptConfig, EndReason
│   ├── evaluate_p.py        # metrics and baseline policies
│   ├── pure_pursuit.py      # geometric guidance baseline
│   ├── ppo_policy.py        # SB3 PPO inference wrapper
│   └── visualize.py         # 3D plots and animations (Plotly)
├── tests/
│   └── test_intercept_env.py
├── demo_p.py                # evaluation + 3D demo flight
├── train_ppo.py             # PPO training script
├── pyproject.toml
└── .pre-commit-config.yaml
```

## Development

```bash
pip install -e ".[dev]"
pre-commit install   # run the checks on every commit

pytest               # tests
ruff check .         # lint
ruff format .        # format
```

---

## Team

| Member | Role | Responsibilities |
| --- | --- | --- |
| **Stanislav Konovalenko** | Simulation and environment | JSBSim setup and aircraft model stability, moving target generator (linear and sinusoidal trajectories), early termination rules (ground collision, critical angles) |
| **Roman Pempus** | RL logic, PPO and baseline | Action and observation spaces, reward shaping, Gymnasium wrapper, offline PPO training, Pure Pursuit baseline |
| **Bohdan Pavlunyshyn** | Infrastructure, metrics and visualization | Repository structure and configuration, evaluation metrics and TensorBoard / W&B tracking, flight trajectory visualization |

**Mentor:** Vladyslav Humennyi

## References

1. J. Pan, J. Xing, R. Reiter, Y. Zhai, E. Aljalbout, D. Scaramuzza. *Learning on the Fly: Rapid Policy Adaptation via Differentiable Simulation.* 2025. [arXiv:2508.21065](https://arxiv.org/abs/2508.21065)
2. J. Xing, Z. Zhu, Y. Ren, I. Geles, Y. Zhai, R. Reiter, D. Scaramuzza. *Continual Robot Policy Learning via Variational Neural Dynamics.* 2026. [arXiv:2606.27353](https://arxiv.org/abs/2606.27353)
3. E. Bøhn, E. M. Coates, S. Moe, T. A. Johansen. *Deep Reinforcement Learning Attitude Control of Fixed-Wing UAVs Using Proximal Policy Optimization.* ICUAS 2019. [arXiv:1911.05478](https://arxiv.org/abs/1911.05478)
4. A. P. Pope et al. *Hierarchical Reinforcement Learning for Air-to-Air Combat.* 2021. [arXiv:2105.00990](https://arxiv.org/abs/2105.00990)

## License

`TBD`

This project uses [JSBSim](https://github.com/JSBSim-Team/jsbsim), which is licensed under GNU LGPL v2.1.
