import mujoco
import numpy as np
from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR
from stable_baselines3 import PPO

env = TARSEnv(DEFAULT_MODEL_PATH_STR)
model = PPO.load("tars_policy.zip", env=env)

obs, _ = env.reset()
x_start = env.data.qpos[0]

for step in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    
    if step % 20 == 0:
        x = env.data.qpos[0]
        foot_contacts = env._foot_on_ground_flags()
        print(f"Step {step:3d}: x={x:.4f} (Δ={x-x_start:.4f}) "
              f"contacts={[int(c) for c in foot_contacts]} "
              f"phase={env.phase}")
    
    if terminated or truncated:
        print(f"Episode ended at step {step}")
        break

print(f"Total forward movement: {env.data.qpos[0] - x_start:.4f}m")
