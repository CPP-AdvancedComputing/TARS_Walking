import numpy as np
from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR
from stable_baselines3 import PPO

env = TARSEnv(DEFAULT_MODEL_PATH_STR)
model = PPO.load("tars_policy.zip", env=env)

# Run 3 episodes, track x movement during lift vs plant
for ep in range(3):
    obs, _ = env.reset()
    x_start = env.data.qpos[0]
    
    lift_dx = 0.0  # x gained while foot_2 is off ground
    plant_dx = 0.0  # x gained while foot_2 is on ground
    lift_steps = 0
    plant_steps = 0
    prev_x = x_start
    
    for step in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        x = env.data.qpos[0]
        dx = x - prev_x
        contacts = env._foot_on_ground_flags()
        
        if not contacts[2]:  # foot_2 lifted
            lift_dx += dx
            lift_steps += 1
        else:
            plant_dx += dx
            plant_steps += 1
        
        prev_x = x
        if terminated or truncated:
            break
    
    total = env.data.qpos[0] - x_start
    print(f"Episode {ep}: total_dx={total:+.4f}m  "
          f"lift_dx={lift_dx:+.5f} ({lift_steps} steps)  "
          f"plant_dx={plant_dx:+.5f} ({plant_steps} steps)")
    if lift_steps > 0:
        print(f"  Mean dx when foot_2 lifted: {lift_dx/lift_steps:+.6f} m/step")
    if plant_steps > 0:
        print(f"  Mean dx when foot_2 planted: {plant_dx/plant_steps:+.6f} m/step")
    print()
