"""Train 50k steps then diagnose what the policy actually does."""
import numpy as np
from stable_baselines3 import PPO
from tars_env import TARSEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

URDF = r"C:\Users\anike\tars-urdf\tars_mjcf.xml"

print("Training 50k steps...")
env = DummyVecEnv([lambda: Monitor(TARSEnv(URDF))])
model = PPO("MlpPolicy", env, n_steps=2048, batch_size=128,
            learning_rate=3e-4, ent_coef=0.01, verbose=0)
model.learn(50_000)
print("Done training.\n")

test_env = TARSEnv(URDF)

# 3 episodes summary
for ep in range(3):
    obs, _ = test_env.reset()
    total_r = 0
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = test_env.step(action)
        total_r += r
        if term or trunc:
            break
    x = test_env.data.qpos[0]
    y = test_env.data.qpos[1]
    print(f"Ep{ep}: steps={step+1:4d}  reward={total_r:7.1f}  "
          f"x={x:+.4f}m  y={y:+.4f}m  term={term}")

# Detailed step trace
print("\n--- Step-by-step trace ---")
obs, _ = test_env.reset()
for step in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, _ = test_env.step(action)
    if step % 25 == 0:
        x = test_env.data.qpos[0]
        tilt = 2 * np.arccos(np.clip(abs(test_env.data.body("internals").xquat[0]), 0, 1))
        hips = [test_env.data.joint(f"hip_revolute_l{i}").qpos[0] for i in range(4)]
        knees = [test_env.data.joint(f"knee_prismatic_l{i}").qpos[0] for i in range(4)]
        shoulders = [test_env.data.joint(f"shoulder_prismatic_l{i}").qpos[0] for i in range(4)]
        print(f"  t={step:3d}  x={x:+.5f}  tilt={tilt:.2f}  "
              f"hips=[{hips[0]:+.2f},{hips[1]:+.2f},{hips[2]:+.2f},{hips[3]:+.2f}]  "
              f"knees=[{knees[0]:+.2f},{knees[1]:+.2f},{knees[2]:+.2f},{knees[3]:+.2f}]  "
              f"shoulders=[{shoulders[0]:+.3f},{shoulders[1]:+.3f},{shoulders[2]:+.3f},{shoulders[3]:+.3f}]")
    if term or trunc:
        print(f"  ENDED at step {step+1}, terminated={term}")
        break

if step == 299:
    print(f"  Ran full 300 steps, final x={test_env.data.qpos[0]:+.5f}")

