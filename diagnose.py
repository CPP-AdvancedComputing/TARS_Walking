"""
Quick diagnostic: runs a few episodes with random/zero/trained actions
and prints reward component breakdowns to find what's broken.
"""
import numpy as np
import mujoco
from tars_env import TARSEnv
from stable_baselines3 import PPO
import os

env = TARSEnv(r"C:\Users\anike\tars-urdf\tars_mjcf.xml")

def run_episode(env, policy="zero", max_steps=500):
    """Run one episode, track every reward component."""
    obs, _ = env.reset()
    
    totals = {
        "alive": 0, "velocity": 0, "upright": 0, "height": 0,
        "contact": 0, "smooth": 0, "energy": 0,
        "fall": 0, "total": 0
    }
    step_count = 0
    heights = []
    tilts = []
    
    for step in range(max_steps):
        if policy == "zero":
            action = np.zeros(12)
        elif policy == "random":
            action = env.action_space.sample()
        elif policy == "standing":
            # Just hold the standing pose
            action = np.zeros(12)
        else:
            action, _ = policy.predict(obs, deterministic=True)
        
        # Manually compute reward components (mirror the env's step logic)
        prev_action = env.prev_action.copy()
        
        obs, reward, terminated, truncated, _ = env.step(action)
        step_count += 1
        
        # Recompute components for logging
        height = env.data.body("internals").xpos[2]
        quat = env.data.body("internals").xquat
        tilt = 2 * np.arccos(np.clip(abs(quat[0]), 0, 1))
        fallen = tilt > 1.0 or height < -0.1  # old code check
        
        velocity_r = 3.0 * env.data.qvel[0]
        upright_r = 0.5 * quat[0] ** 2
        free_z = env.data.qpos[2]
        height_drop = env.initial_height - free_z
        height_r = 0.3 * max(1.0 - 10.0 * abs(height_drop), 0.0)
        
        n_feet = 0
        for j in range(4):
            gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"servo_l{j}_foot")
            if env.data.geom(gid).xpos[2] < 0.06:
                n_feet += 1
        contact_r = 0.25 * n_feet
        
        smooth_p = 0.005 * np.sum(np.square(action - prev_action))
        energy_p = 0.001 * np.sum(np.square(action))
        fall_p = -10.0 if (tilt > 1.0 or height_drop > 0.15) else 0.0
        
        totals["alive"] += 0.5
        totals["velocity"] += velocity_r
        totals["upright"] += upright_r
        totals["height"] += height_r
        totals["contact"] += contact_r
        totals["smooth"] += smooth_p
        totals["energy"] += energy_p
        totals["fall"] += fall_p
        totals["total"] += reward
        
        heights.append(height)
        tilts.append(np.degrees(tilt))
        
        if terminated or truncated:
            break
    
    return step_count, totals, heights, tilts

print("=" * 70)
print("TARS REWARD DIAGNOSTIC")
print("=" * 70)

# 1. Zero action (standing still)
print("\n--- ZERO ACTION (do nothing) ---")
steps, t, heights, tilts = run_episode(env, "zero", 500)
print(f"  Survived: {steps} steps")
print(f"  Height: start={heights[0]:.3f}, end={heights[-1]:.3f}, min={min(heights):.3f}")
print(f"  Tilt: max={max(tilts):.1f} deg")
print(f"  Reward breakdown (totals over episode):")
for k, v in t.items():
    print(f"    {k:>12}: {v:>10.3f}  (avg/step: {v/steps:>8.4f})")

# 2. Random actions
print("\n--- RANDOM ACTIONS ---")
steps, t, heights, tilts = run_episode(env, "random", 500)
print(f"  Survived: {steps} steps")
print(f"  Height: start={heights[0]:.3f}, end={heights[-1]:.3f}, min={min(heights):.3f}")
print(f"  Tilt: max={max(tilts):.1f} deg")
print(f"  Reward breakdown (totals over episode):")
for k, v in t.items():
    print(f"    {k:>12}: {v:>10.3f}  (avg/step: {v/steps:>8.4f})")

# 3. Trained model (if exists)
if os.path.exists("tars_policy.zip"):
    print("\n--- TRAINED MODEL ---")
    model = PPO.load("tars_policy", env=env)
    steps, t, heights, tilts = run_episode(env, model, 500)
    print(f"  Survived: {steps} steps")
    print(f"  Height: start={heights[0]:.3f}, end={heights[-1]:.3f}, min={min(heights):.3f}")
    print(f"  Tilt: max={max(tilts):.1f} deg")
    print(f"  Reward breakdown (totals over episode):")
    for k, v in t.items():
        print(f"    {k:>12}: {v:>10.3f}  (avg/step: {v/steps:>8.4f})")

# 4. Check some physics sanity
print("\n--- PHYSICS SANITY CHECK ---")
obs, _ = env.reset()
print(f"  Initial height (internals body): {env.initial_height:.4f}")
print(f"  Spawn height offset: {env.spawn_height:.4f}")

# Check joint ranges actually work
for i in range(4):
    sp = env.data.joint(f"shoulder_prismatic_l{i}").qpos[0]
    hp = env.data.joint(f"hip_revolute_l{i}").qpos[0]
    print(f"  Leg {i}: shoulder={sp:.4f}, hip={hp:.4f}")

foot_heights = []
for i in range(4):
    gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"servo_l{i}_foot")
    fz = env.data.geom(gid).xpos[2]
    foot_heights.append(fz)
    print(f"  Foot {i} height: {fz:.4f}")

print(f"\n  Feet on ground? (expect ~0.04): {['YES' if abs(h-0.04) < 0.05 else 'NO' for h in foot_heights]}")

# 5. Check what shoulder prismatics actually do
print("\n--- SHOULDER PRISMATIC RANGE TEST ---")
obs, _ = env.reset()
# Push shoulder 0 to max
action = np.zeros(12)
action[0] = 1.0  # max shoulder_prismatic_l0
for _ in range(50):
    obs, _, _, _, _ = env.step(action)

sp0 = env.data.joint("shoulder_prismatic_l0").qpos[0]
gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "servo_l0_foot")
fz = env.data.geom(gid).xpos[2]
print(f"  After maxing shoulder_l0: joint={sp0:.4f}, foot_z={fz:.4f}")
print(f"  (shoulder range is -0.15 to 0.15 mapped from action -1 to 1)")

print("\n" + "=" * 70)
print("KEY QUESTIONS:")
print("  1. Does 'do nothing' survive 500 steps? (basic stability)")
print("  2. Is gait reward dominating? (lift/plant >> velocity)")
print("  3. Are shoulder prismatics actually lifting feet?")
print("  4. Is forward velocity ever positive?")
print("=" * 70)

