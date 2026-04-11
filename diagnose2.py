"""Quick diagnostic: watch what happens during one episode step-by-step."""
import numpy as np
import mujoco
from tars_env import TARSEnv
from stable_baselines3 import PPO
import os

env = TARSEnv(r"C:\Users\anike\tars-urdf\tars_mjcf.xml")

# Use trained model if available, otherwise zero actions
if os.path.exists("tars_policy.zip"):
    model = PPO.load("tars_policy", env=env)
    use_model = True
    print("Using trained model")
else:
    use_model = False
    print("Using zero actions")

obs, _ = env.reset()
print(f"Spawn qpos[2]: {env.data.qpos[2]:.4f}")
print(f"Initial height (qpos[2]): {env.initial_height:.4f}")
print(f"Internals body pos: {env.data.body('internals').xpos}")

print(f"\n{'step':>5} {'qpos_z':>8} {'body_z':>8} {'tilt_deg':>8} {'yaw_deg':>8} "
      f"{'vx':>8} {'vy':>8} {'vz':>8} {'wyaw':>8} {'reward':>8} {'foot_zs':>30}")

for step in range(300):
    if use_model:
        action, _ = model.predict(obs, deterministic=True)
    else:
        action = np.zeros(12)
    
    obs, reward, terminated, truncated, _ = env.step(action)
    
    qpos_z = env.data.qpos[2]
    body_z = env.data.body("internals").xpos[2]
    quat = env.data.body("internals").xquat
    
    # Tilt angle
    tilt = 2 * np.arccos(np.clip(abs(quat[0]), 0, 1))
    
    # Yaw from quaternion (rotation about z-axis)
    # quat = [w, x, y, z]
    yaw = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]),
                     1 - 2*(quat[2]**2 + quat[3]**2))
    
    vx, vy, vz = env.data.qvel[0], env.data.qvel[1], env.data.qvel[2]
    wyaw = env.data.qvel[5]  # angular vel about z
    
    # Foot heights
    foot_zs = []
    for i in range(4):
        gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"servo_l{i}_foot")
        foot_zs.append(env.data.geom(gid).xpos[2])
    
    if step % 10 == 0 or terminated:
        print(f"{step:>5} {qpos_z:>8.4f} {body_z:>8.4f} {np.degrees(tilt):>8.1f} "
              f"{np.degrees(yaw):>8.1f} {vx:>8.3f} {vy:>8.3f} {vz:>8.3f} "
              f"{wyaw:>8.3f} {reward:>8.2f} {[f'{z:.3f}' for z in foot_zs]}")
    
    if terminated:
        print(f"\n*** TERMINATED at step {step} ***")
        print(f"  tilt={np.degrees(tilt):.1f} deg, height_drop={env.initial_height - qpos_z:.4f}")
        
        # Check: which body parts are below floor?
        print(f"\n  Bodies below z=0:")
        for i in range(env.model.nbody):
            bz = env.data.body(i).xpos[2]
            bname = env.model.body(i).name
            if bz < 0:
                print(f"    {bname}: z={bz:.4f}")
        break
    if truncated:
        print(f"\n*** TRUNCATED at step {step} (survived full episode!) ***")
        break

# Check collision geometry setup
print(f"\n--- COLLISION SETUP ---")
print(f"Total geoms: {env.model.ngeom}")
colliding = []
for i in range(env.model.ngeom):
    ct = env.model.geom_contype[i]
    ca = env.model.geom_conaffinity[i]
    if ct > 0 or ca > 0:
        name = env.model.geom(i).name
        colliding.append(name)
        print(f"  {name}: contype={ct}, conaffinity={ca}")
print(f"Only {len(colliding)} geoms can collide. Rest pass through floor!")

