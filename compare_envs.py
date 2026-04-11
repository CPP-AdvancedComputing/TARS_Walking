"""Compare env.reset() state vs test reproduction."""
import numpy as np
import mujoco
from tars_env import TARSEnv

URDF = r'C:\Users\anike\tars-urdf\tars_mjcf.xml'

# --- Method 1: TARSEnv ---
env = TARSEnv(URDF)
obs, _ = env.reset()
print("=== TARSEnv ===")
print(f"spawn_height: {env.spawn_height:.6f}")
print(f"qpos[:7]: {env.data.qpos[:7]}")
print(f"ncon: {env.data.ncon}")
for c in range(env.data.ncon):
    g1, g2 = env.data.contact[c].geom1, env.data.contact[c].geom2
    print(f"  {env.model.geom(g1).name} <> {env.model.geom(g2).name}: dist={env.data.contact[c].dist:.6f}")
fz = [env.data.geom(f'servo_l{j}_foot').xpos[2] for j in range(4)]
print(f"foot z: {fz}")
h_initial = env.data.qpos[2]

# Run through env.step 200 times with zero action
env2 = TARSEnv(URDF)
obs2, _ = env2.reset()
for i in range(200):
    obs_, r_, t_, tr_, _ = env2.step(np.zeros(12))
    if i % 50 == 0 or t_:
        z = env2.data.qpos[2]
        tilt = 2*np.arccos(np.clip(abs(env2.data.body('internals').xquat[0]),0,1))
        ncon = env2.data.ncon
        print(f"ENV step {i:3d}: z={z:.5f}, tilt={tilt:.5f}, ncon={ncon}")
    if t_ or tr_:
        print(f"  FELL at step {i+1}")
        break

print()

# --- Method 2: Raw reproduce ---
print("=== RAW REPRODUCE ===")
model = env.model  # Use the SAME model
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
data.qpos[2] = env.spawn_height
a = 0.2
for i in range(4):
    sign = -1 if i % 2 == 0 else 1
    data.joint(f'hip_revolute_l{i}').qpos[0] = sign * a
ctrl = np.zeros(12)
ctrl[4] = -a; ctrl[5] = a; ctrl[6] = -a; ctrl[7] = a
data.ctrl[:] = ctrl
for _ in range(200):
    mujoco.mj_step(model, data)
    data.qvel[0:3] *= 0.95
    data.qvel[3:6] *= 0.9
data.qvel[:] = 0

print(f"qpos[:7]: {data.qpos[:7]}")
print(f"ncon: {data.ncon}")
for c in range(data.ncon):
    g1, g2 = data.contact[c].geom1, data.contact[c].geom2
    print(f"  {model.geom(g1).name} <> {model.geom(g2).name}: dist={data.contact[c].dist:.6f}")
fz2 = [data.geom(f'servo_l{j}_foot').xpos[2] for j in range(4)]
print(f"foot z: {fz2}")

# Run 200 raw steps
for i in range(200):
    data.ctrl[:] = ctrl
    for _ in range(5):
        mujoco.mj_step(model, data)
    tilt = 2*np.arccos(np.clip(abs(data.body('internals').xquat[0]),0,1))
    z = data.qpos[2]
    if i % 50 == 0:
        print(f"RAW step {i:3d}: z={z:.5f}, tilt={tilt:.5f}, ncon={data.ncon}")
    if tilt > 0.5:
        print(f"RAW step {i:3d}: z={z:.5f}, tilt={tilt:.5f} - FELL")
        break

