import mujoco
import numpy as np
from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR

env = TARSEnv(DEFAULT_MODEL_PATH_STR)
env.reset()

plant_ids = [0, 2]  # phase 0 planted legs
swing_ids = [1, 3]  # phase 0 swing legs (intentionally airborne)

# Planted leg joint positions vs ctrl
print("=== PLANTED LEGS (l0, l2) — JOINT POSITIONS ===")
for leg_id in plant_ids:
    for prefix, offset in [("shoulder_prismatic", 0), ("hip_revolute", 4), ("knee_prismatic", 8)]:
        name = f"{prefix}_l{leg_id}"
        qpos = env.data.joint(name).qpos[0]
        ctrl = env.data.ctrl[offset + leg_id]
        print(f"  {name}: qpos={qpos:.4f} ctrl={ctrl:.4f} error={qpos-ctrl:.4f}")

# Planted foot heights
print("\n=== PLANTED FOOT HEIGHTS ===")
for leg_id in plant_ids:
    foot_pos = env.data.geom(f"servo_l{leg_id}_foot").xpos
    print(f"  servo_l{leg_id}_foot: z={foot_pos[2]:.4f}")

# Swing foot heights (for reference — expected to be airborne)
print("\n=== SWING FOOT HEIGHTS (expected airborne) ===")
for leg_id in swing_ids:
    foot_pos = env.data.geom(f"servo_l{leg_id}_foot").xpos
    print(f"  servo_l{leg_id}_foot: z={foot_pos[2]:.4f}")

# Contact report — only flag unexpected contacts
print("\n=== CONTACTS ===")
mujoco.mj_forward(env.model, env.data)
print(f"  Total contacts: {env.data.ncon}")
for i in range(env.data.ncon):
    c = env.data.contact[i]
    g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
    g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
    print(f"  {g1} <-> {g2} dist={c.dist:.4f}")

# Torso z at reset
print(f"\n=== TORSO AT RESET ===")
print(f"  torso_z = {env.data.qpos[2]:.6f}")

# 50-step stability trace — focus on planted leg forces + torso_z
print("\n=== 50-STEP STABILITY TRACE ===")
z_values = []
for step in range(50):
    mujoco.mj_step(env.model, env.data)
    z_values.append(float(env.data.qpos[2]))
    if step % 5 == 0:
        torso_z = env.data.qpos[2]
        hip0_qpos = env.data.joint("hip_revolute_l0").qpos[0]
        hip0_ctrl = env.data.ctrl[4 + 0]
        hip2_qpos = env.data.joint("hip_revolute_l2").qpos[0]
        hip2_ctrl = env.data.ctrl[4 + 2]
        print(f"  step {step:2d}: torso_z={torso_z:.6f}  "
              f"hip_l0: qpos={hip0_qpos:.4f} ctrl={hip0_ctrl:.4f} err={hip0_qpos-hip0_ctrl:.4f}  "
              f"hip_l2: qpos={hip2_qpos:.4f} ctrl={hip2_ctrl:.4f} err={hip2_qpos-hip2_ctrl:.4f}")

z_range = max(z_values) - min(z_values)
print(f"\nTotal z_range over 50 steps: {z_range:.6f} m")
print(f"Target: < 0.005 m")
print(f"Result: {'PASS' if z_range < 0.005 else 'FAIL'}")
