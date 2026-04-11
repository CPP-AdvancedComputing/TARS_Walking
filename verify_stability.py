import mujoco
import numpy as np
from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR

env = TARSEnv(DEFAULT_MODEL_PATH_STR)
env.reset()

print(f"\n=== SETTLE LOOP: {getattr(env, '_settle_steps', 'N/A')} steps ===")

print("\n=== POST-RESET JOINT POSITIONS ===")
for name in env.joint_names:
    qpos = env.data.joint(name).qpos[0]
    ctrl_idx = env.joint_names.index(name)
    ctrl = env.data.ctrl[ctrl_idx]
    print(f"  {name}: qpos={qpos:.4f} ctrl={ctrl:.4f} error={qpos-ctrl:.4f}")
print(f"  torso_z at reset: {env.data.qpos[2]:.6f}")

z_values = []
for step in range(50):
    mujoco.mj_step(env.model, env.data)
    z_values.append(float(env.data.qpos[2]))
    if step in (0, 4, 9, 14, 19, 24, 49):
        print(f"Step {step+1:2d}: torso_z = {env.data.qpos[2]:.6f}")

z_range = max(z_values) - min(z_values)
print(f"\nTotal z_range over 50 steps: {z_range:.6f} m")
print(f"Target: < 0.005 m")
print(f"Result: {'PASS' if z_range < 0.005 else 'FAIL'}")

print("\n=== KNEE L1 FORCE BREAKDOWN AT RESET ===")
env2 = TARSEnv(DEFAULT_MODEL_PATH_STR)
env2.reset()
dofadr = env2.model.joint("knee_prismatic_l1").dofadr[0]
print(f"  knee_l1 qpos:           {env2.data.joint('knee_prismatic_l1').qpos[0]:.4f}")
print(f"  knee_l1 ctrl:           {env2.data.ctrl[env2.joint_names.index('knee_prismatic_l1')]:.4f}")
print(f"  qfrc_actuator:          {env2.data.qfrc_actuator[dofadr]:.4f}")
print(f"  qfrc_constraint:        {env2.data.qfrc_constraint[dofadr]:.4f}")
print(f"  qfrc_passive:           {env2.data.qfrc_passive[dofadr]:.4f}")
print(f"  joint range:            [{env2.model.jnt_range[env2.model.joint('knee_prismatic_l1').id][0]:.4f}, {env2.model.jnt_range[env2.model.joint('knee_prismatic_l1').id][1]:.4f}]")
foot_z = env2.data.geom("servo_l1_foot").xpos[2]
print(f"  servo_l1_foot z:        {foot_z:.4f}")
print(f"  floor contact l1:       {any(env2.data.contact[i].geom1 == env2.foot_geom_ids[1] or env2.data.contact[i].geom2 == env2.foot_geom_ids[1] for i in range(env2.data.ncon))}")

