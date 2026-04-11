import mujoco
import numpy as np
from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR

env = TARSEnv(DEFAULT_MODEL_PATH_STR)
env.reset()

# Print actuator properties
print("=== ACTUATOR PROPERTIES ===")
for i in range(env.model.nu):
    name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    kp = env.model.actuator_gainprm[i, 0]
    force_range = env.model.actuator_forcerange[i]
    print(f"  {name}: kp={kp:.1f}, forcerange=[{force_range[0]:.1f}, {force_range[1]:.1f}]")

# Print joint damping
print("\n=== JOINT DAMPING ===")
for name in env.joint_names:
    jid = env.model.joint(name).id
    dofadr = env.model.jnt_dofadr[jid]
    damping = env.model.dof_damping[dofadr]
    print(f"  {name}: damping={damping:.3f}")

# Print total robot mass
print(f"\n=== TOTAL MASS ===")
print(f"  {env.model.body_mass.sum():.4f} kg")
print(f"  Weight force: {env.model.body_mass.sum() * 9.81:.2f} N")

# Run 50 steps and print ctrl vs qpos each step for one knee
print("\n=== KNEE L0 CTRL vs QPOS vs FORCE (every 5 steps) ===")
knee_idx = env.joint_names.index("knee_prismatic_l0")
actuator_idx = knee_idx  # should match
for step in range(50):
    mujoco.mj_step(env.model, env.data)
    if step % 5 == 0:
        ctrl = env.data.ctrl[actuator_idx]
        qpos = env.data.joint("knee_prismatic_l0").qpos[0]
        qfrc = env.data.qfrc_actuator[env.model.joint("knee_prismatic_l0").dofadr[0]]
        torso_z = env.data.qpos[2]
        print(f"  step {step:2d}: ctrl={ctrl:.4f} qpos={qpos:.4f} actuator_force={qfrc:.3f} torso_z={torso_z:.6f}")

print("\n=== HIP JOINT TRACKING (every 5 steps) ===")
env2 = TARSEnv(DEFAULT_MODEL_PATH_STR)
env2.reset()
for step in range(50):
    mujoco.mj_step(env2.model, env2.data)
    if step % 5 == 0:
        torso_z = env2.data.qpos[2]
        for leg_id in range(4):
            hip_ctrl = env2.data.ctrl[4 + leg_id]
            hip_qpos = env2.data.joint(f"hip_revolute_l{leg_id}").qpos[0]
            hip_force = env2.data.qfrc_actuator[
                env2.model.joint(f"hip_revolute_l{leg_id}").dofadr[0]
            ]
            print(f"  step {step:2d} l{leg_id}: ctrl={hip_ctrl:.4f} qpos={hip_qpos:.4f} "
                  f"error={hip_qpos - hip_ctrl:.4f} force={hip_force:.3f}")
        print(f"  step {step:2d} torso_z={torso_z:.6f}")
        print()
