import mujoco
import numpy as np
from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR

env = TARSEnv(DEFAULT_MODEL_PATH_STR)

# Reset and settle
mujoco.mj_resetData(env.model, env.data)
env.reset()

# Check joint ranges vs current positions
print("=== JOINT RANGE CHECK ===")
for name in env.joint_names:
    jid = env.model.joint(name).id
    lo, hi = env.model.jnt_range[jid]
    qpos = env.data.joint(name).qpos[0]
    ctrl_idx = env.joint_names.index(name)
    ctrl = env.data.ctrl[ctrl_idx]
    at_lo = qpos <= lo + 0.001
    at_hi = qpos >= hi - 0.001
    flag = " *** AT LIMIT ***" if (at_lo or at_hi) else ""
    print(f"  {name}: range=[{lo:.3f},{hi:.3f}] qpos={qpos:.4f} ctrl={ctrl:.4f}{flag}")

# Check for internal collisions
print("\n=== CONTACT REPORT ===")
mujoco.mj_forward(env.model, env.data)
print(f"  Total contacts: {env.data.ncon}")
for i in range(env.data.ncon):
    c = env.data.contact[i]
    g1_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
    g2_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
    print(f"  contact {i}: {g1_name} <-> {g2_name} dist={c.dist:.4f}")

# Check body positions for l1 and l2 chains
print("\n=== BODY POSITIONS FOR PROBLEM LEGS ===")
for leg_id in (1, 2):
    for body_prefix in ("active_carriage_l", "servo_l", "fixed_carriage_l"):
        body_name = f"{body_prefix}{leg_id}"
        try:
            pos = env.data.body(body_name).xpos
            print(f"  {body_name}: xpos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        except:
            print(f"  {body_name}: NOT FOUND")

# Check foot positions and heights
print("\n=== FOOT HEIGHTS ===")
for i in range(4):
    foot_pos = env.data.geom(f"servo_l{i}_foot").xpos
    print(f"  servo_l{i}_foot: z={foot_pos[2]:.4f} (full pos=[{foot_pos[0]:.4f},{foot_pos[1]:.4f},{foot_pos[2]:.4f}])")

# Check if l1 knee joint is being blocked by geometry
print("\n=== L1 KNEE DETAILED ===")
jid = env.model.joint("knee_prismatic_l1").id
lo, hi = env.model.jnt_range[jid]
qpos = env.data.joint("knee_prismatic_l1").qpos[0]
print(f"  joint range: [{lo:.4f}, {hi:.4f}]")
print(f"  current qpos: {qpos:.4f}")
print(f"  ctrl target: {env.data.ctrl[env.joint_names.index('knee_prismatic_l1')]:.4f}")
dofadr = env.model.joint("knee_prismatic_l1").dofadr[0]
print(f"  qfrc_actuator: {env.data.qfrc_actuator[dofadr]:.4f}")
print(f"  qfrc_constraint: {env.data.qfrc_constraint[dofadr]:.4f}")
print(f"  qfrc_passive: {env.data.qfrc_passive[dofadr]:.4f}")
print(f"  qfrc_applied: {env.data.qfrc_applied[dofadr]:.4f}")
