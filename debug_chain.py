import mujoco, numpy as np
from tars_env import TARSEnv

env = TARSEnv(r"C:\Users\anike\tars-urdf\tars_mjcf.xml")
obs, _ = env.reset()
m, d = env.model, env.data

print("=== BODY HIERARCHY ===")
for i in range(m.nbody):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
    parent = m.body_parentid[i]
    pname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, parent) or f"body_{parent}"
    pos = m.body_pos[i]
    print(f"  {name} (id={i}) -> parent: {pname} (id={parent}), pos={pos}")

print("\n=== JOINT -> BODY MAPPING ===")
for i in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
    body_id = m.jnt_bodyid[i]
    bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
    jtype = ["free","ball","slide","hinge"][m.jnt_type[i]]
    print(f"  {name}: type={jtype}, body={bname}")

print("\n=== FOOT GEOM -> BODY MAPPING ===")
for i in range(4):
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"servo_l{i}_foot")
    body_id = m.geom_bodyid[gid]
    bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
    print(f"  servo_l{i}_foot -> body: {bname} (id={body_id})")

print("\n=== AFTER RESET: BODY + FOOT POSITIONS ===")
internals_z = d.body("internals").xpos[2]
print(f"  internals: {d.body('internals').xpos}")
for i in range(4):
    servo_name = f"servo_l{i}"
    bpos = d.body(servo_name).xpos
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"servo_l{i}_foot")
    fpos = d.geom_xpos[gid]
    print(f"  {servo_name} body: {bpos}, foot geom: {fpos}")

# Step 50 steps with zero action and check again
for _ in range(50):
    d.ctrl[:] = env.standing_ctrl
    mujoco.mj_step(m, d)

print("\n=== AFTER 50 STEPS (standing ctrl): ===")
print(f"  internals: {d.body('internals').xpos}")
for i in range(4):
    servo_name = f"servo_l{i}"
    bpos = d.body(servo_name).xpos
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"servo_l{i}_foot")
    fpos = d.geom_xpos[gid]
    dist = np.linalg.norm(bpos - d.body('internals').xpos)
    print(f"  {servo_name} body: {bpos}, foot: {fpos}, dist_from_body={dist:.3f}")

