import numpy as np, mujoco
from tars_env import TARSEnv
env = TARSEnv(r"C:\Users\anike\tars-urdf\tars_mjcf.xml")
obs, _ = env.reset()
d = env.data; m = env.model
print(f"qpos z (free joint): {d.qpos[2]:.4f}")
print(f"body internals z: {d.body('internals').xpos[2]:.4f}")
print(f"initial_height: {env.initial_height:.4f}")
print()

for name in ["servo_1","servo_2","servo_3","servo_4"]:
    foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name+"_foot")
    pos = d.geom_xpos[foot_id]
    print(f"{name}_foot: z={pos[2]:.4f}")

floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
print(f"\nfloor geom id: {floor_id}")

# Check contype/conaffinity for floor and feet
print(f"floor contype={m.geom_contype[floor_id]}, conaffinity={m.geom_conaffinity[floor_id]}")
for name in ["servo_1_foot","servo_2_foot","servo_3_foot","servo_4_foot","body_collision"]:
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    print(f"{name}: contype={m.geom_contype[gid]}, conaffinity={m.geom_conaffinity[gid]}")

# Check all geom groups
print(f"\nTotal geoms: {m.ngeom}")
groups = {}
for i in range(m.ngeom):
    g = int(m.geom_group[i])
    groups[g] = groups.get(g, 0) + 1
print(f"Geom groups: {groups}")

# Step and check
for i in range(5):
    mujoco.mj_step(m, d)
print(f"\nAfter 5 steps:")
print(f"qpos z: {d.qpos[2]:.4f}")
print(f"ncon: {d.ncon}")
for c in range(min(d.ncon, 10)):
    g1name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[c].geom1) or str(d.contact[c].geom1)
    g2name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[c].geom2) or str(d.contact[c].geom2)
    print(f"  contact {c}: {g1name} <-> {g2name} at z={d.contact[c].pos[2]:.4f}")

for i in range(200):
    mujoco.mj_step(m, d)
print(f"\nAfter 205 steps:")
print(f"qpos z: {d.qpos[2]:.4f}")
for name in ["servo_1","servo_2","servo_3","servo_4"]:
    foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name+"_foot")
    pos = d.geom_xpos[foot_id]
    print(f"{name}_foot: z={pos[2]:.4f}")

