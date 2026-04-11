"""Test different foot geometries for standing stability."""
import numpy as np
import mujoco
from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR, load_tars_spec

MODEL_PATH = DEFAULT_MODEL_PATH_STR

foot_configs = [
    {'label': 'sphere r=0.04 (current)', 'type': mujoco.mjtGeom.mjGEOM_SPHERE, 'size': [0.04, 0, 0]},
    {'label': 'sphere r=0.08',           'type': mujoco.mjtGeom.mjGEOM_SPHERE, 'size': [0.08, 0, 0]},
    {'label': 'box 4x4x1cm',            'type': mujoco.mjtGeom.mjGEOM_BOX,    'size': [0.04, 0.04, 0.01]},
    {'label': 'box 6x6x1cm',            'type': mujoco.mjtGeom.mjGEOM_BOX,    'size': [0.06, 0.06, 0.01]},
    {'label': 'cylinder r3cm h1cm',      'type': mujoco.mjtGeom.mjGEOM_CYLINDER, 'size': [0.03, 0.005, 0]},
    {'label': 'cylinder r4cm h1cm',      'type': mujoco.mjtGeom.mjGEOM_CYLINDER, 'size': [0.04, 0.005, 0]},
]

for cfg in foot_configs:
    spec = load_tars_spec(MODEL_PATH)
    joint_names = [
        'shoulder_prismatic_l0','shoulder_prismatic_l1','shoulder_prismatic_l2','shoulder_prismatic_l3',
        'hip_revolute_l0','hip_revolute_l1','hip_revolute_l2','hip_revolute_l3',
        'knee_prismatic_l0','knee_prismatic_l1','knee_prismatic_l2','knee_prismatic_l3',
    ]
    joint_ranges = []
    for name in joint_names:
        actuator = spec.add_actuator()
        actuator.name = name + '_act'
        actuator.target = name
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.forcelimited = True
        if 'hip' in name:
            kp, kd = 150.0, 15.0
            actuator.forcerange = [-30.0, 30.0]
            joint_ranges.append((-1.57, 1.57))
        else:
            kp, kd = 400.0, 20.0
            actuator.forcerange = [-50.0, 50.0]
            joint_ranges.append((-0.15, 0.15))
        actuator.gainprm[0] = kp
        actuator.biasprm[1] = -kp
        actuator.biasprm[2] = -kd

    for joint in spec.joints:
        if joint.name in joint_names:
            if 'hip' in joint.name:
                joint.damping = 12.0
            else:
                joint.damping = 8.0

    body_by_name = {body.name: body for body in spec.bodies}
    for leg_id in range(4):
        foot_body = body_by_name.get(f'foot_l{leg_id}')
        if foot_body is not None:
            parent_body = foot_body
            foot_pos = [0.0, 0.0, 0.0]
        else:
            lower_body_name = f'fixed_carriage_l{leg_id}'
            parent_body = body_by_name.get(lower_body_name)
            if parent_body is None:
                continue
            foot_pos = TARSEnv.LOWER_FOOT_OFFSETS[lower_body_name]

        foot = parent_body.add_geom()
        foot.name = f'servo_l{leg_id}_foot'
        foot.type = cfg['type']
        foot.size = cfg['size']
        foot.pos = foot_pos
        foot.friction = [2.0, 0.05, 0.05]
        foot.solref = [-1000, -100]
        foot.solimp = [0.99, 0.99, 0.001, 0.5, 2.0]
        foot.rgba = [0, 0, 0, 0]

    for body in spec.bodies:
        if body.name == 'internals':
            coll = body.add_geom()
            coll.name = 'body_collision'
            coll.type = mujoco.mjtGeom.mjGEOM_BOX
            coll.size = [0.05, 0.15, 0.12]
            coll.pos = [0, 0.2, 0.42]
            coll.rgba = [0, 0, 0, 0]
            coll.mass = 0
            break

    spec.option.timestep = 0.002
    spec.option.iterations = 20
    model = spec.compile()
    data = mujoco.MjData(model)

    # Disable all mesh collisions
    for i in range(model.ngeom):
        model.geom_contype[i] = 0
        model.geom_conaffinity[i] = 0
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    model.geom_contype[floor_id] = 1
    model.geom_conaffinity[floor_id] = 1
    for i in range(4):
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'servo_l{i}_foot')
        model.geom_contype[fid] = 1
        model.geom_conaffinity[fid] = 1
    hull_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'body_collision')
    model.geom_contype[hull_id] = 1
    model.geom_conaffinity[hull_id] = 1

    # Find spawn height properly (same as TARSEnv._find_spawn_height)
    mujoco.mj_resetData(model, data)
    a = 0.2
    for i in range(4):
        sign = -1 if i % 2 == 0 else 1
        data.joint(f'hip_revolute_l{i}').qpos[0] = sign * a
    mujoco.mj_forward(model, data)
    min_foot_z = float('inf')
    for i in range(4):
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'servo_l{i}_foot')
        min_foot_z = min(min_foot_z, data.geom(fid).xpos[2])
    spawn_height = -min_foot_z + 0.04

    # Reset with standing pose (same as TARSEnv.reset)
    ctrl = np.zeros(12)
    ctrl[4] = -a; ctrl[5] = a; ctrl[6] = -a; ctrl[7] = a

    mujoco.mj_resetData(model, data)
    data.qpos[2] = spawn_height
    for i in range(4):
        sign = -1 if i % 2 == 0 else 1
        data.joint(f'hip_revolute_l{i}').qpos[0] = sign * a
    data.ctrl[:] = ctrl
    for _ in range(200):
        mujoco.mj_step(model, data)
        data.qvel[0:3] *= 0.95
        data.qvel[3:6] *= 0.9
    data.qvel[:] = 0

    # Run for 1000 env steps (5000 physics steps)
    survived = 0
    for i in range(1000):
        data.ctrl[:] = ctrl
        for _ in range(5):
            mujoco.mj_step(model, data)
        tilt = 2 * np.arccos(np.clip(abs(data.body('internals').xquat[0]), 0, 1))
        if tilt > 0.5:
            break
        survived = i + 1

    ncon = data.ncon
    lbl = cfg['label']
    print(f'{lbl:30s}: survived={survived:4d}/1000 steps, final_tilt={tilt:.4f}, contacts={ncon}')
