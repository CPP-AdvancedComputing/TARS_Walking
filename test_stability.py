import numpy as np, mujoco
from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR, load_tars_spec

configs = [
    {'label': 'CURRENT',              'foot': 0.04, 'hip': 0.2, 'stiff': 1000,  'damp': 100},
    {'label': 'bigger feet',          'foot': 0.06, 'hip': 0.2, 'stiff': 1000,  'damp': 100},
    {'label': 'wider+bigger',         'foot': 0.06, 'hip': 0.4, 'stiff': 1000,  'damp': 100},
    {'label': 'wider+bigger+stiff',   'foot': 0.06, 'hip': 0.4, 'stiff': 5000,  'damp': 300},
    {'label': 'wider+BIG+stiff',      'foot': 0.08, 'hip': 0.4, 'stiff': 5000,  'damp': 300},
    {'label': 'wider+BIG+VERYSTI',    'foot': 0.08, 'hip': 0.4, 'stiff': 10000, 'damp': 500},
]

for cfg in configs:
    spec = load_tars_spec(DEFAULT_MODEL_PATH_STR)
    joint_names = [
        'shoulder_prismatic_l0','shoulder_prismatic_l1','shoulder_prismatic_l2','shoulder_prismatic_l3',
        'hip_revolute_l0','hip_revolute_l1','hip_revolute_l2','hip_revolute_l3',
        'knee_prismatic_l0','knee_prismatic_l1','knee_prismatic_l2','knee_prismatic_l3',
    ]
    for name in joint_names:
        actuator = spec.add_actuator()
        actuator.name = name + '_act'
        actuator.target = name
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.gainprm[0] = 50
        actuator.biasprm[0] = 0
        actuator.biasprm[1] = -50
        actuator.biasprm[2] = -2
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE

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
        foot.type = mujoco.mjtGeom.mjGEOM_SPHERE
        foot.size = [cfg['foot'], 0, 0]
        foot.pos = foot_pos
        foot.friction = [2.0, 0.05, 0.05]
        foot.solref = [-cfg['stiff'], -cfg['damp']]
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

    spec.option.timestep = 0.002
    spec.option.iterations = 20
    model = spec.compile()
    data = mujoco.MjData(model)

    a = cfg['hip']
    ctrl = np.zeros(12)
    ctrl[4] = -a; ctrl[5] = a; ctrl[6] = -a; ctrl[7] = a

    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.7
    for i in range(4):
        sign = -1 if i % 2 == 0 else 1
        data.joint(f'hip_revolute_l{i}').qpos[0] = sign * a
    data.ctrl[:] = ctrl
    for _ in range(200):
        mujoco.mj_step(model, data)
        data.qvel[0:3] *= 0.95
        data.qvel[3:6] *= 0.9
    data.qvel[:] = 0

    survived = 0
    for i in range(1000):
        data.ctrl[:] = ctrl
        for _ in range(5):
            mujoco.mj_step(model, data)
        tilt = 2 * np.arccos(np.clip(abs(data.body('internals').xquat[0]), 0, 1))
        if tilt > 0.5:
            break
        survived = i + 1

    lbl = cfg['label']
    print(f'{lbl:25s}: survived={survived:4d} steps, final_tilt={tilt:.3f}')
