import mujoco
from tars_model import DEFAULT_MODEL_PATH_STR, load_tars_spec

def load_tars(urdf_path):
    spec = load_tars_spec(urdf_path)
    
    joint_names = [
        "shoulder_prismatic_l0", "shoulder_prismatic_l1",
        "shoulder_prismatic_l2", "shoulder_prismatic_l3",
        "hip_revolute_l0", "hip_revolute_l1",
        "hip_revolute_l2", "hip_revolute_l3",
        "knee_prismatic_l0", "knee_prismatic_l1",
        "knee_prismatic_l2", "knee_prismatic_l3",
    ]
    
    for name in joint_names:
        actuator = spec.add_actuator()
        actuator.name = name + "_act"
        actuator.target = name
        actuator.gainprm[0] = 100.0
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    
    model = spec.compile()
    print("Actuators:", model.nu)
    return model

model = load_tars(DEFAULT_MODEL_PATH_STR)
