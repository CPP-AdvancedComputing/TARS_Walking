import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR


JOINTS_TO_COMPARE = (
    "shoulder_prismatic_l1",
    "hip_revolute_l1",
    "knee_prismatic_l1",
    "shoulder_prismatic_l3",
    "hip_revolute_l3",
    "knee_prismatic_l3",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit leg-1 vs leg-3 geometry and sign-relevant joint definitions.")
    parser.add_argument("--urdf", default="robot_mujoco.urdf", help="URDF file to inspect.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH_STR, help="MJCF/URDF model to load in MuJoCo.")
    return parser.parse_args()


def fmt(values):
    return "[" + ", ".join(str(v) for v in values) + "]"


def joint_info_from_urdf(urdf_path: Path):
    root = ET.parse(urdf_path).getroot()
    info = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in JOINTS_TO_COMPARE:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        info[name] = {
            "origin_xyz": origin.attrib.get("xyz", "") if origin is not None else "",
            "origin_rpy": origin.attrib.get("rpy", "") if origin is not None else "",
            "axis": axis.attrib.get("xyz", "") if axis is not None else "",
            "lower": limit.attrib.get("lower", "") if limit is not None else "",
            "upper": limit.attrib.get("upper", "") if limit is not None else "",
        }
    return info


def main():
    args = parse_args()
    urdf_path = Path(args.urdf)
    info = joint_info_from_urdf(urdf_path)
    print("URDF joint audit:")
    for joint_name in JOINTS_TO_COMPARE:
        print(joint_name, info.get(joint_name, {}))

    env = TARSEnv(model_path=args.model_path)
    env.reset()
    print("\nMuJoCo pose audit:")
    for leg_id in (1, 3):
        body = env.data.body(f"active_carriage_l{leg_id}")
        foot = env.data.geom(f"servo_l{leg_id}_foot")
        shoulder = env.data.joint(f"shoulder_prismatic_l{leg_id}").qpos[0]
        hip = env.data.joint(f"hip_revolute_l{leg_id}").qpos[0]
        knee = env.data.joint(f"knee_prismatic_l{leg_id}").qpos[0]
        print(
            f"l{leg_id}: body={fmt(round(float(x), 4) for x in body.xpos)} "
            f"foot={fmt(round(float(x), 4) for x in foot.xpos)} "
            f"qpos={fmt(round(float(x), 4) for x in (shoulder, hip, knee))}"
        )

    print("\nPhase support audit:")
    for phase in (0, 1):
        mujoco.mj_resetData(env.model, env.data)
        env._set_phase_pose(phase)
        desired = [int(x) for x in env._desired_foot_contacts_for_time(phase=phase, phase_timer=0)]
        actual = [int(x) for x in env._foot_on_ground_flags()]
        heights = [round(env._foot_world_height(i), 4) for i in range(4)]
        print(f"phase {phase}: desired={desired} actual={actual} heights={heights}")


if __name__ == "__main__":
    main()
