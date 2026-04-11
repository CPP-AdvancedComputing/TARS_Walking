import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from tars_env import TARSEnv
from tars_gait_reference import (
    CYCLE_DURATION,
    DISPLAY_PLANTED_LEGS,
    DISPLAY_TO_MODEL_LEG,
    FORWARD_DURATION,
    HOLD_DURATION,
    PHASE_1,
    PHASE_2,
    RETURN_DURATION,
    TORSO_PITCH_AMPLITUDE,
    TORSO_X_SHIFT_AMPLITUDE,
    TORSO_Z_BOB_AMPLITUDE,
    ctrl_targets_from_phase_pose,
    ease_in_out_sine,
)
from tars_model import DEFAULT_MODEL_PATH_STR


ACTUATOR_TO_JOINT = {
    "shoulder": "shoulder_prismatic",
    "hip": "hip_revolute",
    "knee": "knee_prismatic",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show TARS performing a looping crutch-gait keyframe demo in the passive viewer."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH_STR, help="Model file to load.")
    parser.add_argument("--sleep", type=float, default=0.01, help="Viewer loop sleep in seconds.")
    return parser.parse_args()


def print_actuator_list(model):
    print("Actuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  [{i}] {name}")


def phase_pose_to_ctrl_targets(env, phase_pose):
    neutral_by_joint = {
        joint_name: float(env.standing_ctrl[index])
        for index, joint_name in enumerate(env.joint_names)
    }
    return ctrl_targets_from_phase_pose(env.joint_names, neutral_by_joint, phase_pose)


def print_phase_targets(model, phase_name, ctrl_targets):
    print(f"\n{phase_name} ctrl targets:")
    for i, value in enumerate(ctrl_targets):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  [{i}] {name}: {value:.3f}")


def print_leg_layout(env):
    model = env.model
    data = env.data
    leg_positions = []
    for leg_id in range(4):
        body_name = f"active_carriage_l{leg_id}"
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        pos = data.xpos[bid].copy()
        leg_positions.append((leg_id, pos))

    print("\nLeg layout (active_carriage body positions):")
    for leg_id, pos in leg_positions:
        print(f"  l{leg_id}: xpos={pos}")

    y_sorted = sorted(leg_positions, key=lambda item: item[1][1])
    print("Leg order by body Y (low -> high): " + " -> ".join(f"l{leg_id}" for leg_id, _ in y_sorted))
    print(
        "Demo leg mapping: "
        + ", ".join(f"display l{display_leg} -> model l{model_leg}" for display_leg, model_leg in DISPLAY_TO_MODEL_LEG.items())
    )
    print("This demo uses display legs: planted = l0,l3 ; swing = l1,l2")


def actuator_name_to_joint_name(actuator_name):
    joint_kind, leg_suffix = actuator_name.split("_l")
    return f"{ACTUATOR_TO_JOINT[joint_kind]}_l{leg_suffix}"


def quat_about_y(angle_radians):
    half = 0.5 * angle_radians
    return np.array([np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float64)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


class CrutchGaitDemo:
    def __init__(self, env):
        self.env = env
        self.model = env.model
        self.data = env.data
        self.base_qpos = self.data.qpos[:7].copy()
        self.rest_base_qpos = self.base_qpos.copy()
        self.phase1_ctrl = phase_pose_to_ctrl_targets(self.env, PHASE_1)
        self.phase2_ctrl = phase_pose_to_ctrl_targets(self.env, PHASE_2)
        self.base_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.rest_base_qpos[:] = self.base_qpos
        self.outer_torso_mocap_id = self._find_outer_torso_mocap_id()
        self.outer_torso_rest_pos, self.outer_torso_rest_quat = self._current_outer_torso_pose()

    def _find_outer_torso_mocap_id(self):
        try:
            body_id = self.model.body("outer_torso_visual").id
        except KeyError:
            return None
        mocap_id = int(self.model.body_mocapid[body_id])
        return mocap_id if mocap_id >= 0 else None

    def _current_outer_torso_pose(self):
        if self.outer_torso_mocap_id is None:
            return None, None
        return (
            self.data.mocap_pos[self.outer_torso_mocap_id].copy(),
            self.data.mocap_quat[self.outer_torso_mocap_id].copy(),
        )

    def _apply_ctrl_targets(self, ctrl_targets, sim_time=None, base_pose=None):
        self.data.qvel[:] = 0.0
        if self.model.na:
            self.data.act[:] = 0.0
        if sim_time is not None:
            self.data.time = float(sim_time)

        for i, value in enumerate(ctrl_targets):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            joint_name = actuator_name_to_joint_name(actuator_name)
            self.data.joint(joint_name).qpos[0] = float(value)

        if base_pose is None:
            self.data.qpos[:7] = self.base_qpos
        else:
            self.data.qpos[:7] = base_pose
        self.data.qvel[:6] = 0.0
        self.data.ctrl[:] = ctrl_targets
        mujoco.mj_forward(self.model, self.data)

    def snap_phase1_to_visible_height(self):
        self._apply_ctrl_targets(self.phase1_ctrl, sim_time=0.0)
        min_foot_z = min(
            float(self.data.geom(f"servo_l{leg_id}_foot").xpos[2])
            for leg_id in range(4)
        )
        target_z = self.env.FOOT_HALF_HEIGHT + self.env.FOOT_SPAWN_CLEARANCE
        height_delta = target_z - min_foot_z
        self.base_qpos[2] += height_delta
        self.rest_base_qpos[:] = self.base_qpos
        if self.outer_torso_mocap_id is not None:
            self.outer_torso_rest_pos[2] += height_delta
            self.data.mocap_pos[self.outer_torso_mocap_id] = self.outer_torso_rest_pos
            self.data.mocap_quat[self.outer_torso_mocap_id] = self.outer_torso_rest_quat
        self._apply_ctrl_targets(self.phase1_ctrl, sim_time=0.0)

    def interpolated_targets(self, sim_time):
        phase_time = sim_time % CYCLE_DURATION
        if phase_time < FORWARD_DURATION:
            eased = ease_in_out_sine(phase_time / FORWARD_DURATION)
            return (1.0 - eased) * self.phase1_ctrl + eased * self.phase2_ctrl
        if phase_time < FORWARD_DURATION + HOLD_DURATION:
            return self.phase2_ctrl
        back_time = phase_time - FORWARD_DURATION - HOLD_DURATION
        eased = ease_in_out_sine(back_time / RETURN_DURATION)
        return (1.0 - eased) * self.phase2_ctrl + eased * self.phase1_ctrl

    def torso_phase_scalar(self, sim_time):
        phase_time = sim_time % CYCLE_DURATION
        if phase_time < FORWARD_DURATION:
            eased = ease_in_out_sine(phase_time / FORWARD_DURATION)
            return 1.0 - 2.0 * eased
        if phase_time < FORWARD_DURATION + HOLD_DURATION:
            return -1.0
        back_time = phase_time - FORWARD_DURATION - HOLD_DURATION
        eased = ease_in_out_sine(back_time / RETURN_DURATION)
        return -1.0 + 2.0 * eased

    def base_pose_for_time(self, sim_time):
        scalar = self.torso_phase_scalar(sim_time)
        base_pose = self.rest_base_qpos.copy()
        base_pose[0] -= TORSO_X_SHIFT_AMPLITUDE * scalar
        base_pose[2] += TORSO_Z_BOB_AMPLITUDE * (1.0 - scalar * scalar)
        base_pose[3:7] = quat_about_y(-TORSO_PITCH_AMPLITUDE * scalar)
        return base_pose

    def outer_torso_pose_for_time(self, sim_time):
        if self.outer_torso_mocap_id is None:
            return None, None
        scalar = self.torso_phase_scalar(sim_time)
        pos = self.outer_torso_rest_pos.copy()
        pos[0] += TORSO_X_SHIFT_AMPLITUDE * scalar
        pos[2] += 0.5 * TORSO_Z_BOB_AMPLITUDE * (1.0 - scalar * scalar)
        quat = quat_multiply(quat_about_y(TORSO_PITCH_AMPLITUDE * scalar), self.outer_torso_rest_quat)
        return pos, quat

    def planted_leg_compensated_targets(self, sim_time):
        targets = self.interpolated_targets(sim_time).copy()
        torso_pitch = -TORSO_PITCH_AMPLITUDE * self.torso_phase_scalar(sim_time)
        for display_leg in DISPLAY_PLANTED_LEGS:
            model_leg = DISPLAY_TO_MODEL_LEG[display_leg]
            hip_actuator_index = 3 * model_leg + 1
            targets[hip_actuator_index] -= torso_pitch
        return targets

    def apply_for_time(self, sim_time):
        targets = self.planted_leg_compensated_targets(sim_time)
        outer_pos, outer_quat = self.outer_torso_pose_for_time(sim_time)
        if outer_pos is not None:
            self.data.mocap_pos[self.outer_torso_mocap_id] = outer_pos
            self.data.mocap_quat[self.outer_torso_mocap_id] = outer_quat
        self._apply_ctrl_targets(targets, sim_time=sim_time, base_pose=self.base_pose_for_time(sim_time))
        return targets


def configure_camera(viewer, data):
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = data.body("internals").xpos
    viewer.cam.lookat[1] += 0.18
    viewer.cam.lookat[2] += 0.15
    viewer.cam.distance = 1.9
    viewer.cam.azimuth = 138.0
    viewer.cam.elevation = -14.0


def main():
    args = parse_args()
    env = TARSEnv(args.model)
    env.reset()

    # Freeze the torso in place for a clean demo pose.
    env.model.opt.gravity[:] = 0.0

    demo = CrutchGaitDemo(env)

    print_actuator_list(env.model)
    print_phase_targets(env.model, "Phase 1", demo.phase1_ctrl)
    print_phase_targets(env.model, "Phase 2", demo.phase2_ctrl)
    print_leg_layout(env)
    print("\nMode: crutch gait keyframe demo with locked base.")
    print("Timing: 2.0s forward swing, 0.5s hold, 2.0s return, looping.")

    demo.snap_phase1_to_visible_height()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        configure_camera(viewer, env.data)
        start_time = time.perf_counter()

        while viewer.is_running():
            sim_time = time.perf_counter() - start_time
            demo.apply_for_time(sim_time)
            viewer.sync()
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
