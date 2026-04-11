import argparse
import time
from dataclasses import dataclass, field

import mujoco
import mujoco.viewer
import numpy as np

from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR


@dataclass
class PoseState:
    shoulder: float
    hip_centered: float
    knee: float


@dataclass
class KeyframeConfig:
    cycle_seconds: float = 2.4
    hold_fraction: float = 0.12
    upright: PoseState = field(default_factory=lambda: PoseState(
        shoulder=0.0,
        hip_centered=0.0,
        knee=0.0,
    ))
    lean: PoseState = field(default_factory=lambda: PoseState(
        shoulder=-0.18,
        hip_centered=0.9,
        knee=-0.18,
    ))
    freeze_base: bool = True
    max_tilt_radians: float = 1.15
    min_body_height: float = 0.05


class TARSKeyframeController:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.lower = np.array([lo for lo, _ in env.joint_ranges], dtype=np.float64)
        self.upper = np.array([hi for _, hi in env.joint_ranges], dtype=np.float64)
        self.base_qpos = np.array([0.0, 0.0, env.spawn_height, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.upright_targets = self.pose_to_ctrl_targets(self.config.upright)
        self.lean_targets = self.pose_to_ctrl_targets(self.config.lean)

    def pose_to_ctrl_targets(self, pose):
        ctrl_targets = self.env.standing_ctrl.copy()
        for leg_id in range(4):
            ctrl_targets[leg_id] = pose.shoulder
            ctrl_targets[4 + leg_id] = self.env.standing_ctrl[4 + leg_id] + pose.hip_centered
            ctrl_targets[8 + leg_id] = pose.knee
        return np.clip(ctrl_targets, self.lower, self.upper)

    def interpolation_alpha(self, sim_time):
        cycle = max(self.config.cycle_seconds, 1e-6)
        phase = (sim_time % cycle) / cycle
        hold = np.clip(self.config.hold_fraction, 0.0, 0.45)
        ramp_span = max(0.5 - hold, 1e-6)

        if phase < hold:
            return 0.0
        if phase < 0.5:
            t = (phase - hold) / ramp_span
            return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))
        if phase < 0.5 + hold:
            return 1.0
        t = (phase - (0.5 + hold)) / ramp_span
        return 0.5 + 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))

    def interpolated_ctrl_targets(self, sim_time):
        alpha = self.interpolation_alpha(sim_time)
        return (1.0 - alpha) * self.upright_targets + alpha * self.lean_targets, alpha

    def apply_joint_targets(self, joint_targets, sim_time=None):
        joint_targets = np.clip(np.asarray(joint_targets, dtype=np.float64), self.lower, self.upper)
        self.env.data.qvel[:] = 0.0
        if self.env.model.na:
            self.env.data.act[:] = 0.0
        if sim_time is not None:
            self.env.data.time = float(sim_time)
        for joint_index, joint_name in enumerate(self.env.joint_names):
            self.env.data.joint(joint_name).qpos[0] = joint_targets[joint_index]
        if self.config.freeze_base:
            self.env.data.qpos[:7] = self.base_qpos
            self.env.data.qvel[:6] = 0.0
        self.env.data.ctrl[:] = joint_targets
        mujoco.mj_forward(self.env.model, self.env.data)
        return joint_targets

    def snap_to_upright(self):
        self.env.data.qpos[:7] = self.base_qpos
        self.apply_joint_targets(self.upright_targets, sim_time=0.0)

        min_foot_z = min(
            float(self.env.data.geom(f"servo_l{leg_id}_foot").xpos[2])
            for leg_id in range(4)
        )
        target_z = self.env.FOOT_HALF_HEIGHT + self.env.FOOT_SPAWN_CLEARANCE
        self.env.data.qpos[2] += target_z - min_foot_z
        self.base_qpos[2] = self.env.data.qpos[2]
        self.apply_joint_targets(self.upright_targets, sim_time=0.0)
        return self.upright_targets

    def enforce_base_pose(self):
        if not self.config.freeze_base:
            return
        self.env.data.qpos[:7] = self.base_qpos
        self.env.data.qvel[:6] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)

    def body_is_unstable(self):
        quat = self.env.data.body("internals").xquat
        tilt = 2.0 * np.arccos(np.clip(abs(quat[0]), 0.0, 1.0))
        body_height = float(self.env.data.body("internals").xpos[2])
        return tilt > self.config.max_tilt_radians or body_height < self.config.min_body_height, tilt, body_height

    def current_joint_qpos(self):
        return np.array([
            float(self.env.data.joint(joint_name).qpos[0])
            for joint_name in self.env.joint_names
        ], dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keyframe TARS between an upright pose and a walking-lean pose using actuators."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH_STR, help="Model file to load.")
    parser.add_argument("--sleep", type=float, default=0.01, help="Viewer loop sleep in seconds.")
    parser.add_argument("--trace-every", type=int, default=25, help="Print diagnostics every N viewer steps.")
    parser.add_argument("--cycle-seconds", type=float, default=KeyframeConfig.cycle_seconds)
    parser.add_argument("--hold-fraction", type=float, default=KeyframeConfig.hold_fraction)
    parser.add_argument("--lean-shoulder", type=float, default=KeyframeConfig.lean.shoulder)
    parser.add_argument("--lean-hip", type=float, default=KeyframeConfig.lean.hip_centered)
    parser.add_argument("--lean-knee", type=float, default=KeyframeConfig.lean.knee)
    parser.add_argument("--free-base", action="store_true", help="Let the torso move physically instead of locking the base for clean pose viewing.")
    parser.add_argument("--no-auto-reset", action="store_true", help="Do not reset if the body tips over.")
    return parser.parse_args()


def build_config(args):
    return KeyframeConfig(
        cycle_seconds=args.cycle_seconds,
        hold_fraction=args.hold_fraction,
        upright=PoseState(shoulder=0.0, hip_centered=0.0, knee=0.0),
        lean=PoseState(
            shoulder=args.lean_shoulder,
            hip_centered=args.lean_hip,
            knee=args.lean_knee,
        ),
        freeze_base=not args.free_base,
    )


def print_trace(step, env, controller, ctrl_targets, alpha):
    joint_qpos = controller.current_joint_qpos()
    centered_hips = np.array([
        float(env.data.joint(f"hip_revolute_l{i}").qpos[0] - env.standing_ctrl[4 + i])
        for i in range(4)
    ], dtype=np.float64)
    centered_shoulders = np.array([
        float(env.data.joint(f"shoulder_prismatic_l{i}").qpos[0] - env.standing_ctrl[i])
        for i in range(4)
    ], dtype=np.float64)
    centered_knees = np.array([
        float(env.data.joint(f"knee_prismatic_l{i}").qpos[0] - env.standing_ctrl[8 + i])
        for i in range(4)
    ], dtype=np.float64)
    unstable, tilt, body_height = controller.body_is_unstable()
    print(
        f"step={step:4d} t={env.data.time:6.3f}s alpha={alpha:.3f} "
        f"x={env.data.qpos[0]:+.4f} tilt={tilt:.3f} body_z={body_height:.3f} unstable={unstable}"
    )
    print("  joint_names=" + ", ".join(env.joint_names))
    print("  ctrl_targets=" + np.array2string(ctrl_targets, precision=3, suppress_small=True))
    print("  joint_qpos=" + np.array2string(joint_qpos, precision=3, suppress_small=True))
    print("  centered_shoulders=" + np.array2string(centered_shoulders, precision=3, suppress_small=True))
    print("  centered_hips=" + np.array2string(centered_hips, precision=3, suppress_small=True))
    print("  centered_knees=" + np.array2string(centered_knees, precision=3, suppress_small=True))


def main():
    args = parse_args()
    env = TARSEnv(args.model)
    env.reset()
    controller = TARSKeyframeController(env, build_config(args))
    if controller.config.freeze_base:
        env.model.opt.gravity[:] = 0.0
    controller.snap_to_upright()

    print(f"Viewing model: {args.model}")
    print("Mode: direct-qpos keyframe interpolation between upright and walking-lean poses.")
    print("No learned gait and no physics stepping are used in this script.")
    print(f"Base locked: {controller.config.freeze_base}")
    print("Joint names: " + ", ".join(env.joint_names))
    print("Upright qpos targets: " + np.array2string(controller.upright_targets, precision=3, suppress_small=True))
    print("Lean qpos targets: " + np.array2string(controller.lean_targets, precision=3, suppress_small=True))

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        try:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = env.model.camera("full_body").id
        except KeyError:
            viewer.cam.lookat[:] = env.data.body("internals").xpos
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -18
            viewer.cam.azimuth = 135

        step = 0
        start_time = time.perf_counter()
        while viewer.is_running():
            sim_time = time.perf_counter() - start_time
            ctrl_targets, alpha = controller.interpolated_ctrl_targets(sim_time)
            controller.apply_joint_targets(ctrl_targets, sim_time=sim_time)

            if step % args.trace_every == 0:
                print_trace(step, env, controller, ctrl_targets, alpha)

            unstable, _, _ = controller.body_is_unstable()
            if unstable and not args.no_auto_reset:
                env.reset()
                controller.snap_to_upright()
                start_time = time.perf_counter()

            if viewer.cam.type != mujoco.mjtCamera.mjCAMERA_FIXED:
                viewer.cam.lookat[:] = env.data.body("internals").xpos
            viewer.sync()
            time.sleep(args.sleep)
            step += 1


if __name__ == "__main__":
    main()
