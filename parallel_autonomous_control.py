import argparse
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR


@dataclass
class ParallelMotionConfig:
    # Edit these defaults if you want a different autonomous parallel motion.
    frequency_hz: float = 0.55
    shoulder_bias: float = -0.050
    shoulder_amplitude: float = 0.030
    shoulder_phase: float = -0.5 * np.pi
    hip_bias: float = 0.000
    hip_amplitude: float = 0.420
    hip_phase: float = 0.0
    knee_bias: float = -0.060
    knee_amplitude: float = 0.040
    knee_phase: float = 0.5 * np.pi
    max_tilt_radians: float = 1.10
    min_body_height: float = 0.05


class ParallelAutonomousController:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.lower = np.array([lo for lo, _ in env.joint_ranges], dtype=np.float64)
        self.upper = np.array([hi for _, hi in env.joint_ranges], dtype=np.float64)

    def shared_centered_deltas(self, sim_time):
        phase = 2.0 * np.pi * self.config.frequency_hz * sim_time
        return np.array([
            self.config.shoulder_bias + self.config.shoulder_amplitude * np.sin(phase + self.config.shoulder_phase),
            self.config.hip_bias + self.config.hip_amplitude * np.sin(phase + self.config.hip_phase),
            self.config.knee_bias + self.config.knee_amplitude * np.sin(phase + self.config.knee_phase),
        ], dtype=np.float64)

    def control_targets(self, sim_time):
        deltas = self.shared_centered_deltas(sim_time)
        ctrl_targets = self.env.standing_ctrl.copy()
        for leg_id in range(4):
            ctrl_targets[leg_id] = self.env.standing_ctrl[leg_id] + deltas[0]
            ctrl_targets[4 + leg_id] = self.env.standing_ctrl[4 + leg_id] + deltas[1]
            ctrl_targets[8 + leg_id] = self.env.standing_ctrl[8 + leg_id] + deltas[2]
        return np.clip(ctrl_targets, self.lower, self.upper)

    def snap_robot_to_motion(self, sim_time=0.0):
        ctrl_targets = self.control_targets(sim_time)
        self.env.data.qvel[:] = 0.0
        self.env.data.act[:] = 0.0 if self.env.model.na else self.env.data.act
        for joint_index, joint_name in enumerate(self.env.joint_names):
            self.env.data.joint(joint_name).qpos[0] = ctrl_targets[joint_index]
        self.env.data.qpos[0] = 0.0
        self.env.data.qpos[1] = 0.0
        self.env.data.qpos[2] = self.env.spawn_height
        self.env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        mujoco.mj_forward(self.env.model, self.env.data)

        min_foot_z = min(
            float(self.env.data.geom(f"servo_l{leg_id}_foot").xpos[2])
            for leg_id in range(4)
        )
        target_z = self.env.FOOT_HALF_HEIGHT + self.env.FOOT_SPAWN_CLEARANCE
        self.env.data.qpos[2] += target_z - min_foot_z
        self.env.data.qvel[:] = 0.0
        self.env.data.ctrl[:] = ctrl_targets
        mujoco.mj_forward(self.env.model, self.env.data)
        return ctrl_targets

    def body_is_unstable(self):
        quat = self.env.data.body("internals").xquat
        tilt = 2.0 * np.arccos(np.clip(abs(quat[0]), 0.0, 1.0))
        body_height = float(self.env.data.body("internals").xpos[2])
        return tilt > self.config.max_tilt_radians or body_height < self.config.min_body_height, tilt, body_height


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drive all four TARS legs with the same autonomous parallel motion."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH_STR, help="Model file to load.")
    parser.add_argument("--sleep", type=float, default=0.01, help="Viewer loop sleep in seconds.")
    parser.add_argument("--trace-every", type=int, default=25, help="Print diagnostics every N viewer steps.")
    parser.add_argument("--frequency", type=float, default=ParallelMotionConfig.frequency_hz)
    parser.add_argument("--shoulder-bias", type=float, default=ParallelMotionConfig.shoulder_bias)
    parser.add_argument("--shoulder-amp", type=float, default=ParallelMotionConfig.shoulder_amplitude)
    parser.add_argument("--hip-bias", type=float, default=ParallelMotionConfig.hip_bias)
    parser.add_argument("--hip-amp", type=float, default=ParallelMotionConfig.hip_amplitude)
    parser.add_argument("--knee-bias", type=float, default=ParallelMotionConfig.knee_bias)
    parser.add_argument("--knee-amp", type=float, default=ParallelMotionConfig.knee_amplitude)
    parser.add_argument("--no-auto-reset", action="store_true", help="Do not reset after the body tips over.")
    return parser.parse_args()


def build_config(args):
    return ParallelMotionConfig(
        frequency_hz=args.frequency,
        shoulder_bias=args.shoulder_bias,
        shoulder_amplitude=args.shoulder_amp,
        hip_bias=args.hip_bias,
        hip_amplitude=args.hip_amp,
        knee_bias=args.knee_bias,
        knee_amplitude=args.knee_amp,
    )


def print_trace(step, env, controller, ctrl_targets):
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
        f"step={step:4d} t={env.data.time:6.3f}s x={env.data.qpos[0]:+.4f} "
        f"tilt={tilt:.3f} body_z={body_height:.3f} unstable={unstable}"
    )
    print("  ctrl_targets=" + np.array2string(ctrl_targets, precision=3, suppress_small=True))
    print("  centered_shoulders=" + np.array2string(centered_shoulders, precision=3, suppress_small=True))
    print("  centered_hips=" + np.array2string(centered_hips, precision=3, suppress_small=True))
    print("  centered_knees=" + np.array2string(centered_knees, precision=3, suppress_small=True))


def main():
    args = parse_args()
    env = TARSEnv(args.model)
    env.reset()
    controller = ParallelAutonomousController(env, build_config(args))
    controller.snap_robot_to_motion(sim_time=0.0)

    print(f"Viewing model: {args.model}")
    print("Mode: all four legs use the same centered autonomous motion.")
    print("This is a debug controller, not the walking gait or PPO policy.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = env.data.body("internals").xpos
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -18
        viewer.cam.azimuth = 135

        step = 0
        while viewer.is_running():
            ctrl_targets = controller.control_targets(env.data.time)
            env.data.ctrl[:] = ctrl_targets
            for _ in range(env.FRAME_SKIP):
                mujoco.mj_step(env.model, env.data)

            if step % args.trace_every == 0:
                print_trace(step, env, controller, ctrl_targets)

            unstable, _, _ = controller.body_is_unstable()
            if unstable and not args.no_auto_reset:
                env.reset()
                controller.snap_robot_to_motion(sim_time=0.0)

            viewer.cam.lookat[:] = env.data.body("internals").xpos
            viewer.sync()
            time.sleep(args.sleep)
            step += 1


if __name__ == "__main__":
    main()
