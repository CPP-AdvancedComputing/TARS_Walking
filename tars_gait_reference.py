from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhasePose:
    shoulder: float
    hip: float
    knee: float


@dataclass(frozen=True)
class LegPhasePose:
    l0: PhasePose
    l1: PhasePose
    l2: PhasePose
    l3: PhasePose


FORWARD_DURATION = 2.0
HOLD_DURATION = 0.5
RETURN_DURATION = 2.0
CYCLE_DURATION = FORWARD_DURATION + HOLD_DURATION + RETURN_DURATION

# Visual/demo-facing numbering in left-to-right order on screen.
DISPLAY_TO_MODEL_LEG = {
    0: 0,
    1: 3,
    2: 2,
    3: 1,
}
MODEL_TO_DISPLAY_LEG = {model_leg: display_leg for display_leg, model_leg in DISPLAY_TO_MODEL_LEG.items()}

DISPLAY_PLANTED_LEGS = (0, 3)
DISPLAY_SWING_LEGS = (1, 2)

MODEL_PLANTED_LEGS = tuple(DISPLAY_TO_MODEL_LEG[display_leg] for display_leg in DISPLAY_PLANTED_LEGS)
MODEL_SWING_LEGS = tuple(DISPLAY_TO_MODEL_LEG[display_leg] for display_leg in DISPLAY_SWING_LEGS)

TORSO_X_SHIFT_AMPLITUDE = 0.03
TORSO_Z_BOB_AMPLITUDE = 0.01
TORSO_PITCH_AMPLITUDE = 0.14
OSCILLATOR_HIP_AMPLITUDE = 0.45
OSCILLATOR_SHOULDER_AMPLITUDE = 0.0
OSCILLATOR_KNEE_AMPLITUDE = 0.0
OSCILLATOR_OUTER_SCALE = 1.0
OSCILLATOR_MIDDLE_SCALE = 1.0


PHASE_1 = LegPhasePose(
    l0=PhasePose(shoulder=0.0, hip=0.0, knee=0.0),
    l1=PhasePose(shoulder=0.0, hip=0.785, knee=0.0),
    l2=PhasePose(shoulder=0.0, hip=0.785, knee=0.0),
    l3=PhasePose(shoulder=0.0, hip=0.0, knee=0.0),
)

PHASE_2 = LegPhasePose(
    l0=PhasePose(shoulder=0.0, hip=0.0, knee=0.0),
    l1=PhasePose(shoulder=0.0, hip=-0.785, knee=0.0),
    l2=PhasePose(shoulder=0.0, hip=-0.785, knee=0.0),
    l3=PhasePose(shoulder=0.0, hip=0.0, knee=0.0),
)


def display_phase_pose_dict(phase_pose):
    return {
        0: phase_pose.l0,
        1: phase_pose.l1,
        2: phase_pose.l2,
        3: phase_pose.l3,
    }


def joint_name(kind, leg_id):
    return f"{kind}_l{leg_id}"


def desired_joint_targets_from_phase_pose(neutral_by_joint, phase_pose):
    desired_by_joint = dict(neutral_by_joint)
    for display_leg_id, pose in display_phase_pose_dict(phase_pose).items():
        model_leg_id = DISPLAY_TO_MODEL_LEG[display_leg_id]
        desired_by_joint[joint_name("shoulder_prismatic", model_leg_id)] = (
            neutral_by_joint[joint_name("shoulder_prismatic", model_leg_id)] + pose.shoulder
        )
        desired_by_joint[joint_name("hip_revolute", model_leg_id)] = (
            neutral_by_joint[joint_name("hip_revolute", model_leg_id)] + pose.hip
        )
        desired_by_joint[joint_name("knee_prismatic", model_leg_id)] = (
            neutral_by_joint[joint_name("knee_prismatic", model_leg_id)] + pose.knee
        )
    return desired_by_joint


def ctrl_targets_from_phase_pose(joint_names, neutral_by_joint, phase_pose):
    desired_by_joint = desired_joint_targets_from_phase_pose(neutral_by_joint, phase_pose)
    return np.array([desired_by_joint[joint_name] for joint_name in joint_names], dtype=np.float64)


def ease_in_out_sine(alpha):
    alpha = np.clip(alpha, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * alpha)


def oscillator_phase_angle(phase, phase_timer, phase_steps):
    total_steps = max(int(2 * phase_steps), 1)
    cycle_step = (int(phase) % 2) * int(phase_steps) + float(phase_timer)
    return 2.0 * np.pi * (cycle_step / total_steps)


def oscillator_phase_pose_for_progress(phase, phase_timer, phase_steps):
    signal = float(np.sin(oscillator_phase_angle(phase, phase_timer, phase_steps)))
    outer_signal = -OSCILLATOR_OUTER_SCALE * signal
    middle_signal = OSCILLATOR_MIDDLE_SCALE * signal

    outer_pose = PhasePose(
        shoulder=OSCILLATOR_SHOULDER_AMPLITUDE * outer_signal,
        hip=OSCILLATOR_HIP_AMPLITUDE * outer_signal,
        knee=OSCILLATOR_KNEE_AMPLITUDE * outer_signal,
    )
    middle_pose = PhasePose(
        shoulder=OSCILLATOR_SHOULDER_AMPLITUDE * middle_signal,
        hip=OSCILLATOR_HIP_AMPLITUDE * middle_signal,
        knee=OSCILLATOR_KNEE_AMPLITUDE * middle_signal,
    )
    return LegPhasePose(
        l0=outer_pose,
        l1=middle_pose,
        l2=middle_pose,
        l3=outer_pose,
    )


def oscillator_ctrl_targets_for_phase_progress(joint_names, neutral_by_joint, phase, phase_timer, phase_steps):
    return ctrl_targets_from_phase_pose(
        joint_names,
        neutral_by_joint,
        oscillator_phase_pose_for_progress(phase, phase_timer, phase_steps),
    )


def reference_ctrl_targets_for_phase_progress(phase, phase_timer, phase_steps, phase1_ctrl, phase2_ctrl):
    progress = 0.0 if phase_steps <= 0 else float(np.clip(phase_timer / phase_steps, 0.0, 1.0))
    eased = ease_in_out_sine(progress)
    if phase == 0:
        return (1.0 - eased) * phase1_ctrl + eased * phase2_ctrl
    return (1.0 - eased) * phase2_ctrl + eased * phase1_ctrl
