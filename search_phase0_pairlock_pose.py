import itertools

import mujoco
import numpy as np

from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR


PHASE = 0
SETTLE_STEPS = 300
OUTER_SHOULDER_OFFSETS = (-0.10, -0.05, 0.0, 0.05)
OUTER_HIP_OFFSETS = (-0.50, -0.25, 0.0, 0.25, 0.50)
OUTER_KNEE_OFFSETS = (-0.10, -0.05, 0.0, 0.05)
MIDDLE_SHOULDER_OFFSETS = (-0.10, -0.05, 0.0, 0.05)
MIDDLE_HIP_OFFSETS = (-0.50, -0.25, 0.0, 0.25, 0.50)
MIDDLE_KNEE_OFFSETS = (-0.10, -0.05, 0.0, 0.05)


def fmt_pattern(values):
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


def apply_pair_offsets(ctrl, pair, shoulder, hip, knee):
    for leg_id in pair:
        ctrl[leg_id] += shoulder
        ctrl[4 + leg_id] += hip
        ctrl[8 + leg_id] += knee


def evaluate_offsets(env, outer_shoulder, outer_hip, outer_knee, middle_shoulder, middle_hip, middle_knee):
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[:7] = env.model.qpos0[:7]
    env.data.qpos[2] = env.spawn_height
    env.data.qpos[3:7] = [1, 0, 0, 0]

    ctrl = env.phase_reference_ctrl[PHASE].copy()
    apply_pair_offsets(ctrl, (0, 3), outer_shoulder, outer_hip, outer_knee)
    apply_pair_offsets(ctrl, (1, 2), middle_shoulder, middle_hip, middle_knee)
    ctrl = env._enforce_tripedal_pair_lock(ctrl)

    for index, joint_name in enumerate(env.joint_names):
        env.data.joint(joint_name).qpos[0] = ctrl[index]
    env.data.ctrl[:] = ctrl
    mujoco.mj_forward(env.model, env.data)
    env._ground_support_feet(PHASE)
    mujoco.mj_forward(env.model, env.data)
    env._settle_with_high_gains(n_steps=SETTLE_STEPS)
    mujoco.mj_forward(env.model, env.data)

    foot_on_ground = env._foot_on_ground_flags()
    contact_counts = env._foot_floor_contact_counts()
    fully_planted, planted_quality = env._foot_fully_planted_flags(
        foot_on_ground=foot_on_ground,
        contact_counts=contact_counts,
    )
    desired = [int(x) for x in env._desired_foot_contacts_for_time(phase=PHASE, phase_timer=0)]
    actual = [int(x) for x in foot_on_ground]
    planted = [int(x) for x in fully_planted]
    heights = [float(env._foot_world_height(i)) for i in range(4)]
    centered_hips = env._centered_hip_thetas()

    score = 0.0
    score += 200.0 * planted[0]
    score += 200.0 * planted[3]
    score -= 160.0 * planted[1]
    score -= 160.0 * planted[2]
    score += 80.0 * actual[0]
    score += 80.0 * actual[3]
    score -= 60.0 * actual[1]
    score -= 60.0 * actual[2]
    score += 40.0 * min(heights[1], 0.04)
    score += 40.0 * min(heights[2], 0.04)
    score -= 30.0 * max(heights[0], 0.0)
    score -= 30.0 * max(heights[3], 0.0)
    score -= 10.0 * abs(float(centered_hips[0] - centered_hips[3]))
    score -= 10.0 * abs(float(centered_hips[1] - centered_hips[2]))
    score += 20.0 * float(np.mean(planted_quality[[0, 3]]))
    score -= 20.0 * float(np.mean(planted_quality[[1, 2]]))

    return {
        "score": score,
        "desired": desired,
        "actual": actual,
        "planted": planted,
        "planted_quality": [float(v) for v in planted_quality],
        "heights": heights,
        "centered_hips": [float(v) for v in centered_hips],
    }


def main():
    env = TARSEnv(model_path=DEFAULT_MODEL_PATH_STR)
    results = []
    for values in itertools.product(
        OUTER_SHOULDER_OFFSETS,
        OUTER_HIP_OFFSETS,
        OUTER_KNEE_OFFSETS,
        MIDDLE_SHOULDER_OFFSETS,
        MIDDLE_HIP_OFFSETS,
        MIDDLE_KNEE_OFFSETS,
    ):
        result = evaluate_offsets(env, *values)
        results.append((result["score"], values, result))

    results.sort(reverse=True, key=lambda item: item[0])
    print(f"tested={len(results)}")
    for rank, (score, values, result) in enumerate(results[:15], start=1):
        outer_s, outer_h, outer_k, middle_s, middle_h, middle_k = values
        print(
            f"rank={rank} score={score:.3f} "
            f"outer=({outer_s:+.3f},{outer_h:+.3f},{outer_k:+.3f}) "
            f"middle=({middle_s:+.3f},{middle_h:+.3f},{middle_k:+.3f}) "
            f"desired={fmt_pattern(result['desired'])} actual={fmt_pattern(result['actual'])} "
            f"planted={fmt_pattern(result['planted'])} "
            f"qualities={[round(v, 3) for v in result['planted_quality']]} "
            f"heights={[round(v, 4) for v in result['heights']]}"
        )


if __name__ == "__main__":
    main()
