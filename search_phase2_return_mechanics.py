import itertools

import mujoco

from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR


PHASE = 2
STEPS = 70
OUTER_HIP_OFFSETS = (-0.50, -0.25, 0.0, 0.25, 0.50)
OUTER_KNEE_OFFSETS = (-0.10, 0.0, 0.10)
MIDDLE_HIP_OFFSETS = (-0.50, -0.25, 0.0, 0.25)
MIDDLE_KNEE_OFFSETS = (-0.10, 0.0, 0.10)


def run_candidate(env, outer_hip, outer_knee, middle_hip, middle_knee):
    original_phase_bias = env.PHASE_CTRL_BIAS[PHASE].copy()
    try:
        patched = original_phase_bias.copy()
        for leg_id in (0, 3):
            patched[4 + leg_id] += outer_hip
            patched[8 + leg_id] += outer_knee
        for leg_id in (1, 2):
            patched[4 + leg_id] += middle_hip
            patched[8 + leg_id] += middle_knee
        env.PHASE_CTRL_BIAS[PHASE] = patched
        env.phase_reference_ctrl = env._build_phase_reference_ctrl()
        env._calibrate_phase_control_corrections()

        mujoco.mj_resetData(env.model, env.data)
        env._set_phase_pose(PHASE)
        env._initialize_rod_targets()
        env._begin_phase(PHASE)
        env.data.ctrl[:] = env.control_targets_for_action(env.zero_action(), phase=PHASE)
        mujoco.mj_forward(env.model, env.data)
        env.initial_height = float(env.data.qpos[2])
        env.initial_body_height = float(env.data.body("internals").xpos[2])
        env.prev_body_z = float(env.data.qpos[2])
        env.prev_body_vz = float(env.data.qvel[2])
        env.prev_x = float(env.data.qpos[0])
        env.stagnation_x = float(env.data.qpos[0])
        env.stagnation_timer = 0
        env.step_count = 0

        best_outer_fwd = -1e9
        best_middle_edge = -1.0
        best_next_contact = -1.0
        best_next_low = -1.0
        best_swing_clear = -1.0
        phase0_switch_step = None

        for step in range(STEPS):
            foot_on_ground = env._foot_on_ground_flags()
            planted_flags, planted_qualities = env._foot_fully_planted_flags(foot_on_ground=foot_on_ground)
            track_x = [env._track_vector_body(i)[0] for i in range(4)]
            outer_fwd = 0.5 * (track_x[0] + track_x[3])
            middle_edge = max(
                float(foot_on_ground[1] and not planted_flags[1]),
                float(foot_on_ground[2] and not planted_flags[2]),
            )
            best_outer_fwd = max(best_outer_fwd, float(outer_fwd))
            best_middle_edge = max(best_middle_edge, float(middle_edge))
            best_next_contact = max(best_next_contact, float(env.last_next_plant_contact_fraction))
            best_next_low = max(best_next_low, float(env.last_next_plant_low_fraction))
            best_swing_clear = max(best_swing_clear, float(env.last_switch_swing_clear_fraction))

            ready, _, _, _ = env._phase_switch_ready(PHASE)
            if env.phase == 0:
                phase0_switch_step = step
                break
            env.step(env.zero_action())

        score = (
            4000.0 * best_outer_fwd
            + 1200.0 * best_middle_edge
            + 600.0 * best_next_contact
            + 400.0 * best_next_low
            + 800.0 * best_swing_clear
        )
        if phase0_switch_step is not None:
            score += 4000.0 - 20.0 * phase0_switch_step

        return {
            "score": score,
            "switch_step": phase0_switch_step,
            "best_outer_fwd": best_outer_fwd,
            "best_middle_edge": best_middle_edge,
            "best_next_contact": best_next_contact,
            "best_next_low": best_next_low,
            "best_swing_clear": best_swing_clear,
        }
    finally:
        env.PHASE_CTRL_BIAS[PHASE] = original_phase_bias
        env.phase_reference_ctrl = env._build_phase_reference_ctrl()
        env._calibrate_phase_control_corrections()


def main():
    env = TARSEnv(model_path=DEFAULT_MODEL_PATH_STR)
    results = []
    for candidate in itertools.product(
        OUTER_HIP_OFFSETS,
        OUTER_KNEE_OFFSETS,
        MIDDLE_HIP_OFFSETS,
        MIDDLE_KNEE_OFFSETS,
    ):
        outcome = run_candidate(env, *candidate)
        results.append((outcome["score"], candidate, outcome))

    results.sort(reverse=True, key=lambda item: item[0])
    print(f"tested={len(results)}")
    for rank, (score, candidate, outcome) in enumerate(results[:20], start=1):
        outer_hip, outer_knee, middle_hip, middle_knee = candidate
        print(
            f"rank={rank} score={score:.2f} "
            f"outer=({outer_hip:+.3f},{outer_knee:+.3f}) "
            f"middle=({middle_hip:+.3f},{middle_knee:+.3f}) "
            f"switch_step={outcome['switch_step']} "
            f"best_outer_fwd={outcome['best_outer_fwd']:.3f} "
            f"best_middle_edge={outcome['best_middle_edge']:.2f} "
            f"best_next_contact={outcome['best_next_contact']:.2f} "
            f"best_next_low={outcome['best_next_low']:.2f} "
            f"best_swing_clear={outcome['best_swing_clear']:.2f}"
        )


if __name__ == "__main__":
    main()
