import itertools

import numpy as np

from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR


PHASE = 0
STEPS = 60

OUTER_SHOULDER_OFFSETS = (-0.05, 0.0, 0.05)
OUTER_HIP_OFFSETS = (0.15, 0.25, 0.35)
OUTER_KNEE_OFFSETS = (-0.10, -0.05, 0.0)
MIDDLE_SHOULDER_OFFSETS = (-0.15, -0.10, -0.05)
MIDDLE_HIP_OFFSETS = (-0.35, -0.25, -0.15)
MIDDLE_KNEE_OFFSETS = (-0.15, -0.10, -0.05)


def run_candidate(env, outer_offset, middle_offset):
    original = {
        pair: offset.copy()
        for pair, offset in env.PHASE_PAIR_CTRL_OFFSETS.get(PHASE, {}).items()
    }
    try:
        env.PHASE_PAIR_CTRL_OFFSETS[PHASE] = {
            (0, 3): np.asarray(outer_offset, dtype=np.float64),
            (1, 2): np.asarray(middle_offset, dtype=np.float64),
        }
        env.phase_reference_ctrl = env._build_phase_reference_ctrl()
        env._calibrate_phase_control_corrections()
        env.reset(start_phase=PHASE)

        target_mask = np.asarray([1, 0, 0, 1], dtype=np.float64)
        best_match = -1.0
        mean_match = 0.0
        mean_outer_contact = 0.0
        mean_middle_clear = 0.0
        mean_reward = 0.0
        switched = False
        for step in range(STEPS):
            foot_on_ground = env._foot_on_ground_flags()
            planted_flags, _ = env._foot_fully_planted_flags(foot_on_ground=foot_on_ground)
            planted = np.asarray(planted_flags, dtype=np.float64)
            match = 1.0 - float(np.mean(np.abs(planted - target_mask)))
            best_match = max(best_match, match)
            mean_match += match
            mean_outer_contact += 0.5 * (planted[0] + planted[3])
            mean_middle_clear += 1.0 - 0.5 * (planted[1] + planted[2])
            _, reward, terminated, truncated, _ = env.step(env.zero_action())
            mean_reward += float(reward)
            if env.phase != PHASE:
                switched = True
                break
            if terminated or truncated:
                break

        denom = max(step + 1, 1)
        score = (
            400.0 * (mean_match / denom)
            + 200.0 * (mean_outer_contact / denom)
            + 200.0 * (mean_middle_clear / denom)
            + 20.0 * best_match
            + 0.1 * (mean_reward / denom)
        )
        if switched:
            score -= 500.0
        return {
            "score": score,
            "best_match": best_match,
            "mean_match": mean_match / denom,
            "mean_outer_contact": mean_outer_contact / denom,
            "mean_middle_clear": mean_middle_clear / denom,
            "mean_reward": mean_reward / denom,
            "switched": switched,
        }
    finally:
        env.PHASE_PAIR_CTRL_OFFSETS[PHASE] = original
        env.phase_reference_ctrl = env._build_phase_reference_ctrl()
        env._calibrate_phase_control_corrections()


def main():
    env = TARSEnv(model_path=DEFAULT_MODEL_PATH_STR)
    results = []
    for candidate in itertools.product(
        OUTER_SHOULDER_OFFSETS,
        OUTER_HIP_OFFSETS,
        OUTER_KNEE_OFFSETS,
        MIDDLE_SHOULDER_OFFSETS,
        MIDDLE_HIP_OFFSETS,
        MIDDLE_KNEE_OFFSETS,
    ):
        outer_offset = candidate[:3]
        middle_offset = candidate[3:]
        outcome = run_candidate(env, outer_offset, middle_offset)
        results.append((outcome["score"], outer_offset, middle_offset, outcome))

    results.sort(reverse=True, key=lambda item: item[0])
    print(f"tested={len(results)}")
    for rank, (score, outer_offset, middle_offset, outcome) in enumerate(results[:20], start=1):
        print(
            f"rank={rank} score={score:.3f} "
            f"outer=({outer_offset[0]:+.3f},{outer_offset[1]:+.3f},{outer_offset[2]:+.3f}) "
            f"middle=({middle_offset[0]:+.3f},{middle_offset[1]:+.3f},{middle_offset[2]:+.3f}) "
            f"best_match={outcome['best_match']:.3f} "
            f"mean_match={outcome['mean_match']:.3f} "
            f"outer_contact={outcome['mean_outer_contact']:.3f} "
            f"middle_clear={outcome['mean_middle_clear']:.3f} "
            f"mean_reward={outcome['mean_reward']:.3f} "
            f"switched={int(outcome['switched'])}"
        )


if __name__ == "__main__":
    main()
