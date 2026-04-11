import argparse

import mujoco

from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose static support integrity for each gait phase.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH_STR, help="MJCF/URDF model to inspect.")
    parser.add_argument("--hold-steps", type=int, default=20, help="MuJoCo steps to hold each phase after applying zero action.")
    return parser.parse_args()


def fmt_list(values):
    return "[" + ", ".join(f"{float(value):+.4f}" for value in values) + "]"


def main():
    args = parse_args()
    env = TARSEnv(model_path=args.model_path)

    for phase in (0, 1):
        mujoco.mj_resetData(env.model, env.data)
        env._set_phase_pose(phase)
        env._initialize_rod_targets()
        env._begin_phase(phase)
        env.data.ctrl[:] = env._scale_action(env.zero_action(), phase=phase)
        for _ in range(max(args.hold_steps, 0)):
            mujoco.mj_step(env.model, env.data)

        desired = env._desired_foot_contacts_for_time(phase=phase, phase_timer=env.phase_timer)
        actual = env._foot_on_ground_flags()
        heights = [env._foot_world_height(i) for i in range(4)]
        body_pos = env.data.body("internals").xpos.copy()

        print(f"\nphase {phase}")
        print(f"  desired contacts: {[int(v) for v in desired]}")
        print(f"  actual contacts:  {[int(v) for v in actual]}")
        print(f"  foot heights:     {fmt_list(heights)}")
        print(f"  body position:    {fmt_list(body_pos)}")


if __name__ == "__main__":
    main()
