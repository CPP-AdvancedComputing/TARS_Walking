import argparse
import time
from pathlib import Path

import numpy as np
from mujoco import viewer as mj_viewer
from stable_baselines3 import PPO

from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR


def parse_args():
    parser = argparse.ArgumentParser(description="Slow-motion gait inspector for TARS.")
    parser.add_argument(
        "--mode",
        choices=("zero", "policy"),
        default="zero",
        help="Use zero actions or a trained PPO policy.",
    )
    parser.add_argument(
        "--policy-path",
        default="tars_policy.zip",
        help="Policy checkpoint to load when --mode policy is used.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH_STR,
        help="MJCF/URDF model to inspect.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=120,
        help="Number of env action-steps to inspect.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.20,
        help="Wall-clock delay after each action-step.",
    )
    parser.add_argument(
        "--phase-pause-sec",
        type=float,
        default=0.75,
        help="Extra pause when the phase changes.",
    )
    return parser.parse_args()


def format_contacts(values):
    return "[" + ",".join(str(int(v)) for v in values) + "]"


def main():
    args = parse_args()
    env = TARSEnv(model_path=args.model_path)
    policy = None
    if args.mode == "policy":
        policy_path = Path(args.policy_path)
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        policy = PPO.load(str(policy_path))

    obs, _ = env.reset()
    prev_phase = env.phase
    print(f"Inspector mode: {args.mode}")
    print(f"Steps: {args.steps}")
    print(f"Sleep per step: {args.sleep_sec:.2f}s")
    print(f"Extra pause on phase switch: {args.phase_pause_sec:.2f}s")

    with mj_viewer.launch_passive(env.model, env.data) as viewer:
        viewer.sync()
        time.sleep(max(args.sleep_sec, 0.0))
        for step in range(args.steps):
            if not viewer.is_running():
                break

            if policy is None:
                action = env.zero_action()
            else:
                action, _ = policy.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)
            desired = env._desired_foot_contacts_for_time(phase=env.phase, phase_timer=env.phase_timer)
            actual = env._foot_on_ground_flags()
            heights = [env._foot_world_height(i) for i in range(4)]

            print(
                f"step={step + 1:03d} "
                f"phase={env.phase} timer={env.phase_timer:02d} "
                f"reward={reward:+.3f} "
                f"desired={format_contacts(desired)} "
                f"actual={format_contacts(actual)} "
                f"heights={[round(h, 4) for h in heights]}"
            )

            viewer.sync()
            time.sleep(max(args.sleep_sec, 0.0))

            if env.phase != prev_phase:
                print(f"phase switch: {prev_phase} -> {env.phase}")
                time.sleep(max(args.phase_pause_sec, 0.0))
                prev_phase = env.phase

            if terminated or truncated:
                print(
                    f"episode reset after step {step + 1}: "
                    f"terminated={terminated} truncated={truncated}"
                )
                obs, _ = env.reset()
                viewer.sync()
                time.sleep(max(args.phase_pause_sec, 0.0))
                prev_phase = env.phase


if __name__ == "__main__":
    main()
