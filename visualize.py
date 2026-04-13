import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    imageio = None

try:
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover
    PPO = None


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize and diagnose TARS gait rollouts.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH_STR)
    parser.add_argument("--policy", default=None, help="Optional PPO policy zip.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--phase", type=int, default=None, help="Optional forced start phase.")
    parser.add_argument("--diag", action="store_true", help="Print periodic per-step diagnostics.")
    parser.add_argument("--phase-diag", action="store_true", help="Print summarized phase contact diagnostics.")
    parser.add_argument("--trace-every", type=int, default=10)
    parser.add_argument("--viewer", action="store_true", help="Open passive MuJoCo viewer.")
    parser.add_argument("--sleep-sec", type=float, default=0.02)
    parser.add_argument("--out", default=None, help="Optional output video/gif path.")
    parser.add_argument("--no-render", action="store_true", help="Skip offscreen rendering even if --out is set.")
    return parser.parse_args()


def configure_camera(camera, env):
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = env.data.body("internals").xpos
    camera.lookat[1] += 0.12
    camera.lookat[2] += 0.18
    camera.distance = 2.0
    camera.azimuth = 138.0
    camera.elevation = -15.0


def resolve_policy(policy_arg):
    if policy_arg is None:
        return None
    if PPO is None:
        raise ImportError("stable_baselines3 is required to load a policy.")
    path = Path(policy_arg)
    if not path.exists():
        raise FileNotFoundError(f"Policy not found: {path}")
    return PPO.load(str(path))


def build_renderer(env, output_path, no_render):
    if output_path is None or no_render:
        return None
    return mujoco.Renderer(env.model, height=540, width=960)


def render_frame(renderer, env):
    cam = mujoco.MjvCamera()
    configure_camera(cam, env)
    renderer.update_scene(env.data, camera=cam)
    return renderer.render().copy()


def save_video(frames, output_path):
    if output_path is None or not frames:
        return
    if imageio is None:
        raise ImportError("imageio is required to save rollout video.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=20)


def action_for_step(env, model, obs):
    if model is not None:
        action, _ = model.predict(obs, deterministic=True)
        return action
    return np.zeros(env.action_space.shape, dtype=np.float32)


def collect_phase_diag(env, phase_stats):
    phase = int(env.phase)
    entry = phase_stats.setdefault(
        phase,
        {
            "steps": 0,
            "contacts": np.zeros(4, dtype=np.float64),
            "planted": np.zeros(4, dtype=np.float64),
            "desired": np.asarray(env.PHASE_CONTACT_MASKS.get(phase, np.zeros(4)), dtype=np.float64),
        },
    )
    contacts = np.asarray(env._foot_on_ground_flags(), dtype=np.float64)
    planted, _ = env._foot_fully_planted_flags(contacts)
    entry["steps"] += 1
    entry["contacts"] += contacts
    entry["planted"] += np.asarray(planted, dtype=np.float64)


def print_diag(env, step, reward):
    contacts = [int(v) for v in env._foot_on_ground_flags()]
    planted, planted_quality = env._foot_fully_planted_flags(env._foot_on_ground_flags())
    centered = env._centered_hip_thetas()
    print(
        f"step={step:03d} phase={env.phase} timer={env.phase_timer:02d} "
        f"x={env.data.qpos[0]:+.4f} reward={reward:+.3f} "
        f"contacts={contacts} planted={[int(v) for v in planted]}"
    )
    print(
        f"  theta outer=({centered[0]:+.4f},{centered[3]:+.4f}) "
        f"middle=({centered[1]:+.4f},{centered[2]:+.4f}) "
        f"forward={np.round([env._foot_forward_offset(i) for i in range(4)], 4).tolist()}"
    )
    print(f"  planted_quality={np.round(planted_quality, 3).tolist()}")


def print_phase_diag(phase_stats):
    for phase in sorted(phase_stats):
        stats = phase_stats[phase]
        steps = max(stats["steps"], 1)
        actual = (stats["contacts"] / steps).round(2).tolist()
        planted = (stats["planted"] / steps).round(2).tolist()
        desired = stats["desired"].round(2).tolist()
        print(
            f"Phase {phase}: actual={actual} planted={planted} desired={desired} steps={stats['steps']}"
        )


def main():
    args = parse_args()
    env = TARSEnv(model_path=args.model_path)
    model = resolve_policy(args.policy)
    obs, _ = env.reset(start_phase=args.phase)
    phase_stats = {}
    frames = []
    renderer = build_renderer(env, args.out, args.no_render)

    viewer = None
    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    try:
        for step in range(args.steps):
            collect_phase_diag(env, phase_stats)
            if renderer is not None:
                frames.append(render_frame(renderer, env))
            if args.diag and step % max(args.trace_every, 1) == 0:
                print_diag(env, step, 0.0)

            action = action_for_step(env, model, obs)
            obs, reward, terminated, truncated, _ = env.step(action)

            if args.diag and step % max(args.trace_every, 1) == 0:
                print_diag(env, step, reward)

            if viewer is not None:
                viewer.sync()
                time.sleep(max(args.sleep_sec, 0.0))

            if terminated or truncated:
                break
    finally:
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()

    if args.phase_diag:
        print_phase_diag(phase_stats)

    if args.out is not None:
        save_video(frames, args.out)
        print(f"Saved rollout to {args.out}")

    summary = {
        "final_phase": int(env.phase),
        "final_phase_timer": int(env.phase_timer),
        "final_x": float(env.data.qpos[0]),
        "phase_diag": {
            str(phase): {
                "steps": int(stats["steps"]),
                "actual": (stats["contacts"] / max(stats["steps"], 1)).round(4).tolist(),
                "planted": (stats["planted"] / max(stats["steps"], 1)).round(4).tolist(),
                "desired": stats["desired"].round(4).tolist(),
            }
            for phase, stats in sorted(phase_stats.items())
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
