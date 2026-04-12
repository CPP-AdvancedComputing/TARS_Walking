import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR

try:
    import mujoco.viewer
except ImportError:  # pragma: no cover
    mujoco = None

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover
    imageio = None

try:
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover
    PPO = None


ROOT = Path(__file__).resolve().parent
PAIR_OUTER = (0, 3)
PAIR_MIDDLE = (1, 2)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render and verify TARS against the high-priority pair-lock / effective-tripedal gait constraint."
        )
    )
    parser.add_argument("--mode", choices=("reference", "zero", "policy"), default="reference")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH_STR)
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--start-phase", type=int, default=0, help="Canonical phase to start verification from.")
    parser.add_argument("--viewer", action="store_true", help="Show the passive MuJoCo viewer.")
    parser.add_argument("--sleep-sec", type=float, default=0.02)
    parser.add_argument("--trace-every", type=int, default=10)
    parser.add_argument("--render-gif", default=None, help="Optional GIF path for offscreen rendering.")
    parser.add_argument("--report-json", default="run_logs/tripedal_pair_report.json")
    parser.add_argument("--theta-tol", type=float, default=0.05)
    parser.add_argument("--joint-tol", type=float, default=0.08)
    parser.add_argument("--foot-tol", type=float, default=0.05)
    return parser.parse_args()


def resolve_policy_path(policy_arg):
    if policy_arg is None:
        default_zip = ROOT / "tars_policy.zip"
        if default_zip.exists():
            return default_zip
        checkpoint_dir = ROOT / "checkpoints"
        if checkpoint_dir.is_dir():
            candidates = sorted(checkpoint_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
            if candidates:
                return candidates[0]
        raise FileNotFoundError("No policy file found. Pass --policy-path PATH.")
    path = Path(policy_arg)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    return path


def configure_camera(camera, env):
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = env.data.body("internals").xpos
    camera.lookat[1] += 0.10
    camera.lookat[2] += 0.18
    camera.distance = 2.0
    camera.azimuth = 138.0
    camera.elevation = -15.0


def pair_joint_metrics(env, pair):
    a, b = pair
    qpos = env._current_joint_state()
    shoulder_diff = abs(float(qpos[a] - qpos[b]))
    hip_diff = abs(float(qpos[4 + a] - qpos[4 + b]))
    knee_diff = abs(float(qpos[8 + a] - qpos[8 + b]))
    foot_track_diff = float(np.linalg.norm(env._foot_track_world(a) - env._foot_track_world(b)))
    return {
        "shoulder_diff": shoulder_diff,
        "hip_diff": hip_diff,
        "knee_diff": knee_diff,
        "theta_diff": hip_diff,
        "foot_track_diff": foot_track_diff,
    }


def _body_frame_foot_tracks(env):
    origin = np.asarray(env.data.body("internals").xpos, dtype=np.float64)
    rotation = env._body_rotation().T
    return {
        leg_id: rotation @ (env._foot_track_world(leg_id) - origin)
        for leg_id in range(4)
    }


def snapshot_metrics(env, step, reward, baseline):
    outer = pair_joint_metrics(env, PAIR_OUTER)
    middle = pair_joint_metrics(env, PAIR_MIDDLE)
    centered_hips = env._centered_hip_thetas()
    foot_forward_offsets = [float(env._foot_forward_offset(i)) for i in range(4)]
    contacts = [int(v) for v in env._foot_on_ground_flags()]
    planted, planted_quality = env._foot_fully_planted_flags(env._foot_on_ground_flags())
    outer_mean_theta = float(np.mean(centered_hips[list(PAIR_OUTER)]))
    middle_mean_theta = float(np.mean(centered_hips[list(PAIR_MIDDLE)]))
    current_tracks = _body_frame_foot_tracks(env)
    outer_delta_diff = float(
        np.linalg.norm(
            (current_tracks[PAIR_OUTER[0]] - baseline["foot_track_body"][PAIR_OUTER[0]])
            - (current_tracks[PAIR_OUTER[1]] - baseline["foot_track_body"][PAIR_OUTER[1]])
        )
    )
    middle_delta_diff = float(
        np.linalg.norm(
            (current_tracks[PAIR_MIDDLE[0]] - baseline["foot_track_body"][PAIR_MIDDLE[0]])
            - (current_tracks[PAIR_MIDDLE[1]] - baseline["foot_track_body"][PAIR_MIDDLE[1]])
        )
    )
    return {
        "step": int(step),
        "phase": int(env.phase),
        "phase_timer": int(env.phase_timer),
        "reward": float(reward),
        "x": float(env.data.qpos[0]),
        "outer_pair": outer,
        "middle_pair": middle,
        "outer_centered_hip_mean": outer_mean_theta,
        "middle_centered_hip_mean": middle_mean_theta,
        "pair_mean_theta_gap": abs(outer_mean_theta - middle_mean_theta),
        "outer_pair_delta_diff": outer_delta_diff,
        "middle_pair_delta_diff": middle_delta_diff,
        "foot_forward_offsets": foot_forward_offsets,
        "outer_forward_mean": float(0.5 * (foot_forward_offsets[0] + foot_forward_offsets[3])),
        "middle_forward_mean": float(0.5 * (foot_forward_offsets[1] + foot_forward_offsets[2])),
        "outer_forward_pair_diff": abs(foot_forward_offsets[0] - foot_forward_offsets[3]),
        "middle_forward_pair_diff": abs(foot_forward_offsets[1] - foot_forward_offsets[2]),
        "contacts": contacts,
        "planted": [int(v) for v in planted],
        "planted_quality": [float(v) for v in planted_quality],
    }


def update_summary(summary, metrics):
    for pair_key in ("outer_pair", "middle_pair"):
        pair_metrics = metrics[pair_key]
        for metric_key, value in pair_metrics.items():
            stat_key = f"{pair_key}.{metric_key}"
            summary["max"][stat_key] = max(summary["max"].get(stat_key, 0.0), float(value))
            summary["sum"][stat_key] = summary["sum"].get(stat_key, 0.0) + float(value)
    summary["steps"] += 1
    for scalar_key in ("pair_mean_theta_gap", "outer_pair_delta_diff", "middle_pair_delta_diff"):
        value = float(metrics[scalar_key])
        summary["max"][scalar_key] = max(summary["max"].get(scalar_key, 0.0), value)
        summary["sum"][scalar_key] = summary["sum"].get(scalar_key, 0.0) + value
    for scalar_key in (
        "outer_forward_mean",
        "middle_forward_mean",
        "outer_forward_pair_diff",
        "middle_forward_pair_diff",
    ):
        value = float(abs(metrics[scalar_key])) if scalar_key.endswith("_mean") else float(metrics[scalar_key])
        summary["max"][scalar_key] = max(summary["max"].get(scalar_key, 0.0), value)
        summary["sum"][scalar_key] = summary["sum"].get(scalar_key, 0.0) + value


def finalize_summary(summary, args):
    means = {}
    for key, total in summary["sum"].items():
        means[key] = total / max(summary["steps"], 1)
    max_stats = summary["max"]
    verdict = {
        "outer_theta_ok": max_stats.get("outer_pair.theta_diff", np.inf) <= args.theta_tol,
        "middle_theta_ok": max_stats.get("middle_pair.theta_diff", np.inf) <= args.theta_tol,
        "outer_joint_pairing_ok": max(
            max_stats.get("outer_pair.shoulder_diff", np.inf),
            max_stats.get("outer_pair.hip_diff", np.inf),
            max_stats.get("outer_pair.knee_diff", np.inf),
        ) <= args.joint_tol,
        "middle_joint_pairing_ok": max(
            max_stats.get("middle_pair.shoulder_diff", np.inf),
            max_stats.get("middle_pair.hip_diff", np.inf),
            max_stats.get("middle_pair.knee_diff", np.inf),
        ) <= args.joint_tol,
        "outer_foot_pairing_ok": max_stats.get("outer_pair_delta_diff", np.inf) <= args.foot_tol,
        "middle_foot_pairing_ok": max_stats.get("middle_pair_delta_diff", np.inf) <= args.foot_tol,
    }
    verdict["pass"] = all(verdict.values())
    return {"max": max_stats, "mean": means, "verdict": verdict, "steps": summary["steps"]}


def maybe_make_renderer(env, render_gif):
    if render_gif is None:
        return None
    return mujoco.Renderer(env.model, height=540, width=960)


def render_frame(renderer, env):
    cam = mujoco.MjvCamera()
    configure_camera(cam, env)
    renderer.update_scene(env.data, camera=cam)
    return renderer.render().copy()


def maybe_save_gif(frames, output_path):
    if output_path is None:
        return
    if imageio is None:
        raise ImportError("imageio is required to save GIF output.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=20)


def print_trace(metrics):
    print(
        f"step={metrics['step']:03d} phase={metrics['phase']} timer={metrics['phase_timer']:02d} "
        f"x={metrics['x']:+.4f} reward={metrics['reward']:+.3f} "
        f"contacts={metrics['contacts']} planted={metrics['planted']}"
    )
    print(
        f"  outer(theta={metrics['outer_pair']['theta_diff']:.4f}, shoulder={metrics['outer_pair']['shoulder_diff']:.4f}, "
        f"hip={metrics['outer_pair']['hip_diff']:.4f}, knee={metrics['outer_pair']['knee_diff']:.4f}, "
        f"foot={metrics['outer_pair']['foot_track_diff']:.4f})"
    )
    print(
        f"  middle(theta={metrics['middle_pair']['theta_diff']:.4f}, shoulder={metrics['middle_pair']['shoulder_diff']:.4f}, "
        f"hip={metrics['middle_pair']['hip_diff']:.4f}, knee={metrics['middle_pair']['knee_diff']:.4f}, "
        f"foot={metrics['middle_pair']['foot_track_diff']:.4f})"
    )
    print(
        f"  pair_mean_theta_gap={metrics['pair_mean_theta_gap']:.4f} "
        f"outer_mean={metrics['outer_centered_hip_mean']:.4f} "
        f"middle_mean={metrics['middle_centered_hip_mean']:.4f}"
    )
    print(
        f"  pair_delta_diff(outer={metrics['outer_pair_delta_diff']:.4f}, "
        f"middle={metrics['middle_pair_delta_diff']:.4f})"
    )
    print(
        f"  forward_mean(outer={metrics['outer_forward_mean']:+.4f}, "
        f"middle={metrics['middle_forward_mean']:+.4f}) "
        f"forward_pair_diff(outer={metrics['outer_forward_pair_diff']:.4f}, "
        f"middle={metrics['middle_forward_pair_diff']:.4f})"
    )


def main():
    args = parse_args()
    env = TARSEnv(model_path=args.model_path)
    obs, _ = env.reset(start_phase=args.start_phase)

    policy = None
    if args.mode == "policy":
        if PPO is None:
            raise ImportError("stable_baselines3 is required for --mode policy")
        policy = PPO.load(str(resolve_policy_path(args.policy_path)))

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {"max": {}, "sum": {}, "steps": 0}
    trace = []
    frames = []
    renderer = maybe_make_renderer(env, args.render_gif)
    baseline = {
        "foot_track_body": _body_frame_foot_tracks(env),
    }
    initial_metrics = snapshot_metrics(env, -1, 0.0, baseline)
    print("initial_state")
    print_trace(initial_metrics)

    if args.viewer:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            configure_camera(viewer.cam, env)
            for step in range(args.steps):
                if args.mode in ("reference", "zero"):
                    action = env.zero_action()
                else:
                    action = policy.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, _ = env.step(action)
                metrics = snapshot_metrics(env, step, reward, baseline)
                trace.append(metrics)
                update_summary(summary, metrics)
                if step % max(args.trace_every, 1) == 0:
                    print_trace(metrics)
                if renderer is not None:
                    frames.append(render_frame(renderer, env))
                viewer.sync()
                time.sleep(max(args.sleep_sec, 0.0))
                if terminated or truncated:
                    obs, _ = env.reset(start_phase=args.start_phase)
                    baseline = {
                        "foot_track_body": _body_frame_foot_tracks(env),
                    }
                    configure_camera(viewer.cam, env)
    else:
        for step in range(args.steps):
            if args.mode in ("reference", "zero"):
                action = env.zero_action()
            else:
                action = policy.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            metrics = snapshot_metrics(env, step, reward, baseline)
            trace.append(metrics)
            update_summary(summary, metrics)
            if step % max(args.trace_every, 1) == 0:
                print_trace(metrics)
            if renderer is not None:
                frames.append(render_frame(renderer, env))
            if terminated or truncated:
                obs, _ = env.reset(start_phase=args.start_phase)
                baseline = {
                    "foot_track_body": _body_frame_foot_tracks(env),
                }

    final_summary = finalize_summary(summary, args)
    payload = {
        "mode": args.mode,
        "start_phase": int(args.start_phase),
        "steps_requested": args.steps,
        "initial_state": initial_metrics,
        "summary": final_summary,
        "trace": trace,
    }
    report_path.write_text(json.dumps(payload, indent=2))
    maybe_save_gif(frames, args.render_gif)

    print("")
    print("Summary")
    print(json.dumps(final_summary, indent=2))
    print(f"Report: {report_path}")
    if args.render_gif:
        print(f"Render: {args.render_gif}")


if __name__ == "__main__":
    main()
