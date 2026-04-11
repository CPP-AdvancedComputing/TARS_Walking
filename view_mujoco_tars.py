import argparse
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR

try:
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover - optional during simple viewing
    PPO = None


ROOT = Path(__file__).resolve().parent
PLANT_RGBA = np.array([0.18, 0.82, 0.28, 0.95], dtype=np.float32)
SWING_RGBA = np.array([0.95, 0.65, 0.12, 0.95], dtype=np.float32)
ANCHOR_RGBA = np.array([0.93, 0.93, 0.97, 0.85], dtype=np.float32)
FOOT_TRACE_RGBA = np.array([0.12, 0.12, 0.12, 0.65], dtype=np.float32)
TARGET_RGBA = np.array([0.92, 0.22, 0.22, 0.85], dtype=np.float32)
IDENTITY_MAT = np.eye(3, dtype=np.float64).reshape(-1)
ZERO_POS = np.zeros(3, dtype=np.float64)
ZERO_SIZE = np.zeros(3, dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(description="View TARS in MuJoCo with gait-aware overlays.")
    parser.add_argument(
        "--mode",
        choices=("static", "gait", "nominal", "policy"),
        default="gait",
        help="`gait` replays the training-time nominal gait, `policy` uses the latest PPO policy, `static` freezes the reset pose.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH_STR,
        help="Model file to load. Defaults to the generated MJCF.",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Optional policy path for `--mode policy`. Defaults to the newest checkpoint/policy in this repo.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.01,
        help="Viewer loop sleep in seconds.",
    )
    parser.add_argument(
        "--trace-every",
        type=int,
        default=25,
        help="Print rollout diagnostics every N env steps.",
    )
    return parser.parse_args()


def _policy_candidates(root):
    candidates = []
    final_zip = root / "tars_policy.zip"
    if final_zip.exists():
        candidates.append(final_zip)

    checkpoint_dir = root / "checkpoints"
    if checkpoint_dir.is_dir():
        for child in checkpoint_dir.iterdir():
            if child.suffix == ".zip":
                candidates.append(child)
    return candidates


def _resolve_policy_path(policy_arg):
    if policy_arg:
        path = Path(policy_arg)
        if path.suffix != ".zip" and path.with_suffix(".zip").exists():
            path = path.with_suffix(".zip")
        if not path.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")
        return str(path.with_suffix("")) if path.suffix == ".zip" else str(path)

    candidates = _policy_candidates(ROOT)
    if not candidates:
        raise FileNotFoundError(
            "No policy found. Train one first or pass --policy PATH."
        )
    latest = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    print(f"Loading latest policy: {latest}")
    return str(latest.with_suffix(""))


def _init_connector_geom(scene, slot, rgba):
    geom = scene.geoms[slot]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        ZERO_SIZE,
        ZERO_POS,
        IDENTITY_MAT,
        rgba,
    )
    scene.ngeom += 1
    return geom


def _add_connector(scene, from_point, to_point, rgba, width, geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = _init_connector_geom(scene, scene.ngeom, rgba)
    mujoco.mjv_connector(
        geom,
        geom_type,
        width,
        np.asarray(from_point, dtype=np.float64),
        np.asarray(to_point, dtype=np.float64),
    )


def _add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        IDENTITY_MAT,
        rgba,
    )
    scene.ngeom += 1


def _draw_leg_overlays(viewer, env, phase):
    scene = viewer.user_scn
    scene.ngeom = 0

    for leg_id in range(4):
        state = env.leg_visual_state(leg_id, phase=phase)
        rod_rgba = PLANT_RGBA if state["role"] == "plant" else SWING_RGBA
        _add_connector(scene, state["upper_anchor"], state["lower_anchor"], rod_rgba, 0.007)
        _add_connector(scene, state["panel_anchor"], state["foot_center"], FOOT_TRACE_RGBA, 0.0025)
        if state["desired_lower_anchor"] is not None:
            _add_connector(scene, state["upper_anchor"], state["desired_lower_anchor"], TARGET_RGBA, 0.0035)
            _add_sphere(scene, state["desired_lower_anchor"], 0.008, TARGET_RGBA)
        _add_sphere(scene, state["upper_anchor"], 0.012, ANCHOR_RGBA)
        _add_sphere(scene, state["panel_anchor"], 0.010, rod_rgba)


def _hide_static_helper_rods(env):
    for leg_id in range(4):
        body = env.model.body(f"fixed_carriage_l{leg_id}")
        geomadr = int(body.geomadr[0]) if hasattr(body.geomadr, "__len__") else int(body.geomadr)
        geomnum = int(body.geomnum[0]) if hasattr(body.geomnum, "__len__") else int(body.geomnum)
        for geom_id in range(geomadr, geomadr + geomnum):
            geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and geom_name.startswith("servo_l"):
                continue
            if env.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                env.model.geom_rgba[geom_id, 3] = 0.0


def _print_trace(step, mode, env, reward, residual_action, ctrl_targets, phase_used):
    swing_ids, plant_ids = env.phase_pairs(phase_used)
    centered_hips = np.array([
        float(env.data.joint(f"hip_revolute_l{i}").qpos[0] - env.standing_ctrl[4 + i])
        for i in range(4)
    ], dtype=np.float64)
    print(
        f"step={step:4d} mode={mode:>6s} phase={phase_used} "
        f"swing={swing_ids} plant={plant_ids} "
        f"x={env.data.qpos[0]:+.4f} tilt={2 * np.arccos(np.clip(abs(env.data.body('internals').xquat[0]), 0, 1)):.3f} "
        f"reward={reward:+.3f}"
    )
    print(
        "  residual_action="
        + np.array2string(np.asarray(residual_action), precision=3, suppress_small=True)
    )
    print(
        "  ctrl_targets="
        + np.array2string(np.asarray(ctrl_targets), precision=3, suppress_small=True)
    )
    print(
        "  centered_hips="
        + np.array2string(centered_hips, precision=3, suppress_small=True)
    )
    print(
        f"  theta_diff(swing={env.last_swing_theta_diff:.3f}, plant={env.last_plant_theta_diff:.3f}) "
        f"theta_target_err(swing={env.last_swing_theta_target_error:.3f}, plant={env.last_plant_theta_target_error:.3f}) "
        f"theta_target_reward={env.last_theta_target_reward:.3f}"
    )
    print(
        f"  rod_diff(swing={env.last_swing_rod_diff:.3f}, plant={env.last_plant_rod_diff:.3f}) "
        f"rod_target_err(swing={env.last_swing_rod_target_error:.3f}, plant={env.last_plant_rod_target_error:.3f}) "
        f"rod_length_err(swing={env.last_swing_rod_length_error:.3f}, plant={env.last_plant_rod_length_error:.3f}) "
        f"rod_target_reward={env.last_rod_target_reward:.3f}"
    )
    print(
        f"  swing_foot_forward_mean={env.last_swing_foot_forward_mean:.3f} "
        f"swing_foot_forward_diff={env.last_swing_foot_forward_diff:.3f} "
        f"swing_foot_forward_reward={env.last_swing_foot_forward_reward:.3f}"
    )


def main():
    args = parse_args()
    mode = "gait" if args.mode == "nominal" else args.mode

    env = TARSEnv(args.model)
    _hide_static_helper_rods(env)
    obs, _ = env.reset()
    policy = None
    if mode == "policy":
        if PPO is None:
            raise ImportError("stable_baselines3 is required for --mode policy")
        policy = PPO.load(_resolve_policy_path(args.policy), env=env)

    print(f"Viewing model: {args.model}")
    print(f"Viewer mode: {mode}")
    if mode == "gait":
        print("Using the training-time nominal gait prior (zero residual action).")
    elif mode == "policy":
        print("Using the trained PPO policy output as the residual action.")
    else:
        print("Showing the settled mid-gait reset pose without stepping.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = env.data.body("internals").xpos
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20

        step = 0
        last_reward = 0.0
        last_action = env.zero_action()
        last_ctrl_targets = env.control_targets_for_action(last_action, phase=env.phase)
        last_phase = env.phase

        while viewer.is_running():
            if mode == "static":
                mujoco.mj_forward(env.model, env.data)
            else:
                phase_used = env.phase
                if mode == "policy":
                    action, _ = policy.predict(obs, deterministic=True)
                else:
                    action = env.zero_action()
                action = np.asarray(action, dtype=np.float64)
                ctrl_targets = env.control_targets_for_action(action, phase=phase_used)
                obs, last_reward, terminated, truncated, _ = env.step(action)
                last_action = action
                last_ctrl_targets = ctrl_targets
                last_phase = phase_used

                if step % args.trace_every == 0:
                    _print_trace(step, mode, env, last_reward, last_action, last_ctrl_targets, last_phase)

                step += 1
                if terminated or truncated:
                    print(
                        f"episode ended at step {step} "
                        f"x={env.data.qpos[0]:+.4f} terminated={terminated} truncated={truncated}"
                    )
                    obs, _ = env.reset()
                    step = 0
                    last_reward = 0.0
                    last_action = env.zero_action()
                    last_ctrl_targets = env.control_targets_for_action(last_action, phase=env.phase)
                    last_phase = env.phase

            _draw_leg_overlays(viewer, env, phase=last_phase)
            viewer.cam.lookat[:] = env.data.body("internals").xpos
            viewer.sync()
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
