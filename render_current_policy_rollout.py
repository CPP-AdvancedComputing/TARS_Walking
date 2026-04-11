from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from tars_env import TARSEnv


RENDER_DIR = Path("render_cleanup_check")
DEFAULT_MODEL = Path("tars_policy.zip")
DEFAULT_VIDEO = RENDER_DIR / "current_policy_rollout.mp4"
DEFAULT_CONTACT_SHEET = RENDER_DIR / "current_policy_rollout_contact_sheet.png"
DEFAULT_REPORT = RENDER_DIR / "current_policy_pivot_report.txt"


def build_camera(env: TARSEnv) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.9
    cam.azimuth = 138.0
    cam.elevation = -14.0
    cam.lookat[:] = env.data.body("internals").xpos
    cam.lookat[1] += 0.18
    cam.lookat[2] += 0.15
    return cam


def draw_label(rgb: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 520, 46), fill=(0, 0, 0))
    draw.text((10, 12), text, fill=(255, 255, 255))
    return np.asarray(img)


def make_contact_sheet(frames: list[np.ndarray], labels: list[str], output_path: Path) -> None:
    thumbs = []
    for frame, label in zip(frames, labels):
        thumb = Image.fromarray(frame).resize((320, 180))
        draw = ImageDraw.Draw(thumb)
        draw.rectangle((0, 0, 220, 30), fill=(0, 0, 0))
        draw.text((8, 8), label, fill=(255, 255, 255))
        thumbs.append(thumb)

    rows = 2
    cols = int(np.ceil(len(thumbs) / rows))
    sheet = Image.new("RGB", (cols * 320, rows * 180), color=(20, 20, 20))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 320
        y = (idx // cols) * 180
        sheet.paste(thumb, (x, y))
    sheet.save(output_path)


def analyze_pivots(env: TARSEnv, trace: dict[int, list[dict[str, np.ndarray]]]) -> str:
    lines = []
    lines.append("Current policy pivot analysis")
    lines.append("")
    for leg in (1, 2):
        entries = trace[leg]
        upper_world = np.array([entry["upper_world"] for entry in entries], dtype=np.float64)
        upper_body = np.array([entry["upper_body"] for entry in entries], dtype=np.float64)
        lower_body = np.array([entry["lower_body"] for entry in entries], dtype=np.float64)

        upper_world_disp = np.linalg.norm(upper_world[-1] - upper_world[0])
        upper_world_span = np.ptp(upper_world, axis=0)
        upper_body_dev = np.max(np.linalg.norm(upper_body - upper_body[0], axis=1))
        lower_body_x_span = float(np.ptp(lower_body[:, 0]))
        lower_body_z_span = float(np.ptp(lower_body[:, 2]))
        lower_body_final = lower_body[-1]
        upper_body_ref = upper_body[0]

        lines.append(f"Leg l{leg}")
        lines.append(f"  Upper anchor world displacement: {upper_world_disp:.6f} m")
        lines.append(
            "  Upper anchor world span: "
            f"[{upper_world_span[0]:.6f}, {upper_world_span[1]:.6f}, {upper_world_span[2]:.6f}] m"
        )
        lines.append(f"  Upper anchor body-frame max deviation: {upper_body_dev:.9f} m")
        lines.append(
            "  Upper anchor body-frame reference: "
            f"[{upper_body_ref[0]:.6f}, {upper_body_ref[1]:.6f}, {upper_body_ref[2]:.6f}]"
        )
        lines.append(
            "  Lower tracking point body-frame final: "
            f"[{lower_body_final[0]:.6f}, {lower_body_final[1]:.6f}, {lower_body_final[2]:.6f}]"
        )
        lines.append(f"  Lower tracking point body-frame x span: {lower_body_x_span:.6f} m")
        lines.append(f"  Lower tracking point body-frame z span: {lower_body_z_span:.6f} m")
        lines.append("")

    lines.append("Interpretation")
    lines.append(
        "  The upper anchor is static in the internals/body frame if its body-frame deviation stays near zero."
    )
    lines.append(
        "  If the upper anchor moves in world space at the same time, that is root-body motion, not a bad pivot."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(DEFAULT_MODEL))
    parser.add_argument("--steps", type=int, default=180)
    args = parser.parse_args()

    RENDER_DIR.mkdir(exist_ok=True)

    env = TARSEnv("tars_mjcf.xml")
    model = PPO.load(args.policy, env=env, print_system_info=False)
    obs, _ = env.reset()

    renderer = mujoco.Renderer(env.model, 360, 640)
    cam = build_camera(env)
    writer = cv2.VideoWriter(
        str(DEFAULT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (640, 360),
    )

    trace = {1: [], 2: []}
    saved_frames: list[np.ndarray] = []
    saved_labels: list[str] = []
    sample_steps = {0, 30, 60, 90, 120, 150}

    try:
        for step in range(args.steps + 1):
            for leg in (1, 2):
                upper_world = env._rod_upper_anchor_world(leg)
                upper_body = env._world_vector_to_body(
                    upper_world - np.asarray(env.data.body("internals").xpos, dtype=np.float64)
                )
                lower_world = env._foot_track_world(leg)
                lower_body = env._world_vector_to_body(lower_world - upper_world)
                trace[leg].append(
                    {
                        "upper_world": upper_world.copy(),
                        "upper_body": upper_body.copy(),
                        "lower_body": lower_body.copy(),
                    }
                )

            renderer.update_scene(env.data, camera=cam)
            rgb = renderer.render().copy()
            label = (
                f"step {step:03d}  phase={env.phase}  timer={env.phase_timer}  "
                f"contacts={env._foot_on_ground_flags()}"
            )
            frame = draw_label(rgb, label)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if step in sample_steps:
                saved_frames.append(frame)
                saved_labels.append(f"step {step}")

            if step == args.steps:
                break

            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        writer.release()
        renderer.close()
        env.close()

    make_contact_sheet(saved_frames, saved_labels, DEFAULT_CONTACT_SHEET)
    DEFAULT_REPORT.write_text(analyze_pivots(env, trace), encoding="utf-8")

    print(f"video={DEFAULT_VIDEO.resolve()}")
    print(f"contact_sheet={DEFAULT_CONTACT_SHEET.resolve()}")
    print(f"report={DEFAULT_REPORT.resolve()}")


if __name__ == "__main__":
    main()
