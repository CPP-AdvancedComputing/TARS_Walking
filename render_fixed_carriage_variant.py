import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont

from tars_env import TARSEnv
from keyframe_tars_states import TARSKeyframeController, KeyframeConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Render one fixed-carriage quat variant.")
    parser.add_argument("--base-xml", required=True)
    parser.add_argument("--quat", required=True)
    parser.add_argument("--pos", default="0 0 0")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--azimuth", type=float, default=180.0)
    parser.add_argument("--elevation", type=float, default=-8.0)
    parser.add_argument("--distance", type=float, default=1.7)
    parser.add_argument("--lookat-dx", type=float, default=0.0)
    parser.add_argument("--lookat-dy", type=float, default=0.18)
    parser.add_argument("--lookat-dz", type=float, default=0.10)
    parser.add_argument("--hide-collision", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    base_xml = Path(args.base_xml)
    out_path = Path(args.out)

    tree = ET.parse(base_xml)
    root = tree.getroot()
    for leg_id in range(4):
        body = root.find(f".//body[@name='fixed_carriage_l{leg_id}']")
        for geom in body.findall("geom"):
            if geom.get("mesh") == "fixed_carriage":
                geom.set("quat", args.quat)
                geom.set("pos", args.pos)
            elif args.hide_collision and geom.get("type") in {"box", "cylinder"}:
                geom.set("rgba", "0 0 0 0")

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, dir=base_xml.parent) as tmp:
        temp_xml = Path(tmp.name)
    tree.write(temp_xml, encoding="utf-8", xml_declaration=False)

    try:
        model = mujoco.MjModel.from_xml_path(str(temp_xml))
        if model.nu != 12:
            raise RuntimeError(f"Unexpected actuator count: {model.nu}")

        env = TARSEnv(str(temp_xml))
        env.reset()
        controller = TARSKeyframeController(env, KeyframeConfig())
        if controller.config.freeze_base:
            env.model.opt.gravity[:] = 0.0
        controller.snap_to_upright()
        controller.apply_joint_targets(controller.upright_targets, sim_time=0.0)

        renderer = mujoco.Renderer(env.model, 320, 320)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        lookat = env.data.body("internals").xpos.copy()
        lookat[0] += args.lookat_dx
        lookat[1] += args.lookat_dy
        lookat[2] += args.lookat_dz
        cam.lookat[:] = lookat
        cam.distance = args.distance
        cam.azimuth = args.azimuth
        cam.elevation = args.elevation

        renderer.update_scene(env.data, camera=cam)
        img = Image.fromarray(renderer.render().copy())
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.rectangle((6, 6, 220, 36), fill=(255, 255, 255))
        draw.multiline_text((10, 9), args.label, fill=(0, 0, 0), font=font, spacing=2)
        img.save(out_path)
    finally:
        try:
            temp_xml.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
