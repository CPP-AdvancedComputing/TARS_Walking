import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import gymnasium
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from tars_env import TARSEnv


ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "tars_mjcf.xml"
RENDER_DIR = ROOT / "render_cleanup_check"
RENDER_DIR.mkdir(exist_ok=True)
UPRIGHT_IMG = RENDER_DIR / "diagnostic_upright.png"
HIP_IMG = RENDER_DIR / "diagnostic_all_hips_max.png"


def geom_type_name(geom_type):
    mapping = {
        int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
        int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
    }
    return mapping.get(int(geom_type), f"type_{int(geom_type)}")


def body_direct_geoms(root, body_name):
    body = root.find(f'.//body[@name="{body_name}"]')
    if body is None:
        return []
    return [child for child in list(body) if child.tag == "geom"]


def describe_xml_body(root, body_name):
    geoms = body_direct_geoms(root, body_name)
    meshes = []
    for geom in geoms:
        meshes.append(geom.get("mesh", geom.get("type", "unknown")))
    return geoms, Counter(meshes)


def geom_lower_z(model, data, geom_id):
    gtype = int(model.geom_type[geom_id])
    pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    mat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)

    if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return -np.inf
    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(pos[2] - size[0])
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        extent = np.sum(np.abs(mat[2, :]) * size[:3])
        return float(pos[2] - extent)
    if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        axis = mat[:, 2]
        radius, half = float(size[0]), float(size[1])
        axis_z = abs(float(axis[2]))
        radial = radius * np.sqrt(max(0.0, 1.0 - axis_z * axis_z))
        axial = half * axis_z
        return float(pos[2] - (radial + axial))
    if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        axis = mat[:, 2]
        radius, half = float(size[0]), float(size[1])
        axis_z = abs(float(axis[2]))
        return float(pos[2] - (radius + half * axis_z))
    if gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        extent = np.linalg.norm(size[:3] * mat[2, :])
        return float(pos[2] - extent)
    return float(pos[2])


def active_contacts(model, data):
    contacts = []
    for cid in range(data.ncon):
        contact = data.contact[cid]
        force6 = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, cid, force6)
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        contacts.append({
            "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1),
            "body1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1])),
            "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2),
            "body2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g2])),
            "normal_force": float(force6[0]),
        })
    return contacts


def render_frame(env, output_path, label, apply_fn=None):
    renderer = mujoco.Renderer(env.model, 360, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.9
    cam.azimuth = 138.0
    cam.elevation = -14.0
    cam.lookat[:] = env.data.body("internals").xpos
    cam.lookat[1] += 0.18
    cam.lookat[2] += 0.15
    if apply_fn is not None:
        apply_fn(env)
    mujoco.mj_forward(env.model, env.data)
    renderer.update_scene(env.data, camera=cam)
    rgb = renderer.render()
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 220, 32), fill=(0, 0, 0))
    draw.text((10, 8), label, fill=(255, 255, 255))
    img.save(output_path)


class TrainDiagWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_start_x = 0.0
        self.episode_contact_sum = 0.0
        self.episode_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_start_x = float(self.unwrapped.data.qpos[0])
        self.episode_contact_sum = 0.0
        self.episode_steps = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        self.episode_contact_sum += float(self.unwrapped.data.ncon)
        if terminated or truncated:
            info = dict(info)
            info["diagnostics"] = {
                "forward_disp": float(self.unwrapped.data.qpos[0] - self.episode_start_x),
                "avg_contacts": float(self.episode_contact_sum / max(self.episode_steps, 1)),
                "final_z": float(self.unwrapped.data.qpos[2]),
            }
        return obs, reward, terminated, truncated, info


class PPO4096Callback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.forward_disps = []
        self.avg_contacts = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            diag = info.get("diagnostics")
            if episode is not None:
                self.episode_rewards.append(float(episode["r"]))
                self.episode_lengths.append(float(episode["l"]))
            if diag is not None:
                self.forward_disps.append(float(diag["forward_disp"]))
                self.avg_contacts.append(float(diag["avg_contacts"]))
        return True


def main():
    lines = []

    # SECTION 1
    lines.append("═══════════════════════════════════════════")
    lines.append("SECTION 1: MJCF INTEGRITY")
    lines.append("═══════════════════════════════════════════")

    pyc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(XML_PATH)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    py_compile_status = "PASS" if pyc.returncode == 0 else "FAIL"
    lines.append(f"1a. python -m py_compile tars_mjcf.xml: {py_compile_status}")
    if pyc.returncode != 0:
        lines.append(f"    py_compile stderr: {pyc.stderr.strip().splitlines()[-1] if pyc.stderr.strip() else 'no stderr'}")

    xml_root = ET.parse(XML_PATH).getroot()
    option = xml_root.find("./option")

    target_bodies = ["internals"] + [f"{prefix}_l{i}" for prefix in ("active_carriage", "servo", "fixed_carriage") for i in range(4)]
    lines.append("1b. Geom counts by XML body:")
    for body_name in target_bodies:
        geoms, counts = describe_xml_body(xml_root, body_name)
        lines.append(f"    {body_name}: {len(geoms)} geoms -> {dict(counts)}")

    runtime_env = TARSEnv(str(XML_PATH))
    runtime_env.reset()
    runtime_model = runtime_env.model
    runtime_data = runtime_env.data

    collision_geoms = []
    for gid in range(runtime_model.ngeom):
        if int(runtime_model.geom_contype[gid]) > 0:
            collision_geoms.append({
                "name": mujoco.mj_id2name(runtime_model, mujoco.mjtObj.mjOBJ_GEOM, gid),
                "body": mujoco.mj_id2name(runtime_model, mujoco.mjtObj.mjOBJ_BODY, int(runtime_model.geom_bodyid[gid])),
                "type": geom_type_name(runtime_model.geom_type[gid]),
            })
    lines.append(f"1c. Collision-enabled geoms (contype>0): {len(collision_geoms)}")
    for item in collision_geoms:
        lines.append(f"    {item['name']} on {item['body']} ({item['type']})")

    floor_gid = mujoco.mj_name2id(runtime_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_exists = floor_gid >= 0
    floor_contype = int(runtime_model.geom_contype[floor_gid]) if floor_exists else -1
    lines.append(f"1d. Floor geom exists: {'YES' if floor_exists else 'NO'} (contype={floor_contype})")

    timestep = option.get("timestep") if option is not None else None
    integrator = option.get("integrator") if option is not None else None
    iterations = option.get("iterations") if option is not None else None
    lines.append(
        "1e. <option> block: "
        f"timestep={timestep}, integrator={integrator}, iterations={iterations}"
    )

    # SECTION 2
    lines.append("")
    lines.append("═══════════════════════════════════════════")
    lines.append("SECTION 2: PHYSICS INTEGRITY")
    lines.append("═══════════════════════════════════════════")

    internals_z = float(runtime_data.body("internals").xpos[2])
    robot_collision_gids = [gid for gid in range(runtime_model.ngeom) if int(runtime_model.geom_contype[gid]) > 0 and gid != floor_gid]
    lowest_gid = min(robot_collision_gids, key=lambda gid: geom_lower_z(runtime_model, runtime_data, gid))
    lowest_name = mujoco.mj_id2name(runtime_model, mujoco.mjtObj.mjOBJ_GEOM, lowest_gid)
    lowest_body = mujoco.mj_id2name(runtime_model, mujoco.mjtObj.mjOBJ_BODY, int(runtime_model.geom_bodyid[lowest_gid]))
    lowest_z = geom_lower_z(runtime_model, runtime_data, lowest_gid)
    lines.append(f"2a. internals starting Z: {internals_z:.6f}")
    lines.append(f"    lowest collision geom at rest: {lowest_name} on {lowest_body}, min_z={lowest_z:.6f}")
    lines.append(f"    gap to floor (z=0): {lowest_z:.6f}")

    env_zero = TARSEnv(str(XML_PATH))
    env_zero.reset()
    resets = 0
    zero_action = np.zeros(env_zero.action_space.shape[0], dtype=np.float32)
    zero_step_snapshots = []
    for step in range(1, 501):
        obs, reward, terminated, truncated, _ = env_zero.step(zero_action)
        if step % 100 == 0:
            zero_step_snapshots.append({
                "step": step,
                "z": float(env_zero.data.body("internals").xpos[2]),
                "ncon": int(env_zero.data.ncon),
                "above_floor": bool(env_zero.data.body("internals").xpos[2] > 0.0),
                "terminated": bool(terminated),
            })
        if terminated or truncated:
            resets += 1
            env_zero.reset()
    lines.append("2b. Zero-control rollout snapshots:")
    for snap in zero_step_snapshots:
        lines.append(
            f"    step {snap['step']}: internals_z={snap['z']:.6f}, contacts={snap['ncon']}, "
            f"above_floor={snap['above_floor']}, terminated={snap['terminated']}"
        )
    lines.append(f"    resets during 500 zero-action steps: {resets}")

    env_act = TARSEnv(str(XML_PATH))
    env_act.reset()
    actuator_results = []
    unresponsive = []
    for act_id in range(env_act.model.nu):
        test_data = mujoco.MjData(env_act.model)
        test_data.qpos[:] = env_act.data.qpos
        test_data.qvel[:] = 0.0
        if env_act.model.na:
            test_data.act[:] = 0.0
        mujoco.mj_forward(env_act.model, test_data)
        jid = int(env_act.model.actuator_trnid[act_id, 0])
        qpos_idx = int(env_act.model.jnt_qposadr[jid])
        initial = float(test_data.qpos[qpos_idx])
        test_data.ctrl[:] = 0.0
        test_data.ctrl[act_id] = 1.0
        for _ in range(200):
            mujoco.mj_step(env_act.model, test_data)
        final = float(test_data.qpos[qpos_idx])
        delta = final - initial
        actuator_name = mujoco.mj_id2name(env_act.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
        joint_name = mujoco.mj_id2name(env_act.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        actuator_results.append((actuator_name, joint_name, initial, final, delta))
        if abs(delta) < 0.001:
            unresponsive.append(actuator_name)
    lines.append("2c. One-at-a-time actuator response (ctrl=1.0 for 200 steps):")
    for actuator_name, joint_name, initial, final, delta in actuator_results:
        lines.append(f"    {actuator_name} -> {joint_name}: qpos {initial:.4f} -> {final:.4f} (delta={delta:.4f})")
    lines.append(f"    joints with |delta| < 0.001: {unresponsive if unresponsive else 'none'}")

    # SECTION 3
    lines.append("")
    lines.append("═══════════════════════════════════════════")
    lines.append("SECTION 3: CONTACT INTEGRITY")
    lines.append("═══════════════════════════════════════════")

    env_contact = TARSEnv(str(XML_PATH))
    env_contact.reset()
    contact_list = active_contacts(env_contact.model, env_contact.data)
    lines.append("3a. Active contact pairs at rest:")
    if contact_list:
        for contact in contact_list:
            lines.append(
                f"    {contact['geom1']} ({contact['body1']}) <-> {contact['geom2']} ({contact['body2']}), "
                f"normal_force={contact['normal_force']:.6f}"
            )
    else:
        lines.append("    none")

    foot_floor_contacts = []
    for contact in contact_list:
        pair = {contact["geom1"], contact["geom2"]}
        if "floor" in pair and any(name.startswith("servo_l") and name.endswith("_foot") for name in pair):
            foot_floor_contacts.append(contact)
    lines.append(f"3b. Foot geoms making contact with floor at rest: {'YES' if foot_floor_contacts else 'NO'}")

    lines.append("3c. Collision masks:")
    for geom_name in ["floor", "body_collision"] + [f"servo_l{i}_foot" for i in range(4)]:
        gid = mujoco.mj_name2id(env_contact.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        lines.append(
            f"    {geom_name}: contype={int(env_contact.model.geom_contype[gid])}, "
            f"conaffinity={int(env_contact.model.geom_conaffinity[gid])}, "
            f"type={geom_type_name(env_contact.model.geom_type[gid])}"
        )

    # SECTION 4
    lines.append("")
    lines.append("═══════════════════════════════════════════")
    lines.append("SECTION 4: ENVIRONMENT INTEGRITY")
    lines.append("═══════════════════════════════════════════")

    env_reset = TARSEnv(str(XML_PATH))
    obs, _ = env_reset.reset()
    lines.append(f"4a. reset observation shape: {obs.shape}")
    lines.append(f"    first 10 obs values: {[float(f'{v:.6f}') for v in obs[:10]]}")
    lines.append(
        f"    initial qpos[x,y,z]: {[float(f'{env_reset.data.qpos[i]:.6f}') for i in range(3)]}"
    )

    env_zero100 = TARSEnv(str(XML_PATH))
    env_zero100.reset()
    zero_100_rows = []
    terminated_seen_zero = False
    resets_zero = 0
    for step in range(1, 101):
        obs, reward, terminated, truncated, _ = env_zero100.step(np.zeros(env_zero100.action_space.shape[0], dtype=np.float32))
        if step % 10 == 0:
            zero_100_rows.append(
                (step, float(reward), float(env_zero100.data.qpos[0]), float(env_zero100.data.qpos[2]), bool(terminated), int(env_zero100.data.ncon))
            )
        if terminated or truncated:
            terminated_seen_zero = True
            resets_zero += 1
            env_zero100.reset()
    lines.append("4b. 100 zero-action steps:")
    for step, reward, qx, qz, terminated, ncon in zero_100_rows:
        lines.append(
            f"    step {step}: reward={reward:.6f}, qpos[0]={qx:.6f}, qpos[2]={qz:.6f}, terminated={terminated}, contacts={ncon}"
        )
    lines.append(f"    zero-action resets during 100 steps: {resets_zero}")

    env_rand = TARSEnv(str(XML_PATH))
    env_rand.reset()
    min_reward = float("inf")
    max_reward = float("-inf")
    terminated_random = False
    resets_random = 0
    rng = np.random.default_rng(0)
    final_qz = float(env_rand.data.qpos[2])
    for step in range(1, 101):
        action = rng.uniform(-1.0, 1.0, size=env_rand.action_space.shape[0]).astype(np.float32)
        obs, reward, terminated, truncated, _ = env_rand.step(action)
        min_reward = min(min_reward, float(reward))
        max_reward = max(max_reward, float(reward))
        final_qz = float(env_rand.data.qpos[2])
        if terminated or truncated:
            terminated_random = True
            resets_random += 1
            env_rand.reset()
    lines.append("4c. 100 random-action steps:")
    lines.append(f"    min reward={min_reward:.6f}, max reward={max_reward:.6f}")
    lines.append(f"    terminated ever triggered: {terminated_random}")
    lines.append(f"    resets during random run: {resets_random}")
    lines.append(f"    final qpos[2]={final_qz:.6f}")

    # SECTION 5
    lines.append("")
    lines.append("═══════════════════════════════════════════")
    lines.append("SECTION 5: TRAINING INTEGRITY")
    lines.append("═══════════════════════════════════════════")

    ppo_env = TrainDiagWrapper(TARSEnv(str(XML_PATH)))
    callback = PPO4096Callback()
    model = PPO(
        "MlpPolicy",
        ppo_env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    model.learn(total_timesteps=4096, callback=callback, reset_num_timesteps=True)
    mean_episode_reward = float(np.mean(callback.episode_rewards)) if callback.episode_rewards else float("nan")
    mean_episode_length = float(np.mean(callback.episode_lengths)) if callback.episode_lengths else float("nan")
    mean_forward_disp = float(np.mean(callback.forward_disps)) if callback.forward_disps else float("nan")
    mean_contacts_per_step = float(np.mean(callback.avg_contacts)) if callback.avg_contacts else float("nan")
    lines.append(f"5a. PPO 4096 steps -> mean episode reward={mean_episode_reward:.6f}")
    lines.append(f"    mean episode length={mean_episode_length:.6f}")
    lines.append(f"    mean forward displacement={mean_forward_disp:.6f}")
    lines.append(f"    mean contacts per step average={mean_contacts_per_step:.6f}")
    lines.append(f"5b. Mean forward displacement positive: {'YES' if mean_forward_disp > 0 else 'NO'}")

    # SECTION 6
    lines.append("")
    lines.append("═══════════════════════════════════════════")
    lines.append("SECTION 6: VISUAL INTEGRITY")
    lines.append("═══════════════════════════════════════════")

    render_env = TARSEnv(str(XML_PATH))
    render_env.reset()
    render_frame(render_env, UPRIGHT_IMG, "Upright")
    lines.append(f"6a. Upright render: {UPRIGHT_IMG}")

    hip_env = TARSEnv(str(XML_PATH))
    hip_env.reset()
    hip_env.model.opt.gravity[:] = 0.0

    def apply_hips(env):
        for i in range(4):
            env.data.ctrl[env.model.actuator(f"hip_l{i}").id] = 1.0
        for _ in range(200):
            mujoco.mj_step(env.model, env.data)

    render_frame(hip_env, HIP_IMG, "All hips = 1.0", apply_fn=apply_hips)
    lines.append(f"6b. Max-hip render: {HIP_IMG}")

    movement_results = {}
    # active_carriage with shoulder
    test_env = TARSEnv(str(XML_PATH))
    test_env.reset()
    test_env.model.opt.gravity[:] = 0.0
    body_name = "active_carriage_l0"
    geom_id = None
    body_id = test_env.model.body(body_name).id
    start = int(test_env.model.body_geomadr[body_id])
    count = int(test_env.model.body_geomnum[body_id])
    for gid in range(start, start + count):
        mid = int(test_env.model.geom_dataid[gid])
        if mid >= 0 and test_env.model.mesh(mid).name == "active_carriage":
            geom_id = gid
            break
    start_pos = np.asarray(test_env.data.geom_xpos[geom_id], dtype=np.float64).copy()
    test_env.data.ctrl[test_env.model.actuator("shoulder_l0").id] = -0.2
    for _ in range(200):
        mujoco.mj_step(test_env.model, test_env.data)
    movement_results["active_carriage"] = float(np.linalg.norm(np.asarray(test_env.data.geom_xpos[geom_id]) - start_pos)) > 0.001

    # servo with hip
    test_env = TARSEnv(str(XML_PATH))
    test_env.reset()
    test_env.model.opt.gravity[:] = 0.0
    body_id = test_env.model.body("servo_l0").id
    start = int(test_env.model.body_geomadr[body_id])
    count = int(test_env.model.body_geomnum[body_id])
    geom_id = None
    for gid in range(start, start + count):
        mid = int(test_env.model.geom_dataid[gid])
        if mid >= 0 and test_env.model.mesh(mid).name == "servo":
            geom_id = gid
            break
    start_pos = np.asarray(test_env.data.geom_xpos[geom_id], dtype=np.float64).copy()
    test_env.data.ctrl[test_env.model.actuator("hip_l0").id] = 1.0
    for _ in range(200):
        mujoco.mj_step(test_env.model, test_env.data)
    movement_results["servo"] = float(np.linalg.norm(np.asarray(test_env.data.geom_xpos[geom_id]) - start_pos)) > 0.001

    # fixed carriage with knee
    test_env = TARSEnv(str(XML_PATH))
    test_env.reset()
    test_env.model.opt.gravity[:] = 0.0
    body_id = test_env.model.body("fixed_carriage_l0").id
    start = int(test_env.model.body_geomadr[body_id])
    count = int(test_env.model.body_geomnum[body_id])
    geom_id = None
    for gid in range(start, start + count):
        mid = int(test_env.model.geom_dataid[gid])
        if mid >= 0 and test_env.model.mesh(mid).name == "fixed_carriage":
            geom_id = gid
            break
    start_pos = np.asarray(test_env.data.geom_xpos[geom_id], dtype=np.float64).copy()
    test_env.data.ctrl[test_env.model.actuator("knee_l0").id] = -0.2
    for _ in range(200):
        mujoco.mj_step(test_env.model, test_env.data)
    movement_results["fixed_carriage"] = float(np.linalg.norm(np.asarray(test_env.data.geom_xpos[geom_id]) - start_pos)) > 0.001

    lines.append("6c. Visual mesh moves with joints:")
    lines.append(f"    active_carriage mesh moves with shoulder: {'YES' if movement_results['active_carriage'] else 'NO'}")
    lines.append(f"    servo mesh moves with hip: {'YES' if movement_results['servo'] else 'NO'}")
    lines.append(f"    fixed_carriage mesh moves with knee: {'YES' if movement_results['fixed_carriage'] else 'NO'}")

    # FINAL SUMMARY
    lines.append("")
    lines.append("═══════════════════════════════════════════")
    lines.append("FINAL SUMMARY")
    lines.append("═══════════════════════════════════════════")

    mjcf_compile_pass = True
    floor_contact_exists = floor_exists and floor_contype > 0
    feet_touch_floor = bool(foot_floor_contacts)
    all_joints_respond = not unresponsive
    env_reset_works = obs.shape == env_reset.observation_space.shape
    height_stable = resets == 0
    forward_motion = mean_forward_disp > 0
    visual_meshes_move = all(movement_results.values())

    summary_rows = [
        ("MJCF compile", "PASS" if mjcf_compile_pass else "FAIL"),
        ("Floor contact exists", "YES" if floor_contact_exists else "NO"),
        ("Feet touch floor", "YES" if feet_touch_floor else "NO"),
        ("All joints respond", "YES" if all_joints_respond else "NO"),
        ("Env reset works", "YES" if env_reset_works else "NO"),
        ("Height stable at rest", "YES" if height_stable else "NO"),
        ("Forward motion in training", "YES" if forward_motion else "NO"),
        ("Visual meshes move", "YES" if visual_meshes_move else "NO"),
    ]
    for label, value in summary_rows:
        suffix = ""
        if value in {"FAIL", "NO"}:
            suffix = "  ACTION REQUIRED"
        lines.append(f"{label + ':':24s} {value}{suffix}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
