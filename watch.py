import os
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from tars_env import TARSEnv


def _policy_candidates(root):
    candidates = []
    final_zip = os.path.join(root, "tars_policy.zip")
    if os.path.exists(final_zip):
        candidates.append(final_zip)

    checkpoint_dir = os.path.join(root, "checkpoints")
    if os.path.isdir(checkpoint_dir):
        for name in os.listdir(checkpoint_dir):
            if name.endswith(".zip"):
                candidates.append(os.path.join(checkpoint_dir, name))
    return candidates


def _latest_policy_path(root):
    candidates = _policy_candidates(root)
    if not candidates:
        raise FileNotFoundError(
            "No policy or checkpoints found. Train first with: python train.py"
        )
    latest = max(candidates, key=os.path.getmtime)
    print(f"Loading policy: {latest}")
    return latest[:-4] if latest.endswith(".zip") else latest


ROOT = r"C:\Users\anike\tars-urdf"
URDF = os.path.join(ROOT, "tars_mjcf.xml")

env = TARSEnv(URDF)
policy_path = _latest_policy_path(ROOT)
model = PPO.load(policy_path, env=env)

obs, _ = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    viewer.cam.lookat[:] = env.data.body("internals").xpos
    viewer.cam.distance = 2.0
    viewer.cam.elevation = -20

    step = 0
    while viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        viewer.cam.lookat[:] = env.data.body("internals").xpos
        viewer.sync()
        time.sleep(0.01)

        if step % 25 == 0:
            tilt = 2 * np.arccos(
                np.clip(abs(env.data.body("internals").xquat[0]), 0, 1)
            )
            print(
                f"step={step:4d}  x={env.data.qpos[0]:+.4f}  "
                f"reward={reward:+.3f}  action_norm={np.linalg.norm(action):.3f}  "
                f"tilt={tilt:.3f}"
            )
        step += 1

        if terminated or truncated:
            print(
                f"episode ended at step {step}  "
                f"x={env.data.qpos[0]:+.4f}  terminated={terminated}  truncated={truncated}"
            )
            obs, _ = env.reset()
            step = 0

