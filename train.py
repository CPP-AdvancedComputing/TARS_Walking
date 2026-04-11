import importlib.util
import os
import threading
import time

import numpy as np
from mujoco import viewer as mj_viewer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR
from training_helpers import BestWalkCallback, CurriculumCallback

TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "100000"))
MODEL_NAME = "tars_policy"
TENSORBOARD_LOG_DIR = "./tb_logs"
TENSORBOARD_ENABLED = importlib.util.find_spec("tensorboard") is not None
LIVE_VIEWER_ENABLED = os.environ.get("LIVE_VIEWER", "1") != "0"
LIVE_VIEWER_SYNC_STEPS = int(os.environ.get("LIVE_VIEWER_SYNC_STEPS", "10"))
LIVE_VIEWER_SYNC_SLEEP = float(os.environ.get("LIVE_VIEWER_SYNC_SLEEP", "0.0"))
TRAIN_DEVICE = os.environ.get("TRAIN_DEVICE", "cpu")
os.environ.setdefault("TARS_REWARD_PROFILE", "foundation")
os.environ.setdefault("TARS_MAX_EPISODE_STEPS", "2000")


class LiveViewerBridge:
    def __init__(self, env):
        self.env = env
        self.sync_sleep = max(0.0, LIVE_VIEWER_SYNC_SLEEP)
        self.sync_requested = threading.Event()
        self.stop_requested = threading.Event()
        self.ready = threading.Event()
        self.closed = threading.Event()
        self.thread = None
        self.error = None

    def start(self):
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        lock = getattr(self.env, "viewer_lock", None)
        try:
            with mj_viewer.launch_passive(self.env.model, self.env.data) as viewer:
                self.ready.set()
                while viewer.is_running() and not self.stop_requested.is_set():
                    if self.sync_requested.wait(timeout=0.02):
                        self.sync_requested.clear()
                        if lock is None:
                            viewer.sync()
                        else:
                            with lock:
                                viewer.sync()
                        if self.sync_sleep > 0.0:
                            time.sleep(self.sync_sleep)
                    else:
                        time.sleep(0.02)
        except Exception as exc:
            self.error = exc
            self.ready.set()
        finally:
            self.closed.set()

    def request_sync(self):
        if not self.closed.is_set():
            self.sync_requested.set()

    def stop(self):
        self.stop_requested.set()
        self.sync_requested.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


class LiveViewerCallback(BaseCallback):
    def __init__(self, viewer_bridge, sync_every_steps=LIVE_VIEWER_SYNC_STEPS):
        super().__init__()
        self.viewer_bridge = viewer_bridge
        self.sync_every_steps = max(1, int(sync_every_steps))

    def _on_training_start(self):
        self.viewer_bridge.start()
        self.viewer_bridge.ready.wait(timeout=5.0)
        if self.viewer_bridge.error is not None:
            print(f"Live viewer failed to start: {self.viewer_bridge.error}")
            return
        print("Live viewer ready.")
        self.viewer_bridge.request_sync()

    def _on_step(self):
        if self.n_calls % self.sync_every_steps == 0:
            self.viewer_bridge.request_sync()
        return True

    def _on_training_end(self):
        self.viewer_bridge.stop()


def attach_viewer_lock(env):
    env.viewer_lock = threading.RLock()
    original_step = env.step
    original_reset = env.reset

    def locked_step(action):
        with env.viewer_lock:
            return original_step(action)

    def locked_reset(*args, **kwargs):
        with env.viewer_lock:
            return original_reset(*args, **kwargs)

    env.step = locked_step
    env.reset = locked_reset
    return env

class EpisodeCallback(BaseCallback):
    def __init__(self, starting_episodes=0):
        super().__init__()
        self.episode_count = starting_episodes

    def _on_step(self):
        for done in self.locals["dones"]:
            if done:
                self.episode_count += 1
        return True
    
    def _on_training_end(self):
        print(f"\nTotal attempts ran: {self.episode_count:,}")


class TensorboardMetricsCallback(BaseCallback):
    def _on_step(self):
        env = self.training_env.envs[0].unwrapped
        quat = env.data.body("internals").xquat
        tilt = 2 * np.arccos(np.clip(abs(quat[0]), 0, 1))

        self.logger.record("gait/x", float(env.data.qpos[0]))
        self.logger.record("gait/tilt", float(tilt))
        self.logger.record("gait/phase", float(env.phase))
        self.logger.record("gait/phase_timer", float(env.phase_timer))
        self.logger.record("gait/swing_theta_diff", float(env.last_swing_theta_diff))
        self.logger.record("gait/plant_theta_diff", float(env.last_plant_theta_diff))
        self.logger.record("gait/theta_sync_reward", float(env.last_theta_sync_reward))
        self.logger.record("gait/swing_theta_target_error", float(env.last_swing_theta_target_error))
        self.logger.record("gait/plant_theta_target_error", float(env.last_plant_theta_target_error))
        self.logger.record("gait/theta_target_reward", float(env.last_theta_target_reward))
        self.logger.record("gait/swing_leg_diff", float(env.last_swing_leg_diff))
        self.logger.record("gait/plant_leg_diff", float(env.last_plant_leg_diff))
        self.logger.record("gait/leg_sync_reward", float(env.last_leg_sync_reward))
        self.logger.record("gait/swing_rod_diff", float(env.last_swing_rod_diff))
        self.logger.record("gait/plant_rod_diff", float(env.last_plant_rod_diff))
        self.logger.record("gait/rod_sync_reward", float(env.last_rod_sync_reward))
        self.logger.record("gait/swing_rod_target_error", float(env.last_swing_rod_target_error))
        self.logger.record("gait/plant_rod_target_error", float(env.last_plant_rod_target_error))
        self.logger.record("gait/swing_rod_length_error", float(env.last_swing_rod_length_error))
        self.logger.record("gait/plant_rod_length_error", float(env.last_plant_rod_length_error))
        self.logger.record("gait/rod_target_reward", float(env.last_rod_target_reward))
        self.logger.record("gait/swing_foot_forward_mean", float(env.last_swing_foot_forward_mean))
        self.logger.record("gait/swing_foot_forward_diff", float(env.last_swing_foot_forward_diff))
        self.logger.record("gait/swing_foot_forward_reward", float(env.last_swing_foot_forward_reward))
        self.logger.record("gait/swing_vertical_checkpoint_mean", float(env.last_swing_vertical_checkpoint_mean))
        self.logger.record("gait/swing_lateral_alignment_mean", float(env.last_swing_lateral_alignment_mean))
        self.logger.record("gait/swing_lateral_drift_penalty", float(env.last_swing_lateral_drift_penalty))
        self.logger.record("gait/rectangle_checkpoint_reward", float(env.last_rectangle_checkpoint_reward))
        self.logger.record("gait/rectangle_vertical_mean", float(env.last_rectangle_vertical_mean))
        self.logger.record("gait/rectangle_contact_mean", float(env.last_rectangle_contact_mean))
        self.logger.record("gait/swing_reach_reset_reward", float(env.last_swing_reach_reset_reward))
        self.logger.record("gait/contact_pattern_reward", float(env.last_contact_pattern_reward))
        self.logger.record("gait/contact_match_count", float(env.last_contact_match_count))
        self.logger.record("gait/contact_pair_bonus", float(env.last_contact_pair_bonus))
        self.logger.record("gait/reference_hip_reward", float(env.last_gait_reference_hip_reward))
        self.logger.record("gait/reference_leg_reward", float(env.last_gait_reference_leg_reward))
        self.logger.record("gait/scored_phase", float(env.last_scored_phase))
        self.logger.record("gait/support_quality", float(env.last_support_quality))
        self.logger.record("gait/progress_gate", float(env.last_progress_gate))
        self.logger.record("gait/body_hull_contact", float(env.last_body_hull_contact))
        self.logger.record("gait/phase_contact_quality", float(env.last_phase_contact_quality))
        self.logger.record("gait/phase_contact_penalty", float(env.last_phase_contact_penalty))
        self.logger.record("gait/early_swing_clearance_quality", float(env.last_early_swing_clearance_quality))
        self.logger.record("gait/early_swing_touchdown_penalty", float(env.last_early_swing_touchdown_penalty))
        self.logger.record("gait/forward_lean_penalty", float(env.last_forward_lean_penalty))
        self.logger.record("gait/vertical_oscillation_penalty", float(env.last_vertical_oscillation_penalty))
        self.logger.record("gait/vertical_stability_gate", float(env.last_vertical_stability_gate))
        self.logger.record("gait/oscillator_opposition_error", float(env.last_oscillator_opposition_error))
        self.logger.record("gait/oscillator_opposition_reward", float(env.last_oscillator_opposition_reward))

        for i in range(4):
            foot_height = float(env.data.geom(f"servo_l{i}_foot").xpos[2])
            self.logger.record(f"gait/foot_{i}_height", foot_height)
            self.logger.record(f"gait/foot_{i}_contact", float(env.last_foot_contacts[i]))
            self.logger.record(f"gait/desired_foot_{i}_contact", float(env.last_desired_contacts[i]))

        return True


def main():
    env = TARSEnv(DEFAULT_MODEL_PATH_STR)
    if LIVE_VIEWER_ENABLED:
        attach_viewer_lock(env)

    checkpoint = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix="tars"
    )

    episode_tracker = EpisodeCallback(starting_episodes=0)
    tb_metrics = TensorboardMetricsCallback()
    curriculum = CurriculumCallback(TOTAL_TIMESTEPS)
    best_walker = BestWalkCallback(
        save_path="./checkpoints/tars_best_walk",
        model_path=DEFAULT_MODEL_PATH_STR,
        eval_freq=20_000,
        n_eval_episodes=3,
    )

    callbacks = [checkpoint, episode_tracker, tb_metrics, curriculum, best_walker]
    if LIVE_VIEWER_ENABLED:
        callbacks.append(LiveViewerCallback(LiveViewerBridge(env)))

    ppo_kwargs = dict(
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    if TENSORBOARD_ENABLED:
        ppo_kwargs["tensorboard_log"] = TENSORBOARD_LOG_DIR

    model = PPO("MlpPolicy", env, verbose=1, device=TRAIN_DEVICE, **ppo_kwargs)
    print(f"Starting fresh {TOTAL_TIMESTEPS:,}-step run!")
    print(f"Training device: {TRAIN_DEVICE}")
    print(f"Reward profile: {env.REWARD_PROFILE}")
    print(f"Episode max steps: {env.MAX_EPISODE_STEPS}")
    if LIVE_VIEWER_ENABLED:
        print(f"Live viewer enabled (sync every {LIVE_VIEWER_SYNC_STEPS} training steps).")
    else:
        print("Live viewer disabled.")
    if TENSORBOARD_ENABLED:
        print(f"TensorBoard log dir: {TENSORBOARD_LOG_DIR}")
    else:
        print("TensorBoard is not installed. Training will continue without dashboard logging.")

    learn_kwargs = dict(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        reset_num_timesteps=True,
    )
    if TENSORBOARD_ENABLED:
        learn_kwargs["tb_log_name"] = "fresh_run"

    model.learn(**learn_kwargs)
    model.save(MODEL_NAME)

    with open("episode_count.txt", "w") as f:
        f.write(str(episode_tracker.episode_count))


if __name__ == "__main__":
    main()
