"""
Quick 100k-step training to verify learning works.
Should see reward increase within minutes if the env is correct.
"""
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from tars_env import TARSEnv
from tars_model import DEFAULT_MODEL_PATH_STR
from training_helpers import BestWalkCallback, CurriculumCallback
import numpy as np
import time

TOTAL_TIMESTEPS = 100_000

class QuickDiagCallback(BaseCallback):
    """Print reward stats every rollout."""
    def __init__(self):
        super().__init__()
        self.ep_rewards = []
        self.ep_lengths = []
        self.start_time = time.time()
        self.last_print = 0
        
    def _on_step(self):
        env = self.training_env.envs[0].unwrapped
        self.logger.record("gait/swing_lift_reward", getattr(env, "last_swing_lift_reward", 0.0))
        self.logger.record("gait/swing_plant_penalty", getattr(env, "last_swing_plant_penalty", 0.0))
        self.logger.record("gait/contact_match_count", getattr(env, "last_contact_match_count", 0.0))
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                if "episode" in info:
                    self.ep_rewards.append(info["episode"]["r"])
                    self.ep_lengths.append(info["episode"]["l"])
        
        if self.num_timesteps - self.last_print >= 2048 and len(self.ep_rewards) > 0:
            self.last_print = self.num_timesteps
            recent_r = self.ep_rewards[-20:]
            recent_l = self.ep_lengths[-20:]
            elapsed = time.time() - self.start_time
            print(f"  Step {self.num_timesteps:>6d} | "
                  f"Reward: {np.mean(recent_r):>8.1f} (std {np.std(recent_r):>5.1f}) | "
                  f"Length: {np.mean(recent_l):>5.0f} | "
                  f"Eps: {len(self.ep_rewards):>4d} | "
                  f"{elapsed:.0f}s", flush=True)
        return True

def make_env():
    env = TARSEnv(DEFAULT_MODEL_PATH_STR)
    return Monitor(env)

env = DummyVecEnv([make_env])

print("=" * 70, flush=True)
print("QUICK TRAINING TEST (100k steps)", flush=True)
print("Watch the Reward column - it should increase over time", flush=True)
print("=" * 70, flush=True)

model = PPO(
    "MlpPolicy", env, verbose=1,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
)

callback = QuickDiagCallback()
curriculum = CurriculumCallback(TOTAL_TIMESTEPS)
best_walker = BestWalkCallback(
    save_path="./checkpoints/tars_quick_best_walk",
    model_path=DEFAULT_MODEL_PATH_STR,
    eval_freq=10_000,
    n_eval_episodes=2,
)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[callback, curriculum, best_walker])
model.save("tars_policy_quick_last")
print("Saved latest quick-run policy to tars_policy_quick_last.zip", flush=True)
model.save("tars_policy_quick_rodaware")
print("saved to tars_policy_quick_rodaware.zip")

print("\n" + "=" * 70, flush=True)
if len(callback.ep_rewards) > 20:
    first_20 = np.mean(callback.ep_rewards[:20])
    last_20 = np.mean(callback.ep_rewards[-20:])
    first_len = np.mean(callback.ep_lengths[:20])
    last_len = np.mean(callback.ep_lengths[-20:])
    print(f"First 20 eps:  reward={first_20:>7.1f}  length={first_len:>5.0f}", flush=True)
    print(f"Last 20 eps:   reward={last_20:>7.1f}  length={last_len:>5.0f}", flush=True)
    print(f"Improvement:          {last_20 - first_20:>+7.1f}         {last_len - first_len:>+5.0f}", flush=True)
    if last_20 > first_20 + 5:
        print("LEARNING IS WORKING!", flush=True)
    elif last_len > first_len + 10:
        print("Survival is improving - learning is working but slow.", flush=True)
    else:
        print("No clear improvement yet.", flush=True)
else:
    print(f"Only {len(callback.ep_rewards)} episodes completed.", flush=True)
print("=" * 70, flush=True)

