"""Quick train + evaluate to see what policy does."""
import numpy as np, sys
from stable_baselines3 import PPO
from tars_env import TARSEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

URDF = r"C:\Users\anike\tars-urdf\tars_mjcf.xml"

print("Training 50k...", flush=True)
env = DummyVecEnv([lambda: Monitor(TARSEnv(URDF))])
model = PPO("MlpPolicy", env, n_steps=2048, batch_size=128,
            learning_rate=3e-4, ent_coef=0.01, verbose=0)
model.learn(50_000)
print("Done.\n", flush=True)

test_env = TARSEnv(URDF)

for ep in range(3):
    obs, _ = test_env.reset()
    total_r = 0
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = test_env.step(action)
        total_r += r
        if term or trunc: break
    x = test_env.data.qpos[0]
    tilt = 2*np.arccos(np.clip(abs(test_env.data.body('internals').xquat[0]),0,1))
    print(f'ep{ep}: steps={step+1}, x={x:+.3f}m, tilt={tilt:.3f}, reward={total_r:.1f}', flush=True)

# Detailed step trace
obs, _ = test_env.reset()
print("\nStep trace:", flush=True)
for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, _ = test_env.step(action)
    if step < 3 or step % 30 == 0 or term:
        x = test_env.data.qpos[0]
        tilt = 2*np.arccos(np.clip(abs(test_env.data.body('internals').xquat[0]),0,1))
        fz = [test_env.data.geom(f'servo_l{j}_foot').xpos[2] for j in range(4)]
        print(f't={step:3d} x={x:+.4f} tilt={tilt:.3f} fz=[{fz[0]:.3f},{fz[1]:.3f},{fz[2]:.3f},{fz[3]:.3f}] r={r:.2f}', flush=True)
    if term or trunc: break

print("DONE", flush=True)
sys.stdout.flush()

