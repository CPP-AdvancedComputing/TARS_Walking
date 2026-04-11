"""Sanity check: standing still vs shoulder lifting vs movement."""
import numpy as np
from tars_env import TARSEnv

URDF = r'C:\Users\anike\tars-urdf\tars_mjcf.xml'

# 1. Standing still (zero action)
env = TARSEnv(URDF)
obs, _ = env.reset()
total_r = 0
for i in range(1000):
    obs, r, term, trunc, _ = env.step(np.zeros(12))
    total_r += r
    if term or trunc:
        break
print(f'Standing still: {i+1} steps, total_reward={total_r:.1f}, avg={total_r/(i+1):.3f}/step')

# 2. Shoulders lifted (action pushes shoulders up)
env2 = TARSEnv(URDF)
obs, _ = env2.reset()
total_r2 = 0
action = np.zeros(12)
action[0:4] = 1.0  # push shoulders up (max)
for i in range(1000):
    obs, r, term, trunc, _ = env2.step(action)
    total_r2 += r
    if term or trunc:
        break
print(f'Shoulders up:   {i+1} steps, total_reward={total_r2:.1f}, avg={total_r2/(i+1):.3f}/step')

# 3. Hips oscillating (try to trigger gait reward)
env3 = TARSEnv(URDF)
obs, _ = env3.reset()
total_r3 = 0
for i in range(1000):
    action = np.zeros(12)
    phase = (i % 100) / 100.0
    if phase < 0.5:
        action[0] = 0.5; action[3] = 0.5  # pair A shoulders up
    else:
        action[1] = 0.5; action[2] = 0.5  # pair B shoulders up
    obs, r, term, trunc, _ = env3.step(action)
    total_r3 += r
    if term or trunc:
        break
print(f'Alternating:    {i+1} steps, total_reward={total_r3:.1f}, avg={total_r3/(i+1):.3f}/step')

