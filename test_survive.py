import numpy as np
from tars_env import TARSEnv

URDF = r'C:\Users\anike\tars-urdf\tars_mjcf.xml'

# Test: Disable stagnation - see how long robot survives physically
env = TARSEnv(URDF)
obs, _ = env.reset()
for i in range(1000):
    obs, r, term, trunc, _ = env.step(np.zeros(12))
    env.stagnation_x = env.data.qpos[0] - 0.1  # prevent stagnation kill
    if i % 250 == 0 or term:
        tilt = 2 * np.arccos(np.clip(abs(env.data.body('internals').xquat[0]), 0, 1))
        z = env.data.qpos[2]
        print(f't={i:4d}: z={z:.5f} tilt={tilt:.4f}')
    if term or trunc:
        tilt = 2 * np.arccos(np.clip(abs(env.data.body('internals').xquat[0]), 0, 1))
        print(f'FELL at step {i+1}, tilt={tilt:.3f}')
        break
else:
    tilt = 2 * np.arccos(np.clip(abs(env.data.body('internals').xquat[0]), 0, 1))
    print(f'SURVIVED 1000 steps! final_tilt={tilt:.4f}')

