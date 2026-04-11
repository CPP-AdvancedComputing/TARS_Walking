import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from tars_env import TARSEnv

env = TARSEnv(r"C:\Users\anike\tars-urdf\tars_mjcf.xml")
ppo = PPO("MlpPolicy", env, verbose=1)

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    def training_callback(_locals, _globals):
        viewer.sync()
        return True
    
    ppo.learn(total_timesteps=100000, callback=training_callback)

ppo.save("tars_policy")
