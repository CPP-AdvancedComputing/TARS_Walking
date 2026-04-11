from tars_env import TARSEnv

env = TARSEnv(r"C:\Users\anike\tars-urdf\tars_mjcf.xml")
obs = env.reset()
print("Object resetted")
for i in range (5):
    action = env.action_space.sample()
    obs, reward, terminated, trunacted, info = env.step(action)
    print(f"Step {i} | reward: {reward} | terminated {terminated}")
