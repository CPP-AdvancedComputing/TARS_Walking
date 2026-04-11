import mujoco
from mujoco import viewer
import time
import os

xml_path = os.path.abspath("robot.urdf")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Optional: start the robot a little above ground if the model allows it
# data.qpos[0:3] = [0, 0, 1.0]

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.01)