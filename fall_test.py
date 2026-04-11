import pybullet as p
import pybullet_data
import time
import os

# Start GUI
physicsClient = p.connect(p.GUI)

# Optional: nicer camera defaults
p.resetDebugVisualizerCamera(
    cameraDistance=2.5,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.5]
)

# Gravity
p.setGravity(0, 0, -9.81)

# Built-in search path for plane.urdf
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Ground plane
plane_id = p.loadURDF("plane.urdf")

# Path to your robot
robot_path = os.path.abspath("robot.urdf")

# Load robot a little above the floor so it can fall
robot_id = p.loadURDF(
    robot_path,
    basePosition=[0, 0, 1.0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=False
)

# Simulate
while True:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)