# Minimal Panda tele-op sanity check (no RL, for GUI & IK verification)
import time
import numpy as np
import pybullet as p
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# Simple table
table_uid = p.loadURDF("table/table.urdf", [0.5, 0, 0.0], useFixedBase=True)

# Load Panda (Franka)
panda = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)

# Helper: control gripper open/close (finger joints 9 and 10 on Panda)
def control_gripper(open_ratio: float):
    # open_ratio in [0,1]; 0 = closed, 1 = fully open
    spread = 0.04 * float(np.clip(open_ratio, 0, 1))
    p.setJointMotorControl2(panda, 9, p.POSITION_CONTROL, targetPosition=spread, force=10)
    p.setJointMotorControl2(panda, 10, p.POSITION_CONTROL, targetPosition=spread, force=10)

# Move end-effector by IK to a few waypoints above the table
waypoints = [
    np.array([0.45, 0.0, 0.45]),
    np.array([0.55, 0.15, 0.30]),
    np.array([0.55, -0.15, 0.30]),
    np.array([0.45, 0.0, 0.45]),
]

ee_link = 11  # Panda end-effector link index
for i, wp in enumerate(waypoints):
    # Open/close gripper just to show it works
    control_gripper(1.0 if i % 2 == 0 else 0.0)

    # Compute IK for target position (keep a neutral orientation)
    orn = p.getQuaternionFromEuler([np.pi, 0, 0])  # tool pointing down
    joint_positions = p.calculateInverseKinematics(panda, ee_link, wp, orn)

    # Apply joint positions for the 7 arm joints (0..6)
    for j in range(7):
        p.setJointMotorControl2(panda, j, p.POSITION_CONTROL, joint_positions[j], force=200)

    # Step for ~1s to reach
    for _ in range(240):
        p.stepSimulation()
        time.sleep(1.0/240.0)

print("Sanity check done. Close the GUI window to exit.")
while p.isConnected():
    time.sleep(0.1)