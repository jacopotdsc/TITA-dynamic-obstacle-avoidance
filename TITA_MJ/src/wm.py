import sys
sys.path.insert(0, '/home/ubuntu/Desktop/repo_rl//TITA_MJ/compiled')

import wm
import numpy as np

wlk = wm.WalkingManager()


robot_state = wm.RobotState()
robot_state.position = np.array([0.0, 0.0, 0.45])
robot_state.orientation = np.array([0.0, 0.0, 0.0, 1.0])
robot_state.linear_velocity = np.zeros(3)
robot_state.angular_velocity = np.zeros(3)
robot_state.total_force = np.zeros(3)

joint_names = [
    "joint_left_leg_1", "joint_left_leg_2", "joint_left_leg_3", "joint_left_leg_4",
    "joint_right_leg_1", "joint_right_leg_2", "joint_right_leg_3", "joint_right_leg_4"
]

armatures = { k : 0.01 for k in joint_names}

for name in joint_names:
    jd = wm.JointData()
    jd.pos = 0.0   
    jd.vel = 0.0   
    jd.acc = 0.0   
    jd.eff = 0.0   
    robot_state.joint_state[name] = jd

cp_left = np.array([0.0, 0.3, 0.0])
cp_right = np.array([0.0, -0.3, 0.0])
robot_state.contact_points = [cp_left, cp_right]

cf_left = np.array([0.0, 0.0, 50.0])  # 50N verticale
cf_right = np.array([0.0, 0.0, 50.0])
robot_state.contact_forces = [cf_left, cf_right]

manager = wm.WalkingManager()
success = manager.init(robot_state, armatures)

if success:
    print("WalkingManager initialized successfully.")
else:
    print("Failed to initialize WalkingManager.")

#cmd = wm.JointCommand()

for i in range(5):
    try:
        torque = manager.update(robot_state)
    except Exception as e:
        print("Error during update:", e)
        break
    print(f"Step {i+1}: Joint Commands: {torque}")
print("------- Loop finished")

for j in torque:
    if j in torque:
        print("funzioan")
    print(type(j))
    print(f"Joint: {j}")
    print(f"  Torque: {j[1]}")