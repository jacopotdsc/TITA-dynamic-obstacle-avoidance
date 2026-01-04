import mujoco
import mujoco.viewer
import time

import sys
ctrl_path = "/home/ubuntu/Desktop/repo_rl/TITA-dynamic-obstacle-avoidance/TITA_MJ/compiled/"
sys.path.insert(0, ctrl_path)
import wm

# path_tita_only = "/home/ubuntu/miniconda3/envs/mujoco_rl/lib/python3.12/site-packages/mujoco_playground/_src/locomotion/tita/xmls/tita_mjx.xml"
base_string = "/home/ubuntu/miniconda3/envs/mujoco_rl/lib/python3.12/site-packages/mujoco_playground/"
path1 = base_string + "_src/locomotion/tita/xmls/scene_mjx_flat_terrain.xml"
path2 = base_string + "_src/locomotion/go1/xmls/scene_mjx_flat_terrain.xml"
path3 = base_string + "tesi/tita/urdf/tita_description.urdf"
path4 = base_string + "tesi/test_python/tita_converted.xml"
path5 = "/home/ubuntu/miniconda3/envs/tianshou/lib/python3.12/site-packages/gymnasium/envs/mujoco/assets/tita_mjx.xml"
path6 = "/home/ubuntu/Desktop/repo_rl/TITA-dynamic-obstacle-avoidance/TITA_MJ/tita_mj_description/tita.xml"
path = path6

model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)
#default_pose = model.keyframe("home").qpos
#default_ctrl = default_pose[7:]
#data.qpos = default_pose
#data.ctrl = default_ctrl
joint_targets = {
    "joint_left_leg_1": 0.0,
    "joint_left_leg_2": 0.5,
    "joint_left_leg_3": -1.0,
    "joint_left_leg_4": 0.0,
    "joint_right_leg_1": 0.0,
    "joint_right_leg_2": 0.5,
    "joint_right_leg_3": -1.0,
    "joint_right_leg_4": 0.0
}

data.qpos[0] = 0.0
data.qpos[1] = 0.0
data.qpos[2] = 0.399 + 0.05 - 0.005 

data.qpos[3] = 1.0  # w (parte scalare)
data.qpos[4] = 0.0  # x
data.qpos[5] = 0.0  # y
data.qpos[6] = 0.0  # z

for joint_name, angle in joint_targets.items():
    
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

    if joint_id != -1:
        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] = angle
    else:
        print(f"[WARNING] Giunto non trovato: {joint_name}")

mujoco.mj_forward(model, data)

viewer = mujoco.viewer.launch_passive(model, data) 
viewer.cam.distance = 15.0
viewer.cam.azimuth = 30
viewer.cam.elevation = -20

_actuated_joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
    for i in range(model.nu)
]

print(_actuated_joint_names)

armatures = {}
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    dof_adr = model.jnt_dofadr[i]
    if name and dof_adr >= 0:
        val = model.dof_armature[dof_adr]
    armatures[name] = val

initial_robot_state = wm.robot_state_from_mujoco(model, data)
walking_manager = wm.WalkingManager()
walking_manager.init(initial_robot_state, armatures)

start_real = time.time()
start_sim = data.time

frame_idx = 0
frame_th = 10
while True:
    time.sleep(0.0)
    try:
        if viewer.is_running:
            frame_idx += 1

            real_diff = time.time() - start_real
            sim_diff = data.time - start_sim

            #if frame_idx % 100 == 0: 
            #    print(f"RTF: {sim_diff / real_diff:.2f}x")
                
            start_real = time.time()
            start_sim = data.time
            
            robot_state = wm.robot_state_from_mujoco(model, data)
            #print(robot_state)
            result_update = walking_manager.update(robot_state)
            torque = result_update.cmd
            mpc_solution = result_update.solution
            print("MPC solution com:")
            print(mpc_solution.com)
            print("MPC solution pc:")
            print(mpc_solution.pc)
            print("------")
            print(mpc_solution.com.pos)
            print(mpc_solution.com.vel)
            print(mpc_solution.com.acc)
            print("------")
            print(mpc_solution.pc.pos)
            print(mpc_solution.pc.vel)
            print(mpc_solution.pc.acc)

            if frame_idx >= 2:
                exit(0)
            #for ( key, val) in torque:
            #    print(f"{key}: {val}")
        
            torque_sorted = []
            for joint_name in _actuated_joint_names:
                val = torque[joint_name]

                #print(f"{joint_name}: {val:.3f}"    )

                if frame_idx >= frame_th:
                    #print(f"{joint_name}: {val:.3f}"    )
                    torque_sorted.append(val)

            #print(torque_sorted)
            if frame_idx >= frame_th:
                data.ctrl = torque_sorted

            mujoco.mj_step(model, data)
            #print(f"prev: {[f'{x:.3f}' for x in body_coordinate]}, new: {[f'{x:.3f}' for x in body_coordinate_new]}")
            viewer.sync()
        else:
            break
    except Exception as e:
        break

# close
viewer.close()