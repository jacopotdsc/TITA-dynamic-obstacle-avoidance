import mujoco
import mujoco.viewer
import time
import numpy as np

np.set_printoptions(precision=2, suppress=True)

def get_sensor_data(
        model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str
    ) -> np.ndarray:
        """Gets sensor data given sensor name."""
        sensor_id = model.sensor(sensor_name).id
        sensor_adr = model.sensor_adr[sensor_id]
        sensor_dim = model.sensor_dim[sensor_id]
        return data.sensordata[sensor_adr : sensor_adr + sensor_dim]

def _get_obs(model, data) -> np.ndarray:
        """Get the current observation."""
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        # Variable definition for readability
        height = qpos[2] #np.array(qpos[2])
        orientation = qpos[3:7]
        linvel = get_sensor_data(model, data, "local_linvel")
        linacc = get_sensor_data(model, data, "local_linacc")
        joint_angles = qpos[7:]
        joint_vel = qvel[6:]
        joint_acc = data.qacc[6:]
        joint_torque_controller_normalized = np.zeros_like(joint_vel) # info["tita_controller_output"]  / abs(self.model.actuator_forcerange[:, 1])
        command = np.zeros(3)
        
        print("\n--- TITA ROBOT STATE ---")
        print(f"Height:              {height:.4f} m")
        print(f"Orientation (quat):  {orientation}")
        print(f"Linear Velocity:     {linvel}")
        print(f"Linear Acc (Gyro):   {linacc}")
        print(f"Joint Angles:        {joint_angles}")
        print(f"Joint Velocity:      {joint_vel}")
        print(f"Joint Acceleration:  {joint_acc}")
        print(f"Command (Input):     {command}")
        print("-" * 30)
        print(f"Controller Output:   {joint_torque_controller_normalized}")
        print("------------------------\n")

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
default_pose = model.keyframe("home").qpos
default_ctrl = default_pose[7:]
#data.qpos = default_pose
#data.ctrl = default_ctrl

data.qpos[:] = np.copy(default_pose)
data.qvel[:] = np.zeros(model.nv)
mujoco.mj_forward(model, data)

print("Target Motori inviato:", data.ctrl)

body_coordinate = data.qpos[0:3]
print("Initial body coordinate:", body_coordinate)
print("Using ctrl values:", data.ctrl)
print(f"Total mass: {model.body_mass.sum():.2f} kg")

print("\n--- Link details ---")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    mass = model.body_mass[i]

    if name == "world": continue
    
    print(f"Link: {name:20} | Mass: {mass:.3f} kg")

print("\n--- Joint details ---")
actuated_joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
    for i in range(model.nu)
]
print("Joint Names:", actuated_joint_names)
print("-------------------------------\n")

for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    qpos_adr = model.jnt_qposadr[i]
    dof_adr = model.jnt_dofadr[i]
    
    print(f"{i:<5} | {name:<20} | {qpos_adr:<12} | {dof_adr:<12}")
print("-------------------------------\n")

print(model.actuator_ctrlrange[:, 0])
print(model.actuator_ctrlrange[:, 1])
print(data.actuator_force )
print("\n--- Actuator Ranges (Ctrl & Force) ---")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if name is None: name = f"actuator_{i}"
    
    ctrl_lim = model.actuator_ctrllimited[i]
    force_lim = model.actuator_forcelimited[i]
    act_lim = model.actuator_actlimited[i] # used for muscles actrange parameter
    
    s_ctrl = 'ON' if ctrl_lim else 'OFF'
    s_force =  'ON' if force_lim else 'OFF'
    s_act = 'ON' if act_lim else 'OFF'

    ctrl_range = model.actuator_ctrlrange[i]
    force_range = model.actuator_forcerange[i]

    print(f"{name} ->", end="")
    if s_ctrl == "ON":
        print(f" Ctrl range: {ctrl_range},", end="")
    else:
        print(f" Ctrl range: {s_ctrl}", end="")
    if s_force == "ON":
        print(f" Force range: {force_range},", end="")
    else:
        print(f" Force range: {s_force}", end="")

    print()

print("-------------------------------------------------\n")

damping = model.dof_damping[6:]
actuator_gainprtm = model.actuator_gainprm[:, 0]
actuator_biasprm = model.actuator_biasprm[:, 1]
print("Damping:", damping)
print("Actuator gainprm:", actuator_gainprtm)
print("Actuator biasprm:", actuator_biasprm)

print("\n-----------------------------------------------------")
viewer = mujoco.viewer.launch_passive(model, data) 

while viewer.is_running():
    try:
        print("Stepping simulation...")
        _get_obs(model, data)
        print("---------------")
        mujoco.mj_step(model, data)
        viewer.sync()
    except KeyboardInterrupt:
        break

viewer.close()