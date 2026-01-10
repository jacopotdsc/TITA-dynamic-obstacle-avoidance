import mujoco
import mujoco.viewer
import time
import numpy as np

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
data.qpos[2] = 0.4 #0.399 + 0.05 - 0.005 

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
print(f"Starting height: {data.qpos[2]}")
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

# ------------------------------
dt = model.opt.timestep
torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso") # Assicurati che il nome sia corretto
torso_mass = model.body_mass[torso_body_id]

# Configurazione (estratta dai tuoi parametri)
kick_wait_times = [1.0, 3.0]
kick_durations = [0.05, 0.2]
velocity_kick = [0.0, 3.0]

# Stato iniziale della perturbazione
info = {
    "steps_since_last_pert": 0,
    "pert_steps": 0,
    "pert_mag": 1.0, #np.random.uniform(velocity_kick[0], velocity_kick[1]),
    "pert_duration_seconds": np.random.uniform(kick_durations[0], kick_durations[1]),
    "pert_dir": np.array([1, 0, 0]), # Direzione iniziale
}
# Calcolo steps
info["steps_until_next_pert"] = int(np.random.uniform(kick_wait_times[0], kick_wait_times[1]) / dt)
info["pert_duration"] = int(info["pert_duration_seconds"] / dt)

# Mock di "self" per usare la tua funzione originale
class Perturbator:
    def __init__(self, model, data, info, dt, torso_id, torso_mass, viewer):
        self.model = model
        self.data = data
        self.info = info
        self.dt = dt
        self._torso_body_id = torso_id
        self._torso_mass = torso_mass
        self.viewer = viewer    
        self.current_force = np.array([0.0, 0.0, 1.0])


    def _maybe_apply_perturbation(self):
        def gen_dir() -> np.ndarray:
            angle = np.random.uniform(low=0.0, high=np.pi * 2)
            dir_force = np.array([np.cos(angle), np.sin(angle), 0.0])
            return  dir_force

        def apply_pert():
            t = self.info["pert_steps"] * self.dt
            t = 1
            u_t = np.sin(np.pi * t / self.info["pert_duration_seconds"])
            # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
            force = (
                u_t  # (unitless)
                * self._torso_mass  # kg
                * self.info["pert_mag"]  # m/s
                / self.info["pert_duration_seconds"]  # 1/s
            )

            # Lateral force vector, total latera magnitude:
            #   150 N: gentle push
            #   200 N: soft and noticeable
            #   300 N: noticeable, 
            # Top-down force vector, total vertical magnitude:
            #   10000 N: light 
            #   11000 N: gentle
            #   15000 N: noticeable 
            #   20000 N: moderate
            #   50000 N: pretty strong
            force = 11000.0
            print(self.info["pert_dir"], force)
            self.data.xfrc_applied[self._torso_body_id, :3] = force * self.info["pert_dir"]

            if self.info["pert_steps"] >= self.info["pert_duration"]:
                self.info["steps_since_last_pert"]  = 0

            self.info["pert_steps"] += 1

        def wait():
            self.info["steps_since_last_pert"] += 1
            xfrc_applied = np.zeros((self.model.nbody, 6))
            self.data.xfrc_applied[self._torso_body_id, :3] = 0.0

            if self.info["steps_since_last_pert"] >= self.info["steps_until_next_pert"]:
                self.info['pert_steps'] = 0
                
            if self.info["steps_since_last_pert"] >= self.info["steps_until_next_pert"]:
                self.info["pert_dir"] = gen_dir()

        if self.info["steps_since_last_pert"] >= self.info["steps_until_next_pert"]:
            apply_pert()
        else:
            wait()

perturbator = Perturbator(model, data, info, dt, torso_body_id, torso_mass, viewer)

def draw_perturbation_arrow(viewer, perturbator):
    if viewer is not None and viewer.is_running:
        f_norm = np.linalg.norm(perturbator.current_force)
        if f_norm > 0.0:
            pos = perturbator.data.xpos[perturbator._torso_body_id]
            viewer.add_marker(
                pos=pos,
                mat=np.eye(3).flatten(),
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=[0.02, 0.02, f_norm * 0.02],
                rgba=[1, 0, 0, 0.8],
                label="KICK"
            )
# --------------------------------
while True:
    time.sleep(0.0)
    try:
        if viewer.is_running:
            frame_idx += 1

            real_diff = time.time() - start_real
            sim_diff = data.time - start_sim

            #if frame_idx % 100 == 0: 
            #    print(f"RTF: {sim_diff / real_diff:.2f}x")

            perturbator._maybe_apply_perturbation()
                
            start_real = time.time()
            start_sim = data.time
            
            robot_state = wm.robot_state_from_mujoco(model, data)
            #print(robot_state)
            result_update = walking_manager.update(robot_state, np.array([0.0, 0.0, 0.40]))
            torque = result_update.cmd
            #if frame_idx == 0 or frame_idx % 100 == 0:
            #    print(torque)
            mpc_solution = result_update.solution
            
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
            #draw_perturbation_arrow(viewer, perturbator)
            #print(f"prev: {[f'{x:.3f}' for x in body_coordinate]}, new: {[f'{x:.3f}' for x in body_coordinate_new]}")
            viewer.sync()
        else:
            break
    except Exception as e:
        break

# close
viewer.close()