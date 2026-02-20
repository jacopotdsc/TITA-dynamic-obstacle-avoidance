import mujoco
import mujoco.viewer
import time
import numpy as np
import ctypes
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
path7 = "/home/ubuntu/Desktop/repo_rl/TITA-dynamic-obstacle-avoidance/TITA_MJ/tita_mj_description/tita_world.xml"
path = path7

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
# Print initial joint states in the order of actuated joint names
print("Initial joint states (actuated order):")
for name in _actuated_joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid != -1:
        qpos_adr = model.jnt_qposadr[jid]
        q = data.qpos[qpos_adr] if qpos_adr >= 0 else float("nan")
        dof_adr = model.jnt_dofadr[jid]
        qvel = data.qvel[dof_adr] if dof_adr >= 0 else float("nan")
        print(f"  {name:30s} pos: {q: .5f} vel: {qvel: .5f}")
    else:
        print(f"  {name:30s} - joint not found")

armatures = {}
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    dof_adr = model.jnt_dofadr[i]
    if name and dof_adr >= 0:
        val = model.dof_armature[dof_adr]
    armatures[name] = val

print("model ptr:", model)
print("data ptr:", data)
print("id: ", id(model), id(data))

print("Initiazlizing robot state and walking manager...")
m_ptr = model._address  # puntatore interno
d_ptr = data._address   # puntatore interno

print(f"Model pointer: {m_ptr}, Data pointer: {d_ptr}")
print(f"Model pointer (hex): {hex(m_ptr)}, Data pointer (hex): {hex(d_ptr)}")

initial_robot_state = wm.robot_state_from_mujoco(m_ptr, d_ptr)

print("Initial robot state:")
print(initial_robot_state)
walking_manager = wm.WalkingManager()

wp = wm.WalkingPlanner(0.0, 0.0, 0.0, 0.25, 0.49)
res_init = walking_manager.init(initial_robot_state, armatures, wp)

wp_variables = wp.get_variables()
print(wp_variables.keys())
print("\nWalking Planner reference trajectories:")
print(f"vel_lin: {wp_variables['v']}, vel_z: {wp_variables['vz']}, vel_ang: {wp_variables['omega']}, z_min: {wp_variables['z_min']}, z_max: {wp_variables['z_max']}")


x_ref = wp.get_x_ref()  # [NX x N_STEP]
u_ref = wp.get_u_ref()  # [NU x (N_STEP-1)]
NX, N_STEP = x_ref.shape

dt = wp_variables['dt']  # 0.002
T = wp_variables['T']    # 13 s

# campiona ogni secondo
times = np.arange(0, T+1e-6)
for t in times:
    step = int(t/dt) #int(t / dt)
    step = min(step, N_STEP-1)
    x = x_ref[:, step]
    if step < u_ref.shape[1]:
        u = u_ref[:, step]
    else:
        u = u_ref[:, -1]  # ultimo comando disponibile
    print(f"[t={t:.1f}s] x={x[:3]} v={x[11]:.2f} vz={x[5]:.2f} omega={x[12]:.2f} u={u}")

print("\nSample x_ref at specific times:")
t0   = 0
tmid = int(1000 * T / 2)
t34 = int(1000 * T * 3 / 4)
tend = int(1000 * T)

for t_ms in [t0, tmid, t34, tend]:
    x_ref = wp.get_xref_at_time_ms(t_ms).reshape(-1)  # <-- FIX

    vz    = x_ref[5]
    v     = x_ref[11]
    omega = x_ref[12]

    print(f"x_ref @ t={t_ms} ms -> v={v:.3f}, vz={vz:.3f}, omega={omega:.3f}")

start_idx = max(0, N_STEP - 3)

print(f"\nFull x_ref from step {start_idx} to {N_STEP}:")
x_ref_full = wp.get_x_ref()  # shape (nx, N_STEP)
N_STEP = x_ref_full.shape[1]
for step in range(start_idx, N_STEP):
    x = x_ref_full[:, step].reshape(-1)  # vettore 1D
    vz    = x[5]
    v     = x[11]
    omega = x[12]
    print(f"step {step} -> v={v:.3f}, vz={vz:.3f}, omega={omega:.3f}")

start_real = time.time()
start_sim = data.time

frame_idx = 0
frame_th = 0

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
    "pert_duration_seconds": 0.1, #np.random.uniform(kick_durations[0], kick_durations[1]),
    "pert_dir": np.array([1, 0, 0]), # Direzione iniziale
}
# Calcolo steps
info["steps_until_next_pert"] = int(np.random.uniform(kick_wait_times[0], kick_wait_times[1]) / dt)
info["pert_duration_steps"] = int(info["pert_duration_seconds"] / dt)

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
            u_t = np.sin(np.pi * t / self.info["pert_duration_seconds"])
            # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
            max_force = 1000.0
            force = max_force * u_t

            # Lateral force vector, total latera magnitude:
            #   150 N: gentle push
            #   190 N: noticeable
            #   200 N: hard, bring to NaN
            # Front-back force vector, total longitudinal magnitude:
            #   200 N: gentle push, FEASIBLE
            # Top-down force vector, total vertical magnitude:
            #   10000 N: light 
            #   11000 N: gentle
            #   15000 N: noticeable 
            #   20000 N: moderate
            #   50000 N: pretty strong
            #print(self.info["pert_dir"], force)
            self.data.xfrc_applied[self._torso_body_id, :3] = force * self.info["pert_dir"]

            if self.info["pert_steps"] >= self.info["pert_duration_steps"]:
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

# --------------------------------
np.set_printoptions(precision=5, suppress=True)
while True:
    time.sleep(0.0)
    try:
        if viewer.is_running:

            print(f"Frame {frame_idx}:")

            real_diff = time.time() - start_real
            sim_diff = data.time - start_sim
        
            #if frame_idx % 100 == 0: 
            #    print(f"RTF: {sim_diff / real_diff:.2f}x")

            #perturbator._maybe_apply_perturbation()

            for name in _actuated_joint_names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid != -1:
                    qpos_adr = model.jnt_qposadr[jid]
                    q = data.qpos[qpos_adr] if qpos_adr >= 0 else float("nan")
                    dof_adr = model.jnt_dofadr[jid]
                    qvel = data.qvel[dof_adr] if dof_adr >= 0 else float("nan")
                    print(f"  {name:30s} pos: {q: .5f} vel: {qvel: .5f}")
                else:
                    print(f"  {name:30s} - joint not found")
                
            start_real = time.time()
            start_sim = data.time
            
            robot_state = wm.robot_state_from_mujoco(model._address, data._address)
            # Also print joint values coming from robot_state (for comparison)
            print("robot_state joints (actuated order):")
            for name in _actuated_joint_names:
                try:
                    jd = robot_state.joint_state[name]
                    pos_rs = jd.pos
                    vel_rs = jd.vel
                    print(f"  {name:30s} pos_rs: {pos_rs: .5f} vel_rs: {vel_rs: .5f}")
                except Exception:
                    print(f"  {name:30s} - not present in robot_state")
            
            pos_des = np.array([0.0, 0.0, 0.4])
            result_update = walking_manager.update(robot_state, pos_des)

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
            
            print("number of joints:", model.njnt)
            for i, joint_name in enumerate(_actuated_joint_names):
                q = data.qpos[model.jnt_qposadr[1+i]]  
                tau = data.qfrc_actuator[6+i] 
                print(f"  {joint_name:15s} | pos: {q: .5f} | torque: {tau: .5f}")
            print("-------")

            #print(torque_sorted)
            if frame_idx >= frame_th:
                if not np.isnan(torque_sorted).any():
                    data.ctrl[:] = torque_sorted
                else:
                    print(f"Warning: NaN nei torque, frame {frame_idx}, skipping assignment")

            print(f"Applied control: {data.ctrl}")
            
            #print("--------\nframe:", frame_idx)
            #print("ctrl:", data.ctrl)
            #print("torque:  ", torque_sorted)
            #print("mpc_sol:", mpc_solution)
            #draw_perturbation_arrow(viewer, perturbator)
            #print(f"prev: {[f'{x:.3f}' for x in body_coordinate]}, new: {[f'{x:.3f}' for x in body_coordinate_new]}")
            mujoco.mj_step(model, data)
            viewer.sync()

            frame_idx += 1
        else:
            break
    except Exception as e:
        break

# close
viewer.close()