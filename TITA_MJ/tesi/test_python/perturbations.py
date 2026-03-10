import mujoco
import mujoco.viewer
import time
import numpy as np
import ctypes
import sys
ctrl_path = "/home/ubuntu/Desktop/repo_rl/TITA-dynamic-obstacle-avoidance/TITA_MJ/compiled/"
sys.path.insert(0, ctrl_path)
import wm

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import csv
import pandas as pd
import argparse


import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- CLI args ---
parser = argparse.ArgumentParser(description="Run perturbations and save outputs")
parser.add_argument('--name', type=str, default='perturbation', help='Base name for output CSV/plots')
parser.add_argument('--compare', nargs=2, metavar=('CSV_A', 'CSV_B'),
                    help='Optional: compare two existing CSV files (provide two paths)')
parser.add_argument('--len', type=int, default=None, help='Terminate viewer after N iterations (default: 1000)')
args = parser.parse_args()
output_basename = args.name

def _ensure_csv_name(name):
    if name is None:
        return None
    s = str(name)
    return s if s.lower().endswith('.csv') else s + '.csv'

# initial normalized csv name from --name
output_csv_filename = _ensure_csv_name(output_basename)

# If the provided --name contains a path segment, create a subfolder under
# perturbation_outputs using the path head and use the tail as the basename
# for saved files. Example: --name exp1/run1 -> create
# perturbation_outputs/exp1/ and use 'run1' as file base.
script_dir = os.path.dirname(__file__)
perturb_outputs_root = os.path.join(script_dir, "perturbation_outputs")
if os.sep in output_basename:
    head, tail = os.path.split(output_basename)
    subfolder = head
    file_base = tail if tail else head
else:
    raise RuntimeError("Need to specify folder name: 'python3 perturbation.py --name folder_name/exp_name'")

save_dir = os.path.join(perturb_outputs_root, subfolder) if subfolder else perturb_outputs_root
os.makedirs(save_dir, exist_ok=True)

# normalized CSV filename (based on file_base)
output_csv_filename = _ensure_csv_name(file_base)
output_file_base = file_base

def save_perturbation_plot(history_perturb, save_dir=None, filename="perturbation_plot.png"):
    """Save a perturbation plot (X, Y, Z) to a folder next to this script.

    history_perturb: array-like shape (N,3) or (3,) for single step
    save_dir: optional target folder; if None, uses ./perturbation_outputs next to this file
    filename: image file name
    """
    arr = np.array(history_perturb)
    if arr.ndim == 1:
        if arr.size == 3:
            arr = arr.reshape(1, 3)
        else:
            raise ValueError("history_perturb must have 3 elements or shape (N,3)")

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "perturbation_outputs")
    os.makedirs(save_dir, exist_ok=True)

    frames = np.arange(len(arr))
    plt.figure(figsize=(10, 6))
    plt.plot(frames, arr[:, 0], color='red', label='Perturb_X')
    plt.plot(frames, arr[:, 1], color='green', label='Perturb_Y')
    plt.plot(frames, arr[:, 2], color='blue', label='Perturb_Z')
    plt.title("External Perturbations Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Force/Torque Value")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    saved = os.path.join(save_dir, filename)
    plt.savefig(saved)
    plt.close()
    print(f"Saved {saved}")

def save_perturbation_csv(history_perturb, save_dir=None, filename="perturbation.csv"):
    """Write perturbation history to CSV with columns: force, dir_x, dir_y, dir_z.

    history_perturb: array-like shape (N,3) or (3,) for single step
    save_dir: optional target folder; if None, uses ./perturbation_outputs next to this file
    filename: csv file name
    """
    arr = np.array(history_perturb)
    if arr.ndim == 1:
        if arr.size == 3:
            arr = arr.reshape(1, 3)
        else:
            raise ValueError("history_perturb must have 3 elements or shape (N,3)")

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "perturbation_outputs")
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, filename)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["force", "dir_x", "dir_y", "dir_z"])
        for row in arr:
            force = float(np.linalg.norm(row))
            if force > 0:
                writer.writerow([force, float(row[0])/force, float(row[1])/force, float(row[2])/force])
            else:
                writer.writerow([force, float(row[0]), float(row[1]), float(row[2])])


    print(f"CSV saved in: {csv_path}")

def save_perturbation_force2d(history_perturb, save_dir=None, filename="perturbation_force2d.png", max_force=None):
    """Save a 2D plot of applied forces: distance from origin = magnitude (up to max_force),
    angle = direction of the force (atan2(y,x)).

    history_perturb: array-like shape (N,3) or (3,) for single step
    save_dir: optional target folder; if None, uses ./perturbation_outputs next to this file
    filename: image file name
    max_force: optional maximum radius to draw (if None uses provided global `max_force` or observed max)
    """
    arr = np.array(history_perturb)
    if arr.ndim == 1:
        if arr.size == 3:
            arr = arr.reshape(1, 3)
        else:
            raise ValueError("history_perturb must have 3 elements or shape (N,3)")

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "perturbation_outputs")
    os.makedirs(save_dir, exist_ok=True)

    # Use only X,Y components for 2D plot
    xy = arr[:, :2]
    mags = np.linalg.norm(xy, axis=1)

    # observed maximum magnitude from data (used to size the square)
    observed_max = float(np.max(mags) if mags.size else 1.0)

    # Determine plotting max force: prefer explicit arg, else use observed max
    if max_force is None:
        max_force_val = observed_max
    else:
        max_force_val = float(max_force)

    # Compute square half-side using requested formula: half = observed_max*2 + 5
    half = observed_max + 5.0

    # dir: unit vector of XY direction; points placed at position = dir * max_force_val
    dirs = np.zeros_like(xy)
    norms = np.linalg.norm(xy, axis=1)
    nonzero = norms > 0
    dirs[nonzero] = (xy[nonzero].T / norms[nonzero]).T

    points = dirs * max_force_val

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    rect = Rectangle((-half, -half), 2*half, 2*half, fill=False, linestyle='--', color='gray')
    ax.add_patch(rect)

    # scatter the points; if multiple points share the same position, stack them vertically
    scatter_xy = points.copy()
    # group by rounded coordinates to detect duplicates
    rounded = np.round(points, decimals=8)
    coord_to_indices = {}
    for idx, (xv, yv) in enumerate(rounded):
        key = (float(xv), float(yv))
        coord_to_indices.setdefault(key, []).append(idx)

    ax.scatter(scatter_xy[:, 0], scatter_xy[:, 1], c='tab:blue', s=30, label='applied force (max*dir)')

    # draw origin
    ax.scatter([0], [0], c='k', s=20, label='origin')

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Perturbation forces (2D): position = max_force * direction')
    ax.set_aspect('equal', 'box')
    ax.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    saved = os.path.join(save_dir, filename)
    plt.savefig(saved)
    plt.close()
    print(f"Saved {saved}")

def compare_perturbation_csvs(csv_a, csv_b, save_dir=None, filename="compare_perturbations.png", max_force=None):

    def load_xy(csv_file):
        data = np.genfromtxt(csv_file, delimiter=",", names=True)
        m = data["force"]
        x = data["force_x"]
        y = data["force_y"]
        return m, np.column_stack((x, y))

    m_a, xy_a = load_xy(csv_a)
    m_b, xy_b = load_xy(csv_b)

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "perturbation_outputs")
    os.makedirs(save_dir, exist_ok=True)

    observed_max = float(max(np.max(m_a), np.max(m_b)))

    if max_force is None:
        max_force_val = observed_max
    else:
        max_force_val = float(max_force)

    half = observed_max + 5.0

    def project_points(xy, max_force):
        dirs = np.zeros_like(xy)
        norms = np.linalg.norm(xy, axis=1)
        nonzero = norms > 0
        dirs[nonzero] = (xy[nonzero].T / norms[nonzero]).T
        return dirs * max_force[:,None]

    points_a = project_points(xy_a, m_a)
    points_b = project_points(xy_b, m_b)

    mask_a = m_a > 0
    mask_b = m_b > 0

    points_a_nz = points_a[mask_a]
    points_b_nz = points_b[mask_b]

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    rect = Rectangle((-half, -half), 2*half, 2*half,
                     fill=False, linestyle='--', color='gray')
    ax.add_patch(rect)

    #ax.scatter(points_a[:, 0], points_a[:, 1], c='tab:blue', s=30, label='CSV A')
    #ax.scatter(points_b[:, 0], points_b[:, 1], c='tab:red', s=30, label='CSV B')
    ax.plot(points_a_nz[:,0], points_a_nz[:,1], color='tab:orange', linewidth=1, alpha=0.7)
    ax.scatter(points_a_nz[:,0], points_a_nz[:,1], c='tab:orange', s=30, label='NN')

    ax.plot(points_b_nz[:,0], points_b_nz[:,1], color='tab:blue', linewidth=1, alpha=0.7)
    ax.scatter(points_b_nz[:,0], points_b_nz[:,1], c='tab:blue', s=30, label='MPC')
    ax.scatter([0], [0], c='k', s=20, label='origin')

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Perturbation forces comparison')
    ax.set_aspect('equal', 'box')
    ax.legend(loc='upper right')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    saved = os.path.join(save_dir, filename)
    plt.savefig(saved)
    plt.close()

    print(f"Saved {saved}")

path = "/home/ubuntu/Desktop/repo_rl/TITA-dynamic-obstacle-avoidance/TITA_MJ/tita_mj_description/tita_world.xml"

model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)

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
viewer = mujoco.viewer.launch_passive(model, data) 
viewer.cam.distance = 15.0
viewer.cam.azimuth = 30
viewer.cam.elevation = -20

_actuated_joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
    for i in range(model.nu)
]
armatures = {}
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    dof_adr = model.jnt_dofadr[i]
    if name and dof_adr >= 0:
        val = model.dof_armature[dof_adr]
    armatures[name] = val

m_ptr = model._address  # puntatore interno
d_ptr = data._address   # puntatore interno

initial_robot_state = wm.robot_state_from_mujoco(m_ptr, d_ptr)
walking_manager = wm.WalkingManager()

wp = wm.WalkingPlanner(0.0, 0.0, 0.0, 0.4, 0.25, 0.49)
res_init = walking_manager.init(initial_robot_state, armatures, wp)

start_real = time.time()
start_sim = data.time

frame_idx = 0

# ------------------------------
dt = model.opt.timestep
torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso") 
torso_mass = model.body_mass[torso_body_id]

max_force = 10
fixed_perturbation=True
first_perturb = 100
fixed_wait_steps=100
fixed_duration_steps=50
max_perturbations=100

perturbation_directions = {
    str(i): np.array([np.cos(np.deg2rad(i*30)), np.sin(np.deg2rad(i*30)), 0])
    for i in range(12)
}
pert_dir_length = len(list(perturbation_directions))

info = {
    "steps_until_next_pert": 0,
    "pert_duration_seconds": 0,
    "pert_duration_steps": 0,
    "steps_since_last_pert": 0,
    "pert_steps": 0,
    "pert_dir": np.array([0.0, 0.0, 0.0]),
    "n_perturbations_applied": 0,
    "perturbing": False,
    "perturbation_directions": perturbation_directions
}

info["steps_until_next_pert"] = int(fixed_wait_steps)
info["pert_duration_steps"] = int(fixed_duration_steps)
info["pert_duration_seconds"] = float(info["pert_duration_steps"] * dt)

offset_recover = 400
times_perturbation =  fixed_duration_steps*pert_dir_length
times_waiting = fixed_wait_steps*(pert_dir_length-1) 
computed_length = first_perturb + times_perturbation + times_waiting + offset_recover if args.len is None else args.len

class Perturbator:
    def __init__(self, model, data, info, dt, max_force, torso_id, torso_mass, viewer):
        self.model = model
        self.data = data
        self.info = info
        self.max_force = max_force
        self.dt = dt
        self._torso_body_id = torso_id
        self._torso_mass = torso_mass
        self.viewer = viewer    

    def _maybe_apply_perturbation(self, n_frame):
        def gen_dir() -> np.ndarray:
            angle = np.random.uniform(low=0.0, high=np.pi * 2)

            pert_number = len(list(self.info["perturbation_directions"].keys()))
            pert_dir = self.info["n_perturbations_applied"] % pert_number
            dir_force = self.info["perturbation_directions"][str(pert_dir)]
            #dir_force = np.array([np.cos(angle), np.sin(angle), 0.0])
           
            return  dir_force

        def apply_pert():
            t = self.info["pert_steps"] * self.dt
            u_t = np.sin(np.pi * t / self.info["pert_duration_seconds"])

            max_force = abs(self.max_force) 

            force = max_force #* u_t
            data.xfrc_applied[self._torso_body_id, :3] = force * self.info["pert_dir"]
            if self.info["pert_steps"] >= self.info["pert_duration_steps"]:
                self.info["steps_since_last_pert"]  = 0
                self.info["n_perturbations_applied"] += 1

            self.info["pert_steps"] += 1
            self.info["perturb"] = self.data.xfrc_applied.copy()[self._torso_body_id, :3]

        def wait():
            self.info["steps_since_last_pert"] += 1
            self.data.xfrc_applied[self._torso_body_id, :3] = 0.0
            self.info["perturb"] = np.zeros(3)

            if self.info["steps_since_last_pert"] >= self.info["steps_until_next_pert"]:
                self.info['pert_steps'] = 0

        if (n_frame >= first_perturb 
            and self.info["steps_since_last_pert"] >= fixed_wait_steps
            and self.info["n_perturbations_applied"] < max_perturbations
            ):
            if self.info['perturbing'] == False:
                self.info["pert_dir"] = gen_dir()
                self.info["pert_steps"] = 0

            self.info["perturbing"] = True
            apply_pert()
        else:
            self.info["perturbing"] = False
            wait()

perturbator = Perturbator(model, data, info, dt, max_force,torso_body_id, torso_mass, viewer)
np.set_printoptions(precision=5, suppress=True)

history_perturb = []

try:
    while True:
        time.sleep(0.0)

        if not viewer.is_running:
            break

        # terminate after a fixed number of iterations if requested
        if frame_idx >= int(computed_length):
            print(f"Reached maximu length {computed_length}, exiting main loop")
            break

        perturbator._maybe_apply_perturbation(frame_idx)
        cur_pert = info.get("perturb", np.zeros(3))        
        history_perturb.append(np.array(cur_pert, copy=True))

        #if info.get("perturbing", False):
        #    # append only at the first perturbation step to avoid duplicates
        #    if info.get("pert_steps", 0) == 1:
        #        history_perturb.append(np.array(cur_pert, copy=True))

        robot_state = wm.robot_state_from_mujoco(model._address, data._address)
        result_update = walking_manager.update(robot_state)

        torque = result_update.torque
        mpc_solution = result_update.solution

        torque_sorted = []
        for joint_name in _actuated_joint_names:
            val = torque[joint_name]
            torque_sorted.append(val)

        if not np.isnan(torque_sorted).any():
            data.ctrl[:] = torque_sorted
        else:
            print(f"Warning: NaN nei torque, frame {frame_idx}, skipping assignment")

        mujoco.mj_step(model, data)
        viewer.sync()

        frame_idx += 1
except Exception:
    # exit loop on unexpected errors
    pass
finally:
    # save plot and csv of perturbations
    print("\nSimulation ended, saving perturbation data...")
    try:
        save_perturbation_plot(history_perturb, save_dir=save_dir, filename=f"{output_file_base}_plot.png")
    except Exception as e:
        print(f"Failed saving perturbation plot: {e}")
    try:
        save_perturbation_csv(history_perturb, save_dir=save_dir, filename=output_csv_filename)
    except Exception as e:
        print(f"Failed saving perturbation csv: {e}")
    try:
        save_perturbation_force2d(history_perturb, save_dir=save_dir, filename=f"{output_file_base}_force2d.png")
    except Exception as e:
        print(f"Failed saving perturbation force2d: {e}")
    try:
        if args.compare is not None:
            ca = _ensure_csv_name(args.compare[0])
            cb = _ensure_csv_name(args.compare[1])
            
            if not os.path.isabs(ca) and os.sep not in args.compare[0]:
                ca = os.path.join(save_dir, "..", ca)
            if not os.path.isabs(cb) and os.sep not in args.compare[1]:
                cb = os.path.join(save_dir, "..", cb)
            compare_perturbation_csvs(
                ca,
                cb,
                save_dir=save_dir,
                filename=f"{output_file_base}_compare.png"
            )
    except Exception as e:
        print(f"Failed comparing perturbation csvs: {e}")
    try:
        viewer.close()
    except Exception:
        pass