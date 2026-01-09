import os
import sys
import pandas as pd
import git
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import numpy as np

def get_git_root():
    """
    Ritorna la root della repository git corrente.
    """
    try:
        repo = git.Repo(".", search_parent_directories=True)
        return repo.working_tree_dir
    except git.InvalidGitRepositoryError:
        return None

def plot_total_reward(csv_path, plots_dir=None):
    df = pd.read_csv(csv_path)

    episode_starts = df.index[df['frame'] == 0].tolist()
    episode_starts.append(len(df))  
    total_rewards = []
    rewards_per_frame = []

    for i in range(len(episode_starts)-1):
        start_idx = episode_starts[i]
        end_idx = episode_starts[i+1]
        ep_reward = df['reward'].iloc[start_idx:end_idx].sum()
        total_rewards.append(ep_reward)

    last_episode_start = episode_starts[-2] # Penultimo elemento della lista è l'inizio dell'ultimo ep
    last_episode_df = df.iloc[last_episode_start:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # --- GRAFICO 1: Reward totale (X = Episodi) ---
    episodes = range(1, len(total_rewards) + 1)
    ax1.plot(episodes, total_rewards, marker='o', color='tab:blue', label='Tot. Reward')
    ax1.set_title("Cumulative Reward per episodio")
    ax1.set_xlabel("Episode") 
    ax1.set_ylabel("Total reward")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- GRAFICO 2: Reward istantanea (X = Frame) ---
    # Usiamo il conteggio dei frame interno all'episodio
    ax2.plot(last_episode_df['frame'], last_episode_df['reward'], color='tab:red', label='Frame Reward')
    ax2.set_title(f"Reward per frame of last episode (episode {len(total_rewards)})")
    ax2.set_xlabel("Number of frame")
    ax2.set_ylabel("Reward")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    plt.tight_layout()

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, "total_reward.png"))
        print(f"Saved total_reward.png in {plots_dir}")

def plot_robot_height(csv_path, plots_dir=None):
    df = pd.read_csv(csv_path)
    episode_starts = df.index[df['frame'] == 0].tolist()
    episode_starts.append(len(df))
    heights = []

    for i in range(len(episode_starts)-1):
        start_idx = episode_starts[i]
        end_idx = episode_starts[i+1]
        ep_df = df.iloc[start_idx:end_idx]
        heights.append(ep_df['robot_height'].mean())

    plt.figure(figsize=(10,5))
    plt.plot(range(1,len(heights)+1), heights, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Robot Height")
    plt.title("Average Robot Height per Episode")
    plt.grid(True)

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, "robot_height.png"))
        print(f"Saved robot_height.png in {plots_dir}")

def plot_orientation(csv_path, plots_dir=None):
    df = pd.read_csv(csv_path)
    ori_cols = ['ori_x','ori_y','ori_z','ori_w']
    episode_starts = df.index[df['frame'] == 0].tolist()
    episode_starts.append(len(df))
    ori_means = {col: [] for col in ori_cols}
    euler_means = {'roll': [], 'pitch': [], 'yaw': []}

    for i in range(len(episode_starts)-1):
        start_idx = episode_starts[i]
        end_idx = episode_starts[i+1]
        ep_df = df.iloc[start_idx:end_idx]
        quat_data = ep_df[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
        rot = Rotation.from_quat(quat_data)
        rot_euler = rot.as_euler('xyz', degrees=False)
        means = rot_euler.mean(axis=0)
        euler_means['roll'].append(means[0])
        euler_means['pitch'].append(means[1])
        euler_means['yaw'].append(means[2])
        
    plt.figure(figsize=(10,5))
    for label, values in euler_means.items():
        # Converte in array numpy per manipolazione veloce
        val_array = np.array(values)
        
        val_array = np.where(val_array >= np.pi/2, val_array - np.pi, val_array)
        val_array = np.where(val_array <= -np.pi/2, val_array + np.pi, val_array)
        plt.plot(range(1,len(euler_means['roll'])+1), val_array, marker='o', label=label)
    
    plt.ylim(-np.pi/2, np.pi/2)
    plt.xlabel("Episode")
    plt.ylabel("Radians")
    plt.title("Average Orientation per Episode")
    plt.legend()
    plt.grid(True)

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, "orientation.png"))
        print(f"Saved orientation.png in {plots_dir}")

def plot_total_torque(csv_path, plots_dir=None):
    df = pd.read_csv(csv_path)
    torque_cols = [f'joint_torque_{i}' for i in range(1,9)]
    episode_starts = df.index[df['frame'] == 0].tolist()
    episode_starts.append(len(df))
    total_torques = []

    for i in range(len(episode_starts)-1):
        start_idx = episode_starts[i]
        end_idx = episode_starts[i+1]
        ep_df = df.iloc[start_idx:end_idx]
        total_torques.append(ep_df[torque_cols].sum(axis=1).sum())

    plt.figure(figsize=(10,5))
    plt.plot(range(1,len(total_torques)+1), total_torques, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Total Torque N/m")
    plt.title("Total Joint Torque per Episode")
    plt.grid(True)

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, "total_torque.png"))
        print(f"Saved total_torque.png in {plots_dir}")

def plot_joint_torque_last_episode(csv_path, plots_dir=None):
    df = pd.read_csv(csv_path)
    torque_cols = [f'action_{i}' for i in range(1, 9)]
    
    joint_names = ["ankle_pitch", "ankle_roll", "knee", "wheel"]
    legend_left = ["left_" + name for name in joint_names]
    legend_right = ["right_" + name for name in joint_names]

    episode_starts = df.index[df['frame'] == 0].tolist()
    start_idx = episode_starts[-1]
    ep_df = df.iloc[start_idx:]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Sopra: left joints
    for col, legend in zip(torque_cols[:4], legend_left):
        axes[0].plot(ep_df['frame'], ep_df[col], label=legend)
    axes[0].set_ylabel("Torque")
    axes[0].set_title(f"Left Leg Joint Torques (Last Episode {len(episode_starts)})")
    axes[0].legend()
    axes[0].grid(True)

    # Sotto: right joints
    for col, legend in zip(torque_cols[4:], legend_right):
        axes[1].plot(ep_df['frame'], ep_df[col], label=legend)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Torque")
    axes[1].set_title(f"Right Leg Joint Torques (Last Episode {len(episode_starts)})")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, "joint_torque_last_episode.png"))
        print(f"Saved joint_torque_last_episode.png in {plots_dir}")

def plot_total_prev_action(csv_path, plots_dir=None):
    df = pd.read_csv(csv_path)
    action_cols = [f'prev_action_{i}' for i in range(1,9)]
    episode_starts = df.index[df['frame'] == 0].tolist()
    episode_starts.append(len(df))
    total_actions = []

    for i in range(len(episode_starts)-1):
        start_idx = episode_starts[i]
        end_idx = episode_starts[i+1]
        ep_df = df.iloc[start_idx:end_idx]
        total_actions.append(ep_df[action_cols].sum(axis=1).sum())

    plt.figure(figsize=(10,5))
    plt.plot(range(1,len(total_actions)+1), total_actions, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Total Previous Action")
    plt.title("Total Previous Action per Episode N/m")
    plt.grid(True)

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, "total_prev_action.png"))
        print(f"Saved total_prev_action.png in {plots_dir}")


def main(exp_name = None):
    DIR_EXPERIMENT_INFO = "experiment_info"           

    root = get_git_root()
    weights_dir = os.path.join(root, "TITA_MJ", "log", "sac_logs", "weights")

    if exp_name is None:

        print("Usage: python3 plot.py <experiment_name>")
        folders = [f for f in os.listdir(weights_dir) if os.path.isdir(os.path.join(weights_dir, f))]
        if not folders:
            raise RuntimeError(f"Nessuna cartella trovata in {weights_dir}")
        folders.sort()  # ordina alfabeticamente
        exp_name = folders[-1]
        print(f"Using the latest experiment: {exp_name}")

    dir_experiment = os.path.join(weights_dir, exp_name)
    csv_path = os.path.join(dir_experiment, DIR_EXPERIMENT_INFO, "observations.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV non found: {csv_path}")

    plots_dir = os.path.join(os.path.dirname(csv_path), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_total_reward(csv_path, plots_dir)
    plot_robot_height(csv_path, plots_dir)
    plot_orientation(csv_path, plots_dir)
    plot_total_torque(csv_path, plots_dir)
    plot_total_prev_action(csv_path, plots_dir)
    plot_joint_torque_last_episode(csv_path, plots_dir)


if __name__ == "__main__":
    exp_name = sys.argv[1] if len(sys.argv) >= 2 else None
    main(exp_name)
