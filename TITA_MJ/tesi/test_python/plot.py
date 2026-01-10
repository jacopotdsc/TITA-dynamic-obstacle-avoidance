import os
import sys
import pandas as pd
import git
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import numpy as np
import gymnasium as gym

gym.register(
    id="Tita-v0",
    entry_point="gymnasium.envs.mujoco.tita_env:TitaEnv",
    max_episode_steps=1000,
)

env = gym.make("Tita-v0", render_mode=None)
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
        name = "total_and_last_reward.png"
        saved = os.path.join(plots_dir, name)
        plt.savefig(saved)
        print(f"Saved {saved}")

def plot_average_height(csv_path, plots_dir=None):
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
    plt.plot(range(1,len(heights)+1), heights, marker='o', label="average_height")

    plt.axhline(y=env.unwrapped.get_config().reward_config.base_height_target, color='r', linestyle='--', label='height_desired')

    plt.xlabel("Episode")
    plt.ylabel("Average Robot Height")
    plt.title("Average Robot Height per Episode")
    plt.grid(True)
    plt.legend()

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        name = "average_height_per_episode.png"
        saved = os.path.join(plots_dir, name)
        plt.savefig(saved)
        print(f"Saved {saved}")

def plot_average_orientation(csv_path, plots_dir=None):
    df = pd.read_csv(csv_path)
    ori_cols = ['ori_x','ori_y','ori_z','ori_w']
    episode_starts = df.index[df['frame'] == 0].tolist()
    episode_starts.append(len(df))
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
        val_array = np.array(values)
        
        val_array = np.where(val_array >= np.pi/2, val_array - np.pi, val_array)
        val_array = np.where(val_array <= -np.pi/2, val_array + np.pi, val_array)
        plt.plot(range(1,len(euler_means['roll'])+1), val_array, marker='o', label=label)
    
    plt.ylim(-np.pi/2 + 0.1, np.pi/2 - 0.1)
    plt.xlabel("Episode")
    plt.ylabel("Radians")
    plt.title("Average Orientation per Episode")
    plt.legend()
    plt.grid(True)

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        name = "average_orientation_per_episode.png"
        saved = os.path.join(plots_dir, name)
        plt.savefig(saved)
        print(f"Saved {saved}")

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
        name = "total_torque.png"
        saved = os.path.join(plots_dir, name)
        plt.savefig(saved)
        print(f"Saved {saved}")

def plot_last_episode_joint_torque(csv_path, plots_dir=None):
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
        name = "last_episode_joint_torque.png"
        saved = os.path.join(plots_dir, name)
        plt.savefig(saved)
        print(f"Saved {saved}")

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
    plt.ylabel("Total Previous Action N/m")
    plt.title("Total Previous Action per Episode")
    plt.grid(True)

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        name = "total_prev_action.png"
        saved = os.path.join(plots_dir, name)
        plt.savefig(saved)
        print(f"Saved {saved}")

def plot_last_episode_orientation(csv_path, plots_dir=None):
    df = pd.read_csv(csv_path)
    episode_starts = df.index[df['frame'] == 0].tolist()
    start_idx = episode_starts[-1]
    ep_df = df.iloc[start_idx:].copy()

    quat_data = ep_df[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    rot = Rotation.from_quat(quat_data)
    euler_angles = rot.as_euler('xyz', degrees=False)
    
    ep_df['roll'] = euler_angles[:, 0]
    ep_df['pitch'] = euler_angles[:, 1]
    ep_df['yaw'] = euler_angles[:, 2]

    scaled_ori = ep_df[['roll', 'pitch', 'yaw']].copy()
    for col in ['roll', 'pitch', 'yaw']:
        val_array = scaled_ori[col].values
        val_array = np.where(val_array >= np.pi/2, val_array - np.pi, val_array)
        val_array = np.where(val_array <= -np.pi/2, val_array + np.pi, val_array)
        scaled_ori[col] = val_array
    
    plt.figure(figsize=(10, 6))
    plt.plot(ep_df['frame'], scaled_ori['roll'], label='Roll')
    plt.plot(ep_df['frame'], scaled_ori['pitch'], label='Pitch')
    plt.plot(ep_df['frame'], scaled_ori['yaw'], label='Yaw')
    
    plt.title(f"Orientation (Euler Angles) - Last Episode ({len(episode_starts)})")
    plt.xlabel("Frame")
    plt.ylabel("Radians")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        name = "last_episode_orientation.png"
        saved = os.path.join(plots_dir, name)
        plt.savefig(saved)
        print(f"Saved {saved}")

def plot_last_episode_height(csv_path, plots_dir=None):
    df = pd.read_csv(csv_path)
    episode_starts = df.index[df['frame'] == 0].tolist()
    start_idx = episode_starts[-1]
    ep_df = df.iloc[start_idx:]

    plt.figure(figsize=(10, 5))
    plt.plot(ep_df['frame'], ep_df['robot_height'], color='tab:green', label='Robot Height')
    
    plt.title(f"Robot Height - Last Episode ({len(episode_starts)})")
    plt.xlabel("Frame")
    plt.ylabel("Height (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        name = "last_episode_height.png"
        saved = os.path.join(plots_dir, name)
        plt.savefig(saved)
        print(f"Saved {saved}")

def main(exp_name = None):
    DIR_EXPERIMENT_INFO = "experiment_info"           

    root = get_git_root()
    weights_dir = os.path.join(root, "TITA_MJ", "log", "sac_logs", "weights")

    if exp_name is None:
        print("Usage: python3 plot.py <experiment_name>")
        folders = [f for f in os.listdir(weights_dir) if os.path.isdir(os.path.join(weights_dir, f))]
        if not folders:
            raise RuntimeError(f"No folder find in {weights_dir}")
        folders.sort()  
        exp_name = folders[-1]
        print(f"Using the latest experiment: {exp_name}")

    dir_experiment = os.path.join(weights_dir, exp_name)
    csv_path = os.path.join(dir_experiment, DIR_EXPERIMENT_INFO, "observations.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV non found: {csv_path}")

    plots_dir = os.path.join(os.path.dirname(csv_path), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_total_reward(csv_path, plots_dir)
    plot_average_height(csv_path, plots_dir)
    plot_average_orientation(csv_path, plots_dir)
    plot_total_torque(csv_path, plots_dir)
    plot_total_prev_action(csv_path, plots_dir)

    plot_last_episode_joint_torque(csv_path, plots_dir)
    plot_last_episode_orientation(csv_path, plots_dir)
    plot_last_episode_height(csv_path, plots_dir)

if __name__ == "__main__":
    exp_name = sys.argv[1] if len(sys.argv) >= 2 else None
    main(exp_name)
