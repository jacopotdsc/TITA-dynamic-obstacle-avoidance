#!/usr/bin/env python3
"""
Simple PPO training script with vectorized environments.
Demonstrates procedural API usage for training on Pendulum-v1.
"""

import os
import sys
import subprocess
import datetime
import argparse
import re
import gymnasium as gym
import numpy as np
import torch
from torch import nn
import cv2
import git
import time
import copy
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from gymnasium.wrappers import RecordVideo
from torch.distributions import Distribution, Independent, Normal
from scipy.spatial.transform import Rotation
from copy import deepcopy

from tianshou.algorithm import TD3
from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy
from tianshou.algorithm import SAC
from tianshou.exploration import GaussianNoise

from tianshou.algorithm import SAC

from tianshou.algorithm import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.modelfree.sac import SACPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.utils.statistics import RunningMeanStd
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.net.common import ActionReprNetWithVectorOutput
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo
import torch
import torch.nn as nn
import torch.nn.functional as F
from tianshou.utils.net.common import ActionReprNetWithVectorOutput
from tianshou.utils.net.continuous import ContinuousActorProbabilistic
from tianshou.data import Batch

_STR_TRAIN = "train"
_STR_TEST = "test"
_STR_SAC = "sac"
_STR_PPO = "ppo"
LOG_ARRAY = []
BEST_LAST_EPOCH = -1
N_FRAME_STACK = -1
DIR_EXPERIMENT_INFO = "experiment_info"
MAIN_DIR = ""
EPISODE_LENGTH = 1_000

global dir_experiment

global initial_time, ep_prev_time, current_time
initial_time = datetime.datetime.now()
ep_prev_time = initial_time

'''
# Reference: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml
# Each configuration, refer to https://stable-baselines3.readthedocs.io/en/master/modules/sac.html#stable_baselines3.sac.SAC
seed: 42
n_timesteps: !!float 5e8
policy: 'MlpPolicy'
learning_rate: !!float 1e-4
buffer_size: 1000000
batch_size: 256
ent_coef: 'auto_0.001'
gamma: 0.98
tau: !!float 5e-3
train_freq: 10
gradient_steps: -1
learning_starts: 0
n_steps: 3
use_sde: True
normalize_input: True
normalize_value: True
policy_kwargs: "dict(
                  log_std_init=-1,
                  activation_fn=nn.ELU,
                  net_arch=[512,256,128], 
                  clip_mean=1.0, 
                )"
'''
class LoggerTee(object):
    def __init__(self, filename):
        self.line_buffer = ""
        self.last_written_line = ""
        self.terminal = sys.stdout
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        
        if not hasattr(self, 'line_buffer'):
            return

        for char in message:
            if char == '\r':
                self._process_line(self.line_buffer)
                self.line_buffer = ""
            elif char == '\n':
                self._process_line(self.line_buffer)
                self.line_buffer = ""
            else:
                self.line_buffer += char

    def _process_line(self, line):
        clean_line = self.ansi_escape.sub('', line).strip()
        
        if not clean_line:
            return

        is_reward_line = 'test_reward' in clean_line or 'best_reward' in clean_line
        
        is_finished_epoch = '100%' in clean_line and '8000/8000' in clean_line # Adatta il numero totale se varia

        if is_reward_line or is_finished_epoch:
            self.log.write(clean_line + '\n')
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

def format_td(td):
    total_seconds = int(td.total_seconds())
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def setup_auto_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout = LoggerTee(log_path)
    sys.stderr = LoggerTee(log_path)

def log_and_print(*args):
        message = " ".join(map(str, args))
        print(message)
        LOG_ARRAY.append(message)

def parser_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", 
                        nargs='*', 
                        help="Start training. Options: 'ppo' or 'sac' (default: sac)")
    
    group.add_argument("--test", 
                        nargs='*',   
                        help="Start testing. Usage: --test [alg] [exp_name] [render_mode]")

    group.add_argument("--log",
                        nargs='?', 
                        const='yes',      
                        choices=["yes", "no"],
                        help="Start logging. Options: 'ppo' or 'sac' (default: sac)"
                       )
    parser.add_argument("--warmup", 
                        type=int, 
                        default=0,
                        help="Number of warmup steps to run before training (default 0)")
    
    parser.add_argument("--exec",
                        type=int,
                        required=False,   
                        help="Start execution. Usage: --exec [script_name] [args...]")

    parser.add_argument("--len",
                        type=int,
                        help="Length of each episode (default 1000)")
    
    args = parser.parse_args()

    script_task = _STR_TRAIN
    alg_type = _STR_SAC
    render_mode = "human"
    name_weight_name = None
    make_log = False
    warmup_steps = int(args.warmup) if args.warmup > 0 else None
    task_to_display = 0

    if args.train:
        script_task = _STR_TRAIN
        train_args = [item.lower() for item in args.train]

        for item in args.train:
            if item in [_STR_SAC, _STR_PPO]:
                alg_type = item
            else:
                name_weight_name = item
    
    if args.test is not None:
        script_task = _STR_TEST
        test_args = [item.lower() for item in args.test]
        
        for item in test_args:
            if item in [_STR_SAC, _STR_PPO]:
                alg_type = item
            elif item in ['human', 'rgb', 'rgb_array']:
                render_mode = "rgb_array" if item in ['rgb', 'rgb_array'] else "human"
            else:
                name_weight_name = item
    if args.exec is not None:
        task_to_display = args.exec
        print(f"Executing task, taken: {task_to_display}")

    if args.log:
        make_log = True if args.log.lower() == "yes" else False

    global EPISODE_LENGTH
    if args.len is not None and args.len > 0:
        EPISODE_LENGTH = args.len

    print(f"Script started with\n\ttask: {script_task},\n\talgorithm: {alg_type},\n\trender mode: {render_mode}\n\tpid: {os.getpid()}\n")
    return script_task, alg_type, render_mode, make_log, name_weight_name, warmup_steps, task_to_display

def get_git_root():
    try:
        repo = git.Repo(".", search_parent_directories=True)
        return repo.working_tree_dir
    except git.InvalidGitRepositoryError:
        return None
    
def test_fn(num_epoch, step_idx, policy, task, num_test_envs, num_view_test_env, save_plot_dir):
    global BEST_LAST_EPOCH
    BEST_LAST_EPOCH = num_epoch

    global N_FRAME_STACK

    global initial_time, ep_prev_time, current_time
    current_time = datetime.datetime.now()
    epoch_elapsed = current_time - ep_prev_time
    total_elapsed = current_time - initial_time

    log_and_print(
        f"Epoch: {num_epoch}, "
        f"epoch elapsed time: {format_td(epoch_elapsed)}, "
        f"time since start: {format_td(total_elapsed)}\n"
    )

    ep_prev_time = current_time
    

    policy.eval()
    test_collector.reset()
    res = test_collector.collect(n_episode=num_test_envs, render=0)
    mean_len = np.mean(res.lens)
    policy.train()
    buf = test_collector.buffer  
    start = 0
    csv_data = []

    for ep_len in res.lens:
        ep_obs = buf.obs[start:start+ep_len]
        ep_act = buf.act[start:start+ep_len]
        ep_rews = buf.rew[start:start+ep_len]
        for i in range(ep_len):
            o = ep_obs[i]
            a = ep_act[i]

            if N_FRAME_STACK > 1:
                index_right_observation = int(o.shape[0]/N_FRAME_STACK)*(N_FRAME_STACK-1)
                o_last = o[index_right_observation:] 
            else:
                o_last = o
            row = [num_epoch, i, ep_len, mean_len, ep_rews[i]] + list(o_last) + list(a)
            csv_data.append(row)
        start += ep_len
    
    global env_single

    obs_dict_keys = env_single.unwrapped.get_obs_info()[0].keys()
    obs_headers = []

    for key in obs_dict_keys:
        sample_val = np.asarray(env_single.unwrapped.obs_dict[key])
        shape = sample_val.shape

        # scalare -> una entry
        if sample_val.ndim == 0 or (sample_val.ndim == 1 and shape[0] == 1):
            obs_headers.append(key)

        # vettore 3D -> _x _y _z
        elif sample_val.ndim == 1 and shape[0] == 3:
            obs_headers.extend([f"{key}_x", f"{key}_y", f"{key}_z"])

        # vettore 1D generico -> _1 _2 ...
        elif sample_val.ndim == 1:
            obs_headers.extend([f"{key}_{i+1}" for i in range(shape[0])])

        # matrice 2D -> flatten row-major: _1 _2 ...
        elif sample_val.ndim == 2:
            obs_headers.extend([f"{key}_{i+1}" for i in range(sample_val.size)])

        else:
            raise ValueError(f"Unsupported obs shape for key '{key}': {shape}")


    act_headers = [f'action_{i}' for i in range(1, 9)]
    headers = ['epoch','frame', 'episode_length', 'mean_length', 'reward'] + list(obs_headers) + act_headers

    global dir_experiment
    csv_path = os.path.join(dir_experiment, DIR_EXPERIMENT_INFO, "observations.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, pd.DataFrame(csv_data, columns=headers)], ignore_index=True)
    else:
        df = pd.DataFrame(csv_data, columns=headers)

    df.to_csv(csv_path, index=False)

    if False and num_epoch >= 0 and num_epoch % 1 == 0:
        policy.eval()
        test_enviroment(
            task_name=task,
            task_to_display=0,
            policy=deepcopy(policy),
            render_mode=None,
            num_test_envs=num_test_envs,
            save_plot_dir=save_plot_dir,
            training_in_test=True,
            num_epoch=num_epoch
        )
        policy.train()

def save_best(algorithm, alg_type, actor_policy, actor_path, critic_policy, critic_path):
    global BEST_LAST_EPOCH
    if BEST_LAST_EPOCH <= 0:
        return
    
    dir_save_best = os.path.join(os.path.dirname(actor_path), "best_epoch_" + str(BEST_LAST_EPOCH))
    os.makedirs(dir_save_best, exist_ok=True)

    actor_best_path = os.path.join(dir_save_best, os.path.basename(actor_path))
    critic_best_path = os.path.join(dir_save_best, os.path.basename(critic_path))
    torch.save(actor_policy.state_dict(), actor_best_path)
    #print("\nSaved best actor policy")

    if alg_type == _STR_PPO:
        torch.save(critic_policy.state_dict(), critic_best_path)
        #print("Saved best critic policy")
    elif alg_type == _STR_SAC:
        torch.save(critic_policy[0].state_dict(), critic_best_path.replace(".pt", "_1.pt"))
        #print("Saved best critic1 policy")

        torch.save(critic_policy[1].state_dict(), critic_best_path.replace(".pt", "_2.pt"))
        #print("Saved best critic2 policy")
    else:
        raise ValueError("Unsupported algorithm. Choose either 'ppo' or 'sac'.")

def test_enviroment(
        task_name: str,
        task_to_display: int,
        policy: nn.Module,
        render_mode: str = "human",
        num_test_envs: int = 16,
        save_plot_dir: str = None,
        training_in_test: bool = False,
        num_epoch: int = 0
    ):
    
    if training_in_test == True:
        policy.eval()

    if training_in_test == False:
        test_envs = SubprocVectorEnv([lambda i=i: create_wrapped_env(task_name, task_to_execute=i) for i in range(num_test_envs)], )
        collector = Collector(policy, test_envs, exploration_noise=False)
        collector.reset()
        result = collector.collect(n_episode=num_test_envs, render=0)

        print(f"\nResults test environment {task_name} (x{num_test_envs}):")
        print(f"Mean reward:     {result.returns_stat.mean:3f}")
        print(f"Std deviation:   ±{result.returns_stat.std:3f}")
        print(f"Min / Max:        {result.returns_stat.min:.3f} / {result.returns_stat.max:.3f}")
        print(f"Mean length:        {np.mean(result.lens):.1f}")
        print(f"Num episodes:       {len(result.returns)}\n")
        print("Finished testing in vectorized envs. Showing in viewer\n")

    # --- Manual rendering ---
    env = create_wrapped_env(task_name, task_to_execute=task_to_display, render_mode=render_mode)

    # ----- Video recording setup -----
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    trial = 0
    while True:
        exp_dir_name = f"task_{task_to_display}_{trial}"
        save_dir = os.path.join(save_plot_dir, exp_dir_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            break
        trial += 1
    

    #video_folder = os.path.join("videos", f"{task_name}_{timestamp}")
    
    if render_mode == "rgb_array" :
        env = RecordVideo(
            env, 
            video_folder=save_dir,
            name_prefix=f"eval_task_{task_to_display}",
            episode_trigger=lambda episode_id: True 
        )
        print(f"Video recording enabled. File will be saved in: {save_dir}")
    try:
        policy.eval()
        obs, info = env.reset()
        total_reward = 0
        n_frame = 0
        terminated = False
        truncated = False

        logging = {key: [] for key in env.unwrapped.get_obs_info()[0]}
        logging.update({
            'action': [], 
            'reward_info': [], 
            'reward_per_frame': [],
            'n_frame': [],
            'perturb': [],
            'tita_controller_output': [],

            "mpc_sol_com_pos" : [],
            "mpc_sol_com_vel" : [],
            "mpc_sol_com_acc" : [],
            "mpc_sol_pl_pos" : [],
            "mpc_sol_pl_vel" : [],
            "mpc_sol_pl_acc" : [],
            "mpc_sol_pr_pos" : [],
            "mpc_sol_pr_vel" : [],
            "mpc_sol_pr_acc" : [],

            'gt_com_pos': [],
            'gt_com_lin_vel': [],
            'gt_com_lin_acc': [],

            'gt_com_orientation': [],
            'gt_com_ang_vel': [],
            'gt_com_ang_acc': [],

            'left_feet_pos': [],
            'left_feet_vel': [],
            'left_feet_acc': [],

            'right_feet_pos': [],
            'right_feet_vel': [],
            'right_feet_acc': [],
        })


        start_time_inference = []
        end_time_inference = []
        
        start_test_time = time.time()
        while True and (not terminated) and (not truncated):
            batch = Batch(obs=np.array([obs]), info={})
            with torch.no_grad():
                    
                start_time_inference.append(time.time())
                result = policy(batch)
                end_time_inference.append(time.time())
                if n_frame == 0:  
                    inference_time = end_time_inference[0] - start_time_inference[0]
            
            action = result.act[0]            
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            scaled_action = action * env.unwrapped.get_config().action_scale
            obs, reward, terminated, truncated, reward_info = env.step(action)
            env.unwrapped._update_perturbation_visual()

            if render_mode is not None:
                frame = env.render()
                
            total_reward += reward

            # Logging
            reward_full = copy.deepcopy(reward_info)
               
            obs_dict, slices = env.unwrapped.get_obs_info()
            for key, value in obs_dict.items():
                logging[key].append(value.copy() if isinstance(value, np.ndarray) else value)
            
            env_info_dict = reward_info['info'] 
            reward_info.pop('info', None)
            reward_info.pop('n_frame', None)

            logging['action'].append(scaled_action.copy())
            logging['reward_per_frame'].append(reward)

            logging["mpc_sol_com_pos"].append(env_info_dict["mpc_sol_com_pos"])
            logging["mpc_sol_com_vel"].append(env_info_dict["mpc_sol_com_vel"])
            logging["mpc_sol_com_acc"].append(env_info_dict["mpc_sol_com_acc"])
            logging["mpc_sol_pl_pos"].append(env_info_dict["mpc_sol_pl_pos"])
            logging["mpc_sol_pl_vel"].append(env_info_dict["mpc_sol_pl_vel"])
            logging["mpc_sol_pl_acc"].append(env_info_dict["mpc_sol_pl_acc"])
            logging["mpc_sol_pr_pos"].append(env_info_dict["mpc_sol_pr_pos"])
            logging["mpc_sol_pr_vel"].append(env_info_dict["mpc_sol_pr_vel"])
            logging["mpc_sol_pr_acc"].append(env_info_dict["mpc_sol_pr_acc"])
            
            gt_com_pos = env.unwrapped.data.subtree_com[0, :].copy()
            gt_com_lin_vel = env.unwrapped.data.qvel[0:3].copy()
            gt_com_lin_acc = env.unwrapped.data.qacc[0:3].copy()

            q_mj = env.unwrapped.data.qpos[3:7].copy()  
            q_scipy = [q_mj[1], q_mj[2], q_mj[3], q_mj[0]]
            rot = Rotation.from_quat(q_scipy)
            gt_com_orientation = rot.as_euler('xyz', degrees=False)
            gt_com_ang_vel = env.unwrapped.data.qvel[3:6].copy()
            gt_com_ang_acc = env.unwrapped.data.qacc[3:6].copy()

            logging['gt_com_pos'].append(gt_com_pos)
            logging['gt_com_lin_vel'].append(gt_com_lin_vel)
            logging['gt_com_lin_acc'].append(gt_com_lin_acc)
            logging['gt_com_orientation'].append(gt_com_orientation)
            logging['gt_com_ang_vel'].append(gt_com_ang_vel)
            logging['gt_com_ang_acc'].append(gt_com_ang_acc)

            left_feet_pos, left_feet_vel, left_feet_acc = env.unwrapped.get_feet_site_state(env.unwrapped._left_feet_site_id)
            right_feet_pos, right_feet_vel, right_feet_acc = env.unwrapped.get_feet_site_state(env.unwrapped._right_feet_site_id)

            logging['left_feet_pos'].append(left_feet_pos)
            logging['left_feet_vel'].append(left_feet_vel)
            logging['left_feet_acc'].append(left_feet_acc)
            logging['right_feet_pos'].append(right_feet_pos)
            logging['right_feet_vel'].append(right_feet_vel)
            logging['right_feet_acc'].append(right_feet_acc)

            logging['n_frame'] = n_frame
            logging['tita_controller_output'].append(env_info_dict['tita_controller_output'])
            logging['perturb'].append(env_info_dict.get('perturb'))
            reward_info.pop('info', None)
            reward_info.pop('n_frame', None)
            logging['reward_info'].append(reward_info.copy())

            # Rendering and small loggin
            if n_frame == 0 or (n_frame+1) % 100 == 0 or terminated or truncated:
                print("Frame:", n_frame, "Action:", scaled_action, ", reward: ", reward, "Total Reward:", total_reward)
            
            if render_mode is not None and frame is not None and render_mode == "rgb_array":
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                cv2.imshow("Agent Preview (Press 'q' to quit)", frame_bgr)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            n_frame += 1

            # Plotting after episode ends
            if terminated or truncated:
                print(f"Episode terminated. Reward: {total_reward:.2f}")
                print(f"Inference time: {1000*inference_time:.6f} ms first step")

                mean_inference_time = 1000 * np.mean([end - start for start, end in zip(start_time_inference, end_time_inference)])
                std_inference_time = 1000 * np.std([end - start for start, end in zip(start_time_inference, end_time_inference)])
                print(f"inference time: {mean_inference_time:.6f} ± {std_inference_time:.6f} ms")
        
                frames = np.arange(len(np.array(logging['action'])))
                
                joint_names = ["ankle_pitch", "ankle_roll", "knee", "wheel"]
                legend_left = ["left_" + name for name in joint_names]
                legend_right = ["right_" + name for name in joint_names]

                def plot_torques():
                    tita_controller_output = np.array(logging['tita_controller_output'])
                    history_action = np.array(logging['action'])

                    frames = range(len(history_action))
                    
                    joint_names = ["ankle_pitch", "ankle_roll", "knee", "wheel"]
                    legend_left = ["left_" + name for name in joint_names]
                    legend_right = ["right_" + name for name in joint_names]

                    # Griglia 2x2: Righe (Left/Right leg), Colonne (Controller/NN Action)
                    fig, axes = plt.subplots(3, 2, figsize=(18, 10), sharex=True)

                    # --- COLONNA 0: TITA CONTROLLER OUTPUT (Sinistra) ---
                    # Gamba Sinistra
                    for i in range(4):
                        axes[0, 0].plot(frames, tita_controller_output[:, i], label=legend_left[i])
                    axes[0, 0].set_ylabel("Torque [N/m]")
                    axes[0, 0].set_title("Controller Output (Left Leg)")
                    axes[0, 0].legend(loc='upper right', fontsize='small')
                    axes[0, 0].grid(True, alpha=0.3)

                    # Gamba Destra
                    for i in range(4):
                        axes[1, 0].plot(frames, tita_controller_output[:, i+4], label=legend_right[i])
                    axes[1, 0].set_xlabel("Frame")
                    axes[1, 0].set_ylabel("Torque [N/m]")
                    axes[1, 0].set_title("Controller Output (Right Leg)")
                    axes[1, 0].legend(loc='upper right', fontsize='small')
                    axes[1, 0].grid(True, alpha=0.3)

                    total_ctrl_sum = np.sum(np.abs(tita_controller_output), axis=1)
                    axes[2, 0].plot(frames, total_ctrl_sum, color='blue', linewidth=2, label='Total controller torque')
                    axes[2, 0].set_title("Total Controller Effort")
                    axes[2, 0].set_xlabel("Frame")
                    axes[2, 0].set_ylabel("Total Torque")
                    axes[2, 0].legend()
                    axes[2, 0].grid(True, alpha=0.3)


                    # --- COLONNA 1: NN HISTORY ACTION (Destra) ---
                    # Gamba Sinistra
                    for i in range(4):
                        axes[0, 1].plot(frames, history_action[:, i], label=legend_left[i])
                    axes[0, 1].set_ylabel("Action (scaled)")
                    axes[0, 1].set_title("NN Action Output (Left Leg)")
                    axes[0, 1].legend(loc='upper right', fontsize='small')
                    axes[0, 1].grid(True, alpha=0.3)

                    # Gamba Destra
                    for i in range(4):
                        axes[1, 1].plot(frames, history_action[:, i+4], label=legend_right[i])
                    axes[1, 1].set_xlabel("Frame")
                    axes[1, 1].set_ylabel("Action (scaled)")
                    axes[1, 1].set_title("NN Action Output (Right Leg)")
                    axes[1, 1].legend(loc='upper right', fontsize='small')
                    axes[1, 1].grid(True, alpha=0.3)

                    # RIGA 2: Somma Totale Action
                    total_action_sum = np.sum(np.abs(history_action), axis=1)
                    axes[2, 1].plot(frames, total_action_sum, color='blue', linewidth=2, label='Total NN action')
                    axes[2, 1].set_title("Total NN Action Effort")
                    axes[2, 1].set_xlabel("Frame")
                    axes[2, 1].set_ylabel("Total Action")
                    axes[2, 1].legend()
                    axes[2, 1].grid(True, alpha=0.3)

                    plt.tight_layout()
                    
                    #os.makedirs(save_dir, exist_ok=True)
                    name = "render_test_actions_comparison.png"
                    saved = os.path.join(save_dir, name)
                    plt.savefig(saved)
                    print(f"Saved {saved}")

                def plot_reward_info():
                    history_reward_info = np.array(logging['reward_info'])

                    reward_info_keys = list(history_reward_info[0].keys())
                    reward_info_keys = [key for key in reward_info_keys if env.unwrapped.get_config().reward_config.scales.get(key) != 0] 
                    n_keys = len(reward_info_keys)
                    
                    mid = (n_keys + 1) // 2
                    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

                    for idx, key in enumerate(reward_info_keys):
                        if key != "cost_early_termination":
                            ax = axes[0] if idx < mid else axes[1]
                            
                            reward_values = [info[key] for info in history_reward_info]
                            ax.plot(frames, reward_values, label=key)
                    axes[0].set_title("Reward Info")
                    axes[0].set_ylabel("Value")
                    axes[0].legend(loc='upper right', ncol=2) #
                    axes[0].grid(True)

                    # Configurazioni per il grafico inferiore
                    axes[1].set_title("Reward Info")
                    axes[1].set_ylabel("Value")
                    axes[1].set_xlabel("Frame")
                    axes[1].legend(loc='upper right', ncol=2)
                    axes[1].grid(True)

                    plt.tight_layout()
                    #os.makedirs(save_dir, exist_ok=True)
                    name = "render_test_reward_info.png"
                    saved = os.path.join(save_dir, name)
                    plt.savefig(saved)
                    print(f"Saved {saved}")
                    plt.savefig(saved)

                def plot_total_reward_info():
                    reward_per_frame = np.array(logging["reward_per_frame"])
                    total_rewards_arr = []
                    tot_reward = 0

                    for r in reward_per_frame:
                        tot_reward += r
                        total_rewards_arr.append(tot_reward)

                    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

                    # ---- sopra: cumulative reward per episodio ----
                    episodes = np.arange(0, len(total_rewards_arr))
                    axes[0].plot(episodes, total_rewards_arr, label="Total reward")
                    axes[0].set_title("Cumulative Reward per Episode")
                    axes[0].set_xlabel("Episode")
                    axes[0].set_ylabel("Total reward")
                    axes[0].grid(True)
                    axes[0].legend()

                    # ---- sotto: reward per frame ultimo episodio ----
                    axes[1].plot(episodes, reward_per_frame, label="Reward per frame")
                    axes[1].set_title(f"Reward per frame (last episode: {len(total_rewards_arr)})")
                    axes[1].set_xlabel("Frame")
                    axes[1].set_ylabel("Reward")
                    axes[1].grid(True)
                    axes[1].legend()

                    plt.tight_layout()
                    #os.makedirs(save_dir, exist_ok=True)
                    saved = os.path.join(save_dir, "render_test_total_reward.png")
                    plt.savefig(saved)
                    print(f"Saved {saved}")
                
                def plot_com():
                    
                    history_gt_com_pos = np.array(logging['gt_com_pos'])
                    history_gt_com_vel = np.array(logging['gt_com_lin_vel'])
                    history_gt_com_acc = np.array(logging['gt_com_lin_acc'])

                    history_gt_com_orientation = np.array(logging['gt_com_orientation']) 
                    history_gt_com_ang_vel = np.array(logging['gt_com_ang_vel']) 
                    history_gt_com_ang_acc = np.array(logging['gt_com_ang_acc']) 

                    history_mpc_com_pos = np.array(logging['mpc_sol_com_pos'])
                    history_mpc_com_vel = np.array(logging['mpc_sol_com_vel'])
                    history_mpc_com_acc = np.array(logging['mpc_sol_com_acc'])

                    frames = range(len(history_mpc_com_pos))
                    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
                    
                    # Colori e stili
                    colors = ['tab:red', 'tab:green', 'tab:blue']
                    labels_lin = ['X', 'Y', 'Z']
                    labels_ang = ['Roll', 'Pitch', 'Yaw']

                    # --- COLONNA 0: LINEARE (GT vs MPC) ---
                    for i in range(3):
                        # Position (Row 0), Velocity (Row 1), Acceleration (Row 2)
                        data_gt = [history_gt_com_pos, history_gt_com_vel, history_gt_com_acc][i]
                        data_mpc = [history_mpc_com_pos, history_mpc_com_vel, history_mpc_com_acc][i]
                        titles = ["CoM Position [m]", "CoM Velocity [m/s]", "CoM Acceleration [m/s²]"]
                        

                        for j in range(3):
                            data_gt[:, j] = np.where(data_gt[:, j] > np.pi, data_gt[:, j] - 2*np.pi, data_gt[:, j])
                            data_gt[:, j] = np.where(data_gt[:, j] < -np.pi, data_gt[:, j] + 2*np.pi, data_gt[:, j])

                            axes[i, 0].plot(frames, data_gt[:, j], color=colors[j], label=f'GT {labels_lin[j]}')
                            axes[i, 0].plot(frames, data_mpc[:, j], color=colors[j], linestyle='--', alpha=0.6, label=f'MPC {labels_lin[j]}')
                        
                        axes[i, 0].set_title(titles[i])
                        axes[i, 0].legend(loc='upper right', ncol=2, fontsize='small')
                        axes[i, 0].grid(True, alpha=0.3)

                    # --- COLONNA 1: ANGOLARE (Solo GT) ---
                    for i in range(3):
                        data_ang = [history_gt_com_orientation, history_gt_com_ang_vel, history_gt_com_ang_acc][i]
                        titles_ang = ["Orientation [rad]", "Angular Velocity [rad/s]", "Angular Acceleration [rad/s²]"]

                        for j in range(3): # Plot Roll, Pitch, Yaw
                            axes[i, 1].plot(frames, data_ang[:, j], color=colors[j], label=labels_ang[j])
                        
                        axes[i, 1].set_title(titles_ang[i])
                        axes[i, 1].legend(loc='upper right', fontsize='small')
                        axes[i, 1].grid(True, alpha=0.3)

                    # Impostazioni finali
                    axes[2, 0].set_xlabel("Frame")
                    axes[2, 1].set_xlabel("Frame")
                    
                    plt.tight_layout()
                    #os.makedirs(save_dir, exist_ok=True)
                    name = "render_test_com_dynamics.png"
                    saved = os.path.join(save_dir, name)
                    plt.savefig(saved)
                    print(f"Saved {saved}")

                def plot_feet():
                    
                    dead_value = -10
                    history_mpc_sol_pl_pos = np.array(logging['mpc_sol_pl_pos'] if 'mpc_sol_pl_pos' in logging else np.full((len(logging.get('mpc_sol_com_pos', [1])), 3), dead_value))
                    history_mpc_sol_pl_vel = np.array(logging['mpc_sol_pl_vel'] if 'mpc_sol_pl_vel' in logging else np.full((len(logging.get('mpc_sol_com_pos', [1])), 3), dead_value))
                    history_mpc_sol_pl_acc = np.array(logging['mpc_sol_pl_acc'] if 'mpc_sol_pl_acc' in logging else np.full((len(logging.get('mpc_sol_com_pos', [1])), 3), dead_value))
                    history_mpc_sol_pr_pos = np.array(logging['mpc_sol_pr_pos'] if 'mpc_sol_pr_pos' in logging else np.full((len(logging.get('mpc_sol_com_pos', [1])), 3), dead_value))
                    history_mpc_sol_pr_vel = np.array(logging['mpc_sol_pr_vel'] if 'mpc_sol_pr_vel' in logging else np.full((len(logging.get('mpc_sol_com_pos', [1])), 3), dead_value))
                    history_mpc_sol_pr_acc = np.array(logging['mpc_sol_pr_acc'] if 'mpc_sol_pr_acc' in logging else np.full((len(logging.get('mpc_sol_com_pos', [1])), 3), dead_value))
                    history_left_feet_pos = np.array(logging['left_feet_pos'] if 'left_feet_pos' in logging else np.full((len(logging.get('mpc_sol_com_pos', [1])), 3), dead_value))
                    history_left_feet_vel = np.array(logging['left_feet_vel'] if 'left_feet_vel' in logging else np.full((len(logging.get('mpc_sol_com_pos', [1])), 3), dead_value))
                    history_left_feet_acc = np.array(logging['left_feet_acc'] if 'left_feet_acc' in logging else np.full((len(logging.get('mpc_sol_com_pos', [1])), 3), dead_value))
                    history_right_feet_pos = np.array(logging['right_feet_pos'] if 'right_feet_pos' in logging else np.full((len(logging.get('mpc_sol_com_pos', [1])), 3), dead_value))
                    history_right_feet_vel = np.array(logging['right_feet_vel'] if 'right_feet_vel' in logging else np.full((len(logging.get('mpc_sol_com_pos', [1])), 3), dead_value))
                    history_right_feet_acc = np.array(logging['right_feet_acc'] if 'right_feet_acc' in logging else np.full((len(logging.get('mpc_sol_com_pos', [1])), 3), dead_value))

                    frames = range(len(history_mpc_sol_pl_pos))
                    # Creazione griglia: 3 righe (Pos, Vel, Acc) e 2 colonne (Left, Right)
                    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)

                    # --- COLONNA 0: LEFT FOOT (PL) ---
                    # 1. Posizione PL_
                    axes[0, 0].plot(frames, history_mpc_sol_pl_pos[:, 0], color='red', label='mpc_X')
                    axes[0, 0].plot(frames, history_mpc_sol_pl_pos[:, 1], color='green', label='mpc_Y')
                    axes[0, 0].plot(frames, history_mpc_sol_pl_pos[:, 2], color='blue', label='mpc_Z')
                    axes[0, 0].plot(frames, history_left_feet_pos[:, 0], color='red', label='current_X', linestyle='--')
                    axes[0, 0].plot(frames, history_left_feet_pos[:, 1], color='green', label='current_Y', linestyle='--')
                    axes[0, 0].plot(frames, history_left_feet_pos[:, 2], color='blue', label='current_Z', linestyle='--')
                    axes[0, 0].set_title("Left Foot Position")
                    axes[0, 0].set_ylabel("Position [m]")
                    axes[0, 0].legend(loc='upper right'); axes[0, 0].grid(True, alpha=0.3)

                    # 2. Velocità PL
                    axes[1, 0].plot(frames, history_mpc_sol_pl_vel[:, 0], color='red', label='mpc_Vx')
                    axes[1, 0].plot(frames, history_mpc_sol_pl_vel[:, 1], color='green', label='mpc_Vy')
                    axes[1, 0].plot(frames, history_mpc_sol_pl_vel[:, 2], color='blue', label='mpc_Vz')
                    axes[1, 0].plot(frames, history_left_feet_vel[:, 0], color='red', label='current_Vx', linestyle='--')
                    axes[1, 0].plot(frames, history_left_feet_vel[:, 1], color='green', label='current_Vy', linestyle='--')
                    axes[1, 0].plot(frames, history_left_feet_vel[:, 2], color='blue', label='current_Vz', linestyle='--')
                    axes[1, 0].set_title("Left Foot Velocity")
                    axes[1, 0].set_ylabel("Velocity [m/s]")
                    axes[1, 0].legend(loc='upper right'); axes[1, 0].grid(True, alpha=0.3)

                    # 3. Accelerazione PL
                    if not np.all(history_mpc_sol_pl_acc == dead_value):
                        axes[2, 0].plot(frames, history_mpc_sol_pl_acc[:, 0], color='red', label='mpc_Ax')
                        axes[2, 0].plot(frames, history_mpc_sol_pl_acc[:, 1], color='green', label='mpc_Ay')
                        axes[2, 0].plot(frames, history_mpc_sol_pl_acc[:, 2], color='blue', label='mpc_Az')
                    axes[2, 0].plot(frames, history_left_feet_acc[:, 0], color='red', label='current_Ax', linestyle='--')
                    axes[2, 0].plot(frames, history_left_feet_acc[:, 1], color='green', label='current_Ay', linestyle='--')
                    axes[2, 0].plot(frames, history_left_feet_acc[:, 2], color='blue', label='current_Az', linestyle='--')
                    axes[2, 0].set_title("Left Foot Acceleration")
                    axes[2, 0].set_xlabel("Frame")
                    axes[2, 0].set_ylabel("Acc [m/s²]")
                    axes[2, 0].legend(loc='upper right'); axes[2, 0].grid(True, alpha=0.3)

                    # --- COLONNA 1: RIGHT FOOT (PR) ---
                    # 1. Posizione PR
                    axes[0, 1].plot(frames, history_mpc_sol_pr_pos[:, 0], color='orange', label='mpc_X', )
                    axes[0, 1].plot(frames, history_mpc_sol_pr_pos[:, 1], color='purple', label='mpc_Y', )
                    axes[0, 1].plot(frames, history_mpc_sol_pr_pos[:, 2], color='brown', label='mpc_Z', )
                    axes[0, 1].plot(frames, history_right_feet_pos[:, 0], color='orange', label='current_X', linestyle='--')
                    axes[0, 1].plot(frames, history_right_feet_pos[:, 1], color='purple', label='current_Y', linestyle='--')
                    axes[0, 1].plot(frames, history_right_feet_pos[:, 2], color='brown', label='current_Z', linestyle='--')
                    axes[0, 1].set_title("Right Foot Position")
                    axes[0, 1].legend(loc='upper right'); axes[0, 1].grid(True, alpha=0.3)

                    # 2. Velocità PR
                    axes[1, 1].plot(frames, history_mpc_sol_pr_vel[:, 0], color='orange', label='mpc_Vx', )
                    axes[1, 1].plot(frames, history_mpc_sol_pr_vel[:, 1], color='purple', label='mpc_Vy', )
                    axes[1, 1].plot(frames, history_mpc_sol_pr_vel[:, 2], color='brown', label='mpc_Vz', )
                    axes[1, 1].plot(frames, history_right_feet_vel[:, 0], color='orange', label='current_Vx', linestyle='--')
                    axes[1, 1].plot(frames, history_right_feet_vel[:, 1], color='purple', label='current_Vy', linestyle='--')
                    axes[1, 1].plot(frames, history_right_feet_vel[:, 2], color='brown', label='current_Vz', linestyle='--')
                    axes[1, 1].set_title("Right Foot Velocity")
                    axes[1, 1].legend(loc='upper right'); axes[1, 1].grid(True, alpha=0.3)

                    # 3. Accelerazione PR
                    if not np.all(history_mpc_sol_pr_acc == dead_value):
                        axes[2, 1].plot(frames, history_mpc_sol_pr_acc[:, 0], color='orange', label='mpc_Ax', )
                        axes[2, 1].plot(frames, history_mpc_sol_pr_acc[:, 1], color='purple', label='mpc_Ay', )
                        axes[2, 1].plot(frames, history_mpc_sol_pr_acc[:, 2], color='brown', label='mpc_Az', )
                    axes[2, 1].plot(frames, history_right_feet_acc[:, 0], color='orange', label='current_Ax', linestyle='--')
                    axes[2, 1].plot(frames, history_right_feet_acc[:, 1], color='purple', label='current_Ay', linestyle='--')
                    axes[2, 1].plot(frames, history_right_feet_acc[:, 2], color='brown', label='current_Az', linestyle='--')
                    axes[2, 1].set_title("Right Foot (PR) Acceleration")
                    axes[2, 1].set_xlabel("Frame")
                    axes[2, 1].legend(loc='upper right'); axes[2, 1].grid(True, alpha=0.3)

                    plt.tight_layout()
                    #os.makedirs(save_dir, exist_ok=True)
                    name = "render_test_feet_dynamics_comparison.png"
                    saved = os.path.join(save_dir, name)
                    plt.savefig(saved)
                    print(f"Saved {saved}")

                def plot_perturbation():
                    history_perturb = np.array(logging['perturb'])
                    frames = range(len(history_perturb))

                    plt.figure(figsize=(10, 6))
                    
                    # Plot delle 3 componenti (X, Y, Z)
                    plt.plot(frames, history_perturb[:, 0], color='red', label='Perturb_X')
                    plt.plot(frames, history_perturb[:, 1], color='green', label='Perturb_Y')
                    plt.plot(frames, history_perturb[:, 2], color='blue', label='Perturb_Z')

                    # Formattazione
                    plt.title("External Perturbations Over Time")
                    plt.xlabel("Frame")
                    plt.ylabel("Force/Torque Value")
                    plt.legend(loc='upper right')
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    # Salvataggio
                    #os.makedirs(save_dir, exist_ok=True)
                    name = "perturbation_plot.png"
                    saved = os.path.join(save_dir, name)
                    plt.savefig(saved)
                    print(f"Saved {saved}")
                    
                if training_in_test == True:
                    subdir = os.path.join(save_plot_dir, "plots_train_in_test")
                    os.makedirs(subdir, exist_ok=True)
                    save_dir = os.path.join(subdir, f"test_epoch_{num_epoch}")
                
                plotting_task = [
                    plot_reward_info,
                    plot_total_reward_info,
                    plot_com,
                    plot_feet,
                    plot_torques,
                    plot_perturbation
                ]

                for func in plotting_task:
                    try:
                        func()
                    except Exception as e:
                        print(f"ERROR in function {func.__name__}: {e}")
                        pass
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        if training_in_test == True:
            policy.train()

        end_test_time = time.time()
        test_duration = end_test_time - start_test_time
        print(f"Test duration: {test_duration:.2f} seconds")
        env.close()
        cv2.destroyAllWindows() 


        if render_mode == "rgb_array":
            print(f"Videos saved in: {save_plot_dir}")

def print_net_info(name, net, state_shape, action_shape=None):
    if action_shape is not None:
        input_dim = state_shape
        output_dim = action_shape[0]
    else:
        input_dim = state_shape
        output_dim = 1
    print(f"\nNetwork {name} info:")
    print(f"\tinput Size: {input_dim}, output Size: {output_dim}")

def log_enviroment_config(task, env_single: gym.Env):

    config = env_single.unwrapped.get_config()
    space_info = SpaceInfo.from_env(env_single)
    state_shape = space_info.observation_info.obs_shape[0]
    action_shape = space_info.action_info.action_shape
    max_action = space_info.action_info.max_action
    frame_stack = config.frame_stack

    log_and_print(f"Enviroment: {task}")
    log_and_print(f"Episode lenght: {EPISODE_LENGTH}")
    log_and_print(f"Observation space: {int(state_shape/frame_stack)} x {frame_stack} (stacked frames)")
    log_and_print(f"Action space: {action_shape}")
    log_and_print(f"Action size: {max_action}\n")

    log_and_print(f"\tAction Scale: {config.action_scale}")
    log_and_print(f"\tAction Repeat: {config.action_repeat}")
    log_and_print(f"\tFrame Stack: {config.frame_stack}")

    # ---- Reward config ----
    reward_scales = config.reward_config.scales
    for reward_name, scale in reward_scales.items():
        if scale != 0:
            log_and_print(f"\t{reward_name}: {scale}")

     # ---- Perturbation config ----
    if hasattr(config, "pert_config"):
        log_and_print("\nPerturbation config:")
        for k, v in config.pert_config.items():
            log_and_print(f"\t{k}: {v}")

    observation_dict = env_single.unwrapped.get_obs_info()[0]
    log_and_print(f"Observation:")

    for k in observation_dict.keys():
        log_and_print(f"\t{k}")

def dist_fn(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
    loc, scale = loc_scale
    return Independent(Normal(loc, scale), 1)

class LayerNormalizer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.rms = RunningMeanStd() # Quella di tianshou
        self.shape = shape
        self.eps = 1e-8

    def forward(self, obs):
        if self.training:
            self.rms.update(obs.detach().cpu().numpy())
        
        # Converte media e varianza in tensori per il calcolo
        mean = torch.as_tensor(self.rms.mean, device=obs.device, dtype=torch.float32)
        std = torch.as_tensor(np.sqrt(self.rms.var + self.eps), device=obs.device, dtype=torch.float32)
        
        return (obs - mean) / std
    
class TitaNetObsNormalizer(Net):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_normalizer = RunningMeanStd()
        self.eps = 1e-8

    def forward(self, obs, state=None, info={}):

        device = next(self.parameters()).device
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        else:
            obs = obs.to(device)

        if self.training:
            self.input_normalizer.update(obs.detach().cpu().numpy())
        
        mean = torch.as_tensor(self.input_normalizer.mean, device=obs.device, dtype=torch.float32)
        std = torch.as_tensor(np.sqrt(self.input_normalizer.var + self.eps), device=obs.device, dtype=torch.float32)
        
        obs_normalized = (obs - mean) / std
        return super().forward(obs_normalized, state, info)
    

class TransformerActorNet(ActionReprNetWithVectorOutput):

    def __init__(
        self,
        state_dim: int,           
        embedding_sizes: list[int] = [512, 512],
        output_dim: int = 128,    
        seq_len: int = 2,
        hidden_sizes: int = 256,
        n_head: int = 4,
        n_encoder: int = 2,
        mlp_ratio: int = 2,
        activation_mlp: type = nn.ReLU,
        activation_encoder: type = F.relu,
        device: str = "cuda"
    ):
        super().__init__(output_dim)
        self.seq_len = seq_len

        embedding_layers = []
        in_dim = state_dim
        for h in embedding_sizes:
            embedding_layers.append(nn.Linear(in_dim, h))
            embedding_layers.append(activation_mlp)
            in_dim = h
        self.embedding = nn.Sequential(*embedding_layers)
        dim_encoder = embedding_sizes[-1]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_encoder,
            nhead=n_head,
            dim_feedforward=dim_encoder*mlp_ratio,
            activation=activation_encoder,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_encoder
        )

        # output vector per SAC
        self.output_layer = nn.Linear(
            dim_encoder, output_dim
        )

    def forward(self, obs, state=None):
        # obs: [B, state_dim] -> [B, seq_len, state_dim]
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.output_layer.weight.device)

        x = obs.unsqueeze(1).repeat(1, self.seq_len, 1)
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.output_layer(x[:, -1, :])  # prendi solo l'ultima "token"
        return x, state

def create_wrapped_env(task: str, task_to_execute: int, render_mode=None,  ) -> gym.Env:
    env = gym.make(task, render_mode=render_mode, width=1000, height=600, task_to_execute=task_to_execute)
    #env = gym.wrappers.NormalizeObservation(env)  
    #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
    env = gym.wrappers.FrameStackObservation(env, stack_size=env.unwrapped.get_config().frame_stack)
    env = gym.wrappers.FlattenObservation(env)
    return env

def init_layer_orthogonal(m):

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=1.0)
        torch.nn.init.constant_(m.bias, 0.0)

def init_last_layer(m):
    if isinstance(m, torch.nn.Linear):
        with torch.no_grad():
            m.weight.data.mul_(0.01)
            m.bias.data.fill_(0.0)

def main():

    # ----- Parse arguments -----
    script_task, alg_type, render_mode, make_log, name_weight_name, warmup_steps, task_to_display = parser_args()
    
    # ----- Configuration -----
    task = "Tita-v0" #"Pendulum-v1"
    
    if task == "Tita-v0":
        import sys
        sys.path.insert(0, '/home/ubuntu/miniconda3/envs/tianshou_gpu/lib/python3.12/site-packages')

        gym.register(
            id="Tita-v0",
            entry_point="gymnasium.envs.mujoco.tita_env:TitaEnv",
            max_episode_steps=EPISODE_LENGTH,
        )
    
    global env_single
    env_single = create_wrapped_env(task, task_to_execute=task_to_display)

    logdir = os.path.join(get_git_root(), "TITA_MJ", "log", f"{alg_type}_logs")
    device = "cuda"
    lr = 1e-5
    hidden_sizes = [512, 256, 128]
    num_training_envs = 4
    num_test_envs = 2*env_single.unwrapped.get_num_tasks()
    num_view_test_env = 1
    if script_task == _STR_TRAIN:
        training_envs = SubprocVectorEnv( [lambda i=i: create_wrapped_env(task, task_to_execute=i) for i in range(num_training_envs)], )
        test_envs = SubprocVectorEnv([lambda i=i: create_wrapped_env(task, task_to_execute=i) for i in range(num_test_envs)], )

    # ----- Get environment info ----- 
    space_info = SpaceInfo.from_env(env_single)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    max_action = space_info.action_info.max_action

    global N_FRAME_STACK
    N_FRAME_STACK = env_single.unwrapped.get_config().frame_stack

    log_enviroment_config(task, env_single)
    #activation_fn = nn.Softsign
    activation_fn = nn.Tanh

    # ----- Choose algorithm -----
    net_mlp = Net(
        state_shape=state_shape,
        hidden_sizes=hidden_sizes,
        activation=activation_fn,
        #norm_layer=LayerNormalizer
    )

    net_norm = TitaNetObsNormalizer(
        state_shape=state_shape,
        hidden_sizes=hidden_sizes,
        activation=activation_fn,
        norm_layer=LayerNormalizer
    )

    net_transformer = TransformerActorNet(
        state_dim=state_shape[0], 
        embedding_sizes=[512, 192],
        seq_len=2,
        hidden_sizes=hidden_sizes,
        n_head=4,
        n_encoder=1,
        mlp_ratio=2,
        output_dim=128,
        activation_mlp=nn.Tanh(),
        activation_encoder=F.relu,
        device=device
    )

    net = net_mlp

    actor = ContinuousActorProbabilistic(
        preprocess_net=net,
        action_shape=action_shape,
        max_action=max_action,
        unbounded=False, # if true apply tanh to output, else max_action = 1.0
        conditioned_sigma=False, # if true, sigma is output of a simple network, else is a parameter
    )
    #actor.apply(init_layer_orthogonal)
    #actor.mu.apply(init_last_layer)
    #with torch.no_grad():
    #    torch.nn.init.constant_(actor.sigma_param, -3.0)
    actor = actor.to(device)
    print_net_info("Actor", actor, state_shape, action_shape)

    if alg_type == _STR_PPO:
        critic = ContinuousCritic(
            preprocess_net=Net(
                state_shape=state_shape,
                action_shape=action_shape,
                concat=False, # whether the input shape is concatenated by state_shape
                hidden_sizes=hidden_sizes,
                activation=activation_fn,
            ),
            hidden_sizes=hidden_sizes,
        )
        critic = critic.to(device)
        print_net_info("Critic", critic, state_shape)

        policy = ProbabilisticActorPolicy(
            actor=actor,
            dist_fn=dist_fn,
            action_scaling=False,
            action_bound_method="tanh",
            action_space=env_single.action_space,
            deterministic_eval=True,
        )

        optim = AdamOptimizerFactory(lr=lr)
        
        algo = PPO(
            policy=policy,
            critic=critic,
            optim=optim,
            eps_clip=0.2,
            vf_coef=0.5,
            ent_coef=0.0,
            gae_lambda=0.95,
            max_grad_norm=0.5,
            gamma=0.99,
        )
        log_and_print(f"\nPPO Hyperparameters:")
        log_and_print(f"\tEps clip: {algo.eps_clip}")
        log_and_print(f"\tVf coef: {algo.vf_coef}")
        log_and_print(f"\tEnt coef: {algo.ent_coef}")
        log_and_print(f"\tGAE lambda: {algo.gae_lambda}")
        log_and_print(f"\tMax grad norm: {algo.max_grad_norm}")
        log_and_print(f"\tGamma: {algo.gamma}")
    elif alg_type == _STR_SAC:
        critic1 = ContinuousCritic(
            preprocess_net=Net(
                state_shape=state_shape,
                action_shape=action_shape,
                concat=True,
                hidden_sizes=hidden_sizes,
                activation=activation_fn,
            ),
            hidden_sizes=hidden_sizes,
        )
        #critic1.apply(init_layer_orthogonal)
        #critic1.mu.apply(init_last_layer)
        critic1 = critic1.to(device)
        print_net_info("Critic1", critic1, state_shape)

        critic2 = ContinuousCritic(
            preprocess_net=Net(
                state_shape=state_shape,
                action_shape=action_shape,
                concat=True,
                hidden_sizes=hidden_sizes,
                activation=activation_fn,
            ),
            hidden_sizes=hidden_sizes,
        )
        #critic2.apply(init_layer_orthogonal)
        #critic2.mu.apply(init_last_layer)
        critic2 = critic2.to(device)
        print_net_info("Critic2", critic2, state_shape)

        policy = SACPolicy(
            actor=actor,
            exploration_noise=None,
            deterministic_eval=True,
            action_scaling=False,
            action_space=env_single.action_space,
        )

        algo = SAC(
            policy=policy,
            policy_optim=AdamOptimizerFactory(lr=lr),
            critic=critic1,
            critic_optim=AdamOptimizerFactory(lr=lr),
            critic2=critic2,
            critic2_optim=AdamOptimizerFactory(lr=lr),
            tau=0.005,
            gamma=0.99,
            alpha=0.001,
            n_step_return_horizon=5,
        )

        log_and_print(f"\nSAC Hyperparameters:")
        log_and_print(f"\tTau: {algo.tau}")
        log_and_print(f"\tGamma: {algo.gamma}")
        try:
            alpha_val = algo.alpha.value
        except Exception:
            alpha_val = algo.alpha
        log_and_print(f"\tAlpha: {alpha_val}")
        log_and_print(f"\tN-step return horizon: {algo.n_step_return_horizon}")
    else:
        raise ValueError("Unsupported algorithm. Choose either 'ppo' or 'sac'.")
    
    # Print device information for debugging ------
    try:
        print(f"\nDevice chosen for training/models: {device}")
        actor_param = next(actor.parameters())
        actor_dev = actor_param.device
        print(f"\tActor: {actor_dev}", end=", ")
        if alg_type == _STR_PPO:
            critic_param = next(critic.parameters())
            critic_dev = critic_param.device

            print(f"Critic: {critic_dev}")
        elif alg_type == _STR_SAC:
            critic1_param = next(critic1.parameters())
            critic1_dev = critic1_param.device
            critic2_param = next(critic2.parameters())
            critic2_dev = critic2_param.device
            print(f"Critic1: {critic1_dev}", end=", ")
            print(f"Critic2: {critic2_dev}")
        else:
            raise ValueError("Unsupported algorithm. Choose either 'ppo' or 'sac'.")
    except StopIteration:
        actor_dev = torch.device(device)
        if alg_type == _STR_PPO:
            critic_dev = torch.device(device)
        elif alg_type == _STR_SAC:
            critic_dev = torch.device(device)
        else:
            raise ValueError("Unsupported algorithm. Choose either 'ppo' or 'sac'.")

    # ----- Cehck train / test task -----
    if script_task == _STR_TEST:
        print(f"\nStarting testing enviroment: {task}")

        policy.eval()
        root = os.path.join(get_git_root(), "TITA_MJ", "log", f"{alg_type}_logs")
        if name_weight_name is not None:
            exp_name = name_weight_name
        else:
            root =os.path.join(root, "weights")
            folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
            if not folders:
                raise RuntimeError(f"No folder find in {root}")
            folders.sort()  
            exp_name = folders[-1]
            print(f"Using the latest experiment: {exp_name}")

        path_actor = os.path.join(exp_name, "final", "final_actor_state_dict.pt") 
        actor_path = os.path.join(root, path_actor)

        print(f"Loading Actor weights from: {actor_path}")
        if os.path.exists(actor_path):
            actor.load_state_dict(torch.load(actor_path, map_location=device))
        else:
            raise FileNotFoundError(f"File Actor non trovato: {actor_path}")
    
        test_enviroment(
            task_name=task,
            task_to_display=task_to_display,
            policy=policy,
            render_mode=render_mode,
            num_test_envs=num_test_envs,
            save_plot_dir=os.path.join(root, exp_name, DIR_EXPERIMENT_INFO, "plots")
        )

        return  
    elif script_task == _STR_TRAIN:
        log_and_print(f"\nStarting training enviroment {task}")
    else:
        raise ValueError("Either --train or --test must be specified.")

    buffer_total_size = 1_010_000
    buffer_num  = num_training_envs
    stack_num = 1  # env_single.unwrapped.get_config().frame_stack

    buffer_vanilla = VectorReplayBuffer(
        total_size=buffer_total_size,
        buffer_num=buffer_num,
        stack_num=stack_num
    )

    buffer_per = PrioritizedVectorReplayBuffer(
        total_size=buffer_total_size,
        buffer_num=buffer_num,
        alpha=0.6,
        beta=0.4,
        stack_num=stack_num,
    )

    buffer = buffer_vanilla

    log_and_print("\nBuffer parameters:")
    log_and_print(f"\t Total size: {buffer.maxsize:_}")
    log_and_print(f"\t Buffer num: {buffer.buffer_num}")
    log_and_print(f"\t Stack num: {buffer.stack_num}")
    if isinstance(buffer, PrioritizedVectorReplayBuffer):
        
        log_and_print(f"\t Prioritized Buffer alpha: {buffer._alpha}")
        log_and_print(f"\t Prioritized Buffer beta: {buffer._beta}")

    test_buffer = VectorReplayBuffer(
        total_size=EPISODE_LENGTH*num_test_envs, 
        buffer_num=len(test_envs), 
        stack_num=1#env_single.unwrapped.get_config().frame_stack,
    )
    
    # Create collectors (Collector works with DummyVectorEnv)
    train_collector = Collector(
        policy,
        training_envs,
        buffer,
        exploration_noise=False,
    )

    global test_collector
    test_collector = Collector(
        policy,
        test_envs,
        test_buffer,
        exploration_noise=False,
    )

    # ----- Initial data collection -----
    train_collector.reset()
    test_collector.reset()

    train_collector.collect(n_step=1*num_training_envs)
    test_collector.collect(n_step=1*num_test_envs)
    train_batch, _ = train_collector.buffer.sample(1)
    log_and_print(f"Train Buffer - Batch Observation Shape: {train_batch.obs.shape}")
    log_and_print(f"Train Buffer - Single Sample Shape: {train_batch.obs[0].shape}")

    test_batch, _ = test_collector.buffer.sample(1)
    log_and_print(f"Test Buffer - Batch Observation Shape: {test_batch.obs.shape}")
    log_and_print(f"Test Buffer - Single Sample Shape: {test_batch.obs[0].shape}")

    train_collector.reset()
    test_collector.reset()

    if warmup_steps is not None:
        start_warmup_time = datetime.datetime.now()
        log_and_print(f"\nCollecting {warmup_steps:_} warmup steps...")

        steps_per_iter = warmup_steps // 10  # divido in 10 chunk
        try:
            for i in range(10):
                train_collector.collect(n_step=steps_per_iter)
                print(f"\rWarmup progress: {(i + 1) * steps_per_iter:_} / {warmup_steps:_} steps", end="", flush=True)
        except KeyboardInterrupt:
         pass

        end_warmup_time = datetime.datetime.now()
        warmup_total_time = end_warmup_time - start_warmup_time
        log_and_print(f"\nCollected {len(train_collector.buffer)} sampled in time: {format_td(warmup_total_time)}")
   
    # ----- Setup logger using LoggerFactoryDefault -----
    timestamp = datetime.datetime.now().strftime('day_%Y_%m_%d_time_%H_%M_%S')
    run_dir_name = f"{alg_type}_{timestamp}"
    actor_path = os.path.join(logdir,"weights", run_dir_name,  f"actor_state_dict.pt")
    critic_path = os.path.join(logdir, "weights", run_dir_name, f"critic_state_dict.pt")

    global dir_experiment
    dir_experiment = os.path.join(logdir, "weights", run_dir_name)
    
    checkpath_root = os.path.join(get_git_root(), "TITA_MJ", "log", f"{alg_type}_logs")
    checkpath_actor = os.path.join(checkpath_root, name_weight_name if name_weight_name is not None else "", "final")
    if name_weight_name is not None and os.path.exists(checkpath_actor):
        log_and_print(f"\nLoading Actor and Critic weights from: {name_weight_name.split('/')[-1]}")
        actor.load_state_dict(torch.load(os.path.join(os.path.join(checkpath_actor), "final_actor_state_dict.pt"), map_location=device))
        if alg_type == _STR_PPO:
            critic.load_state_dict(torch.load(os.path.join(os.path.join(checkpath_actor), "final_critic_state_dict.pt"), map_location=device))
        elif alg_type == _STR_SAC:
            critic1.load_state_dict(torch.load(os.path.join(os.path.join(checkpath_actor), "final_critic_state_dict_1.pt"), map_location=device))
            critic2.load_state_dict(torch.load(os.path.join(os.path.join(checkpath_actor), "final_critic_state_dict_2.pt"), map_location=device))
        log_and_print(f"Loaded Actor and Critic weights from: {checkpath_actor}")
    else:
        log_and_print(f"Actor file not found: {checkpath_actor}\n -> Continuing training from scratch.")

    logger_factory = LoggerFactoryDefault()
    logger_factory.logger_type = "tensorboard"
    logger = logger_factory.create_logger(
        log_dir=os.path.join(logdir, "weights", run_dir_name),
        experiment_name= alg_type + "_".join(map(str, hidden_sizes)),
        run_id=task + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    # ----- Create trainer and run training ----- 
    if alg_type == _STR_PPO:
        trainer_type = OnPolicyTrainerParams(
                training_collector=train_collector, 
                test_collector=test_collector,  
                logger=logger,
                test_fn=test_fn,
                #stop_fn=lambda mean_rewards: mean_rewards >= 2950.0,
                save_best_fn=partial(save_best, alg_type=alg_type, actor_policy=actor, actor_path=actor_path, critic_policy=critic, critic_path=critic_path),
                test_in_training=False,

                # Know parameters 
                max_epochs=10,   
                batch_size=254,

                # online training: total number of enviroment steps to collect before updated
                # offline training: total number of training step per epoch before update
                epoch_num_steps=100*num_training_envs, #*num_test_envs,   

                # Transition to collect at each collection step
                # before network update update 
                collection_step_num_env_steps=200, #*num_training_envs, 
                # Number of training at each epoch: epoch_num_steps / collection_step_num_env_steps

                # The number of times data are used
                # for gradient updates
                update_step_num_repetitions=2000,

                # Number of episodes to colleact in each test step
                # i.e. number of run for evaluation
                test_step_num_episodes=num_test_envs,
            )
        
        log_and_print("\nPPO Training parameters:")
        log_and_print("\t Num train/test envs: ", num_training_envs, "/", num_test_envs)
        log_and_print("\t learning rate:", lr)
        log_and_print("\t hidden sizes:", hidden_sizes)
        log_and_print("\t Max epochs:", trainer_type.max_epochs)
        log_and_print("\t Batch size:", trainer_type.batch_size)
        log_and_print("\t Epoch num steps:", trainer_type.epoch_num_steps)
        log_and_print("\t Collection step num env steps:", trainer_type.collection_step_num_env_steps)
        log_and_print("\t Update step num repetitions:", trainer_type.update_step_num_repetitions)
        log_and_print("\t Test step num episodes:", trainer_type.test_step_num_episodes, "\n")
    elif alg_type == _STR_SAC: 
        rollout = 20
        trainer_type = OffPolicyTrainerParams(
                training_collector=train_collector, 
                test_collector=test_collector,  
                logger=logger,
                #test_fn=test_fn,
                test_fn=partial(test_fn, policy=deepcopy(actor), task=task, num_test_envs=num_test_envs, num_view_test_env=num_view_test_env, save_plot_dir=os.path.join(os.path.dirname(actor_path), DIR_EXPERIMENT_INFO, "plots") ),
                #stop_fn=lambda mean_rewards: mean_rewards >= 2950.0,
                save_best_fn=partial(save_best, alg_type=alg_type, actor_policy=actor, actor_path=actor_path, critic_policy=[critic1, critic2], critic_path=critic_path),
                test_in_training=False,

                # Know parameters 
                max_epochs=100,    
                batch_size=1024,

                # Total number of training steps to take per epoch
                epoch_num_steps=EPISODE_LENGTH*num_training_envs, 

                # the number of environment steps/transitions to collect in each collection step before the
                # network update within each training step.
                collection_step_num_env_steps=rollout*num_training_envs,
                #collection_step_num_episodes=num_training_envs,
                
                # The number of times data 
                update_step_num_gradient_steps_per_sample=1.0,

                # Number of episodes to colleact in each test step
                # i.e. number of run for evaluation
                test_step_num_episodes=num_test_envs,
            )
        
        log_and_print("\nSAC Trainer parameters:")
        log_and_print("\t Num train/test envs", num_training_envs, "/", num_test_envs)
        log_and_print("\t learning rate:", lr)
        
        log_and_print("\t hidden sizes:", hidden_sizes)
        log_and_print("\t Max epochs:", trainer_type.max_epochs)
        log_and_print("\t Batch size:", trainer_type.batch_size)
        log_and_print("\t Epoch num steps:", trainer_type.epoch_num_steps)
        if trainer_type.collection_step_num_env_steps is not None:
            log_and_print("\t Collection step num env steps:", trainer_type.collection_step_num_env_steps, ", roullout: ", rollout)
        else:
            log_and_print("\t Collection step num episodes:", trainer_type.collection_step_num_episodes, ", episode per enviroment: ", trainer_type.collection_step_num_episodes/(rollout*num_training_envs))
        log_and_print("\t Update step num gradient steps per sample:", trainer_type.update_step_num_gradient_steps_per_sample)
        log_and_print("\t Test step num episodes:", trainer_type.test_step_num_episodes, "\n")
    else:
        raise ValueError("Unsupported algorithm. Choose either 'ppo' or 'sac'.")

    actor_base_dir = os.path.dirname(actor_path)
    setup_auto_logging(os.path.join(actor_base_dir, DIR_EXPERIMENT_INFO, "training_log.txt"))

    start_time = datetime.datetime.now()
    try:
        result_policy = algo.run_training(
            trainer_type
        )
    except KeyboardInterrupt:
        pass
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    log_and_print(f"Total training time: {format_td(total_time)}")

    log_and_print("\nTraining completed!")
    log_and_print(f"Logs saved to {logdir}")

    # ----- Save model weights -----
    log_and_print(f"Saving weights in {logdir}")
    try:
        actor_file_name = os.path.basename(actor_path)
        final_actor_path = os.path.join(actor_base_dir, "final", f"final_{actor_file_name}")
        os.makedirs(os.path.dirname(final_actor_path), exist_ok=True)

        torch.save(actor.state_dict(), final_actor_path)
        log_and_print(f"Saved actor weights to {final_actor_path}")
        if alg_type == _STR_PPO:
            critical_base_dir = os.path.dirname(critic_path)
            critical_file_name = os.path.basename(critic_path)
            final_critic_path = os.path.join(critical_base_dir, "final", f"final_{critical_file_name}")
            
            torch.save(critic.state_dict(), final_critic_path)
            log_and_print(f"Saved critic weights to {final_critic_path}")
        elif alg_type == _STR_SAC:
            critic1_base_dir = os.path.dirname(critic_path)
            critic1_file_name = os.path.basename(critic_path)
            final_critic1_path = os.path.join(critic1_base_dir, "final", f"final_{critic1_file_name.replace('.pt', '_1.pt')}")
            torch.save(critic1.state_dict(), final_critic1_path)
            log_and_print(f"Saved critic1 weights to {final_critic1_path}")

            critic2_base_dir = os.path.dirname(critic_path)
            critic2_file_name = os.path.basename(critic_path)
            final_critic2_path = os.path.join(critic2_base_dir, "final", f"final_{critic2_file_name.replace('.pt', '_2.pt')}")
            torch.save(critic2.state_dict(), final_critic2_path)
            log_and_print(f"Saved critic2 weights to {final_critic2_path}")
        else:
            raise ValueError("Unsupported algorithm. Choose either 'ppo' or 'sac'.")
        
        
    except Exception as e:
        log_and_print("Could not save model weights:", e)
    finally:
        info_file_path = os.path.join(actor_base_dir, DIR_EXPERIMENT_INFO, "experiment_info.txt")
        os.makedirs(os.path.dirname(info_file_path), exist_ok=True)
        with open(info_file_path, "w") as f:
            f.write("\n".join(LOG_ARRAY))
        print(f"\n\tExperiment info saved to {info_file_path}")

        try:
            script_path = os.path.join(get_git_root(), "TITA_MJ", "tesi", "test_python", "plot.py")
            subprocess.run(["python3", script_path, run_dir_name], check=True)
        except subprocess.CalledProcessError as e:
            print("\n\tError on executing plot.py:", e)

        try:
            test_enviroment(
                task_name=task,
                task_to_display=task_to_display,
                policy=policy,
                render_mode=None,
                num_test_envs=num_test_envs,
                save_plot_dir=os.path.join(actor_base_dir, DIR_EXPERIMENT_INFO, "plots")
            )
        except Exception as e:
            print("\n\tError on testing enviroment after training:", e)

if __name__ == "__main__":
    main()
