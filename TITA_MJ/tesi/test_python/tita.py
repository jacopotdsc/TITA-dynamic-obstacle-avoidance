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
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from gymnasium.wrappers import RecordVideo
from torch.distributions import Distribution, Independent, Normal


from tianshou.algorithm import TD3
from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy
from tianshou.algorithm import SAC
from tianshou.exploration import GaussianNoise

from tianshou.algorithm import SAC

from tianshou.algorithm import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.modelfree.sac import SACPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.utils.statistics import RunningMeanStd
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo
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

global dir_experiment

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
                self.line_buffer = ""
            elif char == '\n':
                clean_line = self.ansi_escape.sub('', self.line_buffer).strip()
                # Verifica se la riga pulita è diversa dall'ultima scritta
                if clean_line and clean_line != self.last_written_line:
                    is_partial_tqdm = clean_line.endswith('it/s]')
                    is_complete_tqdm = 'update_step' in clean_line or 'env_step' in clean_line
                    is_reward_line = 'test_reward' in clean_line or 'Initial test' in clean_line

                    if not is_partial_tqdm or is_complete_tqdm or is_reward_line:
                        self.log.write(clean_line + '\n')
                        self.log.flush()
                        self.last_written_line = clean_line
                
                self.line_buffer = ""
            else:
                self.line_buffer += char

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

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
                        nargs='?', 
                        const='sac',      
                        choices=[_STR_PPO, _STR_SAC],
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
    
    args = parser.parse_args()

    script_task = _STR_TRAIN
    alg_type = _STR_SAC
    render_mode = "human"
    test_exp_name = None
    make_log = False

    if args.train:
        script_task = _STR_TRAIN
        alg_type = args.train.lower()
    
    if args.test is not None:
        script_task = _STR_TEST
        test_args = [item.lower() for item in args.test]
        
        for item in test_args:
            if item in [_STR_SAC, _STR_PPO]:
                alg_type = item
            elif item in ['human', 'rgb', 'rgb_array']:
                render_mode = "rgb_array" if item in ['rgb', 'rgb_array'] else "human"
            else:
                test_exp_name = item

    if args.log:
        make_log = True if args.log.lower() == "yes" else False

    print(f"Script started with\n\ttask: {script_task},\n\talgorithm: {alg_type},\n\trender mode: {render_mode}\n\tpid: {os.getpid()}\n")
    return script_task, alg_type, render_mode, make_log, test_exp_name

def get_git_root():
    try:
        repo = git.Repo(".", search_parent_directories=True)
        return repo.working_tree_dir
    except git.InvalidGitRepositoryError:
        return None
    
def test_fn(num_epoch, step_idx):
    global BEST_LAST_EPOCH
    BEST_LAST_EPOCH = num_epoch

    global N_FRAME_STACK

    test_collector.reset()
    res = test_collector.collect(n_episode=1, render=0)
    mean_len = np.mean(res.lens)

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

    obs_headers = [
        # Robot state
        'robot_height', 'ori_w', 'ori_x', 'ori_y', 'ori_z', 'grav_x', 'grav_y', 'grav_z', 
        'lin_vel_x', 'lin_vel_y', 'lin_vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z',

        # Joint positions and velocities
        'joint_angle_1', 'joint_angle_2', 'joint_angle_3', 'joint_angle_4', 
        'joint_angle_5', 'joint_angle_6', 'joint_angle_7', 'joint_angle_8',
        'joint_vel_1', 'joint_vel_2', 'joint_vel_3', 'joint_vel_4', 
        'joint_vel_5', 'joint_vel_6', 'joint_vel_7', 'joint_vel_8',

        # Normalized wbc output
        'joint_torque_1', 'joint_torque_2', 'joint_torque_3', 'joint_torque_4', 
        'joint_torque_5', 'joint_torque_6', 'joint_torque_7', 'joint_torque_8',

        # Previous actions of neural network and user commands
        'prev_action_1', 'prev_action_2', 'prev_action_3', 'prev_action_4', 
        'prev_action_5', 'prev_action_6', 'prev_action_7', 'prev_action_8',
        'user_cmd_vx', 'user_cmd_vy', 'user_cmd_omega'
    ]
    
    act_headers = [f'action_{i}' for i in range(1, 9)]
    headers = ['epoch','frame', 'episode_length', 'mean_length', 'reward'] + obs_headers + act_headers

    global dir_experiment
    csv_path = os.path.join(dir_experiment, DIR_EXPERIMENT_INFO, "observations.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, pd.DataFrame(csv_data, columns=headers)], ignore_index=True)
    else:
        df = pd.DataFrame(csv_data, columns=headers)

    df.to_csv(csv_path, index=False)

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
        policy: nn.Module,
        render_mode: str = "human",
        num_test_envs: int = 16,
        save_dir: str = None,
    ):

    test_envs = SubprocVectorEnv([lambda: create_wrapped_env(task_name) for _ in range(num_test_envs)], )
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
    env = create_wrapped_env(task_name, render_mode=render_mode)

    # ----- Video recording setup -----
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("videos", exist_ok=True)
    video_folder = os.path.join("videos", f"{task_name}_{timestamp}")
    
    if render_mode == "rgb_array" :
        env = RecordVideo(
            env, 
            video_folder=video_folder,
            name_prefix="eval",
            episode_trigger=lambda episode_id: True 
        )
        print(f"Video recording enabled. File will be saved in: {video_folder}")

    try:
        obs, info = env.reset()
        total_reward = 0
        n_frame = 0
        history_action = []
        history_reward_info = []
        terminated = False
        truncated = False

        start_time_inference = []
        end_time_inference = []
        
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
            history_action.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            history_reward_info.append(info)


            if n_frame == 0 or (n_frame+1) % 100 == 0:
                print("Frame:", n_frame, "Action:", action, ", Total Reward:", total_reward)

            if render_mode is not None:
                frame = env.render()
            
            if render_mode is not None and frame is not None and render_mode == "rgb_array":
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                cv2.imshow("Agent Preview (Press 'q' to quit)", frame_bgr)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            n_frame += 1
            if terminated or truncated:
                print(f"Episode terminated. Reward: {total_reward:.2f}")
                print(f"Inference time: {1000*inference_time:.6f} ms first step")

                mean_inference_time = 1000 * np.mean([end - start for start, end in zip(start_time_inference, end_time_inference)])
                std_inference_time = 1000 * np.std([end - start for start, end in zip(start_time_inference, end_time_inference)])
                print(f"inference time: {mean_inference_time:.6f} ± {std_inference_time:.6f} ms")
        
                actions_array = np.array(history_action) # Shape: (N_frames, 8)
                frames = np.arange(len(actions_array))
                
                joint_names = ["ankle_pitch", "ankle_roll", "knee", "wheel"]
                legend_left = ["left_" + name for name in joint_names]
                legend_right = ["right_" + name for name in joint_names]

                def plot_actions():
                    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

                    for i in range(4):
                        axes[0].plot(frames, actions_array[:, i], label=legend_left[i])
                    axes[0].set_ylabel("Action/Torque")
                    axes[0].set_title("Left Leg Joint Actions")
                    axes[0].legend(loc='upper right')
                    axes[0].grid(True)

                    for i in range(4):
                        axes[1].plot(frames, actions_array[:, i+4], label=legend_right[i])
                    axes[1].set_xlabel("Frame")
                    axes[1].set_ylabel("Action/Torque")
                    axes[1].set_title("Right Leg Joint Actions")
                    axes[1].legend(loc='upper right')
                    axes[1].grid(True)

                    plt.tight_layout()
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, "torque_in_render_test.png"))
                    print(f"Saved torque_in_render_test.png in {save_dir}")
                
                def plot_reward_info():
                    reward_info_keys = list(history_reward_info[0].keys())
                    reward_info_keys = [key for key in reward_info_keys if env.unwrapped.get_config().reward_config.scales.get(key) != 0] 
                    n_keys = len(reward_info_keys)
                    
                    mid = (n_keys + 1) // 2
                    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

                    for idx, key in enumerate(reward_info_keys):
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
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, "reward_info_in_render_test.png"))
                    print(f"Saved reward_info_in_render_test.png in {save_dir}")
                plot_actions()
                plot_reward_info()

                #plt.show()

                break
                
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        cv2.destroyAllWindows() 

        if render_mode == "rgb_array":
            print(f"Videos saved in: {video_folder}")

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
    log_and_print(f"Observation space: {int(state_shape/frame_stack)} x {frame_stack} (stacked frames)")
    log_and_print(f"Action space: {action_shape}")
    log_and_print(f"Action size: {max_action}\n")

    log_and_print(f"\tAction Scale: {config.action_scale}")
    log_and_print(f"\tAction Repeat: {config.action_repeat}")
    reward_scales = config.reward_config.scales
    for reward_name, scale in reward_scales.items():
        if scale != 0:
            log_and_print(f"\t{reward_name}: {scale}")

def dist_fn(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
    loc, scale = loc_scale
    return Independent(Normal(loc, scale), 1)

def create_wrapped_env(task: str, render_mode=None) -> gym.Env:
    env = gym.make(task, render_mode=render_mode, width=1000, height=600)
    #env = gym.wrappers.NormalizeObservation(env)  
    #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
    env = gym.wrappers.FrameStackObservation(env, stack_size=env.unwrapped.get_config().frame_stack)
    env = gym.wrappers.FlattenObservation(env)
    return env

def init_layer_orthogonal(m):
    '''
    Docstring for init_layer_orthogonal
    
    Initialize the weights of a linear layer using orthogonal initialization.
    Given a tensor w, it will be initialized in a way that w @ w.T = I  ( or w.T @ w = I).
    Property of orthogonal matrix is that |Wx| = |x|, so it preserves the norm of the input
    as matrix rotation does.
    It is used to aboid the vanishing/exploding gradient problem.
    '''

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=1.0)
        torch.nn.init.constant_(m.bias, 0.0)

def init_last_layer(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=0.01)
        torch.nn.init.constant_(m.bias, 0.0)

def main():

    # ----- Parse arguments -----
    script_task, alg_type, render_mode, make_log, test_exp_name = parser_args()
    
    # ----- Configuration -----
    logdir = os.path.join(get_git_root(), "TITA_MJ", "log", f"{alg_type}_logs")
    device = "cuda"
    task = "Tita-v0" #"Pendulum-v1"
    lr = 0.0000001
    hidden_sizes = [256, 256, 256]
    num_training_envs = 8
    num_test_envs = 1
    num_view_test_env = 1

    if task == "Tita-v0":
        import sys
        sys.path.insert(0, '/home/ubuntu/miniconda3/envs/tianshou_gpu/lib/python3.12/site-packages')

        gym.register(
            id="Tita-v0",
            entry_point="gymnasium.envs.mujoco.tita_env:TitaEnv",
            max_episode_steps=1000,
        )
    
    if script_task == _STR_TRAIN:
        training_envs = SubprocVectorEnv( [lambda: create_wrapped_env(task) for _ in range(num_training_envs)], )
        test_envs = SubprocVectorEnv([lambda: create_wrapped_env(task) for _ in range(num_test_envs)], )

    # ----- Get environment info ----- 
    env_single = create_wrapped_env(task)
    space_info = SpaceInfo.from_env(env_single)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    max_action = space_info.action_info.max_action

    global N_FRAME_STACK
    N_FRAME_STACK = env_single.unwrapped.get_config().frame_stack

    log_enviroment_config(task, env_single)

    # ----- Choose algorithm -----
    net = Net(
        state_shape=state_shape,
        hidden_sizes=hidden_sizes,
        activation=nn.Tanh,
    )
    
    actor = ContinuousActorProbabilistic(
        preprocess_net=net,
        action_shape=action_shape,
        max_action=max_action,
        unbounded=False, # if true apply tanh to output, else max_action = 1.0
        conditioned_sigma=False, # if true, sigma is output of a simple network, else is a parameter
    )
    #actor.apply(init_layer_orthogonal)
    actor.mu.apply(init_last_layer)
    with torch.no_grad():
        torch.nn.init.constant_(actor.sigma_param, -3.0)
    actor = actor.to(device)
    print_net_info("Actor", actor, state_shape, action_shape)

    if alg_type == _STR_PPO:
        critic = ContinuousCritic(
            preprocess_net=Net(
                state_shape=state_shape,
                action_shape=action_shape,
                concat=False, # whether the input shape is concatenated by state_shape
                hidden_sizes=hidden_sizes,
                activation=nn.Tanh,
            ),
            hidden_sizes=hidden_sizes,
        )
        critic = critic.to(device)
        print_net_info("Critic", critic, state_shape)

        policy = ProbabilisticActorPolicy(
            actor=actor,
            dist_fn=dist_fn,
            action_scaling=False,
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
    elif alg_type == _STR_SAC:
        critic1 = ContinuousCritic(
            preprocess_net=Net(
                state_shape=state_shape,
                action_shape=action_shape,
                concat=True,
                hidden_sizes=hidden_sizes,
                activation=nn.Tanh,
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
                activation=nn.Tanh,
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
            alpha=0.1,
            n_step_return_horizon=2,
        )
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
        root = os.path.join(get_git_root(), "TITA_MJ", "log", f"{alg_type}_logs", "saved_weights")
        exp_name = test_exp_name
        path_actor = os.path.join(exp_name, "final", "final_actor_state_dict.pt") 
        actor_path = os.path.join(root, path_actor)

        print(f"Loading Actor weights from: {actor_path}")
        if os.path.exists(actor_path):
            actor.load_state_dict(torch.load(actor_path, map_location=device))
        else:
            raise FileNotFoundError(f"File Actor non trovato: {actor_path}")
    
        test_enviroment(
            task_name=task,
            policy=policy,
            render_mode=render_mode,
            num_test_envs=num_view_test_env,
            save_dir=os.path.join(root, exp_name, DIR_EXPERIMENT_INFO, "plots")
        )

        return  
    elif script_task == _STR_TRAIN:
        log_and_print(f"\nStarting training enviroment {task}")
    else:
        raise ValueError("Either --train or --test must be specified.")

    # ------ Buffer -------
    if alg_type == _STR_PPO:
        buffer = VectorReplayBuffer(
            total_size=1000*num_training_envs,
            buffer_num=num_training_envs,
            stack_num=1#env_single.unwrapped.get_config().frame_stack,
        )
        log_and_print("\nPPO Buffer parameters:")
        log_and_print(f"\t Total size: {buffer.maxsize:_}")
        log_and_print(f"\t Buffer num: {buffer.buffer_num}")
        log_and_print(f"\t Stack num: {buffer.stack_num}")
    elif alg_type == _STR_SAC:
        buffer = VectorReplayBuffer(
            total_size=1000*num_training_envs,
            buffer_num=num_training_envs,
            stack_num=1 #env_single.unwrapped.get_config().frame_stack,
        )
        log_and_print("\nSAC Buffer parameters:")
        log_and_print(f"\t Total size: {buffer.maxsize:_}")
        log_and_print(f"\t Buffer num: {buffer.buffer_num}")
        log_and_print(f"\t Stack num: {buffer.stack_num}")
    else:
        raise ValueError("Unsupported algorithm. Choose either 'ppo' or 'sac'.")
    
    test_buffer = VectorReplayBuffer(
        total_size=20000, 
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

    train_collector.collect(n_step=100*num_training_envs)
    test_collector.collect(n_step=100*num_test_envs)
    train_batch, _ = train_collector.buffer.sample(1)
    log_and_print(f"Train Buffer - Batch Observation Shape: {train_batch.obs.shape}")
    log_and_print(f"Train Buffer - Single Sample Shape: {train_batch.obs[0].shape}")

    test_batch, _ = test_collector.buffer.sample(1)
    log_and_print(f"Test Buffer - Batch Observation Shape: {test_batch.obs.shape}")
    log_and_print(f"Test Buffer - Single Sample Shape: {test_batch.obs[0].shape}")

    train_collector.reset()
    test_collector.reset()

    # ----- Setup logger using LoggerFactoryDefault -----
    timestamp = datetime.datetime.now().strftime('day_%Y_%m_%d_time_%H_%M_%S')
    run_dir_name = f"{alg_type}_{timestamp}"
    actor_path = os.path.join(logdir,"weights", run_dir_name,  f"actor_state_dict.pt")
    critic_path = os.path.join(logdir, "weights", run_dir_name, f"critic_state_dict.pt")

    global dir_experiment
    dir_experiment = os.path.join(logdir, "weights", run_dir_name)
    
    checkpath_root = os.path.join(get_git_root(), "TITA_MJ", "log", "weights_saved")
    checkpath_path_actor = "stand_up_randomize_reset.pt"
    checkpath_actor = os.path.join(checkpath_root, checkpath_path_actor)
    if os.path.exists(checkpath_actor):
        actor.load_state_dict(torch.load(checkpath_actor, map_location=device))
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
                stop_fn=lambda mean_rewards: mean_rewards >= 2950.0,
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
        trainer_type = OffPolicyTrainerParams(
                training_collector=train_collector, 
                test_collector=test_collector,  
                logger=logger,
                test_fn=test_fn,
                stop_fn=lambda mean_rewards: mean_rewards >= 2970.0,
                save_best_fn=partial(save_best, alg_type=alg_type, actor_policy=actor, actor_path=actor_path, critic_policy=[critic1, critic2], critic_path=critic_path),
                
                test_in_training=False,

                # Know parameters 
                max_epochs=4,    
                batch_size=512,

                # Total number of training steps to take per epoch
                epoch_num_steps=10*num_training_envs, 

                # the number of environment steps/transitions to collect in each collection step before the
                # network update within each training step.
                collection_step_num_env_steps=20*num_training_envs,
                #collection_step_num_episodes=1*num_training_envs 
                
                # The number of times data 
                update_step_num_gradient_steps_per_sample=10,

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
            log_and_print("\t Collection step num env steps:", trainer_type.collection_step_num_env_steps, ", roullout: ", trainer_type.collection_step_num_env_steps/num_training_envs)
        else:
            log_and_print("\t Collection step num episodes:", trainer_type.collection_step_num_episodes, ", episode per enviroment: ", trainer_type.collection_step_num_episodes/num_training_envs)
        log_and_print("\t Update step num gradient steps per sample:", trainer_type.update_step_num_gradient_steps_per_sample)
        log_and_print("\t Test step num episodes:", trainer_type.test_step_num_episodes, "\n")
    else:
        raise ValueError("Unsupported algorithm. Choose either 'ppo' or 'sac'.")

    actor_base_dir = os.path.dirname(actor_path)
    setup_auto_logging(os.path.join(actor_base_dir, DIR_EXPERIMENT_INFO, "training_log.txt"))

    try:
        result_policy = algo.run_training(
            trainer_type
        )
    except KeyboardInterrupt:
        pass

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
                policy=policy,
                render_mode=None,
                num_test_envs=num_view_test_env,
                save_dir=os.path.join(actor_base_dir, DIR_EXPERIMENT_INFO, "plots")
            )
        except Exception as e:
            print("\n\tError on testing enviroment after training:", e)

if __name__ == "__main__":
    main()
