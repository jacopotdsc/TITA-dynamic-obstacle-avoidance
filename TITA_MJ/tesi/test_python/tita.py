#!/usr/bin/env python3
"""
Simple PPO training script with vectorized environments.
Demonstrates procedural API usage for training on Pendulum-v1.
"""

import os
import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from torch import nn
import cv2
from functools import partial
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

def parser_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", 
                        nargs='?', 
                        const='sac',      
                        choices=['ppo', 'sac'],
                        help="Start training. Options: 'ppo' or 'sac' (default: sac)")
    
    group.add_argument("--test", 
                        nargs='*',   
                        choices=['sac', 'ppo' 'human', 'rgb'],
                        help="Start testing. Options: 'human' or 'rgb' (default: human)")
    
    args = parser.parse_args()

    script_task = _STR_TEST if args.test is not None else _STR_TRAIN
    alg_type = args.train if args.train else ( _STR_PPO if _STR_PPO in args.test else _STR_SAC )
    render_mode = ("rgb_array" if "rgb_array" in args.test else "human") if args.test is not None else None

    print(f"Script started with\n\ttask: {script_task},\n\talgorithm: {alg_type},\n\trender mode: {render_mode}\n")
    return script_task, alg_type, render_mode

def save_best(algorithm, alg_type, actor_path, critic_path):
    print("\tSaving best model weights\n")
    os.makedirs(os.path.dirname(actor_path), exist_ok=True)
    torch.save(algorithm.policy.actor.state_dict(), actor_path)

    #if alg_type == _STR_PPO:
    #    print("qui")
    #    torch.save(algorithm.critic.state_dict(), critic_path)
    #elif alg_type == _STR_SAC:
    #    torch.save(algorithm.critic.critic1.state_dict(), critic_path.replace(".pt", "_1.pt"))
    #    torch.save(algorithm.critic.critic2.state_dict(), critic_path.replace(".pt", "_2.pt"))
    #else:
    #    raise ValueError("Unsupported algorithm. Choose either 'ppo' or 'sac'.")

def test_enviroment(
        task_name: str,
        policy: nn.Module,
        render_mode: str = "human",
        num_test_envs: int = 16,
    ):

    test_envs = SubprocVectorEnv([lambda: create_wrapped_env(task_name) for _ in range(num_test_envs)], )
    collector = Collector(policy, test_envs, exploration_noise=False)
    collector.reset()
    result = collector.collect(n_episode=num_test_envs, render=0)
 
    print(result)
    print(type(result))
    print(f"\nResults test environment {task_name} (x{num_test_envs}):")
    print(f"Mean reward:     {result.returns_stat.mean:3f}")
    print(f"Std deviation:   ±{result.returns_stat.std:3f}")
    print(f"Min / Max:        {result.returns_stat.min:.3f} / {result.returns_stat.max:.3f}")
    print(f"Mean length:        {np.mean(result.lens):.1f}")
    print(f"Num episodes:       {len(result.returns)}\n")
    print("Finished testing in vectorized envs. Showing in viewer\n")

    # --- Manual rendering ---
    env = gym.make(task_name, render_mode=render_mode)

    try:
        obs, info = env.reset()
        total_reward = 0
        
        while True:
            # 1. Inferenza
            batch = Batch(obs=np.array([obs]), info={})
            with torch.no_grad():
                result = policy(batch)
            action = result.act[0]
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            # 2. Step Ambiente
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # 3. VISUALIZZAZIONE MANUALE E REGISTRAZIONE
            # env.render() restituisce l'array RGB perché mode="rgb_array"
            frame = env.render()
            
            if frame is not None and render_mode == "rgb_array":
                # OpenCV usa BGR invece di RGB, convertiamo per vedere i colori giusti
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Mostra la finestra
                cv2.imshow("Agent Preview (Press 'q' to quit)", frame_bgr)
                
                # Aspetta 1ms e controlla se premi 'q' per uscire
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if terminated or truncated:
                print(f"Episode terminated. Reward: {total_reward:.2f}")
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        cv2.destroyAllWindows() # Chiude la finestra OpenCV
        print("Saved video, closed.")

def dist_fn(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
    loc, scale = loc_scale
    return Independent(Normal(loc, scale), 1)

def create_wrapped_env(task: str) -> gym.Env:
    env = gym.make(task)
    #env = gym.wrappers.NormalizeObservation(env)  
    #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
    return env

def init_layer_orthogonal(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(m.bias, 0.0)

def init_last_layer(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=0.01)
            torch.nn.init.constant_(m.bias, 0.0)

def main():

    # ----- Parse arguments -----
    script_task, alg_type, render_mode = parser_args()
    
    # ----- Configuration -----
    logdir = F"log/{alg_type}_vectorized"
    device = "cuda"
    task = "Tita-v0" #"Pendulum-v1"
    lr = 0.0000001
    hidden_sizes = [256, 256, 256]
    num_training_envs = 16
    num_test_envs = 16

    if task == "Tita-v0":
        import sys
        sys.path.insert(0, '/home/ubuntu/miniconda3/envs/tianshou_gpu/lib/python3.12/site-packages')

        gym.register(
            id="Tita-v0",
            entry_point="gymnasium.envs.mujoco.tita_env:TitaEnv",
            max_episode_steps=1000,
        )
    
    if script_task == _STR_TRAIN:
        print(f"Creating vectorized environments (task={task})...")
        training_envs = SubprocVectorEnv( [lambda: create_wrapped_env(task) for _ in range(num_training_envs)], )
        test_envs = SubprocVectorEnv([lambda: create_wrapped_env(task) for _ in range(num_test_envs)], )

    # ----- Get environment info ----- 
    env_single = create_wrapped_env(task)
    space_info = SpaceInfo.from_env(env_single)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    max_action = space_info.action_info.max_action

    print(f"Enviroment: {task}")
    print(f"Observation space: {state_shape}")
    print(f"Action space: {action_shape}")
    print(f"Action size: {max_action}")

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
    actor.apply(init_layer_orthogonal)
    actor.mu.apply(init_last_layer)
    with torch.no_grad():
            torch.nn.init.constant_(actor.sigma_param, -3.0)
    actor = actor.to(device)

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
            eps_clip=0.1,
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
        critic1 = critic1.to(device)

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
        critic2 = critic2.to(device)

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
        actor_param = next(actor.parameters())
        actor_dev = actor_param.device
        if alg_type == _STR_PPO:
            critic_param = next(critic.parameters())
            critic_dev = critic_param.device
        elif alg_type == _STR_SAC:
            critic_param = next(critic1.parameters())
            critic_dev = critic_param.device
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

    print(f"Device chosen for training/models: {device}")
    print(f"Actor module device: {actor_dev}")
    print(f"Critic module device: {critic_dev}")

    # ----- Cehck train / test task -----
    if script_task == _STR_TEST:
        print(f"Starting testing enviroment: {task}")

        policy.eval()
        root = f"/home/ubuntu/Desktop/repo_rl/TITA-dynamic-obstacle-avoidance/TITA_MJ/log/{alg_type}_vectorized/weights"
        path_actor = "actor_state_dict_256_256_256_sac_2025_12_31_08_28_20_final.pt"   
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
            num_test_envs=num_test_envs,
        )

        return
    elif script_task == _STR_TRAIN:
        print(f"Starting training enviroment {task}")
    else:
        raise ValueError("Either --train or --test must be specified.")

    # ------ Buffer -------
    if alg_type == _STR_PPO:
        buffer = VectorReplayBuffer(
            total_size=1000*num_training_envs,
            buffer_num=num_training_envs,
        )
    elif alg_type == _STR_SAC:
        buffer = VectorReplayBuffer(
            total_size=1000*1000*num_training_envs,
            buffer_num=num_training_envs,
        )
    else:
        raise ValueError("Unsupported algorithm. Choose either 'ppo' or 'sac'.")
    
    # Create collectors (Collector works with DummyVectorEnv)
    train_collector = Collector(
        policy,
        training_envs,
        buffer,
        exploration_noise=False,
    )
    test_collector = Collector(
        policy,
        test_envs,
        exploration_noise=False,
    )
    
    # ----- Setup logger using LoggerFactoryDefault -----
    logger_factory = LoggerFactoryDefault()
    logger_factory.logger_type = "tensorboard"
    logger = logger_factory.create_logger(
        log_dir=logdir,
        experiment_name= alg_type + "_".join(map(str, hidden_sizes)),
        run_id=task + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    hsize_str = "_".join(map(str, hidden_sizes))
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    actor_path = os.path.join(logdir, f"weights/actor_state_dict_{hsize_str}_{alg_type}_{timestamp}.pt")
    critic_path = os.path.join(logdir, f"weights/critic_state_dict_{hsize_str}_{alg_type}_{timestamp}.pt")
    
    # ----- Create trainer and run training -----   
    online_trainer = OnPolicyTrainerParams(
            training_collector=train_collector, 
            test_collector=test_collector,  
            logger=logger,
            save_best_fn=partial(save_best, alg_type=alg_type, actor_path=actor_path, critic_path=critic_path),
            test_in_training=False,

            # Know parameters 
            max_epochs=10,   
            batch_size=254,

            # online training: total number of enviroment steps to collect before updated
            # offline training: total number of training step per epoch before update
            epoch_num_steps=100*num_training_envs, #*num_test_envs,   

            # Transition to collect at each collection step
            # before network update update 
            collection_step_num_env_steps=20*num_training_envs, 
            # Number of training at each epoch: epoch_num_steps / collection_step_num_env_steps

            # The number of times data are used
            # for gradient updates
            update_step_num_repetitions=2000,

            # Number of episodes to colleact in each test step
            # i.e. number of run for evaluation
            test_step_num_episodes=num_test_envs,
        )
    
    offline_trainer = OffPolicyTrainerParams(
            training_collector=train_collector, 
            test_collector=test_collector,  
            logger=logger,
            save_best_fn=partial(save_best, alg_type=alg_type, actor_path=actor_path, critic_path=critic_path),
            
            test_in_training=False,

            # Know parameters 
            max_epochs=10,   
            batch_size=2048,

            # Total number of training steps to take per epoch
            epoch_num_steps=10*num_training_envs, 

            # the number of environment steps/transitions to collect in each collection step before the
            # network update within each training step.
            collection_step_num_env_steps=10*num_training_envs,
            #collection_step_num_episodes=1*num_training_envs 
            
            # The number of times data 
            update_step_num_gradient_steps_per_sample=10,

            # Number of episodes to colleact in each test step
            # i.e. number of run for evaluation
            test_step_num_episodes=num_test_envs,
        )

    if alg_type == _STR_PPO:
        trainer_type = online_trainer
    elif alg_type == _STR_SAC:
        trainer_type = offline_trainer
    else:
        raise ValueError("Unsupported algorithm. Choose either 'ppo' or 'sac'.")

    try:
        result_policy = algo.run_training(
            trainer_type
        )
    except KeyboardInterrupt:
        pass

    print("\nTraining completed!")
    print(f"Logs saved to {logdir}")

    # ----- Save model weights -----
    print(f"Saving weights in {logdir}")
    try:
        os.makedirs(logdir, exist_ok=True)
        torch.save(actor.state_dict(), actor_path.replace(".pt", "_final.pt"))

        if alg_type == _STR_PPO:
            torch.save(critic.state_dict(), critic_path.replace(".pt", "_final.pt"))
        elif alg_type == _STR_SAC:
            torch.save(critic1.state_dict(), critic_path.replace(".pt", "_1_final.pt"))
            torch.save(critic2.state_dict(), critic_path.replace(".pt", "_2_final.pt"))

        print(f"Saved actor weights to {actor_path}")
        print(f"Saved critic weights to {critic_path}")
    except Exception as e:
        print("Could not save model weights:", e)

if __name__ == "__main__":
    main()
