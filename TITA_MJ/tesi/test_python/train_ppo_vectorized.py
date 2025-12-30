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
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo

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

    # ----- Hard-coded configuration -----
    logdir = "log/ppo_vectorized"
    device = "cuda"
    task = "Tita-v0" #"Pendulum-v1"
    lr = 0.000001
    hidden_sizes = [256, 256]
    num_training_envs = 16
    num_test_envs = 1

    if task == "Tita-v0":
        import sys
        sys.path.insert(0, '/home/ubuntu/miniconda3/envs/tianshou_gpu/lib/python3.12/site-packages')

        gym.register(
            id="Tita-v0",
            entry_point="gymnasium.envs.mujoco.tita_env:TitaEnv",
            max_episode_steps=1000,
        )
    
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

    use_ppo = False

    if use_ppo == True:
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
        
        critic = ContinuousCritic(
            preprocess_net=Net(
                state_shape=state_shape,
                action_shape=action_shape,
                concat=True,
                hidden_sizes=hidden_sizes,
                activation=nn.Tanh,
            ),
            hidden_sizes=hidden_sizes,
        )
        #critic.apply(init_layer_orthogonal)
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
    else:

        net_a = Net(
            state_shape=state_shape,
            hidden_sizes=hidden_sizes,
            activation=nn.Tanh,
        )

        actor = ContinuousActorProbabilistic(
            preprocess_net=net_a,
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

        # DEBUG: inspect network dimensions to catch shape mismatches
        try:
            print("[DEBUG] state_shape:", state_shape)
            print("[DEBUG] action_shape:", action_shape)
            print("[DEBUG] net_a output dim:", net_a.get_output_dim())
            print("[DEBUG] actor preprocess output dim:", actor.get_preprocess_net().get_output_dim())
            import torch as _torch
            _x = _torch.zeros(1, *state_shape)
            _logits, _ = actor.get_preprocess_net()(_x)
            print("[DEBUG] preprocess logits shape:", tuple(_logits.shape))
            # mu is an MLP; inspect its final linear weight if available
            try:
                _w = actor.mu.model[-1].weight
                print("[DEBUG] actor.mu final weight shape:", tuple(_w.shape))
            except Exception:
                try:
                    print("[DEBUG] actor.mu weight shape:", tuple(actor.mu.weight.shape))
                except Exception:
                    pass
        except Exception as _e:
            print("[DEBUG] could not run actor shape debug:", _e)

        actor_optim = AdamOptimizerFactory(lr=lr)

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

        # DEBUG: inspect critic1 preprocess and last MLP layer shapes
        try:
            pre = critic1.preprocess
            print("[DEBUG] critic1.preprocess.get_output_dim():", pre.get_output_dim())
            try:
                print("[DEBUG] critic1.preprocess.model:", pre.model)
                for i, m in enumerate(pre.model):
                    if hasattr(m, 'weight'):
                        print(f"[DEBUG] pre.model[{i}] weight shape:", tuple(m.weight.shape))
            except Exception:
                pass
            try:
                print("[DEBUG] critic1.last.model:", critic1.last.model)
                for i, m in enumerate(critic1.last.model):
                    if hasattr(m, 'weight'):
                        print(f"[DEBUG] last.model[{i}] weight shape:", tuple(m.weight.shape))
            except Exception:
                pass
        except Exception as _e:
            print("[DEBUG] could not inspect critic1 nets:", _e)

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

        # DEBUG: inspect critic2 preprocess and last MLP layer shapes
        try:
            pre = critic2.preprocess
            print("[DEBUG] critic2.preprocess.get_output_dim():", pre.get_output_dim())
            try:
                print("[DEBUG] critic2.preprocess.model:", pre.model)
                for i, m in enumerate(pre.model):
                    if hasattr(m, 'weight'):
                        print(f"[DEBUG] pre2.model[{i}] weight shape:", tuple(m.weight.shape))
            except Exception:
                pass
            try:
                print("[DEBUG] critic2.last.model:", critic2.last.model)
                for i, m in enumerate(critic2.last.model):
                    if hasattr(m, 'weight'):
                        print(f"[DEBUG] last2.model[{i}] weight shape:", tuple(m.weight.shape))
            except Exception:
                pass
        except Exception as _e:
            print("[DEBUG] could not inspect critic2 nets:", _e)

        critic1_optim = AdamOptimizerFactory(lr=lr)
        critic2_optim = AdamOptimizerFactory(lr=lr)

        policy = SACPolicy(
            actor=actor,
            exploration_noise=None,
            deterministic_eval=True,
            action_scaling=False,
            action_space=env_single.action_space,
        )

        algo = SAC(
            policy=policy,
            policy_optim=actor_optim,
            critic=critic1,
            critic_optim=critic1_optim,
            critic2=critic2,
            critic2_optim=critic2_optim,
            tau=0.005,
            gamma=0.99,
            alpha=0.1,
            n_step_return_horizon=2,
        )

    # Print device information for debugging ------
    try:
        actor_param = next(actor.parameters())
        actor_dev = actor_param.device
        if use_ppo == True:
            critic_param = next(critic.parameters())
            critic_dev = critic_param.device
        else:
            critic_param = next(critic1.parameters())
            critic_dev = critic_param.device
    except StopIteration:
        actor_dev = torch.device(device)
        if use_ppo == True:
            critic_dev = torch.device(device)
        else:
            critic_dev = torch.device(device)

    print(f"Device chosen for training/models: {device}")
    print(f"Actor module device: {actor_dev}")
    print(f"Critic module device: {critic_dev}")


    if use_ppo == True:
        buffer = VectorReplayBuffer(
            total_size=1000*num_training_envs,
            buffer_num=num_training_envs,
        )
    else:
        buffer = VectorReplayBuffer(
            total_size=1000,
            buffer_num=2*num_training_envs,
        )
    
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
        experiment_name="ppo_vectorized",
        run_id="pendulum"
    )
    
    # ----- Create trainer and run training -----   
    online_trainer = OnPolicyTrainerParams(
            training_collector=train_collector, 
            test_collector=test_collector,  
            logger=logger,
            test_in_training=False,

            # Know parameters 
            max_epochs=100,   
            batch_size=254,

            # online training: total number of enviroment steps to collect before updated
            # offline training: total number of training step per epoch before update
            epoch_num_steps=1000*num_training_envs, #*num_test_envs,   

            # Transition to collect at each collection step
            # before network update update 
            collection_step_num_env_steps=1000*num_training_envs, 
            # Number of training at each epoch: epoch_num_steps / collection_step_num_env_steps

            # The number of times data are used
            # for gradient updates
            update_step_num_repetitions=10,

            # Number of episodes to colleact in each test step
            # i.e. number of run for evaluation
            test_step_num_episodes=num_test_envs,
        )
    
    offline_trainer = OffPolicyTrainerParams(
            training_collector=train_collector, 
            test_collector=test_collector,  
            logger=logger,
            test_in_training=False,

            # Know parameters 
            max_epochs=100,   
            batch_size=254,

            # online training: total number of enviroment steps to collect before updated
            # offline training: total number of training step per epoch before update
            epoch_num_steps=10*num_training_envs, #*num_test_envs,   

            # Transition to collect at each collection step
            # before network update update 
            collection_step_num_env_steps=10*num_training_envs, 
            # Number of training at each epoch: epoch_num_steps / collection_step_num_env_steps

            update_step_num_gradient_steps_per_sample=10,

            # Number of episodes to colleact in each test step
            # i.e. number of run for evaluation
            test_step_num_episodes=num_test_envs,
        )

    if use_ppo == True:
        trainer_type = online_trainer
    else:
        trainer_type = offline_trainer

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
        actor_path = os.path.join(logdir, "actor_state_dict.pt")
        critic_path = os.path.join(logdir, "critic_state_dict.pt")
        torch.save(actor.state_dict(), actor_path)
        torch.save(critic.state_dict(), critic_path)
        print(f"Saved actor weights to {actor_path}")
        print(f"Saved critic weights to {critic_path}")
    except Exception as e:
        print("Could not save model weights:", e)

if __name__ == "__main__":
    main()
