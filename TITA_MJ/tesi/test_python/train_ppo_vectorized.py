#!/usr/bin/env python3
"""
Simple PPO training script with vectorized environments.
Demonstrates procedural API usage for training on Pendulum-v1.
"""

import os
import datetime
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution, Independent, Normal



from tianshou.algorithm import TD3
from tianshou.algorithm.modelfree.ddpg import ContinuousDeterministicPolicy
from tianshou.algorithm import SAC
from tianshou.exploration import GaussianNoise

from tianshou.algorithm import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo

def dist_fn(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)

def main():

    # ----- Hard-coded configuration -----
    logdir = "log/ppo_vectorized"
    device = "cuda"
    task = "Tita-v0" #"Pendulum-v1"
    lr = 1e-3
    hidden_sizes = [256, 64, 256]
    num_training_envs = 8
    num_test_envs = 8

    if task == "Tita-v0":
        import sys
        sys.path.insert(0, '/home/ubuntu/miniconda3/envs/tianshou_gpu/lib/python3.12/site-packages')

        gym.register(
            id="Tita-v0",
            entry_point="gymnasium.envs.mujoco.tita_env:TitaEnv",
            max_episode_steps=1000,
        )
    
    print(f"Creating vectorized environments (task={task})...")
    training_envs = SubprocVectorEnv( [lambda: gym.make(task) for _ in range(num_training_envs)], )
    test_envs = SubprocVectorEnv([lambda: gym.make(task) for _ in range(num_test_envs)], )
    
    # ----- Get environment info ----- 
    env_single = gym.make(task)
    space_info = SpaceInfo.from_env(env_single)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    max_action = space_info.action_info.max_action

    print(f"Enviroment: {task}")
    print(f"Observation space: {state_shape}")
    print(f"Action space: {action_shape}")
    print(f"Action size: {max_action}")


    # ----- Network setup ----- 
    use_ppo = True

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
        actor = actor.to(device)
        
        critic = ContinuousCritic(
            preprocess_net=Net(
                state_shape=state_shape,
                hidden_sizes=hidden_sizes,
                activation=nn.Tanh,
            ),
            hidden_sizes=hidden_sizes,
        )
        critic = critic.to(device)
        
        policy = ProbabilisticActorPolicy(
            actor=actor,
            dist_fn=dist_fn,
            action_scaling=True,
            action_space=env_single.action_space,
            deterministic_eval=True, # If true, no randomness in eval mode for output
        )

        # ----- Create PPO algorithm -----
        optim = AdamOptimizerFactory(lr=lr)
        
        algo = PPO(
            policy=policy,
            critic=critic,
            optim=optim,
            eps_clip=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            gae_lambda=0.95,
            max_grad_norm=0.5,
            gamma=0.99,
        )
    else:

        net_a = Net(
            state_shape=state_shape,
            hidden_sizes=hidden_sizes,
        )
        actor = ContinuousActorDeterministic(
            preprocess_net=net_a,
            action_shape=action_shape,
            max_action=max_action,
        ).to(device)
        actor_optim = AdamOptimizerFactory(lr=lr)

        # critic network
        net_c1 = Net(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
        )
        net_c2 = Net(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
        )
        critic1 = ContinuousCritic(preprocess_net=net_c1).to(device)
        critic1_optim = AdamOptimizerFactory(lr=lr)
        critic2 = ContinuousCritic(preprocess_net=net_c2).to(device)
        critic2_optim = AdamOptimizerFactory(lr=lr)

        policy = ContinuousDeterministicPolicy(
            actor=actor,
            exploration_noise=GaussianNoise(sigma=0.2),
            action_space=env_single.action_space,
        )

        algo = TD3BC(
            policy=policy,
            policy_optim=actor_optim,
            critic=critic1,
            critic_optim=critic1_optim,
            critic2=critic2,
            critic2_optim=critic2_optim,
            tau=tau,
            gamma=gamma,
            policy_noise=policy_noise,
            update_actor_freq=update_actor_freq,
            noise_clip=noise_clip,
            alpha=alpha,
            n_step_return_horizon=n_step,
        )

    # Print device information for debugging ------
    try:
        actor_param = next(actor.parameters())
        actor_dev = actor_param.device
        critic_param = next(critic.parameters())
        critic_dev = critic_param.device
    except StopIteration:
        actor_dev = torch.device(device)
        critic_dev = torch.device(device)

    print(f"Device chosen for training/models: {device}")
    print(f"Actor module device: {actor_dev}")
    print(f"Critic module device: {critic_dev}")


    buffer = VectorReplayBuffer(
        total_size=1000*8,
        buffer_num=num_training_envs,
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
            max_epochs=1,   
            batch_size=256,

            # online training: total number of enviroment steps to collect before updated
            # offline training: total number of training step per epoch before update
            epoch_num_steps=100*1,   

            # Transition to collect at each collection step
            # before network update update 
            collection_step_num_env_steps=100*1, 
            # Number of training at each epoch: epoch_num_steps / collection_step_num_env_steps

            # The number of times data are used
            # for gradient updates
            update_step_num_repetitions=1,

            # Number of episodes to colleact in each test step
            # i.e. number of run for evaluation
            test_step_num_episodes=num_test_envs,
        )
    
    offline_trainer = None  # Not used in on-policy training
    trainer_type = online_trainer

    try:
        result_online_policy = algo.run_training(
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

    # ----- Rendering  -----
    try:
        from tianshou.data import Batch
        import imageio
        import matplotlib.pyplot as plt

        print("Saving GIF ")
        render_env = gym.make(task, render_mode="rgb_array")
        policy.eval()
        epi = 0
        saved_gifs = []
        frames = []

        obs, _ = render_env.reset()

        while True:
            try:
                # try several policy call styles for compatibility
                batch = Batch(obs=np.expand_dims(obs, 0))
                out = policy(batch)
                action = out.act
                # unwrap batch-size-1
                if isinstance(action, (list, tuple)):
                    action = action[0]
                if hasattr(action, "shape") and action.shape[0] == 1:
                    action = action[0]
            except Exception:
                try:
                    out = policy.forward(batch)
                    action = out.act
                except Exception:
                    action = render_env.action_space.sample()

            obs, reward, terminated, truncated, info = render_env.step(action)
            frame = render_env.render()
            if frame is not None:
                frames.append(frame)
            done = terminated or truncated
            if done:
                break

        if frames:
            gif_path = os.path.join(logdir, f"policy_playback_ep{epi}.gif")
            try:
                imageio.mimsave(gif_path, frames, fps=30)
                saved_gifs.append(gif_path)
                print(f"Saved playback GIF: {gif_path}")
            except Exception:
                # fallback: show first frame inline (if possible)
                plt.figure(figsize=(6, 6))
                plt.imshow(frames[0])
                plt.axis("off")
                plt.show()
        else:
            print("No frames recorded for episode")

        if not saved_gifs:
            print("No GIFs saved (imageio may be missing or render returned None).")
    except Exception as e:
        print("Could not run render/playback after training:", e)

if __name__ == "__main__":
    main()
