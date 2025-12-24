import gymnasium as gym
import torch
import numpy as np
import os
from torch import nn
from torch.distributions import Distribution, Independent, Normal

# Tianshou imports
from tianshou.data import Batch
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo

def dist_fn(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)

def watch_agent_separate_weights(
    task_name: str,
    actor_path: str,
    critic_path: str,
    hidden_sizes: list = [64, 64],
    device: str = "cpu"
):
    print(f"--- Setup Environment: {task_name} ---")
    env = gym.make(task_name, render_mode="human")
    
    # Parametri ambiente
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = SpaceInfo.from_env(env).action_info.max_action

    print(f"Obs: {state_shape}, Action: {action_shape}, Max Action: {max_action}")
    
    # ----- Network setup ----- 
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

    # --- 2. CARICAMENTO PESI SEPARATI ---
    
    print(f"Loading Actor weights from: {actor_path}")
    if os.path.exists(actor_path):
        actor.load_state_dict(torch.load(actor_path, map_location=device))
    else:
        raise FileNotFoundError(f"File Actor non trovato: {actor_path}")

    print(f"Loading Critic weights from: {critic_path}")
    if os.path.exists(critic_path):
        critic.load_state_dict(torch.load(critic_path, map_location=device))
    else:
        print("Warning: File Critic non trovato. Procedo senza (per il render basta l'actor).")

    #dist_fn =torch.distributions.Normal
    policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist_fn,
        action_scaling=True,
        action_space=env.action_space,
        deterministic_eval=True, # If true, no randomness in eval mode for output
        )
    policy.eval()

    # --- 4. LOOP DI RENDERING ---
    try:
        obs, info = env.reset()
        total_reward = 0
        
        while True:
            # Creazione Batch (1, obs_dim)
            batch = Batch(obs=np.array([obs]), info={})
            
            with torch.no_grad():
                result = policy(batch)
            
            # Se result.act è un tensore, lo convertiamo; se è numpy, lo usiamo direttamente
            action = np.array([0.0]*8)#result.act[0]
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(f"Episode terminated, total reward: {total_reward:.2f}")
                break

                
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

# --- ESECUZIONE ---
if __name__ == "__main__":
    # Aggiorna i percorsi con i tuoi file reali
    root = "/home/ubuntu/Desktop/repo_rl/TITA-dynamic-obstacle-avoidance/TITA_MJ/log/ppo_vectorized/"
    path_actor = "actor_state_dict.pt"   
    path_critic = "critic_state_dict.pt"

    import sys
    sys.path.insert(0, '/home/ubuntu/miniconda3/envs/tianshou_gpu/lib/python3.12/site-packages')

    gym.register(
        id="Tita-v0",
        entry_point="gymnasium.envs.mujoco.tita_env:TitaEnv",
        max_episode_steps=1000,
    ) 

    watch_agent_separate_weights(
        task_name="Tita-v0", #"Pendulum-v1",
        actor_path=root + path_actor,
        critic_path=root + path_critic,
        hidden_sizes=[256, 64, 256], # Assicurati che siano quelli usati nel training!
        device="cpu"
    )