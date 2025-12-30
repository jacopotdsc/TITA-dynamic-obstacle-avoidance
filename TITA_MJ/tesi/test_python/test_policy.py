import gymnasium as gym
import torch
import numpy as np
import os
from torch import nn
import time
import cv2
from torch.distributions import Distribution, Independent, Normal
from gymnasium.wrappers import RecordVideo

# Tianshou imports
from tianshou.data import Batch
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic
from tianshou.utils.space_info import SpaceInfo

def dist_fn(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
    loc, scale = loc_scale
    return Independent(Normal(loc, scale), 1)

def init_layer_orthogonal(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(m.bias, 0.0)

def init_last_layer(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=0.01)
            torch.nn.init.constant_(m.bias, 0.0)

def watch_agent_separate_weights(
        task_name: str,
        actor_path: str,
        critic_path: str,
        hidden_sizes: list = [64, 64],
        device: str = "cpu"
    ):

    # ----- Environment setup -----
    print(f"--- Setup Environment: {task_name} ---")
    render_mode = "human" #"rgb_array"
    env = gym.make(task_name, render_mode=render_mode)
    
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = SpaceInfo.from_env(env).action_info.max_action

    print(f"Obs: {state_shape}, Action: {action_shape}, Max Action: {max_action}")

    # ----- Video recording setup -----
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("videos", exist_ok=True)
    video_folder = os.path.join("videos", f"{task_name}_{timestamp}")
    
    if render_mode == "rbg_array":
        env = RecordVideo(
            env, 
            video_folder=video_folder,
            name_prefix="eval",
            episode_trigger=lambda episode_id: True 
        )
        print(f"Video recording enabled. Files will be saved in: {video_folder}")
    
    # ----- Network setup ----- 
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
        actor = actor.to(device)

        # ----- Load weights -----
        print(f"Loading Actor weights from: {actor_path}")
        if os.path.exists(actor_path):
            actor.load_state_dict(torch.load(actor_path, map_location=device))
        else:
            raise FileNotFoundError(f"File Actor non trovato: {actor_path}")
        
        policy = ProbabilisticActorPolicy(
            actor=actor,
            dist_fn=dist_fn,
            action_scaling=False,
            action_space=env.action_space,
            deterministic_eval=True, # If true, no randomness in eval mode for output
            )
        policy.eval()

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

        policy = ProbabilisticActorPolicy(
            actor=actor,
            dist_fn=dist_fn,
            action_scaling=False,
            action_space=env.action_space,
            deterministic_eval=True,
        )
        policy.eval()


    print(f"Loading Actor weights from: {actor_path}")
    if os.path.exists(actor_path):
        actor.load_state_dict(torch.load(actor_path, map_location=device))
    else:
        raise FileNotFoundError(f"File Actor non trovato: {actor_path}")
    
    # --- 4. LOOP DI RENDERING ---
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
            print(action)
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
        hidden_sizes=[256, 256],
        device="cpu"
    )