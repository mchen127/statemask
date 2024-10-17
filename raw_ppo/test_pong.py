import argparse
import torch
import gymnasium as gym
from policy import PolicyNetwork
import os

def test(args):
    # Load Pong environment with video recording enabled
    env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, args.video_dir, episode_trigger=lambda e: True)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load policy network
    policy_net = PolicyNetwork(obs_dim, action_dim)
    policy_net.load_state_dict(torch.load(args.model_path))
    policy_net.eval()

    observation, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        state = torch.tensor(observation, dtype=torch.float32)
        action_dist = policy_net(state)
        action = action_dist.sample()

        # Step in the environment
        observation, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward

        # Check if episode is finished
        done = terminated or truncated

    print(f"Total reward for this episode: {total_reward}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained PPO model on Pong')
    parser.add_argument('--model_path', type=str, default='results/pong_model.pth', help='Path to the trained model')
    parser.add_argument('--video_dir', type=str, default='results/video', help='Directory to save the gameplay video')
    
    args = parser.parse_args()
    os.makedirs(args.video_dir, exist_ok=True)
    test(args)
