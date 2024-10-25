import os
import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
from stable_baselines3.common.atari_wrappers import NoopResetEnv
import argparse
import time
from baseline_model import PPO
from test import test


def parse_args():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="PPO training hyperparameters")

    # Add arguments with default values
    parser.add_argument("--mode", type=str, default="default_train", help="Train with or without validation.")
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5", help="env-id")
    parser.add_argument(
        "--total_timesteps", type=int, default=1000000, help="Total number of timesteps"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=128,
        help="Number of steps per environment per update",
    )
    parser.add_argument(
        "--num_envs", type=int, default=8, help="Number of parallel environments"
    )
    parser.add_argument(
        "--num_minibatches", type=int, default=4, help="Number of minibatches"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for future rewards"
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="Lambda for Generalized Advantage Estimation",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2.5e-4,
        help="the learning rate of the optimizer",
    )

    # Parse arguments
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    return args


# Initialize the environment for training
def make_env(env_id):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = NoopResetEnv(env)
        env = ResizeObservation(env, (84, 84))  # Resize to 84x84
        env = GrayScaleObservation(
            env, keep_dim=True
        )  # Convert to grayscale and keep the channel dimension
        return env

    return thunk


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create environments
    print(args.num_envs)
    envs = SyncVectorEnv([make_env(args.env_id) for _ in range(args.num_envs)])
    print(envs.num_envs)
    num_inputs = envs.single_observation_space.shape[2]
    num_outputs = envs.single_action_space.n

    # Initialize PPO model
    model = PPO(num_inputs=num_inputs, num_outputs=num_outputs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_steps = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_dones = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # Main training loop
    for update in range(num_updates):
        print(f"Update {update}")
        # rollout
        for step in range(args.num_steps):
            print(f"Step {step}")
            global_steps += args.num_envs
            obs[step] = next_obs
            dones[step] = next_dones
            
            with torch.no_grad():
                # Reorder dimensions: (batch_size, height, width, channels) -> (batch_size, channels, height, width)
                next_obs_reorder = next_obs.permute(0, 3, 1, 2)  # Moves the last dimension (channels) to the second position
                # print(f"Reordered shape for CNN: {next_obs_reorder.shape}")  # Should print (8, 1, 84, 84)
                dist, value = model(next_obs_reorder)
                # print("dist", dist)
                # print("value", value)
                values[step] = value.flatten()
            action = dist.sample()
            logprob = dist.log_prob(action)
            actions[step] = action
            logprobs[step] = logprob
        # update for 4 epoches

        # update

        # validate

        if args.mode == "default_train":
            # Call test function for validation
            test_results = test(model, env_id=args.env_id, num_episodes=50, save_results="./results")
            print(f"Validation results after episode {update}: {test_results}")

        print(f"Episode {update} completed.")

    torch.save(model.state_dict(), "ppo_baseline.pth")
    print("Training completed and model saved.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
