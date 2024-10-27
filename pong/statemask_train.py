import os
import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
from stable_baselines3.common.atari_wrappers import NoopResetEnv
import argparse
import time
from datetime import datetime
from model import PPO
from statemask_test import test_statemask
from tqdm import tqdm


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="PPO training hyperparameters")

    # Add arguments with default values
    parser.add_argument(
        "--mode",
        type=str,
        default="default_train",
        help="Train with or without validation.",
    )
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5", help="env-id")
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=2_000_000,
        help="Total number of timesteps",
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
    parser.add_argument(
        "--update_epochs", type=int, default=4, help="the K epochs to update the policy"
    )
    parser.add_argument(
        "--norm_adv",
        type=lambda x: bool(str2bool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        "--clip_coef",
        type=float,
        default=0.1,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--clip_vloss",
        type=lambda x: bool(str2bool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
    )
    parser.add_argument(
        "--ent_coef", type=float, default=0.01, help="coefficient of the entropy"
    )
    parser.add_argument(
        "--vf_coef", type=float, default=0.5, help="coefficient of the value function"
    )

    # Add lasso_coef and learning_rate_lagrange_multiplier
    # ================================================================================================================
    parser.add_argument(
        "--lasso_coef",
        type=float,
        default=0.0001,
        help="coefficient of the Lasso regularization",
    )
    parser.add_argument(
        "--learning_rate_lagrange_multiplier",
        type=float,
        default=0.001,
        help="coefficient of the Lasso regularization",
    )
    parser.add_argument(
        "--baseline_exp_total_reward",
        type=float,
        default=0.001,
        help="coefficient of the Lasso regularization",
    )
    # ================================================================================================================

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--target_kl",
        type=float,
        default=None,
        help="the target KL divergence threshold",
    )

    parser.add_argument(
        "--result_path",
        type=str,
        default="./results",
        help="File path to save the test results (JSON format)",
    )
    parser.add_argument(
        "--baseline_model_path",
        type=str,
        default="./checkpoints/baseline/baseline_2024-10-27_12:57:15_0.036.pth",
        help="File path to save the model",
    )
    parser.add_argument(
        "--statemask_model_path",
        type=str,
        default="./checkpoints/statemask",
        help="File path to save the model",
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


def append_json_to_file(file_path, new_data):
    with open(file_path, "a") as f:  # Open the file in append mode
        json.dump(new_data, f)
        f.write("\n")  # Write a newline character to separate JSON objects


def train(args):
    start_time = datetime.now()
    formatted_start_time = start_time.strftime("%Y-%m-%d_%H:%M:%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create environments
    print(args.num_envs)
    envs = SyncVectorEnv([make_env(args.env_id) for _ in range(args.num_envs)])
    print(envs.num_envs)
    num_inputs = envs.single_observation_space.shape[2]
    num_outputs = envs.single_action_space.n

    # Initialize PPO baseline agent
    checkpoint_path = args.baseline_model_path
    baseline_model = PPO(num_inputs=num_inputs, num_outputs=num_outputs).to(device)
    baseline_model.load_state_dict(torch.load(checkpoint_path))
    baseline_model.eval()

    # Initialize satemask agent
    statemask_model = PPO(num_inputs=num_inputs, num_outputs=2).to(device)
    optimizer = optim.Adam(statemask_model.parameters(), lr=args.learning_rate)

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
    num_batches = args.total_timesteps // args.batch_size

    validation_results = []

    # Lagrange Multiplier
    lagrange_multiplier = 0

    # Main training loop
    for batch_cnt in tqdm(range(num_batches), desc="Training Batch"):
        # print(f"Update {batch_cnt}")

        # =============================
        # ||         Rollout          ||
        # =============================
        for step in tqdm(range(args.num_steps), desc="Generating Steps"):
            # print(f"Step {step}")
            global_steps += args.num_envs
            obs[step] = next_obs
            # print("next_dones:")
            # print(next_dones)
            # print(next_dones.shape)
            # print("dones:")
            # print(dones)
            # print(dones.shape)
            assert (
                next_dones.shape == dones[step].shape
            ), "next_dones and dones[step] shape inconsistent"
            dones[step] = next_dones

            with torch.no_grad():
                # Reorder dimensions: (batch_size, height, width, channels) -> (batch_size, channels, height, width)
                next_obs_reorder = next_obs.permute(0, 3, 1, 2)
                # print(f"Reordered shape for CNN: {next_obs_reorder.shape}")  # Should print (8, 1, 84, 84)
                baseline_dist, baseline_value = baseline_model(next_obs_reorder)
                # values[step] = baseline_value.flatten()
                statemask_dist, statemask_value = statemask_model(next_obs_reorder)

            # get actions and log probabilities
            baseline_action = baseline_dist.sample()
            baseline_action_copy = baseline_action.cpu().numpy()
            statemask_action = statemask_dist.sample()
            statemask_action_copy = statemask_action.cpu().numpy()

            actions[step] = statemask_action
            logprob = statemask_dist.log_prob(statemask_action)
            logprobs[step] = logprob

            # Determine actual action
            real_action = []
            for i in range(args.num_envs):
                real_action.append(
                    baseline_action_copy[i]
                    if statemask_action_copy[i] == 0
                    else np.random.choice(num_outputs)
                )

            # perform next step
            next_obs, reward, terminates, truncates, infos = envs.step(real_action)
            # print(terminates)
            done = torch.tensor(
                list(map(lambda x, y: int(x or y), terminates, truncates))
            )
            assert (
                done.shape == terminates.shape and done.shape == truncates.shape
            ), "done has wrong shape"
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_dones = torch.Tensor(done).to(device)
        # done generating trajectories

        # compute advantages
        with torch.no_grad():
            # bootstrap value if not done
            next_obs_reorder = next_obs.permute(0, 3, 1, 2)
            next_values_of_T = baseline_model.get_value(next_obs_reorder).reshape(1, -1)
            # compute advantages with current critic
            estimated_returns = torch.zeros_like(rewards).to(device)
            advantages = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                next_terminates = (
                    dones[t + 1] if (t < args.num_steps - 1) else next_dones
                )  # capture termination before T
                next_values = (
                    values[t + 1] if (t < args.num_steps - 1) else next_values_of_T
                )
                delta = (
                    rewards[t]
                    + args.gamma * next_values * (1 - next_terminates)
                    - values[t]
                )
                advantages[t] = delta + args.gamma * args.gae_lambda * (
                    1 - next_terminates
                ) * (advantages[t + 1] if (t < args.num_steps - 1) else 0)
                estimated_returns[t] = rewards[t] + args.gamma * (
                    1 - next_terminates
                ) * (estimated_returns[t + 1] if (t < args.num_steps - 1) else 0)

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_estimated_returns = estimated_returns.reshape(-1)
        b_values = values.reshape(-1)

        # =============================
        # ||   Update for 4 epochs   ||
        # =============================
        batch_indices = np.arange(args.batch_size)
        loss_buffer = []
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(batch_indices)
            # Update
            for begin in range(0, args.batch_size, args.minibatch_size):
                end = begin + args.minibatch_size
                minibatch_indices = batch_indices[begin:end]
                # Get new dist, value, action, log probabilities
                minibatch_obs_reorder = b_obs[minibatch_indices].permute(0, 3, 1, 2)
                new_dists, new_values = statemask_model(minibatch_obs_reorder)
                new_actions = new_dists.sample()
                no_mask_actions = torch.zeros_like(new_actions)
                no_mask_logprobs = new_dists.log_prob(no_mask_actions)

                new_logprobs = new_dists.log_prob(new_actions)
                logratio = new_logprobs - b_logprobs[minibatch_indices]
                importance_sampling_ratio = (logratio).exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((importance_sampling_ratio - 1.0).abs() > args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                # Normalize advantages
                unnorm_minibatch_advantages = b_advantages[minibatch_indices]
                minibatch_advantages = b_advantages[minibatch_indices]
                if args.norm_adv:
                    minibatch_advantages = (
                        minibatch_advantages - minibatch_advantages.mean()
                    ) / (minibatch_advantages.std() + 1e-8)

                # Policy loss
                actor_loss1 = -minibatch_advantages * importance_sampling_ratio
                actor_loss2 = -minibatch_advantages * torch.clamp(
                    importance_sampling_ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                actor_loss = torch.max(actor_loss1, actor_loss2).mean()

                # Value loss
                new_values = new_values.view(-1)
                if args.clip_vloss:
                    critic_loss_unclipped = (
                        new_values - b_estimated_returns[minibatch_indices]
                    ) ** 2
                    value_clipped = b_values[minibatch_indices] + torch.clamp(
                        new_values - b_values[minibatch_indices],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    critic_loss_clipped = (
                        value_clipped - b_estimated_returns[minibatch_indices]
                    ) ** 2
                    critic_loss_max = torch.max(
                        critic_loss_unclipped, critic_loss_clipped
                    )
                    critic_loss = 0.5 * critic_loss_max.mean()
                else:
                    critic_loss = (
                        0.5
                        * (
                            (new_values - b_estimated_returns[minibatch_indices]) ** 2
                        ).mean()
                    )

                # Entropy bonus
                entropy = new_dists.entropy()
                entropy_loss = entropy.mean()

                # Number of statemasks
                num_masks = torch.sum(no_mask_logprobs.exp()) / (args.minibatch_size)

                # Total loss
                if lagrange_multiplier > 1:
                    actor_loss *= -1

                loss = (
                    actor_loss
                    + args.vf_coef * critic_loss
                    - args.ent_coef * entropy_loss
                    + args.lasso_coef * num_masks
                )

                # Print the losses
                print(
                    f"Batch {batch_cnt}, Epoch {epoch}, Actor Loss: {actor_loss.item():.4f}, Crtitc Loss: {critic_loss.item():.4f}, Entropy Loss: {entropy_loss.item():.4f}, Num masks: {num_masks.item():.4f}, Total Loss: {loss.item():.4f}"
                )

                # Perform update
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    statemask_model.parameters(), args.max_grad_norm
                )
                optimizer.step()

                #
                loss_buffer.append(unnorm_minibatch_advantages.cpu().detach().numpy())

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

            # validation
            if args.mode == "default_train":
                # Call test function for validation
                validation_results = test_statemask(
                    baseline_model=baseline_model,
                    statemask_model=statemask_model,
                    env_id=args.env_id,
                    num_episodes=50,
                )
                os.makedirs(args.result_path, exist_ok=True)
                save_path = f"{args.result_path}/{formatted_start_time}.jsonl"
                append_json_to_file(save_path, validation_results)
                print(
                    f"Validation results after batch {batch_cnt}, epoch {epoch}: {validation_results}"
                )

        lagrange_multiplier -= args.learning_rate_lagrange_multiplier * (
            2 * args.baseline_exp_total_reward
            + np.mean(loss_buffer)
            - np.mean(estimated_returns.cpu().numpy())
        )
        lagrange_multiplier = max(0, lagrange_multiplier)

        print(f"Batch {batch_cnt} completed.")

    evaluation_results = test_statemask(
        baseline_model=baseline_model,
        statemask_model=statemask_model,
        env_id=args.env_id,
        num_episodes=500,
    )

    # Save evaluation results
    os.makedirs(args.result_path, exist_ok=True)
    result_save_path = f"{args.result_path}/{formatted_start_time}.jsonl"
    append_json_to_file(result_save_path, evaluation_results)
    print(
        f"Evaluation results after batch {batch_cnt}, epoch {epoch}: {evaluation_results}"
    )
    # Save PPO baseline model checkpoint
    os.makedirs(args.statemask_model_path, exist_ok=True)
    model_save_path = f"{args.statemask_model_path}/baseline_{formatted_start_time}_{evaluation_results['avg_win_rate']}.pth"
    torch.save(statemask_model.state_dict(), model_save_path)
    print("Training completed and model saved.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
