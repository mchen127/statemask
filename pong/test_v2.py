import os
import PIL
import gymnasium as gym
import torch
import base64
import imageio
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from torch.distributions import Categorical
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv
from model import CNN
from gymnasium.wrappers import RecordVideo

G_GAE = 0.99  # gamma parameter for GAE


class PongAgent:
    def __init__(
        self,
        env_id="ALE/Pong-v5",
        baseline_path=None,
        mask_path=None,
        record_video=False,
    ):
        # Initialize the environment and models
        with gym.make(env_id, render_mode="rgb_array") as env:
            if record_video:
                self.env = RecordVideo(
                    env, video_folder="./recording", episode_trigger=lambda x: True
                )
            else:
                self.env = env
            self.num_inputs = 1
            self.num_outputs = self.env.action_space.n
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load baseline model
        self.baseline_model = CNN(self.num_inputs, self.num_outputs, 256).to(
            self.device
        )
        self._load_model(self.baseline_model, baseline_path)

        # Load mask network
        self.mask_network = CNN(self.num_inputs, 2, 256).to(self.device)
        self._load_model(self.mask_network, mask_path)

        print("PongAgent initialized.")

    def _load_model(self, model, path):
        # Load model weights from a checkpoint if the path is provided
        if path is not None:
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint["state_dict"])
            print(f"Model loaded from {path}")

    def preprocess_state(self, state):
        # Preprocess the state by converting to grayscale, cropping, and resizing
        img = Image.fromarray(state)
        grey_img = img.convert(mode="L")  # Convert to grayscale
        cropped_img = grey_img.crop(
            (0, 34, 160, 194)
        )  # Crop the image to focus on the playing area
        resized_img = cropped_img.resize(
            (84, 84)
        )  # Resize to 84x84 for input consistency
        array_2d = np.asarray(resized_img)
        array_3d = np.expand_dims(array_2d, axis=0)  # Add channel dimension (C*H*W)
        # print("State preprocessed.")
        return array_3d

    def test_baseline(self, i_episode):
        # Test the baseline model on a single episode
        print(f"Starting baseline test for episode {i_episode}")
        state, _ = self.env.reset(
            seed=i_episode
        )  # Reset the environment with a specific seed
        state = self.preprocess_state(state)  # Preprocess the initial state
        total_discounted_reward = 0
        count = 0
        total_reward = 0
        done = False

        while not done:
            # Convert state to tensor and pass through the baseline model
            state_tensor = (
                torch.tensor(np.copy(state), dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            baseline_dist, _ = self.baseline_model(state_tensor)
            action = np.argmax(
                baseline_dist.probs.detach().cpu().numpy()[0]
            )  # Select action with highest probability
            # print(f"Episode {i_episode}, Step {count}: Action taken: {action}")
            count += 1
            next_state, reward, terminated, truncated, _ = self.env.step(
                action
            )  # Take action in the environment
            done = terminated or truncated

            # Calculate discounted reward based on the outcome
            if reward == -1:
                total_discounted_reward = -np.power(G_GAE, count)
                print(
                    f"Episode {i_episode}: Negative reward encountered. Ending episode."
                )
                break
            elif reward == 1:
                total_discounted_reward = np.power(G_GAE, count)
                total_reward += reward
                print(
                    f"Episode {i_episode}: Positive reward encountered. Ending episode."
                )
                break

            # Update state
            next_state = self.preprocess_state(next_state)
            state = next_state
            total_reward += reward

        # Set total_reward to 1 if won, otherwise 0
        total_reward = int(total_reward == 1)
        print(
            f"Episode {i_episode} finished. Total reward: {total_reward}, Total discounted reward: {total_discounted_reward}, Steps: {count}"
        )
        return total_reward, total_discounted_reward, count

    def test_mask(self, i_episode):
        # Test the mask network on a single episode
        print(f"Starting mask test for episode {i_episode}")
        state, _ = self.env.reset(
            seed=i_episode
        )  # Reset the environment with a specific seed
        state = self.preprocess_state(state)  # Preprocess the initial state
        count = 0
        num_mask = 0
        done = False
        total_reward = 0
        mask_pos = []
        action_seq = []
        mask_probs = []

        while not done:
            # Convert state to tensor and pass through both baseline and mask networks
            state_tensor = (
                torch.tensor(np.copy(state), dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            baseline_dist, _ = self.baseline_model(state_tensor)
            mask_dist, _ = self.mask_network(state_tensor)

            # Get actions from both networks
            baseline_action = np.argmax(baseline_dist.probs.detach().cpu().numpy()[0])
            mask_action = np.argmax(mask_dist.probs.detach().cpu().numpy()[0])
            mask_probs.append(mask_dist.probs.detach().cpu().numpy()[0])
            print(
                f"Episode {i_episode}, Step {count}: Baseline action: {baseline_action}, Mask action: {mask_action}"
            )

            # Use mask action to decide whether to take the baseline action or a random action
            if mask_action == 1:
                action = baseline_action
            else:
                action = np.random.choice(6)  # Take a random action if mask_action is 0
                num_mask += 1
                mask_pos.append(count)  # Track the positions where mask was used
                print(
                    f"Episode {i_episode}, Step {count}: Random action taken due to mask."
                )

            action_seq.append(action)  # Record the action sequence
            count += 1
            next_state, reward, terminated, truncated, _ = self.env.step(
                action
            )  # Take action in the environment
            done = terminated or truncated
            next_state = self.preprocess_state(next_state)  # Preprocess the next state
            state = next_state
            total_reward += reward

        # Set total_reward to 1 if won, otherwise 0
        total_reward = int(total_reward == 1)
        print(
            f"Episode {i_episode} finished. Total reward: {total_reward}, Steps: {count}, Num mask used: {num_mask}"
        )
        return total_reward, count, num_mask, mask_pos, action_seq, mask_probs


def main():
    env_id = "ALE/Pong-v5"
    baseline_path = "./ppo_test/baseline/Pong-v0_+0.896_12150.dat"
    mask_path = "./ppo_test/masknet/Pong-v0_+0.898_19660.dat"
    agent = PongAgent(env_id, baseline_path, mask_path, record_video=True)

    N_TESTS = 500
    tmp_rewards = []
    tmp_disc_rewards = []
    tmp_counts = []

    print("=====Test baseline model=====")
    # Test the baseline model over N_TESTS episodes
    for i in range(N_TESTS):
        total_reward, total_disc_reward, count = agent.test_baseline(i)
        tmp_rewards.append(total_reward)
        tmp_disc_rewards.append(total_disc_reward)
        tmp_counts.append(count)

    # Print average results for the baseline model
    print("Average winning rate: ", np.mean(tmp_rewards))
    print("Policy value: ", np.mean(tmp_disc_rewards))

    print("=====Test mask network=====")
    tmp_rewards = []
    # Test the mask network over N_TESTS episodes
    for i in range(N_TESTS):
        total_reward, count, num_mask, mask_pos, action_seq, mask_probs = (
            agent.test_mask(i)
        )
        tmp_rewards.append(total_reward)

    # Print average results for the mask network
    print("Average winning rate: ", np.mean(tmp_rewards))


if __name__ == "__main__":
    main()
