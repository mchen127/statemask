import os
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv
from model import CNN
from gymnasium.wrappers.record_video import RecordVideo

# Constants
G_GAE = 0.99  # gamma param for GAE
H_SIZE = 256
N_TESTS = 500
BASELINE_PATH = "./ppo_test/baseline/Pong-v0_+0.896_12150.dat"
# BASELINE_PATH = "./ppo_test/baseline/Pong-v0_+0.340_100.dat"
MASK_PATH = "./ppo_test/checkpoints/Pong-v0_+0.850_7200.dat"
RECORDING_DIR = "./recording"

# Utility Functions
def make_env():
    """
    Utility function for multiprocessed env.
    """
    def _thunk():
        env = gym.make("Pong-v0").env
        return env
    return _thunk

def grey_crop_resize(state):
    """
    Convert observation to grayscale, crop, and resize.
    """
    img = Image.fromarray(state)
    grey_img = img.convert(mode='L')
    cropped_img = grey_img.crop((0, 34, 160, 194))  # empirically chosen
    resized_img = cropped_img.resize((84, 84))
    array_2d = np.asarray(resized_img)
    array_3d = np.expand_dims(array_2d, axis=0)
    return array_3d  # C*H*W

def load_model(path, model, device):
    """
    Load model from checkpoint.
    """
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    return model

def test_episode(env, model, device, record_video=False, video_prefix="vid"): 
    """
    Test the model for one episode.
    """
    if record_video:
        env = RecordVideo(env, video_folder=RECORDING_DIR, name_prefix=video_prefix)
    state, _ = env.reset()
    state = grey_crop_resize(state)
    
    count, total_reward, total_discounted_reward = 0, 0, 0
    action_seq, mask_probs = [], []
    
    while True:
        if record_video:
            env.render()
        state_tensor = torch.FloatTensor(np.copy(state)).unsqueeze(0).to(device)
        dist, _ = model(state_tensor)
        action = np.argmax(dist.probs.detach().cpu().numpy()[0])
        action_seq.append(action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        count += 1
        total_reward += reward

        if reward == -1:
            total_discounted_reward = - np.power(G_GAE, count)
            break
        elif reward == 1:
            total_discounted_reward = np.power(G_GAE, count)
            break

        state = grey_crop_resize(next_state)
        if done:
            break

    return total_reward, total_discounted_reward, count, action_seq, mask_probs

def setup_environment():
    """
    Setup environment and models.
    """
    env = gym.make("Pong-v0").env
    num_inputs = 1
    num_outputs = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline_model = CNN(num_inputs, num_outputs, H_SIZE).to(device)
    load_model(BASELINE_PATH, baseline_model, device)

    mask_network = CNN(num_inputs, 2, H_SIZE).to(device)
    load_model(MASK_PATH, mask_network, device)

    return env, baseline_model, mask_network, device

def main():
    """
    Main function to test models.
    """
    env, baseline_model, mask_network, device = setup_environment()

    # Test baseline model
    tmp_rewards, tmp_disc_rewards = [], []
    print("=====Test baseline model=====")
    for i in range(N_TESTS):
        total_reward, total_disc_reward, count, _, _ = test_episode(env, baseline_model, device)
        tmp_rewards.append(total_reward)
        tmp_disc_rewards.append(total_disc_reward)
    print("Average winning rate: ", np.mean(tmp_rewards))
    print("Policy value: ", np.mean(tmp_disc_rewards))

    # Test mask network
    tmp_rewards = []
    print("=====Test mask network=====")
    for i in range(N_TESTS):
        total_reward, _, _, _, _ = test_episode(env, mask_network, device, record_video=True, video_prefix=f"vid_mask_{i}")
        tmp_rewards.append(total_reward)
    print("Average winning rate: ", np.mean(tmp_rewards))

    # Generating fid tests
    tmp_rewards, tmp_disc_rewards, tmp_counts = [], [], []
    print("=====Generating fid tests=====")
    for i in range(N_TESTS):
        total_reward, total_disc_reward, count, action_seq, mask_probs = test_episode(env, baseline_model, device)
        tmp_rewards.append(total_reward)
        tmp_counts.append(count)
        tmp_disc_rewards.append(total_disc_reward)

        np.savetxt(f"{RECORDING_DIR}/eps_len_{i}.out", [count])
        np.savetxt(f"{RECORDING_DIR}/act_seq_{i}.out", action_seq)
        np.savetxt(f"{RECORDING_DIR}/mask_probs_{i}.out", mask_probs)

    print("Average winning rate: ", np.mean(tmp_rewards))
    print("Policy value: ", np.mean(tmp_disc_rewards))
    np.savetxt(f"{RECORDING_DIR}/reward_record.out", tmp_rewards)

if __name__ == "__main__":
    main()