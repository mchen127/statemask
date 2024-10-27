import os
import torch
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from model import PPO
import argparse
import json
from datetime import datetime
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
from stable_baselines3.common.atari_wrappers import NoopResetEnv
from tqdm import tqdm


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


def test_baseline(baseline_model, env_id, num_episodes=500):
    # Record start time
    start_time = datetime.now()

    # Create environments for testing
    # envs = SyncVectorEnv([make_env(env_id) for _ in range(num_envs)])
    env = make_env(env_id)()

    win_cnt = 0
    lose_cnt = 0

    for episode in tqdm(range(num_episodes), desc="Testing episodes"):
        state, info = env.reset(seed=42)
        done = False
        step_cnt = 0
        episode_reward = 0

        while not done:
            step_cnt += 1
            state = torch.tensor(state, dtype=torch.float32)
            # Reorder dimensions: (batch_size, height, width, channels) -> (batch_size, channels, height, width)
            state_reorder = state.permute(2, 0, 1).unsqueeze(
                0
            )  # Moves the last dimension (channels) to the second position
            # print(f"Reordered shape for CNN: {state_reorder.shape}")
            dist, value = model(state_reorder)
            action = dist.sample()
            # print(action)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            if reward == 1:
                win_cnt += 1
                break
            elif reward == -1:
                lose_cnt += 1
                break

        # total_reward += episode_reward
        
        # tqdm.write(
        #     f"Episode {episode}, Total Step: {step_cnt}, Result: {'win' if episode_reward == 1 else 'lose'}, Accumulated Win Rate: {win_cnt / (win_cnt + lose_cnt)}"
        # )

    avg_win_rate = win_cnt / (win_cnt + lose_cnt)
    print(f"Average win rate over {num_episodes} episodes: {avg_win_rate}")

    # Record end time
    end_time = datetime.now()

    results = {
        "env_id": env_id,
        "num_episodes": num_episodes,
        "avg_win_rate": avg_win_rate,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return results


if __name__ == "__main__":
    start_time_ = datetime.now()
    formatted_start_time = start_time_.strftime("%Y-%m-%d_%H:%M:%S")
    
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_model_path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--env_id", type=str, default="ALE/Pong-v5", help="Gym environment ID"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=50, help="Number of test episodes"
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="./results/baseline",
        help="File path to save the test results (JSON format)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Load model
    baseline_model = PPO(
        num_inputs=1, num_outputs=6, hidden_size=256
    )  # Adjust these based on your setup
    baseline_model.load_state_dict(torch.load(args.baseline_model_path)["state_dict"])
    baseline_model.eval()

    # Run the test
    results = test_baseline(
        baseline_model=baseline_model,
        env_id=args.env_id,
        num_episodes=args.num_episodes,
        result_path=args.result_path,
    )

    if args.result_path:
        # Ensure save directory exists
        os.makedirs(args.result_path, exist_ok=True)
        save_path = (
            f"{args.result_path}/{formatted_start_time}.jsonl"
        )
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {save_path}")
