import os
import torch
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from baseline_model import PPO
import argparse
import json
from datetime import datetime
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
from stable_baselines3.common.atari_wrappers import NoopResetEnv

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


def test(model, env_id, num_episodes=500, save_results=None):
    # Record start time
    start_time = datetime.now()

    # Create environments for testing
    # envs = SyncVectorEnv([make_env(env_id) for _ in range(num_envs)])
    env = make_env(env_id)()

    win_cnt = 0
    lose_cnt = 0

    for episode in range(num_episodes):
        state, info = env.reset(seed=42)
        done = False
        step_cnt = 0
        episode_reward = 0

        while not done:
            step_cnt += 1
            state = torch.tensor(state, dtype=torch.float32)
            # Reorder dimensions: (batch_size, height, width, channels) -> (batch_size, channels, height, width)
            state_reorder = state.permute(2, 0, 1)  # Moves the last dimension (channels) to the second position
            print(f"Reordered shape for CNN: {state_reorder.shape}") 
            dist, value = model(state_reorder)
            action = dist.sample()
            next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
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
        print(f"Episode {episode}, Total Step: {step_cnt}, Result: {'win' if episode_reward == 1 else 'lose'}, Accumulated Win Rate: {win_cnt / (win_cnt + lose_cnt)}")

    avg_win_rate = win_cnt / (win_cnt + lose_cnt)
    print(f"Average win rate over {num_episodes} episodes: {avg_win_rate}")

    # Record end time
    end_time = datetime.now()

    results = {
        "env_id": args.env_id,
        "num_episodes": args.num_episodes,
        "avg_reward": avg_win_rate,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if args.save_results:
        # Ensure save directory exists
        os.makedirs(args.save_results, exist_ok=True)
        save_path = f"{args.save_results}/{start_time}-{avg_win_rate:.2f}.json"
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {save_path}")

    return results


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5", help="Gym environment ID")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of test episodes")
    parser.add_argument("--save_results", type=str, default="./results", help="File path to save the test results (JSON format)")

    # Parse arguments
    args = parser.parse_args()

    # Load model
    model = PPO(num_inputs=1, num_outputs=6, hidden_size=256)  # Adjust these based on your setup
    model.load_state_dict(torch.load(args.model_path)["state_dict"])
    model.eval()

    # Run the test
    test(model, env_id="ALE/Pong-v5", num_episodes=500, save_results=args.save_results)
