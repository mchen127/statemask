import gymnasium as gym
import numpy as np
import time
import argparse
from gymnasium.wrappers import RecordVideo

# Define the actions
actions = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
    4: "RIGHTFIRE",
    5: "LEFTFIRE"
}

def demo_actions():
    # Loop through each action to visualize
    for action, action_name in actions.items():
        print(f"[DEBUG] Creating environment for action: {action_name} ({action})")
        # Create the Pong environment for each action
        env = gym.make("ALE/Pong-v5", render_mode='rgb_array')
        env = RecordVideo(env, video_folder=f'./pong_action_videos/{action_name}', disable_logger=True)
        
        print(f"[DEBUG] Environment created. Executing action: {action_name} ({action})")
        observation, info = env.reset()
        print(f"[DEBUG] Environment reset. Initial observation shape: {observation.shape}")
        
        
        # Run the action for a few frames to see its effect
        terminated = False
        truncated = False
        for frame in range(30):
            if not (terminated or truncated):
                # env.render()
                print(f"[DEBUG] Step {frame + 1}: Taking action {action_name} ({action})")
                observation, reward, terminated, truncated, info = env.step(action)
                print(f"[DEBUG] Step {frame + 1}: Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
                time.sleep(0.05)  # Add a short delay to visualize clearly
            else:
                print(f"[DEBUG] Episode ended at step {frame + 1}.")
                break
        
        # Close the environment
        env.close()
        print(f"[DEBUG] Environment closed for action: {action_name} ({action})")
    
    print("Recorded videos can be found in the './pong_action_videos' folder.")

def demo_state():
    # Create the Pong environment to visualize different states
    print("[DEBUG] Creating environment for state demonstration.")
    env = gym.make("ALE/Pong-v5", render_mode='rgb_array')
    observation, info = env.reset()
    print(f"[DEBUG] Environment reset. Initial observation shape: {observation.shape}")
    
    
    print("Demonstrating different states of the environment.")
    
    # Run a random policy for a while to showcase different states
    terminated = False
    truncated = False
    for frame in range(5):
        if not (terminated or truncated):
            # env.render()
            action = env.action_space.sample()  # Take random actions to demonstrate different states
            print(f"[DEBUG] Step {frame + 1}: Taking random action {action}")
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"[DEBUG] Step {frame + 1}: Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            # if frame == 0 or frame == 250:
            print(f"[DEBUG] Observation at step {frame + 1}: {observation}")
            time.sleep(0.05)  # Add a short delay to visualize clearly
        else:
            print(f"[DEBUG] Episode ended at step {frame + 1}. Resetting environment.")
            observation, info = env.reset()
            print(f"[DEBUG] Environment reset. New observation shape: {observation.shape}")
            terminated = False
            truncated = False
    
    # Close the environment
    env.close()
    print("State demonstration complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo Atari Pong actions or states.")
    parser.add_argument('mode', choices=['actions', 'state'], help="Mode to demo ('actions' or 'state')")
    args = parser.parse_args()
    
    print(f"[DEBUG] Selected mode: {args.mode}")
    if args.mode == 'actions':
        demo_actions()
    elif args.mode == 'state':
        demo_state()