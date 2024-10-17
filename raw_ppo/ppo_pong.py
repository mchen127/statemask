import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # For the progress bar

# Policy Network with Convolutional Layers
class PolicyNetwork(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(PolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate the size of the flattened output from conv layers
        test_input = torch.zeros(1, input_channels, 210, 160)
        conv_out = self.conv(test_input)
        self.flattened_size = conv_out.view(-1).size(0)

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 512),  # Dynamically calculated input size
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Convert from (batch, height, width, channel) to (batch, channel, height, width)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        logits = self.fc(x)
        return Categorical(logits=logits)

# Value Network with Convolutional Layers
class ValueNetwork(nn.Module):
    def __init__(self, input_channels):
        super(ValueNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate the size of the flattened output from conv layers
        test_input = torch.zeros(1, input_channels, 210, 160)
        conv_out = self.conv(test_input)
        self.flattened_size = conv_out.view(-1).size(0)

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Convert from (batch, height, width, channel) to (batch, channel, height, width)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        value = self.fc(x)
        return value.squeeze()

# PPO Agent remains unchanged
class PPO:
    def __init__(self, policy, value, lr, gamma, eps_clip, vf_coeff, entropy_coeff):
        self.policy = policy
        self.value = value
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.vf_coeff = vf_coeff
        self.entropy_coeff = entropy_coeff

    def compute_advantage(self, rewards, values, next_value, dones):
        advantage = []
        gae = 0
        values = values + [next_value]  # Append next_value to the list of values
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * 0.95 * (1 - dones[i]) * gae
            advantage.insert(0, gae)
        return torch.tensor(advantage, dtype=torch.float32)

    def update(self, log_probs, states, actions, returns, advantages):
        log_probs_old = torch.cat(log_probs).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(4):  # Number of PPO epochs
            log_probs_new = torch.cat([self.policy(state).log_prob(action) for state, action in zip(states, actions)])
            ratio = torch.exp(log_probs_new - log_probs_old)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(self.value(torch.cat(states)), returns)
            entropy = torch.cat([self.policy(state).entropy() for state in states]).mean()

            loss = policy_loss + self.vf_coeff * value_loss - self.entropy_coeff * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Training Function
def train(args):
    env = gym.make('ALE/Pong-v5')
    input_channels = env.observation_space.shape[-1]  # Number of channels in the input image
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(input_channels, action_dim)
    value_net = ValueNetwork(input_channels)

    ppo = PPO(policy_net, value_net, args.lr, args.gamma, args.eps_clip, args.vf_coeff, args.entropy_coeff)

    all_rewards = []

    # Add progress bar using tqdm
    for episode in tqdm(range(args.max_episodes), desc="Training Progress"):
        obs, info = env.reset()
        rewards, log_probs, values, states, actions, dones = [], [], [], [], [], []

        for step in range(args.max_steps):
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Adding batch dimension
            action_dist = policy_net(state)
            action = action_dist.sample()
            value = value_net(state)

            observation, reward, terminated, truncated, info = env.step(action.item())

            # Store the data collected during this step
            rewards.append(reward)
            values.append(value)
            log_probs.append(action_dist.log_prob(action))
            states.append(state)
            actions.append(action)
            dones.append(terminated or truncated)

            obs = observation

            # If the episode has ended, reset to start a new episode
            if terminated or truncated:
                obs, info = env.reset()
                break

        # Convert list of values to tensor for addition with advantages
        values = torch.tensor(values, dtype=torch.float32)
        next_value = value_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        advantages = ppo.compute_advantage(rewards, values.tolist(), next_value, dones)
        returns = advantages + values

        ppo.update(log_probs, states, actions, returns, advantages)
        all_rewards.append(sum(rewards))

    # Save the policy model
    os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
    torch.save(policy_net.state_dict(), args.save_model_path)

    # Plot Training Curve
    plt.plot(all_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Curve')
    plt.savefig(args.save_path)
    plt.close()

# Main Function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO for Pong')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='Clipping epsilon for PPO')
    parser.add_argument('--vf_coeff', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--max_episodes', type=int, default=1000, help='Maximum number of episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--save_path', type=str, default='results/training_curve.png', help='Path to save training curve')
    parser.add_argument('--save_model_path', type=str, default='results/pong_model.pth', help='Path to save the trained model')

    args = parser.parse_args()
    train(args)
