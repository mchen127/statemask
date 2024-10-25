import torch.nn as nn
from torch.distributions import Categorical

class PPO(nn.Module):
    def __init__(self, num_inputs=1, num_outputs=6, hidden_size=256):
        super(PPO, self).__init__()
        self.critic = nn.Sequential(
            nn.Conv2d(in_channels=num_inputs, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=2592, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=1),
        )
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels=num_inputs, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=2592, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=num_outputs),
            nn.Softmax(dim=1),
        )

    def get_value(self, x):
        value = self.critic(x / 255.0)
        return value

    def get_action(self, x):
        probs = self.actor(x / 255.0)
        dist = Categorical(probs)
        action = dist.sample()
        return action

    def forward(self, x):
        value = self.critic(x / 255.0)
        probs = self.actor(x / 255.0)
        dist = Categorical(probs)
        return dist, value
