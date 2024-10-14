import os
import PIL

# import gym
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

# from IPython.display import clear_output
from functools import total_ordering
from tqdm import tqdm  # Import tqdm

# Define constants
ENV_ID = "ALE/Pong-v5"  # Updated to Gymnasium's naming convention
# ENV_ID = "Pong-v0"  # Updated to Gymnasium's naming convention
H_SIZE = 256  # hidden size, linear units of the output layer
L_RATE = 1e-5  # learning rate, gradient coefficient for CNN weight update
L_RATE_LAMBDA = 1e-3  # learning rate of LAMBDA
G_GAE = 0.99  # gamma param for GAE
L_GAE = 0.95  # lambda param for GAE
E_CLIP = 0.2  # clipping coefficient
C_1 = 0.5  # squared loss coefficient
C_2 = 0.01  # entropy coefficient

lambda_1 = 1e-4  # lasso regularization
eta_origin = 0.18054925225205481
N = 4  # simultaneous processing environments
T = 256  # PPO steps
M = 64  # mini batch size
K = 10  # PPO epochs
T_EPOCHS = 2  # each T_EPOCH
N_TESTS = 20  # do N_TESTS tests
TARGET_REWARD = 0.9
TRANSFER_LEARNING = False
BASELINE_PATH = f"./ppo_test/baseline/Pong-v0_+0.896_12150.dat"
PATH = f"./ppo_test/checkpoints/Pong-v0_+0.855_19700.dat"


class CNN(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(CNN, self).__init__()
        self.critic = nn.Sequential(  # The “Critic” estimates the value function
            nn.Conv2d(in_channels=num_inputs, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=2592, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=1),
        )
        self.actor = nn.Sequential(  # The “Actor” updates the policy distribution in the direction suggested by the Critic
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

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        distribution = Categorical(probs)
        return distribution, value


# Create necessary directories
for func in [
    lambda: os.makedirs(os.path.join(".", "ppo_test/checkpoints"), exist_ok=True),
    lambda: os.makedirs(os.path.join(".", "ppo_test/plots"), exist_ok=True),
    lambda: os.makedirs(os.path.join(".", "ppo_test/records"), exist_ok=True),
]:
    try:
        func()
    except Exception as error:
        print(error)
        continue


def make_env():  # this function creates a single environment
    def _thunk():
        env = gym.make(ENV_ID)
        return env

    return _thunk


def normalize(x):
    x -= x.mean()
    x /= x.std() + 1e-8  # prevent division by zero
    return x


def test_env(i_episode, env, baseline_model, model, device):
    # print(f"[test_env] Episode {i_episode} initializing")
    env.reset(seed=i_episode)
    state, _ = env.reset()
    state = grey_crop_resize(state)
    done = False
    total_reward = 0

    # print(f"[test_env] Episode {i_episode} entering while loop")

    while not done:
        state = torch.FloatTensor(np.copy(state)).unsqueeze(0).to(device)
        baseline_distribution, _ = baseline_model(state)
        baseline_action = np.argmax(
            baseline_distribution.probs.detach().cpu().numpy()[0]
        )

        distribution, _ = model(state)
        action = np.argmax(distribution.probs.detach().cpu().numpy()[0])

        # Define action mapping
        if action == 1:
            real_action = baseline_action
        else:
            real_action = np.random.choice(6)

        next_state, reward, terminated, truncated, info = env.step(real_action)
        done = terminated or truncated
        next_state = grey_crop_resize(next_state)
        state = next_state
        total_reward += reward

    return total_reward >= 21


def plot(train_epoch, rewards, save=True):
    plt.close("all")
    plt.figure()
    plt.title(f"{ENV_ID}: Epoch: {train_epoch} -> Reward: {rewards[-1]:.3f}")
    plt.ylabel("Reward")
    plt.xlabel("Epoch")
    plt.plot(rewards)
    plt.grid()
    if save:
        plt.savefig(f"ppo_test/plots/{ENV_ID}_{rewards[-1]:.3f}.png")


def record_video(
    env_id, model, video_length=500, prefix="", video_folder="ppo_test/records/"
):
    eval_env = SubprocVecEnv([lambda: gym.make(env_id)])
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=prefix,
    )

    state = eval_env.reset()
    state = grey_crop_resize_batch(state)
    for _ in tqdm(range(video_length), desc="Recording Video"):
        state = torch.FloatTensor(state).to(device)
        dist, _ = model(state)
        action = dist.sample().cuda() if use_cuda else dist.sample()
        next_state, _, _, _ = eval_env.step(action.cpu().numpy())
        state = grey_crop_resize_batch(next_state)
    eval_env.close()


def grey_crop_resize_batch(state):  # deal with batch observations
    states = []
    for i in state:
        img = Image.fromarray(i)
        grey_img = img.convert(mode="L")
        left = 0
        top = 34  # empirically chosen
        right = 160
        bottom = 194  # empirically chosen
        cropped_img = grey_img.crop((left, top, right, bottom))  # cropped image
        resized_img = cropped_img.resize((84, 84))
        array_2d = np.asarray(resized_img)
        array_3d = np.expand_dims(array_2d, axis=0)
        array_4d = np.expand_dims(array_3d, axis=0)
        states.append(array_4d)
    states_array = np.vstack(states)  # stack into array
    return states_array  # B*C*H*W


def grey_crop_resize(state):  # deal with single observation
    img = Image.fromarray(state)
    grey_img = img.convert(mode="L")
    left = 0
    top = 34  # empirically chosen
    right = 160
    bottom = 194  # empirically chosen
    cropped_img = grey_img.crop((left, top, right, bottom))
    resized_img = cropped_img.resize((84, 84))
    array_2d = np.asarray(resized_img)
    array_3d = np.expand_dims(array_2d, axis=0)
    return array_3d  # C*H*W


def compute_gae(next_value, rewards, masks, values, gamma=G_GAE, lam=L_GAE):
    values = values + [next_value]  # append last value
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(states, actions, log_probs, returns, advantage, unnorm_advantage):
    batch_size = states.size(0)

    for _ in range(batch_size // M):
        rand_start = np.random.randint(0, batch_size - M)
        yield (
            states[rand_start : rand_start + M, :],
            actions[rand_start : rand_start + M, :],
            log_probs[rand_start : rand_start + M, :],
            returns[rand_start : rand_start + M, :],
            advantage[rand_start : rand_start + M, :],
            unnorm_advantage[rand_start : rand_start + M, :],
        )


def ppo_update(
    states,
    actions,
    log_probs,
    returns,
    advantages,
    unnorm_advantage,
    disc_rewards,
    LAMBDA,
    clip_param=E_CLIP,
):
    loss_buff = []

    for _ in range(K):
        for state, action, old_log_probs, return_, advantage, unnorm_adv in ppo_iter(
            states, actions, log_probs, returns, advantages, unnorm_advantage
        ):
            distribution, value = model(state)
            action = action.reshape(1, len(action))  # reshape
            no_mask_acts = torch.ones_like(action)
            no_mask_probs = distribution.log_prob(no_mask_acts)
            no_mask_probs = no_mask_probs.reshape(len(old_log_probs), 1)

            new_log_probs = distribution.log_prob(action)
            new_log_probs = new_log_probs.reshape(len(old_log_probs), 1)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()

            unnorm_actor_loss = -ratio * unnorm_adv.mean()

            critic_loss = (return_ - value).pow(2).mean()
            entropy = distribution.entropy().mean()

            num_masks = torch.sum(no_mask_probs.exp()) / (T // M)

            if LAMBDA > 1:
                actor_loss = -actor_loss

            loss = (
                C_1 * critic_loss + actor_loss - C_2 * entropy + lambda_1 * num_masks
            )  # loss function

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_buff.append(unnorm_actor_loss.cpu().detach().numpy())

    LAMBDA -= L_RATE_LAMBDA * (
        np.mean(loss_buff) - 2 * np.mean(disc_rewards) + 2 * eta_origin
    )
    LAMBDA = max(LAMBDA, 0)

    return LAMBDA


def ppo_train(
    baseline_model,
    model,
    envs,
    device,
    use_cuda,
    test_rewards,
    test_epochs,
    train_epoch,
    best_reward,
    early_stop=False,
):
    LAMBDA = 0  # Lagrange multiplier

    env = gym.make(ENV_ID)
    state = envs.reset()
    state = grey_crop_resize_batch(state)

    print(len(state))
    while_loop_cnt = 0

    # Initialize tqdm progress bar for training epochs
    with tqdm(desc="Training Epochs", unit="epoch") as pbar_outer:
        while not early_stop:
            while_loop_cnt += 1
            pbar_outer.update(1)
            pbar_outer.set_postfix(
                {"Loop Count": while_loop_cnt, "Train Epoch": train_epoch}
            )

            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            disc_rewards = np.zeros(N)

            # Initialize tqdm progress bar for PPO steps
            for t in tqdm(
                range(T),
                desc=f"PPO Steps (Epoch {train_epoch})",
                leave=False,
                unit="step",
            ):
                state = torch.FloatTensor(state).to(device)

                baseline_distribution, baseline_value = baseline_model(state)
                baseline_action = (
                    baseline_distribution.sample().cuda()
                    if use_cuda
                    else baseline_distribution.sample()
                )

                distribution, value = model(state)
                action = (
                    distribution.sample().cuda() if use_cuda else distribution.sample()
                )

                baseline_action_copy = baseline_action.cpu().numpy()
                mask_action_copy = action.cpu().numpy()

                real_actions = []
                for i in range(len(mask_action_copy)):
                    if mask_action_copy[i] == 1:
                        real_actions.append(baseline_action_copy[i])
                    else:
                        real_actions.append(np.random.choice(6))

                next_state, reward, done, _ = envs.step(real_actions)
                for i in range(N):
                    disc_rewards[i] += np.power(G_GAE, t) * reward[i]
                next_state = grey_crop_resize_batch(next_state)
                log_prob = distribution.log_prob(action)
                log_prob_vect = log_prob.reshape(len(log_prob), 1)
                log_probs.append(log_prob_vect)
                action_vect = action.reshape(len(action), 1)
                actions.append(action_vect)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
                states.append(state)
                state = next_state

            next_state = torch.FloatTensor(next_state).to(device)  # last state
            _, next_value = model(next_state)
            returns = compute_gae(next_value, rewards, masks, values)
            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)
            unnorm_advantage = returns - values
            advantage = normalize(unnorm_advantage)
            LAMBDA = ppo_update(
                states,
                actions,
                log_probs,
                returns,
                advantage,
                unnorm_advantage,
                disc_rewards,
                LAMBDA,
            )
            train_epoch += 1

            state = envs.reset()
            state = grey_crop_resize_batch(state)

            # Testing phase every T_EPOCHS
            if train_epoch % T_EPOCHS == 0:
                test_rewards_batch = []
                with tqdm(
                    total=N_TESTS, desc="Testing Episodes", leave=False
                ) as pbar_test:
                    for i in range(N_TESTS):
                        test_reward = test_env(i, env, baseline_model, model, device)
                        test_rewards_batch.append(test_reward)
                        pbar_test.update(1)
                test_reward_mean = np.mean(test_rewards_batch)
                test_rewards.append(test_reward_mean)
                test_epochs.append(train_epoch)
                pbar_outer.set_postfix(
                    {
                        "Epoch": train_epoch,
                        "Mean Test Reward": f"{test_reward_mean:.3f}",
                        "Best Reward": f"{best_reward:.3f}" if best_reward else "None",
                        "Lambda": f"{LAMBDA:.3f}",
                    }
                )

                print(f"Epoch: {train_epoch} -> Reward: {test_reward_mean:.3f}")
                print(f"Current lambda: {LAMBDA:.3f}")

                if best_reward is None or best_reward < test_reward_mean:
                    if best_reward is not None:
                        print(
                            f"Best reward updated: {best_reward:.3f} -> {test_reward_mean:.3f}"
                        )
                    best_reward = test_reward_mean
                    name = f"{ENV_ID}_{test_reward_mean:+.3f}_{train_epoch}.dat"
                    fname = os.path.join(".", "ppo_test/checkpoints", name)
                    states_save = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "test_rewards": test_rewards,
                        "test_epochs": test_epochs,
                    }
                    torch.save(states_save, fname)
                    print("Model saved")

                if test_reward_mean > TARGET_REWARD:
                    print(f"Target reward {TARGET_REWARD} achieved. Stopping training.")
                    early_stop = True

    return


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device:", device)

    envs = [make_env() for _ in range(N)]
    envs = SubprocVecEnv(envs)
    num_inputs = 1
    num_outputs = envs.action_space.n

    baseline_model = CNN(num_inputs, num_outputs, H_SIZE).to(device)
    baseline_model.eval()

    model = CNN(num_inputs, 2, H_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=L_RATE)
    test_rewards = []
    test_epochs = []
    train_epoch = 0
    best_reward = None

    if use_cuda:
        checkpoint = torch.load(BASELINE_PATH)
        baseline_model.load_state_dict(checkpoint["state_dict"])
    else:
        checkpoint = torch.load(
            BASELINE_PATH, map_location=lambda storage, loc: storage
        )
        baseline_model.load_state_dict(checkpoint["state_dict"])

    print("Baseline Model: loaded")

    if TRANSFER_LEARNING:
        if use_cuda:
            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint["state_dict"])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            test_rewards = checkpoint["test_rewards"]
            test_epochs = checkpoint["test_epochs"]
            train_epoch = test_epochs[-1]
            best_reward = test_rewards[-1]
        else:
            checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint["state_dict"])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            test_rewards = checkpoint["test_rewards"]
            test_epochs = checkpoint["test_epochs"]
            train_epoch = test_epochs[-1]
            best_reward = test_rewards[-1]
        print("CNN: loaded")
        print(f"Previous best reward: {best_reward:.3f}")

    print(model)
    print(optimizer)

    ppo_train(
        baseline_model,
        model,
        envs,
        device,
        use_cuda,
        test_rewards,
        test_epochs,
        train_epoch,
        best_reward,
    )

    plot(train_epoch, test_rewards, save=True)

    # Uncomment the following line to record a video after training
    record_video(ENV_ID, model, video_length=6000, prefix='ppo_pong')
