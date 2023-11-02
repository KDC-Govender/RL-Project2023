# Adapted from https://github.com/BrentonBudler/deep-rl-minihack-the-planet/blob/main/A2C.ipynb
import numpy as np
import gym
import minihack
from nle import nethack
import re

import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import flatten

from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

# Pytorch Neural Networks
# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def moving_average(a, n):
    """Calculates the moving average of an array a with a window size n"""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

def display_screen(state):
    """Displays the state as the screen in image form using the 'pixel' observation key"""
    screen = Image.fromarray(np.uint8(state['pixel']))
    display(screen)

def format_state(state):
    """Formats the state according to the input requirements of the Actor Critic Neural Network"""

    # Normalize and reshape for convolutional layer input
    glyphs = state["glyphs"]
    glyphs = glyphs/glyphs.max()
    glyphs = glyphs.reshape((1,1,21,79))

    # Normalize the message and reshape for the fully connected layer input
    message = state["message"]
    if state["message"].max()>0:
        # Occassionally the message is empty which will cause a Zero Division error
        message = message/message.max()
    message = message.reshape((1,len(message)))

    # Normalize the inventory items
    # inventory = state[""]

    state = {"glyphs":glyphs,"message":message}
    return state

def compute_returns(rewards, gamma):
    """Computes the discounted returns of a sequence of rewards achieved in a trajectory"""
    returns = []
    r= 0
    for reward in rewards[::-1]:
        r = reward + gamma*r
        returns.append(r)
    returns.reverse()
    returns = np.array(returns)

    # Standardize Returns
    if np.mean(returns)!= 0:
         returns = returns - np.mean(returns)
    if np.std(returns) != 0:
        returns = returns/ np.std(returns)

    return returns

def plot_results(env_name,scores, color,ylim):
    """Plots the reward attained by an Agent at each step of training in
        an environment for each iteration and average over all iterations"""

    plt.figure(figsize=(8,6))

    # Plot individual iterations
    for score in scores:
        plt.plot(score, alpha =0.1, color=color)

    # Plot mean over all iterations
    mean = np.mean(scores,axis=0)
    plt.plot(mean, color=color,label="Mean Reward")

    plt.title(f"Actor Critic - {env_name}")
    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    plt.yticks(np.arange(ylim[0], ylim[1], 1.00))
    plt.legend(loc=4)
    plt.savefig(f"Actor-Critic-{env_name}.pdf")
    plt.show()

def map_descriptions(mapping_dict, neighbor_descriptions):

    # Initialize an empty list to store the mapped values
    mapped_values = []

    # Loop through the keys to lookup
    for lookup_key in neighbor_descriptions:
        found = False
        for key in mapping_dict.keys():
            if lookup_key in key:
                mapped_values.append(mapping_dict[key])
                found = True
                break
        if not found:
            mapped_values.append(0)

    return mapped_values

class ActorCritic(nn.Module):
    """The Actor Critic Neural Network used to estimate the state value function and action probabilities"""
    def __init__(self,s_size=8,h_size=128, a_size=4):

        # The network architecture follows the popular lenet-5 CNN architeture
        super(ActorCritic, self).__init__()

        # Initialize first set of convolutional and pooling layers with a ReLU activation function
        self.conv1 = Conv2d(in_channels=1, out_channels=20,
                            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Initialize second set of of convolutional and pooling layers with a ReLU activation function
        self.conv2 = Conv2d(in_channels=20, out_channels=50,
                            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Initialize fully connected layers for glyph output after convolutional and pooling layers
        self.fc1 = Linear(in_features=1600, out_features=500)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features=500, out_features=128)
        self.relu4 = ReLU()

        # Initialize fully connected for message input
        self.fc3 = Linear(in_features=256, out_features=128)
        self.relu5 = ReLU()

        # Initialize fully connected for neighbor_descriptions input
        self.fc5 = Linear(in_features=9, out_features=8)
        self.relu7 = ReLU()

        # Initialize fully connected for direction input
        self.fc6 = Linear(in_features=9, out_features=1)
        self.relu8 = ReLU()

        # Initialize fully connected for combination of glyphs, message, crop and direction
        self.fc4 = Linear(in_features=265, out_features=128)
        self.relu6 = ReLU()

        # To estimate the value function of the state
        self.value_layer = nn.Linear(128, 1)

        # To calculate the probability of taking each action in the given state
        self.action_layer = nn.Linear(128, a_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state, neighbor_descriptions, neighbor_directions):
        # Format state
        state = format_state(state)

        # Transform the glyph and state arrays into tensors
        glyphs_t  = torch.from_numpy(state["glyphs"]).float().to(device)
        message_t  = torch.from_numpy(state["message"]).float().to(device)
        neighbor_descriptions  = torch.from_numpy(neighbor_descriptions).float().to(device)
        neighbor_directions  = torch.from_numpy(neighbor_directions).float().to(device)

        # Pass the 2D glyphs input through our convolutional and pooling layers
        glyphs_t = self.conv1(glyphs_t)
        glyphs_t = self.relu1(glyphs_t)
        glyphs_t = self.maxpool1(glyphs_t)
        glyphs_t = self.conv2(glyphs_t)
        glyphs_t = self.relu2(glyphs_t)
        glyphs_t = self.maxpool2(glyphs_t)

        # Platten the output from the final pooling layer and pass it through the fully connected layers
        glyphs_t = glyphs_t.reshape(glyphs_t.shape[0], -1)
        glyphs_t = self.fc1(glyphs_t)
        glyphs_t = self.relu3(glyphs_t)
        glyphs_t = self.fc2(glyphs_t)
        glyphs_t = self.relu4(glyphs_t)

        # Pass the message input through a fully connected layer
        message_t = self.fc3(message_t)
        message_t = self.relu5(message_t)

        # Pass the neighbor_descriptions input through a fully connected layer
        neighbor_descriptions = self.fc5(neighbor_descriptions)
        neighbor_descriptions = self.relu7(neighbor_descriptions)

        # Pass the neighbor_direction input through a fully connected layer
        neighbor_directions = self.fc6(neighbor_directions)
        neighbor_directions = self.relu8(neighbor_directions)

        # Combine glyphs output from convolution and fully connected layers
        # with message output from fully connected layer
        # Cat and Concat are used for different versions of PyTorch
        try:
            combined = torch.cat((glyphs_t,message_t, neighbor_descriptions, neighbor_directions),1)
        except:
            combined = torch.concat([glyphs_t,message_t, neighbor_descriptions, neighbor_directions],1)

        # Pass glyphs and messaged combination through a fully connected layer
        combined = self.fc4(combined)
        combined = self.relu6(combined)

        # Pass the output from the previous fully connected layer through two seperate
        # fully connected layers, one with a single output neuron (to estimate the state value function)
        # and the other with the number of output neurons equal to the number of actions
        # (to estimate the action probabilities)
        state_value = self.value_layer(combined)

        action_probs = self.action_layer(combined)
        action_probs = self.softmax(action_probs)

        return action_probs,state_value
    

def action_index(action, env_actions):
    return env_actions.index(action)


def complete_option(env, option_num, next_state, env_options):
    option_policy, termination_clause, max_steps = env_options[option_num]
    if termination_clause is None:
        next_state, reward, done, info = env.step(
            action_index(option_policy, env.actions)
        )
        next_state = format_state(next_state)
        return next_state, reward, done, info, 1
    is_complete = False
    done = False
    n_steps = 0
    episode_return = 0
    while not (is_complete or done) and n_steps < max_steps:
        action = option_policy.select_action(env, next_state)  # selected from policy
        next_state, reward, done, info = env.step(action)
        is_complete = termination_clause.check_complete(env, next_state)
        next_state = format_state(next_state)
        episode_return += reward
        n_steps += 1
    return next_state, episode_return, done, info, n_steps


def actor_critic(env, model, seed, learning_rate, number_episodes, max_episode_length, gamma, verbose=True, env_options=[]):
    """
    Method to train Actor Critic model.

    Input:
    env: The environment to be used during training
    seed: The random seed for any random operations performed
    learning_rate: The learning rate uesd for the Adam optimizer when training the model
    number_episodes: Number of episodes to train for
    max_episode_length: The maximum number of steps to take in an episode before terminating
    gamma: The discount factor used when calculating the discounted rewards of an episode
    verbose: Print episode reward after each episode

    Returns:
    policy: The neural network model after training that approximates the state value functions and action probabilities
    scores: The cumulative reward achieved by the agent for each episode during traiing
    """
    # Setting random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    env.seed(seed)
    my_dict = {'':0, "floor of a room": 1, "human rogue called Agent": 1, "staircase up": 3, 'staircase down':4}
    directions = ['107', '108', '106', '104', '117', '110', '98', '121', None]
    obj_to_find = "potion"

    # Initialize optimizer for Actor Critic Network
    # if optimizer is None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # List to store the rewards attained in each episode
    scores =[]

    for i in range(number_episodes):
        # Reset environment
        state = env.reset()
        neighbor_descriptions = env.get_neighbor_descriptions()
        mapped_descriptions = np.array(map_descriptions(my_dict, neighbor_descriptions))
        mapped_descriptions = mapped_descriptions.reshape((1,len(mapped_descriptions)))

        # Choose one category to encode (e.g., 'C')
        selected_directions = env.get_object_direction(obj_to_find)
        selected_directions_encoded = np.zeros(len(directions), dtype=int)
        index = directions.index(str(selected_directions) if selected_directions is not None else selected_directions)
        selected_directions_encoded[index] = 1
        selected_directions_encoded = np.array(selected_directions_encoded.reshape((1,len(selected_directions_encoded))))

        # Flag to see if episode has terminated
        done = False

        # Lists to store the rewards acquired, the log_probability of the actions and
        # the value function of the states visited in this episode
        rewards = []
        log_probs = []
        state_values = []

        t = 1

        while t < max_episode_length:

            # Get the probability distribution over actions and
            # estimated state value function from Actor Critic network
            action_probs,state_value = model.forward(state, mapped_descriptions, selected_directions_encoded)
            distribution = torch.distributions.Categorical(action_probs)
            # Sample from the probability distribution to determine which OPTION to take
            option_num = distribution.sample()

            # Take selected action, observe the reward received, the next state
            # and whether or not the episode terminated
            next_state, reward, done, info, num_steps_taken = complete_option(env, option_num, state, env_options)
            # env.step(action.item())

            # next_state = format_state(next_state)

            neighbor_descriptions = env.get_neighbor_descriptions()
            mapped_descriptions = np.array(map_descriptions(my_dict, neighbor_descriptions))
            mapped_descriptions = mapped_descriptions.reshape((1,len(mapped_descriptions)))

            selected_directions = env.get_object_direction(obj_to_find)
            selected_directions_encoded = np.zeros(len(directions), dtype=int)
            index = directions.index(str(selected_directions) if selected_directions is not None else selected_directions)
            selected_directions_encoded[index] = 1
            selected_directions_encoded = np.array(selected_directions_encoded.reshape((1,len(selected_directions_encoded))))

            # Store the reward, log of the probability of the action selected
            # And
            rewards.append(reward)
            log_probs.append(distribution.log_prob(option_num))
            state_values.append(state_value)

            state = next_state
            t += num_steps_taken

            if done:
                break

        # Store the reward acquired in the episode and calculate the discounted return of the episode
        scores.append(np.sum(rewards))
        returns = compute_returns(rewards, gamma)
        returns = torch.from_numpy(returns).float().to(device)

        # Print the episode, the reward acquired in the episode and the mean reward over the last 50 episodes
        if verbose:
            print("Episode:",i,"Reward:",np.sum(rewards),"Average Reward:",np.mean(scores[-50:]),"Steps",t)

        # Calculate the loss for the episode and use it to update the network parameters
        loss = 0
        for logprob, value, reward in zip(log_probs, state_values, returns):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            try:
                reward = reward.resize(1,1)
            except:pass
            value_loss = F.smooth_l1_loss(value, reward)
            loss += (action_loss + value_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Store the policy as the neural network model at the final iteration
    policy = model

    return policy, scores, optimizer

def run_actor_critic(env,number_episodes,max_episode_length,iterations,env_options):
    """Trains Actor Critic model for a number of episodes on a given environment"""
    seeds = np.random.randint(1000, size=iterations)
    scores_arr = []

    for seed in seeds:
        # if ac_model is None:
        # Initialize the Actor Critic Model
        ac_model = ActorCritic(h_size=512, a_size=len(env_options))

        # Train the Actor Critic Model
        policy, scores, optimizer = actor_critic(env=env,
                                    model= ac_model,
                                    env_options=env_options,
                                    seed=seed,
                                    learning_rate=0.02,
                                    number_episodes=number_episodes,
                                    max_episode_length=max_episode_length,
                                    gamma=0.99 ,
                                    verbose=True)

        # Store rewards for this iteration
        scores_arr.append(scores)

    return policy, scores_arr, optimizer

