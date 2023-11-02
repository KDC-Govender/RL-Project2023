# Adapted from https://gist.github.com/ArnolFokam/4fc3fbd3f9c33bf2235dbe77c8a61192

# Import necessary libraries
from pathlib import Path
import gym
import pygame
import numpy as np
import cv2  # OpenCV for video creation
import os
import minihack
from nle import nethack
from pygame.locals import *
from actor_critic_v3_hierarchical_options import (
    ActorCritic,
    format_state,
    DrinkPolicy,
    PotionPolicy,
    MessageTerminationEvent,
    InventoryTerminationEvent,
    map_descriptions,
    complete_option,
    action_index
)
import torch
from nle.nethack import CompassDirection
from custom_reward_manager import RewardManager, InventoryEvent, MessageEvent

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Function to scale an observation to a new size using Pygame
def scale_observation(observation, new_size):
    """
    Scale an observation (image) to a new size using Pygame.
    Args:
        observation (pygame.Surface): The input Pygame observation.
        new_size (tuple): The new size (width, height) for scaling.
    Returns:
        pygame.Surface: The scaled observation.
    """
    return pygame.transform.scale(observation, new_size)


# Function to render the game observation
def render(obs, screen, font, text_color):
    """
    Render the game observation on the Pygame screen.
    Args:
        obs (dict): Observation dictionary containing "pixel" and "message" keys.
        screen (pygame.Surface): The Pygame screen to render on.
        font (pygame.Font): The Pygame font for rendering text.
        text_color (tuple): The color for rendering text.
    """
    img = obs["pixel"]
    msg = obs["message"]
    msg = msg[: np.where(msg == 0)[0][0]].tobytes().decode("utf-8")
    rotated_array = np.rot90(img, k=-1)

    window_size = screen.get_size()
    image_surface = pygame.surfarray.make_surface(rotated_array)
    image_surface = scale_observation(image_surface, window_size)

    screen.fill((0, 0, 0))
    screen.blit(image_surface, (0, 0))

    text_surface = font.render(msg, True, text_color)
    text_position = (
        window_size[0] // 2 - text_surface.get_width() // 2,
        window_size[1] - text_surface.get_height() - 20,
    )
    screen.blit(text_surface, text_position)
    pygame.display.flip()


def render_frame(out, screen, clock, pygame_frame_rate):
    # Capture the current frame and save it to the video
    pygame.image.save(screen, "temp_frame.png")
    frame = cv2.imread("temp_frame.png")
    out.write(frame)

    clock.tick(pygame_frame_rate)

# Function to record a video of agent gameplay
def record_video(
    env, agent, video_filepath, pygame_frame_rate, video_frame_rate, max_timesteps, env_options
):
    """
    Record a video of agent's gameplay and save it as an MP4 file.
    Args:
        env (gym.Env): The environment in which the agent plays.
        agent (object): The agent that interacts with the environment.
        video_filepath (Path): The file path where the video will be saved.
        pygame_frame_rate (int): Frame rate for rendering the video.
        video_frame_rate (int): Frame rate for the output video.
        max_timesteps (int): Maximum number of timesteps to record in the video.
    """
    frame_width = env.observation_space["pixel"].shape[1]
    frame_height = env.observation_space["pixel"].shape[0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(video_filepath), fourcc, video_frame_rate, (frame_width, frame_height)
    )

    pygame.init()
    screen = pygame.display.set_mode((frame_width, frame_height))
    font = pygame.font.Font(None, 36)
    text_color = (255, 255, 255)

    done = False
    obs = env.reset()
    clock = pygame.time.Clock()

    steps = 1

    while not done and steps < max_timesteps:
        # FOR EACH STEP IN SUBPOLICY
        action = agent.act(env, obs)
        option_policy, termination_clause, max_steps = env_options[action]
        if termination_clause is None:
            print(option_policy)
            obs, reward, done, info = env.step(
                action_index(option_policy, env.actions)
            )
            render(obs, screen, font, text_color)
            render_frame(out, screen, clock, pygame_frame_rate)
            steps += 1
            continue
        is_complete = False
        done = False
        n_steps = 0
        episode_return = 0
        while not (is_complete or done) and n_steps < max_steps:
            action = option_policy.select_action(env, obs)  # selected from policy
            if action is None:
                break
            print(env.actions[action].name)
            obs, reward, done, info = env.step(action)
            is_complete = termination_clause.check_complete(env, obs)
            episode_return += reward
            n_steps += 1
            render(obs, screen, font, text_color)
            render_frame(out, screen, clock, pygame_frame_rate)
            steps += 1

    out.release()  # Release the video writer
    cv2.destroyAllWindows()  # Close any OpenCV windows
    os.remove("temp_frame.png")  # Remove the temporary frame file


# Function to visualize agent's gameplay and save it as a video
def visualize(env, agent, pygame_frame_rate, video_frame_rate, save_dir, max_timesteps, env_options):
    """
    Visualize agent's gameplay and save it as a video.
    Args:
        env (gym.Env): The environment in which the agent plays.
        agent (object): The agent that interacts with the environment.
        pygame_frame_rate (int): Frame rate for rendering on the pygame screen.
        video_frame_rate (int): Frame rate for the output video.
        save_dir (str): Directory where the video will be saved.
        max_timesteps (int): Maximum number of timesteps to record in the video.
    """
    os.makedirs(save_dir, exist_ok=True)
    video_filepath = Path(save_dir) / "video.mp4"

    record_video(
        env, agent, video_filepath, pygame_frame_rate, video_frame_rate, max_timesteps, env_options
    )


class CustomPolicy:
    my_dict = {'':0, "floor of a room": 1, "human rogue called Agent": 1, "staircase up": 3, 'staircase down':4}
    directions = ['107', '108', '106', '104', '117', '110', '98', '121', None]
    obj_to_find = "staircase down"
    def __init__(self, policy_file, actions):
        modelA = ActorCritic(h_size=512, a_size=len(actions))
        optimizerA = torch.optim.Adam(modelA.parameters(), lr=0.02)

        checkpoint = torch.load(policy_file)
        modelA.load_state_dict(checkpoint["model_policy"])
        optimizerA.load_state_dict(checkpoint["optimizer"])

        modelA.eval()
        self.policy = modelA.to(device)
        self.actions = actions

    def act(self, env, next_state):
        neighbor_descriptions = env.get_neighbor_descriptions()
        mapped_descriptions = np.array(map_descriptions(self.my_dict, neighbor_descriptions))
        mapped_descriptions = mapped_descriptions.reshape((1,len(mapped_descriptions)))

        # Choose one category to encode (e.g., 'C')
        selected_directions = env.get_object_direction(self.obj_to_find)
        selected_directions_encoded = np.zeros(len(self.directions), dtype=int)
        index = self.directions.index(str(selected_directions) if selected_directions is not None else selected_directions)
        selected_directions_encoded[index] = 1
        selected_directions_encoded = np.array(selected_directions_encoded.reshape((1,len(selected_directions_encoded))))
        # Get the probability distribution over actions and
        # estimated state value function from Actor Critic network

        action_probs,state_value = self.policy.forward(next_state, mapped_descriptions, selected_directions_encoded)
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        return action.item()


if __name__ == "__main__":
    # ------ Setup Environment ------

    # Lava ENV -- extra actions needed for confirming action
    LAVA_ACTIONS = tuple(nethack.CompassDirection) + (
        nethack.Command.PICKUP,
        nethack.Command.QUAFF,
        nethack.Command.FIRE,
    )

    env = gym.make(
        "MiniHack-LavaCross-Levitate-Potion-Pickup-Restricted-v0",
        # des_file=des_file,
        observation_keys=[
            "glyphs",
            "pixel",
            "message",
            "pixel_crop",
            "glyphs_crop",
            "blstats",
            "inv_strs",
        ],
        # reward_manager=reward_manager,
        # reward_lose=-1, # Does not work when reward manager is used
        # actions=LAVA_ACTIONS, # Included in env.
        autopickup=True,
        allow_all_modes=True,  # Enables confirmation message for consuming potion
        max_episode_steps=500,
    )

    # Option 1: Loaded policy that's trained to find and pickup a potion
    termination_clause = InventoryTerminationEvent("potion")
    potion_policy = PotionPolicy(
        "./policy_potion_pickup_with_neighbours_2000.pt", tuple(nethack.CompassDirection)
    )
    # Option 2: Defined policy for consuming a potion in the agent's inventory
    drink_policy = DrinkPolicy("potion")
    levitation_message = MessageTerminationEvent(["You start to float in the air!"])

    # Specify the set of Options, including the primitive actions
    ENV_OPTIONS = [(action, None, 1) for action in env.actions] + [
        (potion_policy, termination_clause, 20),
        (drink_policy, levitation_message, 2),
    ]

    # Visualize trained agent
    visualize(
        env,
        CustomPolicy("policy_potion_pickup_with_options.pt", ENV_OPTIONS),
        pygame_frame_rate=60,
        video_frame_rate=5,
        save_dir="videos",
        max_timesteps=100000,
        env_options=ENV_OPTIONS
    )
