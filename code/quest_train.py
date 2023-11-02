import gym
import minihack
from nle import nethack
from skimage.io import imshow
import numpy as np
from actor_critic_v1 import ActorCritic, run_actor_critic, plot_results
from custom_reward_manager import RewardManager, MessageEvent, InventoryEvent, RelativeCoordEvent
import torch

MOVE_ACTIONS = tuple(nethack.CompassDirection)
NAVIGATE_ACTIONS = MOVE_ACTIONS + (
    nethack.Command.OPEN,
    nethack.Command.KICK,
    nethack.Command.SEARCH,
    nethack.Command.FIGHT,
)

QUEST_ACTIONS = NAVIGATE_ACTIONS + (
    nethack.Command.PICKUP,
    nethack.Command.APPLY,
    nethack.Command.PUTON,
    nethack.Command.WEAR,
    nethack.Command.QUAFF,
    nethack.Command.FIRE,
    nethack.Command.RUSH,
    nethack.Command.ZAP,
)

reward_manager = RewardManager()
reward_manager.add_event(InventoryEvent(5, False, True, False, inv_item="potion"))
reward_manager.add_event(InventoryEvent(5, False, True, False, inv_item="wand"))
reward_manager.add_event(MessageEvent(5, False, True, False, messages=["You start to float in the air!"]))
reward_manager.add_event(RelativeCoordEvent(0.5, True, False, False))

# Final Reward
reward_manager.add_location_event("staircase down", reward = 1000, repeatable = False, terminal_required = True, terminal_sufficient=True)
# reward_manager.add_coordinate_event((11,28), reward = 10, terminal_required = False)
# reward_manager.add_coordinate_event((11,38), reward = 10, terminal_required = False)

# Custom Rewards for long corridors at top and bottom 
# reward_manager.add_coordinate_event((3,27), reward = -2, terminal_required = False)
# reward_manager.add_coordinate_event((3,28), reward = -2, terminal_required = False)
# reward_manager.add_coordinate_event((3,29), reward = -2, terminal_required = False)

# reward_manager.add_coordinate_event((19,27), reward = -2, terminal_required = False)
# reward_manager.add_coordinate_event((19,28), reward = -2, terminal_required = False)
# reward_manager.add_coordinate_event((19,29), reward = -2, terminal_required = False)


env = gym.make(
    "MiniHack-Quest-Hard-v0",
    observation_keys=["glyphs", "pixel", "message", "pixel_crop", "glyphs_crop", "blstats", "inv_strs"],
    reward_manager=reward_manager,
    reward_lose=-5,
    actions=QUEST_ACTIONS,
    autopickup=True,
    allow_all_modes=True,
    max_episode_steps=500,
)

policy, results, optimizer = run_actor_critic(env,number_episodes=5000,max_episode_length=700,iterations=1)
torch.save({"model_policy":policy.state_dict(),"optimizer":optimizer.state_dict()}, "./quest_train_with_rel_coords.pt")
plot_results(env_name="MazeWalk Navigation for 10000 Episodes",scores=results, ylim =(-1.5,1000), color = "teal" )