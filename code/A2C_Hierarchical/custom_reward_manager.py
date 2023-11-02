# --------------- Adapted from minihack.reward_manager --------------- 
# 
# -- Changes to the RewardManager and check_episode_end_call
# -- Addition of 3x new event types
# 

# Copyright (c) Facebook, Inc. and its affiliates.
from __future__ import annotations

import enum
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, List, Tuple
import numpy as np

if TYPE_CHECKING:
    from minihack import MiniHack

from nle.nethack import Command, CompassDirection

Y_cmd = CompassDirection.NW


class EventType(enum.IntEnum):
    MESSAGE = 0
    LOC_ACTION = 1
    COORD = 2
    LOC = 3


COMESTIBLES = [
    "orange",
    "meatball",
    "meat ring",
    "meat stick",
    "kelp frond",
    "eucalyptus leaf",
    "clove of garlic",
    "sprig of wolfsbane",
    "carrot",
    "egg",
    "banana",
    "melon",
    "candy bar",
    "lump of royal jelly",
]


class Event(ABC):
    """An event which can occur in a MiniHack episode.

    This is the base class of all other events.
    """

    def __init__(
        self,
        reward: float,
        repeatable: bool,
        terminal_required: bool,
        terminal_sufficient: bool,
    ):
        """Initialise the Event.

        Args:
            reward (float):
                The reward for the event occuring
            repeatable (bool):
                Whether the event can occur repeated (i.e. if the reward can be
                collected repeatedly
            terminal_required (bool):
                Whether this event is required for the episode to terminate.
            terminal_sufficient (bool):
                Whether this event causes the episode to terminate on its own.
        """
        self.reward = reward
        self.repeatable = repeatable
        self.terminal_required = terminal_required
        self.terminal_sufficient = terminal_sufficient
        self.achieved = False

    @abstractmethod
    def check(self, env, previous_observation, action, observation, past_cells=[]) -> float:
        """Check whether the environment is in the state such that this event
        has occured.

        Args:
            env (MiniHack):
                The MiniHack environment in question.
            previous_observation (tuple):
                The previous state observation.
            action (int):
                The action taken.
            observation (tuple):
                The current observation.
        Returns:
            float: The reward.
        """
        pass

    def reset(self):
        """Reset the event, if there is any state necessary."""
        self.achieved = False

    def _set_achieved(self) -> float:
        if not self.repeatable:
            self.achieved = True
        return self.reward


def _standing_on_top(env, location, past_cells=[]):
    return location in past_cells and not env.screen_contains(location)


class LocActionEvent(Event):
    """An event which checks whether an action is performed at a specified
    location.
    """

    def __init__(
        self,
        *args,
        loc: str,
        action: Command,
    ):
        """Initialise the Event.

        Args:
            loc (str):
                The name of the location to reach.
            action (int):
                The action to perform.
            reward (float):
                The reward for the event occuring
            repeatable (bool):
                Whether the event can occur repeated (i.e. if the reward can be
                collected repeatedly
            terminal_required (bool):
                Whether this event is required for the episode to terminate.
            terminal_sufficient (bool):
                Whether this event causes the episode to terminate on its own.
        """
        super().__init__(*args)
        self.loc = loc
        self.action = action
        self.status = False

    def check(self, env, previous_observation, action, observation, past_cells=[]) -> float:
        del previous_observation, observation
        if env.actions[action] == self.action and _standing_on_top(
            env, self.loc, past_cells
        ):
            self.status = True
        elif env.actions[action] == Y_cmd and self.status:
            return self._set_achieved()
        else:
            self.status = False
        return 0

    def reset(self):
        super().reset()
        self.status = False


class LocEvent(Event):
    """An event which checks whether a specified location is reached."""

    def __init__(self, *args, loc: str):
        super().__init__(*args)
        """Initialise the Event.

        Args:
            loc (str):
                The name of the location to reach.
            reward (float):
                The reward for the event occuring
            repeatable (bool):
                Whether the event can occur repeated (i.e. if the reward can be
                collected repeatedly
            terminal_required (bool):
                Whether this event is required for the episode to terminate.
            terminal_sufficient (bool):
                Whether this event causes the episode to terminate on its own.
        """
        self.loc = loc

    def check(self, env, previous_observation, action, observation, past_cells=[]) -> float:
        del previous_observation, action, observation
        if _standing_on_top(env, self.loc, past_cells):
            return self._set_achieved()
        return 0.0


class CoordEvent(Event):
    """An event which occurs when reaching certain coordinates."""

    def __init__(self, *args, coordinates: Tuple[int, int]):
        """Initialise the Event.

        Args:
            coordinates (tuple):
                The coordinates to reach for the event.
            reward (float):
                The reward for the event occuring
            repeatable (bool):
                Whether the event can occur repeated (i.e. if the reward can be
                collected repeatedly
            terminal_required (bool):
                Whether this event is required for the episode to terminate.
            terminal_sufficient (bool):
                Whether this event causes the episode to terminate on its own.
        """
        super().__init__(*args)
        self.coordinates = coordinates

    def check(self, env, previous_observation, action, observation, past_cells=[]) -> float:
        coordinates = tuple(observation[env._blstats_index][:2])
        # print(f"Current coordinates: {observation[env._blstats_index][:2]}")
        if self.coordinates == coordinates:
            return self._set_achieved()
        return 0.0


class MessageEvent(Event):
    """An event which occurs when any of the `messages` appear."""

    def __init__(self, *args, messages: List[str]):
        """Initialise the Event.

        Args:
            messages (list):
                The messages to be seen to trigger the event.
            reward (float):
                The reward for the event occuring
            repeatable (bool):
                Whether the event can occur repeated (i.e. if the reward can be
                collected repeatedly
            terminal_required (bool):
                Whether this event is required for the episode to terminate.
            terminal_sufficient (bool):
                Whether this event causes the episode to terminate on its own.
        """
        super().__init__(*args)
        self.messages = messages

    def check(self, env, previous_observation, action, observation, past_cells=[]) -> float:
        del previous_observation, action
        try:
          msg = observation[env._original_observation_keys.index("message")]
          curr_msg = msg[: np.where(msg == 0)[0][0]].tobytes().decode("utf-8")
          for msg in self.messages:
              if msg in curr_msg:
                  return self._set_achieved()
        except:
          print("Failed to decode message:")
          print(observation[env._original_observation_keys.index("message")])
        return 0.0


class AbstractRewardManager(ABC):
    """This is the abstract base class for the ``RewardManager`` that is used
    for defining custom reward functions.
    """

    def __init__(self):
        self.terminal_sufficient = None
        self.terminal_required = None

    @abstractmethod
    def collect_reward(self) -> float:
        """Return reward calculated and accumulated in check_episode_end_call,
        and then reset it.

        Returns:
            flaot: The reward.
        """
        raise NotImplementedError

    @abstractmethod
    def check_episode_end_call(
        self, env, previous_observation, action, observation
    ) -> bool:
        """Check if the task has ended, and accumulate any reward from the
        transition in ``self._reward``.

        Args:
            env (MiniHack):
                The MiniHack environment in question.
            previous_observation (tuple):
                The previous state observation.
            action (int):
                The action taken.
            observation (tuple):
                The current observation.
        Returns:
            bool: Boolean whether the episode has ended.

        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset all events, to be called when a new episode occurs."""
        raise NotImplementedError


class RewardManager(AbstractRewardManager):
    """This class is used for managing rewards, events and termination for
    MiniHack tasks.

    Some notes on the ordering or calls in the MiniHack/NetHack base class:

    - ``step(action)`` is called on the environment
    - Within ``step``, first a copy of the last observation is made, and then the
      underlying NetHack game is stepped
    - Then ``_is_episode_end(observation)`` is called to check whether this the
      episode has ended (and this is overridden if we've gone over our
      max_steps, or the underlying NetHack game says we're done (i.e. we died)
    - Then ``_reward_fn(last_observation, observation)`` is called to calculate
      the reward at this time-step
    - if ``end_status`` tells us the game is done, we quit the game
    - then ``step`` returns the observation, calculated reward, done, and some
    statistics.

    All this means that we need to check whether an observation is terminal in
    ``_is_episode_end`` before we're calculating the reward function.

    The call of ``_is_episode_end`` in ``MiniHack`` will call
    ``check_episode_end_call`` in this class, which checks for termination and
    accumulates any reward, which is returned and zeroed in ``collect_reward``.
    """

    def __init__(self):
        self.events: List[Event] = []
        self.custom_reward_functions: List[
            Callable[[MiniHack, Any, int, Any], float]
        ] = []
        self._reward = 0.0
        self.past_cells = []

        # Only used for GroupedRewardManager
        self.terminal_sufficient = None
        self.terminal_required = None

    def add_custom_reward_fn(
        self, reward_fn: Callable[[MiniHack, Any, int, Any], float]
    ) -> None:
        """Add a custom reward function which is called every after step to
        calculate reward.

        The function should be a callable which takes the environment, previous
        observation, action and current observation and returns a float reward.

        Args:
            reward_fn (Callable[[MiniHack, Any, int, Any], float]):
                A reward function which takes an environment, previous
                observation, action, next observation and returns a reward.

        """
        self.custom_reward_functions.append(reward_fn)

    def add_event(self, event: Event):
        """Add an event to be managed by the reward manager.

        Args:
            event (Event):
                The event to be added.
        """
        self.events.append(event)

    def _add_message_event(
        self, msgs, reward, repeatable, terminal_required, terminal_sufficient
    ):
        self.add_event(
            MessageEvent(
                reward,
                repeatable,
                terminal_required,
                terminal_sufficient,
                messages=msgs,
            )
        )

    def _add_loc_action_event(
        self,
        loc,
        action,
        reward,
        repeatable,
        terminal_required,
        terminal_sufficient,
    ):
        try:
            action = Command[action.upper()]
        except KeyError:
            raise KeyError(
                "Action {} is not in the action space.".format(action.upper())
            )

        self.add_event(
            LocActionEvent(
                reward,
                repeatable,
                terminal_required,
                terminal_sufficient,
                loc=loc.lower(),
                action=action,
            )
        )

    def add_eat_event(
        self,
        name: str,
        reward=1,
        repeatable=False,
        terminal_required=True,
        terminal_sufficient=False,
    ):
        """Add an event which is triggered when `name` is eaten.

        Args:
            name (str):
                The name of the object being eaten.
            reward (float):
                The reward for this event. Defaults to 1.
            repeatable (bool):
                Whether this event can be triggered multiple times. Defaults to
                False.
            terminal_required (bool):
                Whether this event is required for termination. Defaults to
                True.
            terminal_sufficient (bool):
                Whether this event is sufficient for termination. Defaults to
                False.
        """
        msgs = [
            f"This {name} is delicious",
            "Blecch!  Rotten food!",
            "last bite of your meal",
        ]
        if name == "apple":
            msgs.append("Delicious!  Must be a Macintosh!")
            msgs.append("Core dumped.")
        if name == "pear":
            msgs.append("Core dumped.")

        self._add_message_event(
            msgs, reward, repeatable, terminal_required, terminal_sufficient
        )

    def add_wield_event(
        self,
        name: str,
        reward=1,
        repeatable=False,
        terminal_required=True,
        terminal_sufficient=False,
    ):
        """Add event which is triggered when a specific weapon is wielded.

        Args:
            name (str):
                The name of the weapon to be wielded.
            reward (float):
                The reward for this event. Defaults to 1.
            repeatable (bool):
                Whether this event can be triggered multiple times. Defaults to
                False.
            terminal_required (bool):
                Whether this event is required for termination. Defaults to
                True.
            terminal_sufficient (bool):
                Whether this event is sufficient for termination. Defaults to
                False.
        """
        msgs = [
            f"{name} wields itself to your hand!",
            f"{name} (weapon in hand)",
        ]
        self._add_message_event(
            msgs, reward, repeatable, terminal_required, terminal_sufficient
        )

    def add_wear_event(
        self,
        name: str,
        reward=1,
        repeatable=False,
        terminal_required=True,
        terminal_sufficient=False,
    ):
        """Add event which is triggered when a specific armor is worn.

        Args:
            name (str):
                The name of the armor to be worn.
            reward (float):
                The reward for this event. Defaults to 1.
            repeatable (bool):
                Whether this event can be triggered multiple times. Defaults to
                False.
            terminal_required (bool):
                Whether this event is required for termination. Defaults to
                True.
            terminal_sufficient (bool):
                Whether this event is sufficient for termination. Defaults to
                False.
        """
        msgs = [f"You are now wearing a {name}"]
        self._add_message_event(
            msgs, reward, repeatable, terminal_required, terminal_sufficient
        )

    def add_amulet_event(
        self,
        reward=1,
        repeatable=False,
        terminal_required=True,
        terminal_sufficient=False,
    ):
        """Add event which is triggered when an amulet is worn.

        Args:
            reward (float):
                The reward for this event. Defaults to 1.
            repeatable (bool):
                Whether this event can be triggered multiple times. Defaults to
                False.
            terminal_required (bool):
                Whether this event is required for termination. Defaults to
                True.
            terminal_sufficient (bool):
                Whether this event is sufficient for termination. Defaults to
                False.
        """
        self._add_message_event(
            ["amulet (being worn)."],
            reward,
            repeatable,
            terminal_required,
            terminal_sufficient,
        )

    def add_kill_event(
        self,
        name: str,
        reward=1,
        repeatable=False,
        terminal_required=True,
        terminal_sufficient=False,
    ):
        """Add event which is triggered when a specified monster is killed.

        Args:
            name (str):
                The name of the monster to be killed.
            reward (float):
                The reward for this event. Defaults to 1.
            repeatable (bool):
                Whether this event can be triggered multiple times. Defaults to
                False.
            terminal_required (bool):
                Whether this event is required for termination. Defaults to
                True.
            terminal_sufficient (bool):
                Whether this event is sufficient for termination. Defaults to
                False.
        """
        self._add_message_event(
            [f"You kill the {name}"],
            reward,
            repeatable,
            terminal_required,
            terminal_sufficient,
        )

    def add_message_event(
        self,
        msgs: List[str],
        reward=1,
        repeatable=False,
        terminal_required=True,
        terminal_sufficient=False,
    ):
        """Add event which is triggered when any of the given messages are seen.

        Args:
            msgs (List[str]):
                The name of the monster to be killed.
            reward (float):
                The reward for this event. Defaults to 1.
            repeatable (bool):
                Whether this event can be triggered multiple times. Defaults to
                False.
            terminal_required (bool):
                Whether this event is required for termination. Defaults to
                True.
            terminal_sufficient (bool):
                Whether this event is sufficient for termination. Defaults to
                False.
        """
        self._add_message_event(
            msgs, reward, repeatable, terminal_required, terminal_sufficient
        )

    def add_positional_event(
        self,
        place_name: str,
        action_name: str,
        reward=1,
        repeatable=False,
        terminal_required=True,
        terminal_sufficient=False,
    ):
        """Add event which is triggered on taking a given action at a given place.

        Args:
            place_name (str):
                The name of the place to trigger the event.
            action_name (int):
                The name of the action to trigger the event.
            reward (float):
                The reward for this event. Defaults to 1.
            repeatable (bool):
                Whether this event can be triggered multiple times. Defaults to
                False.
            terminal_required (bool):
                Whether this event is required for termination. Defaults to
                True.
            terminal_sufficient (bool):
                Whether this event is sufficient for termination. Defaults to
                False.
        """
        self._add_loc_action_event(
            place_name,
            action_name,
            reward,
            repeatable,
            terminal_required,
            terminal_sufficient,
        )

    def add_coordinate_event(
        self,
        coordinates: Tuple[int, int],
        reward=1,
        repeatable=False,
        terminal_required=True,
        terminal_sufficient=False,
    ):
        """Add event which is triggered on when reaching the specified
        coordinates.

        Args:
            coordinates (Tuple[int, int]):
                The coordinates to be reached (tuple of ints).
            reward (float):
                The reward for this event. Defaults to 1.
            repeatable (bool):
                Whether this event can be triggered multiple times. Defaults to
                False.
            terminal_required (bool):
                Whether this event is required for termination. Defaults to
                True.
            terminal_sufficient (bool):
                Whether this event is sufficient for termination. Defaults to
                False.
        """
        self.add_event(
            CoordEvent(
                reward,
                repeatable,
                terminal_required,
                terminal_sufficient,
                coordinates=coordinates,
            )
        )

    def add_location_event(
        self,
        location: str,
        reward=1,
        repeatable=False,
        terminal_required=True,
        terminal_sufficient=False,
    ):
        """Add event which is triggered on reaching a specified location.

        Args:
            name (str):
                The name of the location to be reached.
            reward (float):
                The reward for this event. Defaults to 1.
            repeatable (bool):
                Whether this event can be triggered multiple times. Defaults to
                False.
            terminal_required (bool):
                Whether this event is required for termination. Defaults to
                True.
            terminal_sufficient (bool):
                Whether this event is sufficient for termination. Defaults to
                False.
        """
        self.add_event(
            LocEvent(
                reward,
                repeatable,
                terminal_required,
                terminal_sufficient,
                loc=location,
            )
        )

    def _set_achieved(self, event: Event) -> float:
        if not event.repeatable:
            event.achieved = True
        return event.reward

    def _standing_on_top(self, env, name, past_cells=[]):
        """Returns whether the agents is standing on top of the given object.
        The object name (e.g. altar, sink, fountain) must exist on the map.

        Args:
            env (MiniHack):
                The environment object.
            name (str):
                The name of the object.

        Returns:
            bool: True if the object name is not in the screen descriptions
            with agent info taking the space of the corresponding tile rather
            than the object).
        """

        return name in past_cells and not env.screen_contains(name)

    def check_episode_end_call(
        self, env, previous_observation, action, observation
    ) -> bool:
        reward = 0.0
        for event in self.events:
            if event.achieved:
                continue
            reward += event.check(
                env, previous_observation, action, observation, self.past_cells
            )

        for custom_reward_function in self.custom_reward_functions:
            reward += custom_reward_function(
                env, previous_observation, action, observation
            )
        # ADDITION of neighbouring cells
        self.past_cells = env.get_neighbor_descriptions()
        self._reward = reward
        return self._check_complete()

    def _check_complete(self) -> bool:
        """Checks whether the episode is complete.

        Requires any event which is sufficient to be achieved, OR all required
        events to be achieved."""
        result = True
        for event in self.events:
            # This event is enough, we're done
            if event.achieved and event.terminal_sufficient:
                return True
            # We need this event and we haven't done it, we're not done
            if not event.achieved and event.terminal_required:
                result = False

        # We've achieved all terminal_required events, we're done
        return result

    def collect_reward(self) -> float:
        result = self._reward
        self._reward = 0.0
        return result

    def reset(self):
        self._reward = -1.0
        for event in self.events:
            event.reset()

class InventoryEvent(Event):
    """An event which checks whether a specified object is in the inventory."""

    def __init__(self, *args, inv_item: str):
        super().__init__(*args)
        """Initialise the Event.

        Args:
            inv_item (str):
                The name of the object to gain.
            reward (float):
                The reward for the event occuring
            repeatable (bool):
                Whether the event can occur repeated (i.e. if the reward can be
                collected repeatedly
            terminal_required (bool):
                Whether this event is required for the episode to terminate.
            terminal_sufficient (bool):
                Whether this event causes the episode to terminate on its own.
        """
        self.inv_item = inv_item

    def check(self, env, previous_observation, action, observation, past_cells=[]) -> float:
        del previous_observation, action, observation
        inventory_items = observation[env._original_observation_keys.index("inv_strs")]
        for inv_item in inventory_items:
          if self.inv_item in inv_item[: np.where(inv_item == 0)[0][0]].tobytes().decode("utf-8"):
            return self._set_achieved()
        return 0.0
    

class RelativeCoordEvent(Event):
    """An event which occurs when the agent's coordinates move relative to a start position."""

    def __init__(self, *args):
        """Initialise the Event.

        Args:
            reward (float):
                The reward for the event occuring
            repeatable (bool):
                Whether the event can occur repeated (i.e. if the reward can be
                collected repeatedly
            terminal_required (bool):
                Whether this event is required for the episode to terminate.
            terminal_sufficient (bool):
                Whether this event causes the episode to terminate on its own.
        """
        super().__init__(*args)
        self.current_coordinates = None

    def check(self, env, previous_observation, action, observation, past_cells=[]) -> float:
        del previous_observation, observation
        coordinates = tuple(observation[env._blstats_index][:2])
        if self.current_coordinates is None:
            self.current_coordinates = coordinates
            return 0.0
        if self.current_coordinates[0] < coordinates[0]:
            self.current_coordinates = coordinates
            return self._set_achieved()
        return 0.0
    
class NeighbourEvent(Event):
    """An event which checks whether an item is next to the agent."""

    def __init__(self, *args, item_search: str):
        super().__init__(*args)
        """Initialise the Event.

        Args:
            item_search (str):
                The name of the item that should be next to the agent.
            reward (float):
                The reward for the event occuring
            repeatable (bool):
                Whether the event can occur repeated (i.e. if the reward can be
                collected repeatedly
            terminal_required (bool):
                Whether this event is required for the episode to terminate.
            terminal_sufficient (bool):
                Whether this event causes the episode to terminate on its own.
        """
        self.item_search = item_search

    def check(self, env, previous_observation, action, observation, past_cells=[]) -> float:
        del previous_observation, observation
        neighbour_cells = env.get_neighbor_descriptions()
        if self.item_search in neighbour_cells:
            return self._set_achieved()
        return 0.0