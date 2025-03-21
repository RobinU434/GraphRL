import gymnasium as gym
from gymnasium import spaces
import igraph as ig
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Union, Any
from abc import ABC, abstractmethod


class _RewardFunction(ABC):
    """
    Abstract base class for custom reward functions.
    Users can implement their own reward logic by inheriting from this class.
    """

    @abstractmethod
    def __call__(
        self, current_node: int, action_taken: int, next_node: int, graph: ig.Graph
    ) -> float:
        """
        Calculate the reward based on current state, action, and next state.

        Args:
            current_node: Index of the current node
            action_taken: Index of the edge or action taken
            next_node: Index of the next node
            graph: The igraph Graph instance

        Returns:
            float: The calculated reward
        """
        raise NotImplementedError

    @abstractmethod
    def is_done(
        self, current_node: int, action_taken: int, next_node: int, graph: ig.Graph
    ) -> bool:
        """
        Determine if the episode has terminated based on current state, action, and next state.

        Args:
            current_node: Index of the current node
            action_taken: Index of the edge or action taken
            next_node: Index of the next node
            graph: The igraph Graph instance

        Returns:
            bool: has episode terminated
        """
        raise NotImplementedError


class DefaultReward(_RewardFunction):
    """Default reward function implementation as an example."""

    def __call__(
        self, current_node: int, action_taken: int, next_node: int, graph: ig.Graph
    ) -> float:
        # Simple reward: -1 for each step to encourage short paths
        return -1.0
    
    def is_done(self, current_node, action_taken, next_node, graph):
        return False


class ShortestPathReward(_RewardFunction):
    """Reward function that encourages finding the shortest path to a target node."""

    def __init__(self, target_node: int, success_reward: float = 10.0):
        """
        Initialize the reward function.

        Args:
            target_node: The target node index to reach
            success_reward: Reward for reaching the target node
        """
        self.target_node = target_node
        self.success_reward = success_reward

    def __call__(
        self, current_node: int, action_taken: int, next_node: int, graph: ig.Graph
    ) -> float:
        # Base step penalty
        reward = -1.0

        # Check if reached target
        if next_node == self.target_node:
            reward += self.success_reward

        # Add reward based on edge weight (prefer lower weights)
        edge_weight = graph.es[action_taken]["weight"]
        reward -= edge_weight * 0.1

        return reward
    
    def is_done(self, current_node, action_taken, next_node, graph):
        return current_node == self.target_node
