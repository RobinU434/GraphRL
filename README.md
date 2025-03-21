# GraphRL

A flexible reinforcement learning environment for navigating and learning on graph-structured data, built on top of the Gymnasium interface.

## Overview

This repository provides a robust and flexible framework for reinforcement learning on graphs. It combines the efficiency of `igraph` with the standardized interfaces of `gymnasium` to create an environment where agents can learn to navigate graph structures and solve graph-based tasks.

Key features:

- Built on the Gymnasium API for compatibility with popular RL frameworks
- Uses igraph for efficient graph operations and algorithms
- Support for node and edge features/embeddings
- Customizable reward functions
- Support for stochastic transitions
- Multiple observation modes (dictionary or vector)
- Various graph generation models (Erdős–Rényi, Watts-Strogatz, Barabási–Albert, etc.)
- Direct integration with popular RL libraries like Stable Baselines3

## Installation

```bash
# Clone the repository
git clone https://github.com/username/rl-graph-navigation-gym.git
cd rl-graph-navigation-gym

# Install dependencies
poetry install
```

### Dependencies

- gymnasium
- python-igraph
- numpy
- stable-baselines3 (optional, for RL examples)
- networkx (optional, for visualization and conversion)

## Quick Start

Here's a simple example to get started:

```python
import igraph as ig
import numpy as np
from graph_rl.environment.graph_navigation import GraphNavigationEnv
from graph_rl.reward_function.reward_function import ShortestPathReward
from graph_rl.graph.extended_graph import ExtendedGraph
    
# Create a simple graph
g = ExtendedGraph(directed=False, node_feature_dim=4, edge_feature_dim=2)
g.add_vertices(10)
edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 9)]
g.add_edges(edges)

# Define a reward function - reach node 9
reward_fn = ShortestPathReward(target_node=9, success_reward=10.0)

# Create the environment
env = GraphNavigationEnv(
    graph=g,
    reward_function=reward_fn,
    max_steps=20,
    transition_noise=0.1,
)

# Run a simple episode with random actions
observation, info = env.reset(seed=42)
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    env.render()  # Console rendering
    done = terminated or truncated

print(f"Episode finished with total reward: {total_reward}")
```

## Environment Details

### `GraphNavigationEnv`

The main environment class that implements the Gymnasium interface.

```python
env = GraphNavigationEnv(
    graph=graph,                    # ExtendedGraph Graph object
    reward_function=reward_fn,      # Custom reward function 
    max_steps=100,                  # Maximum episode length
    transition_noise=0.0,           # Probability of random transitions
    render_mode='console',          # 'console' or 'rgb_array'
)
```

#### Observation Space

The observations space is by Default a dictionary with attributes:
- `current_node`: Index of the current node (`int`)
- `node_features`: Features of the current node (`ndarray`)
- `neighbors`: Indices of neighboring nodes (`ndarray`)
- `edge_features`: Features of edges to neighbors(`ndarray`)
- `valid_actions`: Mask for valid actions (`ndarray`)
- `weights`: Steps taken in the episode (`ndarray`)

You can also vectorize this space by using: 
```python
from graph_rl.environment.wrapper import ObsVectorizeWrapper

env = GraphNavigationEnv(
    graph=graph,                    # ExtendedGraph Graph object
    reward_function=reward_fn,      # Custom reward function 
    max_steps=100,                  # Maximum episode length
    transition_noise=0.0,           # Probability of random transitions
    render_mode='console',          # 'console' or 'rgb_array'
)
env = ObsVectorizeWrapper(env) 
```

#### Action Space

Actions correspond to selecting an outgoing edge from the current node.

### Custom Reward Functions

You can define custom reward functions by inheriting from the `RewardFunction` base class:

```python
class MyRewardFunction(RewardFunction):
    def __call__(self, current_node, action_taken, next_node, graph):
        # Custom reward logic
        return reward_value
    def is_done(self, current_node, action_taken, next_node, graph):
        # Custom done logic
        return done
        
```


### Graph Generators

The environment also allows you to generate several different types of graphs with builtin generators: 

- `ErdosRenyiGenerator`: Random graphs
- `WattsStrogatzGenerator`: Small-world networks
- `BarabasiAlbertGenerator`: Scale-free networks
- `StochasticBlockModelGenerator`: Community structure
- `LatticeGenerator`: Regular lattices
- `ForestFireGenerator`: Growing networks with forest fire model

## Integration with Stable Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Check environment compatibility
check_env(env)

# Create and train a PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_graph_navigation")
```

## Examples and Use Cases

- Shortest path finding
- Traveling salesman problem
- Network flow optimization
- Influence maximization
- Recommender systems
- Knowledge graph navigation
- Simulating diffusion processes
- Learning graph embeddings

## Contributing

Contributions are welcome! Pleas            e feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE file](LICENSE) for details.