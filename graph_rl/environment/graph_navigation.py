import gymnasium as gym
from gymnasium import spaces
import numpy as np

from graph_rl.graph.extended_graph import ExtendedGraph
from graph_rl.reward_function.reward_function import _RewardFunction, DefaultReward


class GraphNavigationEnv(gym.Env):
    """
    Reinforcement Learning environment for graph navigation problems
    implemented as a Gymnasium environment.

    Features:
    - Support for node and edge attributes
    - Stochastic transitions between nodes
    - Custom reward functions
    - Based on igraph for efficiency
    - Compliant with Gymnasium interface
    """

    metadata = {"render_modes": ["console", "rgb_array"]}

    def __init__(
        self,
        graph: ExtendedGraph = None,
        reward_function: _RewardFunction = None,
        max_steps: int = 100,
        transition_noise: float = 0.0,
        render_mode: str = "console",
        observation_mode: str = "dict",
    ):
        """
        Initialize the environment.

        Args:
            graph: igraph Graph object. If None, an empty graph is created.
            reward_function: Custom reward function. If None, DefaultReward is used.
            max_steps: Maximum number of steps per episode
            transition_noise: Probability of random transition (0.0 = deterministic)
            render_mode: Mode for rendering ('console' or 'rgb_array')
            observation_mode: Mode for observations ('dict' or 'vector')
            node_feature_dim: Dimension of node features for vector mode
            edge_feature_dim: Dimension of edge features for vector mode
        """
        super().__init__()

        # Initialize the graph
        self.graph = graph if graph is not None else ExtendedGraph()

        # Set reward function
        self.reward_function: _RewardFunction = (
            reward_function if reward_function is not None else DefaultReward()
        )

        # Environment parameters
        self.max_steps = max_steps
        self.transition_noise = transition_noise
        self.render_mode = render_mode
        self.observation_mode = observation_mode

        # Feature dimensions for vector observations
        self.node_feature_dim = self.graph.node_feature_dim
        self.edge_feature_dim = self.graph.edge_feature_dim

        # State variables
        self.current_node = None
        self.steps_taken = 0

        # Validate the graph
        self._validate_graph()

        # Define action and observation spaces
        self._define_spaces()

    def _validate_graph(self):
        """Validate the graph structure and add required attributes if missing."""
        # Check if graph is empty
        if self.graph.vcount() == 0:
            raise ValueError("Graph must contain at least one node")

        # Ensure all nodes have a feature attribute
        if "features" not in self.graph.vertex_attributes():
            features = np.zeros((self.node_feature_dim, self.graph.vcount()))
            features[0] = np.arange(self.graph.vcount())
            self.graph.vs["features"] = features.T
        # Ensure all node features are of the same size = self.node_feature_dim
        else:
            try:
                stack = np.stack(self.graph.vs["features"])
                assert stack.shape == (self.graph.vcount(), self.node_feature_dim)
            except (ValueError, AssertionError):
                raise ValueError(
                    f"all node feature elements must have dimension of: {self.node_feature_dim}"
                )

        # Ensure all edges have a feature attribute
        if "features" not in self.graph.edge_attributes():
            features = np.zeros((self.edge_feature_dim, self.graph.ecount()))
            features[0] = np.arange(self.graph.vcount())
            self.graph.es["features"] = features.T
        # Ensure all edge features are of the same size = self.edge_feature_dim
        else:
            try:
                stack = np.stack(self.graph.es["features"])
                assert stack.shape == (self.graph.ecount(), self.edge_feature_dim)
            except (ValueError, AssertionError):
                raise ValueError(
                    f"all edge feature elements must have dimension of: {self.edge_feature_dim}"
                )

        # Ensure all edges have weights if not present
        if "weight" not in self.graph.edge_attributes():
            self.graph.es["weight"] = [1.0] * self.graph.ecount()

        # Get the maximum number of outgoing edges from any node
        self.max_out_edges = max(
            len(self.graph.incident(v.index, mode="out")) for v in self.graph.vs
        )

        # For vector observations, determine feature dimensions if not specified
        if self.observation_mode == "vector":
            if self.node_feature_dim is None:
                # Count node attributes excluding 'id'
                attrs = [
                    attr for attr in self.graph.vertex_attributes() if attr != "id"
                ]
                self.node_feature_dim = len(attrs) + 1  # +1 for the node ID

            if self.edge_feature_dim is None:
                self.edge_feature_dim = len(self.graph.edge_attributes())

    def _define_spaces(self):
        """Define the action and observation spaces."""
        # Action space: discrete, with number of actions equal to max outgoing edges
        self.action_space = spaces.Discrete(self.max_out_edges)

        # Dictionary observation space
        self.observation_space = spaces.Dict(
            {
                "current_node": spaces.Discrete(self.graph.vcount()),
                "node_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.node_feature_dim,) if self.node_feature_dim else (1,),
                    dtype=float,
                ),
                "neighbors": spaces.Box(
                    low=0,
                    high=self.graph.vcount() - 1,
                    shape=(self.max_out_edges,),
                    dtype=np.int32,
                ),
                "edge_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_out_edges, self.edge_feature_dim)
                    if self.edge_feature_dim
                    else (self.max_out_edges, 1),
                    dtype=float,
                ),
                "valid_actions": spaces.Box(
                    low=0, high=1, shape=(self.max_out_edges,), dtype=np.int8
                ),
                "weights": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.max_out_edges,), dtype=float
                ),
            }
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.

        Args:
            seed: Random seed
            options: Additional options (can contain 'start_node')

        Returns:
            observation: The initial observation
            info: Additional information
        """
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset state variables
        self.steps_taken = 0

        # Get starting node from options if provided
        start_node = None
        if options and "start_node" in options:
            start_node = int(options["start_node"])

        # Set starting node
        if start_node is None:
            self.current_node = np.random.randint(0, self.graph.vcount())
        else:
            if start_node >= self.graph.vcount() or start_node < 0:
                raise ValueError(f"Invalid start_node index: {start_node}")
            self.current_node = start_node

        observation = self._get_observation()
        info = {"is_success": False, "steps_takes": self.steps_taken}

        return observation, info

    def step(self, action):
        """
        Take a step in the environment by following an edge.

        Args:
            action: Index of the edge to follow from current node

        Returns:
            observation: New observation after the step
            reward: Reward for this step
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated (e.g., due to max steps)
            info: Additional information
        """
        assert self.action_space.contains(action), (
            "Action space does not contain given action"
        )
        # Initialize info dict
        info = {"is_success": False}

        # Get available edges from current node
        available_edges = self.graph.incident(self.current_node, mode="out")
        # Check if action is valid
        if action not in available_edges:
            # Invalid action, stay in place and receive penalty
            reward = -10.0  # Penalty for invalid action
            info["is_valid_action"] = False
            info["message"] = "Invalid action selected"
            next_node = self.current_node  # Stay in the same node
            terminated = False
        else:
            # Get the selected edge and its target
            selected_edge_id = available_edges[action]
            edge = self.graph.es[selected_edge_id]

            # Get source and target nodes
            target = self.graph.vs[edge.target]["id"]
            next_node = target

            # Apply stochastic transition if configured
            if self.transition_noise > 0 and np.random.random() < self.transition_noise:
                # Pick a random connected node instead of the intended target
                neighbors = self.graph.neighbors(self.current_node, mode="out")
                if neighbors:
                    next_node = np.random.choice(neighbors)

            # Calculate reward based on current state, action, and next state
            reward = self.reward_function(
                self.current_node, selected_edge_id, next_node, self.graph
            )
            # Reward-Function-determined termination
            terminated = self.reward_function.is_done(
                self.current_node, selected_edge_id, next_node, self.graph
            )

            info["is_valid_action"] = True
            info["edge_taken"] = selected_edge_id

        # Update current node
        self.current_node = next_node
        self.steps_taken += 1

        # Determine if episode has ended
        truncated = self.steps_taken >= self.max_steps  # Time limit reached

        # Get new observation
        observation = self._get_observation()

        # Additional info
        info["steps_taken"] = self.steps_taken

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """
        Construct the observation from the current state.

        Returns:
            Observation in the specified format (dict or vector)
        """
        # Get current node and its attributes
        node = self.graph.vs[self.current_node]

        # Get available actions (outgoing edges)
        available_edges = self.graph.incident(self.current_node, mode="out")

        # Create a mask for valid actions
        valid_actions = np.zeros(self.max_out_edges, dtype=np.int8)
        valid_actions[: len(available_edges)] = 1

        # Get neighbors connected by these edges
        neighbors = np.zeros(self.max_out_edges, dtype=np.int32)

        # Extract edge features and weights
        edge_features = np.zeros(
            (self.max_out_edges, self.edge_feature_dim if self.edge_feature_dim else 1)
        )
        weights = np.zeros(self.max_out_edges)
        for i, edge_id in enumerate(available_edges):
            edge = self.graph.es[edge_id]
            edge_features[i] = edge["features"]
            weights[i] = edge["weight"]
            neighbors[i] = (
                edge.target if edge.target != self.current_node else edge.source
            )

        # Extract node features
        node_features = node["features"]

        # Dictionary observation
        observation = {
            "current_node": self.current_node,
            "node_features": node_features,
            "neighbors": neighbors,
            "edge_features": edge_features,
            "valid_actions": valid_actions,
            "weights": weights,
        }

        return observation

    def render(self):
        """
        Render the current state of the environment.

        Returns:
            Depends on render_mode
        """
        if self.render_mode == "console":
            print(f"Current Node: {self.current_node}")
            print(
                f"Node Attributes: {dict(self.graph.vs[self.current_node].attributes())}"
            )

            # Get available actions
            available_edges = self.graph.incident(self.current_node, mode="out")
            print(f"Available Actions: {len(available_edges)}")

            for i, edge_id in enumerate(available_edges):
                edge = self.graph.es[edge_id]
                target = (
                    edge.target if edge.source == self.current_node else edge.source
                )
                print(
                    f"  Action {i}: -> Node {target} (Edge {edge_id}, "
                    f"Weight: {edge['weight'] if 'weight' in edge.attributes() else 'N/A'})"
                )

            print(f"Steps Taken: {self.steps_taken}/{self.max_steps}")

        elif self.render_mode == "rgb_array":
            # Placeholder for graphical rendering
            # Return a simple representation of the graph state
            # In a real implementation, you might use matplotlib or another library
            # to draw the graph with the current node highlighted
            width, height = 640, 480
            canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

            # This is just a minimal placeholder
            # In practice, implement proper graph visualization here
            return canvas

        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def close(self):
        """Clean up resources."""
        pass


# Example Custom Reward Function


# Example usage with stable-baselines3
def train_with_stable_baselines(env, total_timesteps=10000):
    """Example of how to use this environment with stable-baselines3."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env

        # First, check if the environment is valid
        check_env(env)

        # Create the model
        model = PPO("MlpPolicy", env, verbose=1)

        # Train the agent
        model.learn(total_timesteps=total_timesteps)

        # Save the model
        model.save("ppo_graph_navigation")

        print("Model trained and saved successfully!")
        return model

    except ImportError:
        print("stable-baselines3 not installed. Run: pip install stable-baselines3")
        return None
