from gymnasium import ObservationWrapper
import numpy as np

from graph_rl.environment.graph_navigation import GraphNavigationEnv
from gymnasium import spaces


class ObsVectorizeWrapper(ObservationWrapper):
    def __init__(self, env: GraphNavigationEnv):
        super().__init__(env)
        # 'vector' mode
        # Vector observation space
        # Structure: [current_node_id, node_features, valid_action_mask, edge_features, weights]

        # Calculate total dimension
        node_feat_dim = env.node_feature_dim if env.node_feature_dim else 1
        edge_feat_dim = env.edge_feature_dim if env.edge_feature_dim else 1
        total_dim = (
            1  # Current node ID
            + node_feat_dim  # Node features
            + env.max_out_edges  # Valid action mask
            + env.max_out_edges  # neighbors
            + (env.max_out_edges * edge_feat_dim)  # Edge features
            + env.max_out_edges  # weights
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,),dtype=float
        )

    def observation(self, observation):
        observation = np.concat(
            [
                np.array([observation["current_node"]]),
                observation["node_features"],
                observation["neighbors"],
                observation["edge_features"].flatten(),
                observation["valid_actions"],
                observation["weights"],
            ], dtype=float
        )
        return observation
