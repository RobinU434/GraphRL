import igraph as ig
import numpy as np
from typing import Callable


def gaussian_feature_func_generator(
    salt_dim: int = 0,
) -> Callable[[ig.Vertex, ig.Graph], np.ndarray]:
    """return a function with node features.
    Embedding will be added to feature vector but is uncorrelated from real node embedding.

    Args:
        salt (int): which dimension the
    """

    def gaussian_feature_generator(vertex: ig.Vertex, graph: ig.Graph) -> np.ndarray:
        """generates a vector for node features which should be stored in attributes["features"] of the corresponding node

        Args:
            vertex (ig.Vertex): node to generate the feature vector for
            graph (ig.Graph): graph of the node

        Returns:
            np.ndarray: feature vector
        """
        degree = vertex.degree()
        clustering = graph.transitivity_local_undirected([vertex.index])[0]
        if np.isnan(clustering):
            clustering = 0
        res = np.array([degree, clustering])
        if salt_dim > 0:
            res = np.concat([res, np.random.randn((salt_dim,))])
        return res

    return gaussian_feature_generator
