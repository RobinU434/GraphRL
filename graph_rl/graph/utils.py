import igraph as ig
import numpy as np
from typing import Callable

def gaussian_feature_func_generator(embedding_dim: int) -> Callable[[ig.Vertex, ig.Graph], np.ndarray]:
    """return a function with node features. Embedding will be added to feature vector but is uncorrelated from real node embedding.

    Args:
        embedding_dim (int): which dimension the embedding has
    """

    def gaussian_feature_generator(vertex: ig.Vertex, graph: ig.Graph):
        degree = vertex.degree()
        clustering = graph.transitivity_local_undirected([vertex.index])[0]
        if np.isnan(clustering):
            clustering = 0
        return np.concat(
            [np.array([degree, clustering]), np.random.randn((embedding_dim,))]
        )

    return gaussian_feature_generator
