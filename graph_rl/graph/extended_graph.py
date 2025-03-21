import igraph as ig
import numpy as np
from typing import Optional, Callable, Dict, List, Tuple
import random
from graph_rl.graph.utils import gaussian_feature_func_generator
import networkx as nx


class ExtendedGraph(ig.Graph):
    """
    Extended Graph class that inherits from igraph's Graph class and adds
    additional functionality for features and stochastic transitions.
    """

    def __init__(
        self,
        *args,
        node_feature_dim: int = 32,
        edge_feature_dim: int = 16,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize an extended graph with feature dimensions.

        Args:
            *args: Arguments to pass to the igraph Graph constructor
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            seed: Random seed for reproducibility
            **kwargs: Keyword arguments to pass to the igraph Graph constructor
        """
        # Initialize the parent Graph class
        super().__init__(*args, **kwargs)

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def generate_features(self) -> "ExtendedGraph":
        """
        Generate random features for nodes and edges.

        Returns:
            Self for method chaining
        """
        # Generate node features
        for vertex in self.vs:
            vertex["features"] = np.random.randn(self.node_feature_dim)

        # Generate edge features
        for edge in self.es:
            edge["features"] = np.random.randn(self.edge_feature_dim)

        return self

    def add_stochastic_transition(
        self, transition_fn: Optional[Callable] = None
    ) -> "ExtendedGraph":
        """
        Add stochastic transition probabilities to edges.

        Args:
            transition_fn: Optional function to compute transition probabilities.
                          Takes (source, target, graph) and returns a probability.

        Returns:
            Self for method chaining
        """
        directed = self.is_directed()

        if transition_fn is None:
            # Default: random uniform probabilities
            def transition_fn(source, target, graph):
                return np.random.uniform(0, 1)

        # Assign transition probabilities
        for edge in self.es:
            source, target = edge.tuple
            edge["transition_prob"] = transition_fn(source, target, self)

        # Normalize probabilities
        if directed:
            for v in self.vs:
                out_edges = self.es.select(_source=v.index)
                if len(out_edges) > 0:
                    total_prob = sum(edge["transition_prob"] for edge in out_edges)
                    if total_prob > 0:  # Avoid division by zero
                        for edge in out_edges:
                            edge["transition_prob"] /= total_prob
        else:
            for v in self.vs:
                incident_edges = self.es.select(_incident=v.index)
                if len(incident_edges) > 0:
                    total_prob = sum(edge["transition_prob"] for edge in incident_edges)
                    if total_prob > 0:  # Avoid division by zero
                        for edge in incident_edges:
                            edge["transition_prob"] /= total_prob

        return self

    def get_transition_matrix(self) -> np.ndarray:
        """
        Get the transition probability matrix.

        Returns:
            Transition matrix as numpy array
        """
        n = self.vcount()
        trans_matrix = np.zeros((n, n))

        for edge in self.es:
            source, target = edge.tuple
            if "transition_prob" in edge.attributes():
                trans_matrix[source, target] = edge["transition_prob"]
            else:
                # If no transition probabilities defined, use uniform distribution
                out_degree = self.vs[source].outdegree()
                if out_degree > 0:
                    trans_matrix[source, target] = 1.0 / out_degree

        return trans_matrix

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the adjacency matrix of the graph.

        Returns:
            Adjacency matrix as numpy array
        """
        return np.array(self.get_adjacency().data)

    def get_community_labels(self) -> List[int]:
        """
        Get community labels for vertices if available.

        Returns:
            List of community labels
        """
        if "community" not in self.vs.attributes():
            # If no community labels, run community detection
            communities = self.community_multilevel()
            return communities.membership

        return [v["community"] for v in self.vs]

    def get_node_features(self) -> np.ndarray:
        """
        Get node features as a matrix.

        Returns:
            Matrix of node features where each row corresponds to a node
        """
        if "features" not in self.vs.attributes():
            raise ValueError(
                "Node features not found in the graph. Call generate_features() first."
            )

        n = self.vcount()
        emb_dim = len(self.vs[0]["features"])
        features = np.zeros((n, emb_dim))

        for i in range(n):
            features[i] = self.vs[i]["features"]

        return features

    def get_edge_features(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Get edge features as a dictionary.

        Returns:
            Dictionary mapping (source, target) to edge features
        """
        if "features" not in self.es.attributes():
            raise ValueError(
                "Edge features not found in the graph. Call generate_features() first."
            )

        features = {}
        for edge in self.es:
            source, target = edge.tuple
            features[(source, target)] = edge["features"]

        return features

    def detect_communities(self, algorithm: str = "multilevel") -> List[int]:
        """
        Detect communities in the graph using various algorithms.

        Args:
            algorithm: Community detection algorithm to use
                      ("multilevel", "label_propagation", "leading_eigenvector", "fast_greedy")

        Returns:
            List of community assignments for each vertex
        """
        if algorithm == "multilevel":
            communities = self.community_multilevel()
        elif algorithm == "label_propagation":
            communities = self.community_label_propagation()
        elif algorithm == "leading_eigenvector":
            communities = self.community_leading_eigenvector()
        elif algorithm == "fast_greedy":
            # Only for undirected graphs
            if self.is_directed():
                # Create an undirected copy for fast_greedy
                undirected = self.copy()
                undirected.to_undirected(combine_edges="sum")
                communities = undirected.community_fastgreedy().as_clustering()
            else:
                communities = self.community_fastgreedy().as_clustering()
        else:
            raise ValueError(f"Unknown community detection algorithm: {algorithm}")

        # Store community assignments as vertex attributes
        membership = communities.membership
        for i, comm_id in enumerate(membership):
            self.vs[i]["community"] = comm_id

        return membership

    def add_node_features(
        self, feature_generator: Callable[[ig.Vertex, ig.Graph], np.ndarray] = None
    ) -> "ExtendedGraph":
        """
        Add features to nodes using a custom generator function.

        Args:
            feature_generator: Function that takes a vertex and returns features
                              If None, uses random features

        Returns:
            Self for method chaining
        """
        if feature_generator is None:
            # Default: random features based on degree and local clustering
            feature_generator = gaussian_feature_func_generator(self.node_feature_dim)
        for vertex in self.vs:
            vertex["features"] = feature_generator(vertex, self)

        return self

    def to_networkx(self):
        """
        Convert to a NetworkX graph with all attributes preserved.

        Returns:
            NetworkX graph representation
        """
        # Determine if directed
        if self.is_directed():
            nx_graph = nx.DiGraph()
        else:
            nx_graph = nx.Graph()

        # Add nodes with attributes
        for v in self.vs:
            node_attrs = {
                key: v[key] for key in v.attributes() if not key.startswith("__")
            }
            nx_graph.add_node(v.index, **node_attrs)

        # Add edges with attributes
        for e in self.es:
            source, target = e.tuple
            edge_attrs = {
                key: e[key] for key in e.attributes() if not key.startswith("__")
            }
            nx_graph.add_edge(source, target, **edge_attrs)

        return nx_graph

    def plot_with_communities(self, filename=None, layout=None, **kwargs):
        """
        Plot the graph with community colors.

        Args:
            filename: File to save the plot (if None, shows interactive plot)
            layout: Graph layout to use (if None, uses Fruchterman-Reingold)
            **kwargs: Additional arguments to pass to plot()
        """
        if "community" not in self.vs.attributes():
            self.detect_communities()

        communities = [v["community"] for v in self.vs]
        max_comm = max(communities) + 1

        # Generate colors for communities
        community_colors = [
            f"hsv({h}, 0.8, 0.8)" for h in np.linspace(0, 0.9, max_comm)
        ]
        vertex_colors = [community_colors[c] for c in communities]

        if layout is None:
            layout = self.layout_fruchterman_reingold()

        visual_style = {
            "vertex_size": 10,
            "vertex_color": vertex_colors,
            "vertex_label": None,
            "edge_width": 0.5,
            "bbox": (400, 400),
            "margin": 40,
        }

        # Update with any user kwargs
        return self.plot(filename, layout, **visual_style)
    
    def plot(self, filename=None, layout=None, **kwargs):
        """
        Plot the graph.

        Args:
            filename: File to save the plot (if None, shows interactive plot)
            layout: Graph layout to use (if None, uses Fruchterman-Reingold)
            **kwargs: Additional arguments to pass to plot()
        """
        visual_style = {
            "vertex_size": 10,
            "vertex_label": None,
            "edge_width": 0.5,
            "layout": layout,
            "bbox": (400, 400),
            "margin": 40,
        }

        # Update with any user kwargs
        visual_style.update(kwargs)

        return ig.plot(
            ig.Graph(len(self.vs), self.get_edgelist(), directed=self.is_directed()),
            **visual_style,
        )
