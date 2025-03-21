from graph_rl.graph.extended_graph import ExtendedGraph
import igraph as ig
import numpy as np
from typing import Optional, Union, Callable, List
import random
from abc import ABC, abstractmethod


class _BaseGraphGenerator(ABC):
    """
    Base class for graph generators with common functionality.
    """

    def __init__(
        self,
        n_vertices: int,
        directed: bool = False,
        node_embedding_dim: int = 32,
        edge_embedding_dim: int = 16,
        seed: Optional[int] = None,
    ):
        """
        Initialize the base graph generator.

        Args:
            n_vertices: Number of vertices in the graph
            directed: Whether the graph is directed
            node_embedding_dim: Dimension of node embeddings
            edge_embedding_dim: Dimension of edge embeddings
            seed: Random seed for reproducibility
        """
        self.n_vertices = n_vertices
        self.directed = directed
        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _create_base_graph(self) -> ExtendedGraph:
        """Create a base empty graph with the specified number of vertices."""
        return ExtendedGraph(n=self.n_vertices, directed=self.directed)

    def _generate_embeddings(self, graph: ExtendedGraph) -> ExtendedGraph:
        """Generate random embeddings for nodes and edges."""
        assert graph.node_embedding_dim == self.node_embedding_dim, (
            f"node_embedding_dim have to be equal, graph:{graph.node_embedding_dim} != self:{self.node_embedding_dim}"
        )
        assert graph.edge_embedding_dim == self.edge_embedding_dim, (
            f"edge_embedding_dim have to be equal, graph:{graph.edge_embedding_dim} != self:{self.edge_embedding_dim}"
        )
        # Generate node embeddings
        for i in range(self.n_vertices):
            graph.vs[i]["embedding"] = np.random.randn(self.node_embedding_dim)

        # Generate edge embeddings
        for edge in graph.es:
            edge["embedding"] = np.random.randn(self.edge_embedding_dim)

        return graph

    def add_stochastic_transition(
        self,
        graph: ig.Graph,
        transition_fn: Optional[
            Callable[[ig.Vertex, ig.Vertex, ig.Graph], float]
        ] = None,
    ) -> ig.Graph:
        """
        Add stochastic transition probabilities to the edges.

        Args:
            graph: The graph to add transition probabilities to
            transition_fn: A custom function that takes source and target node
                           and returns a transition weight. Later those weights will
                           be normalized into a probability function across outgoing
                           nodes

        Returns:
            Graph with transition probabilities added to edges
        """
        if transition_fn is None:
            # Default function: uniform random probabilities
            def transition_fn(source, target, graph):
                return np.random.uniform(0, 1)

        # Assign transition probabilities to edges
        for edge in graph.es:
            source, target = edge.tuple
            edge["transition_prob"] = transition_fn(source, target, graph)

        # Normalize transition probabilities for each node
        for v in graph.vs:
            out_edges = graph.es.select(_source=v.index)
            if len(out_edges) == 0:
                continue
            total_prob = sum(edge["transition_prob"] for edge in out_edges)
            for edge in out_edges:
                edge["transition_prob"] /= total_prob

        return graph

    @abstractmethod
    def generate(self) -> ig.Graph:
        """Generate the graph. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the generate method.")


class ErdosRenyiGenerator(_BaseGraphGenerator):
    """
    Generator for Erdos-Renyi random graphs.

    Example:
    >>> generator = ErdosRenyiGenerator(
            n_vertices=100,
            p=0.1,
            directed=False,
            node_embedding_dim=32,
            edge_embedding_dim=16,
            seed=42,
        )
    >>> generator.generate()
    """

    def __init__(
        self,
        n_vertices: int,
        p: float = 0.1,
        directed: bool = False,
        node_embedding_dim: int = 32,
        edge_embedding_dim: int = 16,
        seed: Optional[int] = None,
    ):
        """
        Initialize Erdos-Renyi generator.

        Args:
            n_vertices: Number of vertices
            p: Probability of edge creation
            directed: Whether the graph is directed
            node_embedding_dim: Dimension of node embeddings
            edge_embedding_dim: Dimension of edge embeddings
            seed: Random seed for reproducibility
        """
        super().__init__(
            n_vertices, directed, node_embedding_dim, edge_embedding_dim, seed
        )
        self.p = p

    def generate(self) -> ExtendedGraph:
        """Generate an Erdos-Renyi graph."""
        # Create graph using the G(n,p) model
        graph: ExtendedGraph = ExtendedGraph.Erdos_Renyi(
            n=self.n_vertices,
            p=self.p,
            directed=self.directed,
        )
        graph.node_embedding_dim = self.node_embedding_dim
        graph.edge_embedding_dim = self.edge_embedding_dim

        # Generate embeddings
        graph = self._generate_embeddings(graph)

        return graph


class WattsStrogatzGenerator(_BaseGraphGenerator):
    """
    Generator for Watts-Strogatz small-world graphs.
    
    Example:
    >>> generator = WattsStrogatzGenerator(
            n_vertices=100,
            k=4,
            p=0.1,
            directed=False,
            node_embedding_dim=32,
            edge_embedding_dim=16,
            seed=42,
        )
    >>> generator.generate()

    """

    def __init__(
        self,
        n_vertices: int,
        k: int = 4,
        p: float = 0.1,
        directed: bool = False,
        node_embedding_dim: int = 32,
        edge_embedding_dim: int = 16,
        seed: Optional[int] = None,
    ):
        """
        Initialize Watts-Strogatz generator.

        Args:
            n_vertices: Number of vertices
            k: Each node is connected to k nearest neighbors in ring topology
            p: Rewiring probability
            directed: Whether the graph is directed
            node_embedding_dim: Dimension of node embeddings
            edge_embedding_dim: Dimension of edge embeddings
            seed: Random seed for reproducibility
        """
        super().__init__(
            n_vertices, directed, node_embedding_dim, edge_embedding_dim, seed
        )
        self.k = k
        self.p = p

    def generate(self) -> ExtendedGraph:
        """Generate a Watts-Strogatz small-world graph."""
        # k must be even for the Watts-Strogatz model
        if self.k % 2 != 0:
            self.k += 1

        # Create graph
        graph: ExtendedGraph = ExtendedGraph.Watts_Strogatz(
            dim=1,
            size=self.n_vertices,
            nei=self.k // 2,
            p=self.p,
            loops=False,
            multiple=False,
        )
        graph.node_embedding_dim = self.node_embedding_dim
        graph.edge_embedding_dim = self.edge_embedding_dim

        # Set directionality
        if self.directed:
            graph.to_directed()

        # Generate embeddings
        graph = self._generate_embeddings(graph)

        return graph


class BarabasiAlbertGenerator(_BaseGraphGenerator):
    """
    Generator for Barabasi-Albert scale-free networks.

    Example:
    >>> generator = BarabasiAlbertGenerator(
            n_vertices=100,
            m=2,
            directed=False,
            node_embedding_dim=32,
            edge_embedding_dim=16,
            seed=42,
        )
    >>> generator.generate()
    """

    def __init__(
        self,
        n_vertices: int,
        m: int = 2,
        directed: bool = False,
        node_embedding_dim: int = 32,
        edge_embedding_dim: int = 16,
        seed: Optional[int] = None,
    ):
        """
        Initialize Barabasi-Albert generator.

        Args:
            n_vertices: Number of vertices
            m: Number of edges to add in each step
            directed: Whether the graph is directed
            node_embedding_dim: Dimension of node embeddings
            edge_embedding_dim: Dimension of edge embeddings
            seed: Random seed for reproducibility
        """
        super().__init__(
            n_vertices, directed, node_embedding_dim, edge_embedding_dim, seed
        )
        self.m = m

    def generate(self) -> ExtendedGraph:
        """Generate a Barabasi-Albert scale-free network."""
        # Create graph
        graph: ExtendedGraph = ExtendedGraph.Barabasi(
            n=self.n_vertices,
            m=self.m,
            directed=self.directed,
            node_embedding_dim=self.node_embedding_dim,
            edge_embedding_dim=self.edge_embedding_dim,
            seed=self.seed,
        )
        graph.node_embedding_dim = self.node_embedding_dim
        graph.edge_embedding_dim = self.edge_embedding_dim

        # Generate embeddings
        graph = self._generate_embeddings(graph)

        return graph


class StochasticBlockModelGenerator(_BaseGraphGenerator):
    """
    Generator for Stochastic Block Model graphs for community detection.

    Example:
    >>> generator = StochasticBlockModelGenerator(
            n_vertices=100,
            block_sizes=[25, 25, 25, 25],
            p_matrix=[
                [0.5, 0.1, 0.1, 0.1],
                [0.1, 0.5, 0.1, 0.1],
                [0.1, 0.1, 0.5, 0.1],
                [0.1, 0.1, 0.1, 0.5],
            ],
            directed=False,
            node_embedding_dim=32,
            edge_embedding_dim=16,
            seed=42,
        )
    >>> generator.generate()

    """

    def __init__(
        self,
        n_vertices: int,
        block_sizes: List[int],
        p_matrix: List[List[float]],
        directed: bool = False,
        node_embedding_dim: int = 32,
        edge_embedding_dim: int = 16,
        seed: Optional[int] = None,
    ):
        """
        Initialize Stochastic Block Model generator.

        Args:
            n_vertices: Number of vertices (should equal sum of block_sizes)
            block_sizes: List of community sizes
            p_matrix: Matrix of inter/intra community edge probabilities
            directed: Whether the graph is directed
            node_embedding_dim: Dimension of node embeddings
            edge_embedding_dim: Dimension of edge embeddings
            seed: Random seed for reproducibility
        """
        super().__init__(
            n_vertices, directed, node_embedding_dim, edge_embedding_dim, seed
        )
        self.block_sizes = block_sizes
        self.p_matrix = p_matrix

        # Verify that block sizes sum to n_vertices
        if sum(block_sizes) != n_vertices:
            raise ValueError("Sum of block sizes must equal n_vertices")

        # Verify dimensions of p_matrix
        n_blocks = len(block_sizes)
        if len(p_matrix) != n_blocks or any(len(row) != n_blocks for row in p_matrix):
            raise ValueError(
                "p_matrix must be a square matrix with dimensions matching the number of blocks"
            )

    def generate(self) -> ExtendedGraph:
        """Generate a Stochastic Block Model graph."""
        # Create graph using SBM
        graph: ExtendedGraph = ExtendedGraph.SBM(
            n=self.n_vertices,
            pref_matrix=self.p_matrix,
            block_sizes=self.block_sizes,
            directed=self.directed,
            loops=False,
        )
        graph.node_embedding_dim = self.node_embedding_dim
        graph.edge_embedding_dim = self.edge_embedding_dim

        # Add community labels to vertices
        community_id = 0
        vertex_id = 0
        for size in self.block_sizes:
            for _ in range(size):
                graph.vs[vertex_id]["community"] = community_id
                vertex_id += 1
            community_id += 1

        # Generate embeddings
        graph = self._generate_embeddings(graph)

        return graph


class LatticeGenerator(_BaseGraphGenerator):
    """
    Generator for regular lattice graphs.

    Example:
    >>> generator = ForestFireGenerator(
            dim=2,
            size=10,
            nei=1,
            directed=False,
            node_embedding_dim=32,
            edge_embedding_dim=16,
            seed=42,
        )
    >>> generator.generate()

    """

    def __init__(
        self,
        dim: int = 2,
        size: Union[int, List[int]] = 10,
        nei: int = 1,
        directed: bool = False,
        node_embedding_dim: int = 32,
        edge_embedding_dim: int = 16,
        seed: Optional[int] = None,
    ):
        """
        Initialize Lattice graph generator.

        Args:
            dim: Dimension of the lattice
            size: Size(s) of the lattice in each dimension
            nei: Neighborhood degree
            directed: Whether the graph is directed
            node_embedding_dim: Dimension of node embeddings
            edge_embedding_dim: Dimension of edge embeddings
            seed: Random seed for reproducibility
        """
        # Calculate total number of vertices
        if isinstance(size, int):
            size = [size] * dim
            n_vertices = size[0] ** dim
        else:
            n_vertices = np.prod(size)

        super().__init__(
            n_vertices, directed, node_embedding_dim, edge_embedding_dim, seed
        )
        self.dim = dim
        self.size = size
        self.nei = nei

    def generate(self) -> ExtendedGraph:
        """Generate a lattice graph."""
        # Create lattice graph
        graph: ExtendedGraph = ExtendedGraph.Lattice(
            dim=self.dim,
            size=self.size,
            nei=self.nei,
            directed=self.directed,
            mutual=not self.directed,
            circular=False,
        )
        graph.node_embedding_dim = self.node_embedding_dim
        graph.edge_embedding_dim = self.edge_embedding_dim

        # Generate embeddings
        graph = self._generate_embeddings(graph)

        return graph


class ForestFireGenerator(_BaseGraphGenerator):
    """
    Generator for Forest Fire graphs (a model for growing networks).

    Example:
    >>> generator = ForestFireGenerator(
            fw_prob=0.2,
            bw_factor=0.2,
            ambs=2,
            directed=True,
            node_embedding_dim=32,
            edge_embedding_dim=16,
            seed=42
        )
    >>> er_graph = generator.generate()
    >>> er_graph = er_gen.add_stochastic_transition(er_graph)
    """

    def __init__(
        self,
        n_vertices: int,
        fw_prob: float = 0.2,
        bw_factor: float = 0.2,
        ambs: int = 2,
        directed: bool = True,
        node_embedding_dim: int = 32,
        edge_embedding_dim: int = 16,
        seed: Optional[int] = None,
    ):
        """
        Initialize Forest Fire graph generator.

        Args:
            n_vertices: Number of vertices
            fw_prob: Forward burning probability
            bw_factor: Backward burning ratio
            ambs: Number of ambassador vertices
            directed: Whether the graph is directed (usually True for this model)
            node_embedding_dim: Dimension of node embeddings
            edge_embedding_dim: Dimension of edge embeddings
            seed: Random seed for reproducibility
        """
        super().__init__(
            n_vertices, directed, node_embedding_dim, edge_embedding_dim, seed
        )
        self.fw_prob = fw_prob
        self.bw_factor = bw_factor
        self.ambs = ambs

    def generate(self) -> ExtendedGraph:
        """Generate a Forest Fire graph."""
        # Create Forest Fire graph
        graph: ExtendedGraph = ExtendedGraph.Forest_Fire(
            n=self.n_vertices,
            fw_prob=self.fw_prob,
            bw_factor=self.bw_factor,
            ambs=self.ambs,
            directed=self.directed,
        )
        graph.node_embedding_dim = self.node_embedding_dim
        graph.edge_embedding_dim = self.edge_embedding_dim

        # Generate embeddings
        graph = self._generate_embeddings(graph)

        return graph
