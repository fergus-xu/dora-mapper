"""Base classes for graph representations."""

from typing import Dict, List, Set, Any, Optional, TypeVar, Generic, Iterator
from abc import ABC, abstractmethod


class Node(ABC):
    """Base class for graph nodes."""

    def __init__(self, node_id: str, **attributes: Any) -> None:
        """
        Initialize a node.

        Args:
            node_id: Unique identifier for the node
            **attributes: Additional node attributes
        """
        self.id = node_id
        self.attributes: Dict[str, Any] = attributes

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get a node attribute."""
        return self.attributes.get(key, default)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a node attribute."""
        self.attributes[key] = value

    def to_string(self) -> str:
        """Return a detailed string representation of the node."""
        attrs_str = ", ".join(f"{k}={v}" for k, v in self.attributes.items())
        if attrs_str:
            return f"{self.__class__.__name__}(id={self.id}, {attrs_str})"
        return f"{self.__class__.__name__}(id={self.id})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id})"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.id == other.id


class Edge(ABC):
    """Base class for graph edges."""

    def __init__(
        self,
        edge_id: str,
        source: Node,
        destination: Node,
        **attributes: Any
    ) -> None:
        """
        Initialize an edge.

        Args:
            edge_id: Unique identifier for the edge
            source: Source node
            destination: Destination node
            **attributes: Additional edge attributes
        """
        self.id = edge_id
        self.source = source
        self.destination = destination
        self.attributes: Dict[str, Any] = attributes

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get an edge attribute."""
        return self.attributes.get(key, default)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an edge attribute."""
        self.attributes[key] = value

    def to_string(self) -> str:
        """Return a detailed string representation of the edge."""
        attrs_str = ", ".join(f"{k}={v}" for k, v in self.attributes.items())
        if attrs_str:
            return f"{self.__class__.__name__}({self.source.id} -> {self.destination.id}, {attrs_str})"
        return f"{self.__class__.__name__}({self.source.id} -> {self.destination.id})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.source.id} -> {self.destination.id})"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return False
        return self.id == other.id


N = TypeVar('N', bound=Node)
E = TypeVar('E', bound=Edge)


class Graph(ABC, Generic[N, E]):
    """Base class for graph representations."""

    def __init__(self, name: str = "Graph") -> None:
        """
        Initialize a graph.

        Args:
            name: Name of the graph
        """
        self.name = name
        self._nodes: Dict[str, N] = {}
        self._edges: Dict[str, E] = {}
        self._adjacency_list: Dict[str, Dict[str, Dict[str, Any]]] = {}  # node_id -> neighbor_id -> edge_data
        self._reverse_adjacency_list: Dict[str, Dict[str, Dict[str, Any]]] = {}  # node_id -> predecessor_id -> edge_data
        # NetworkX compatibility: graph-level attributes dict
        self.graph: Dict[str, Any] = {}

    def add_node(self, node: N) -> None:
        """Add a node to the graph."""
        if node.id in self._nodes:
            raise ValueError(f"Node {node.id} already exists in graph")
        self._nodes[node.id] = node
        self._adjacency_list[node.id] = {}
        self._reverse_adjacency_list[node.id] = {}

    def add_edge(self, edge: E) -> None:
        """Add an edge to the graph."""
        if edge.id in self._edges:
            raise ValueError(f"Edge {edge.id} already exists in graph")

        # Ensure source and destination nodes exist
        if edge.source.id not in self._nodes:
            self.add_node(edge.source)
        if edge.destination.id not in self._nodes:
            self.add_node(edge.destination)

        self._edges[edge.id] = edge

        # Store edge data in adjacency dict (NetworkX compatible format)
        edge_data = {
            'edge_id': edge.id,
            'weight': getattr(edge, 'latency', 1)
        }
        # Add any other edge attributes
        for key, value in edge.attributes.items():
            if key not in edge_data:
                edge_data[key] = value

        self._adjacency_list[edge.source.id][edge.destination.id] = edge_data
        self._reverse_adjacency_list[edge.destination.id][edge.source.id] = edge_data

    def get_node(self, node_id: str) -> Optional[N]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[E]:
        """Get an edge by ID."""
        return self._edges.get(edge_id)

    def get_nodes(self) -> List[N]:
        """Get all nodes in the graph."""
        return list(self._nodes.values())

    def get_edges(self) -> List[E]:
        """Get all edges in the graph."""
        return list(self._edges.values())

    def get_successors(self, node_id: str) -> List[N]:
        """Get all successor nodes of a given node."""
        neighbor_ids = self._adjacency_list.get(node_id, {}).keys()
        return [self._nodes[nid] for nid in neighbor_ids]

    def get_predecessors(self, node_id: str) -> List[N]:
        """Get all predecessor nodes of a given node."""
        predecessor_ids = self._reverse_adjacency_list.get(node_id, {}).keys()
        return [self._nodes[nid] for nid in predecessor_ids]

    def get_outgoing_edges(self, node_id: str) -> List[E]:
        """Get all outgoing edges from a node."""
        edge_data_dict = self._adjacency_list.get(node_id, {})
        edge_ids = [data['edge_id'] for data in edge_data_dict.values()]
        return [self._edges[eid] for eid in edge_ids]

    def get_incoming_edges(self, node_id: str) -> List[E]:
        """Get all incoming edges to a node."""
        edge_data_dict = self._reverse_adjacency_list.get(node_id, {})
        edge_ids = [data['edge_id'] for data in edge_data_dict.values()]
        return [self._edges[eid] for eid in edge_ids]

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its associated edges."""
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} not found in graph")

        # Collect all edge IDs connected to this node
        edges_to_remove = set()

        # Outgoing edges
        for edge_data in self._adjacency_list[node_id].values():
            edges_to_remove.add(edge_data['edge_id'])

        # Incoming edges
        for edge_data in self._reverse_adjacency_list[node_id].values():
            edges_to_remove.add(edge_data['edge_id'])

        # Remove all edges
        for edge_id in edges_to_remove:
            if edge_id in self._edges:
                self.remove_edge(edge_id)

        # Remove the node
        del self._nodes[node_id]
        del self._adjacency_list[node_id]
        del self._reverse_adjacency_list[node_id]

    def remove_edge(self, edge_id: str) -> None:
        """Remove an edge from the graph."""
        if edge_id not in self._edges:
            raise ValueError(f"Edge {edge_id} not found in graph")

        edge = self._edges[edge_id]

        # Remove from adjacency dicts
        if edge.destination.id in self._adjacency_list.get(edge.source.id, {}):
            del self._adjacency_list[edge.source.id][edge.destination.id]
        if edge.source.id in self._reverse_adjacency_list.get(edge.destination.id, {}):
            del self._reverse_adjacency_list[edge.destination.id][edge.source.id]

        del self._edges[edge_id]

    def num_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)

    def num_edges(self) -> int:
        """Return the number of edges in the graph."""
        return len(self._edges)

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self._nodes

    def has_edge(self, edge_id: str) -> bool:
        """Check if an edge exists in the graph."""
        return edge_id in self._edges

    def clear(self) -> None:
        """Remove all nodes and edges from the graph."""
        self._nodes.clear()
        self._edges.clear()
        self._adjacency_list.clear()
        self._reverse_adjacency_list.clear()

    def to_string(self) -> str:
        """Return a detailed string representation of the graph."""
        lines = []
        lines.append(f"{self.__class__.__name__}: {self.name}")
        lines.append(f"  Nodes: {self.num_nodes()}")
        lines.append(f"  Edges: {self.num_edges()}")
        lines.append("\n  Nodes:")
        for node in self.get_nodes():
            lines.append(f"    {node.to_string()}")
        lines.append("\n  Edges:")
        for edge in self.get_edges():
            lines.append(f"    {edge.to_string()}")
        return "\n".join(lines)

    @abstractmethod
    def validate(self) -> bool:
        """Validate the graph structure. To be implemented by subclasses."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, nodes={self.num_nodes()}, edges={self.num_edges()})"

    # -- NetworkX Compatible Interface ------------------------------------------
    def __getitem__(self, node_id: str) -> Dict[str, Dict[str, Any]]:
        """Get adjacency dict for NetworkX compatibility."""
        # Directly return the adjacency dict (already in NetworkX format)
        return self._adjacency_list.get(node_id, {})

    def __iter__(self) -> Iterator[str]:
        """Iterate over node IDs."""
        return iter(self._nodes.keys())

    def __len__(self) -> int:
        """Return number of nodes."""
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        """Check if node exists."""
        return node_id in self._nodes

    def is_directed(self) -> bool:
        """Return True (graphs are directed)."""
        return True

    def is_multigraph(self) -> bool:
        """Return False (not a multigraph)."""
        return False

    def successors(self, node_id: str) -> Iterator[str]:
        """Iterator over successor node IDs."""
        for node in self.get_successors(node_id):
            yield node.id

    def neighbors(self, node_id: str) -> Iterator[str]:
        """Iterator over neighbor node IDs (same as successors for directed graphs)."""
        return self.successors(node_id)

    def predecessors(self, node_id: str) -> Iterator[str]:
        """Iterator over predecessor node IDs."""
        for node in self.get_predecessors(node_id):
            yield node.id

    def get_edge_data(self, u: str, v: str, default=None) -> Optional[Dict[str, Any]]:
        """Get edge data between two nodes (NetworkX compatibility)."""
        # Directly lookup from adjacency dict (much faster than iterating edges)
        return self._adjacency_list.get(u, {}).get(v, default)

    def in_degree(self, node_id: Optional[str] = None):
        """
        Return the in-degree of a node or all nodes (NetworkX compatibility).

        Args:
            node_id: Optional node ID. If None, returns iterator of (node_id, degree) tuples.

        Returns:
            If node_id is provided, returns int (in-degree of that node).
            Otherwise, returns iterator of (node_id, degree) tuples.
        """
        if node_id is not None:
            return len(self._reverse_adjacency_list.get(node_id, {}))
        else:
            # NetworkX expects an iterable of (node, degree) tuples
            return ((nid, len(self._reverse_adjacency_list.get(nid, {}))) for nid in self._nodes.keys())

    def out_degree(self, node_id: Optional[str] = None):
        """
        Return the out-degree of a node or all nodes (NetworkX compatibility).

        Args:
            node_id: Optional node ID. If None, returns iterator of (node_id, degree) tuples.

        Returns:
            If node_id is provided, returns int (out-degree of that node).
            Otherwise, returns iterator of (node_id, degree) tuples.
        """
        if node_id is not None:
            return len(self._adjacency_list.get(node_id, {}))
        else:
            # NetworkX expects an iterable of (node, degree) tuples
            return ((nid, len(self._adjacency_list.get(nid, {}))) for nid in self._nodes.keys())

    @property
    def nodes(self) -> Iterator[N]:
        """Iterator over nodes."""
        return iter(self._nodes.values())

    @property
    def edges(self) -> Iterator[E]:
        """Iterator over edges."""
        return iter(self._edges.values())

    @property
    def adj(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Adjacency dict (NetworkX compatible)."""
        return self._adjacency_list
