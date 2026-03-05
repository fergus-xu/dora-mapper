"""Latency specification data structures for scheduling."""

from dataclasses import dataclass
from typing import Dict, NamedTuple, Tuple, Optional
from mapper.graph.dfg import OperationType


class OperationLatencyEdge(NamedTuple):
    """Edge representing network latency between two operation types.

    Attributes:
        src: Source operation type
        sink: Sink (destination) operation type
    """
    src: OperationType
    sink: OperationType


@dataclass
class LatencySpecification:
    """Specification of operation and network latencies for scheduling.

    This class encapsulates latency information needed for scheduling:
    - Operation latencies: execution time for each operation type
    - Network latencies: communication delay between operation types

    Attributes:
        op_latencies: Map from operation type to execution latency
        network_latencies: Map from (src, sink) operation edge to (lower_bound, upper_bound) latency tuple
    """

    op_latencies: Dict[OperationType, int]
    network_latencies: Dict[OperationLatencyEdge, Tuple[int, int]]

    def get_op_latency(self, op_type: OperationType) -> int:
        """Get the execution latency for an operation type.

        Args:
            op_type: The operation type

        Returns:
            Execution latency in cycles

        Raises:
            ValueError: If operation type not found in specification
        """
        if op_type not in self.op_latencies:
            raise ValueError(f"No operation latency found for {op_type}")

        return self.op_latencies[op_type]

    def get_network_latency(self, src: OperationType, sink: OperationType) -> Tuple[int, int]:
        """Get the network latency between two operation types.

        Args:
            src: Source operation type
            sink: Sink (destination) operation type

        Returns:
            Tuple of (lower_bound, upper_bound) network latency in cycles

        Raises:
            ValueError: If edge not found in specification
        """

        # Check if the edge is in the specification.
        edge = OperationLatencyEdge(src, sink)
        if edge not in self.network_latencies:
            raise ValueError(f"No network latency found for {src} to {sink}")

        # Return the lower and upper bound network latencies.
        return self.network_latencies[edge]

    def get_network_latency_lower(self, src: OperationType, sink: OperationType) -> int:
        """Get the lower bound network latency between two operation types.

        Args:
            src: Source operation type
            sink: Sink (destination) operation type

        Returns:
            Lower bound network latency in cycles

        Raises:
            ValueError: If edge not found in specification
        """
        return self.get_network_latency(src, sink)[0]

    def get_network_latency_upper(self, src: OperationType, sink: OperationType) -> int:
        """Get the upper bound network latency between two operation types.

        Args:
            src: Source operation type
            sink: Sink (destination) operation type

        Returns:
            Upper bound network latency in cycles

        Raises:
            ValueError: If edge not found in specification
        """
        return self.get_network_latency(src, sink)[1]

    def has_op_latency(self, op_type: OperationType) -> bool:
        """Check if operation type has a defined latency.

        Args:
            op_type: The operation type

        Returns:
            True if operation latency is defined, False otherwise
        """
        return op_type in self.op_latencies

    def has_network_latency(self, src: OperationType, sink: OperationType) -> bool:
        """Check if network latency is defined for an edge.

        Args:
            src: Source operation type
            sink: Sink (destination) operation type

        Returns:
            True if network latency is defined, False otherwise
        """
        edge = OperationLatencyEdge(src, sink)
        return edge in self.network_latencies

    def num_operations(self) -> int:
        """Get the number of operations with defined latencies.

        Returns:
            Number of operations
        """
        return len(self.op_latencies)

    def num_edges(self) -> int:
        """Get the number of edges with defined network latencies.

        Returns:
            Number of edges
        """
        return len(self.network_latencies)
