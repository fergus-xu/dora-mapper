from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from mapper.graph.mrrg import MRRGNode, OperandTag
    from mapper.graph.dfg import DFGNode

from mapper.graph.dfg import DFGEdge
from mapper.graph.hyperdfg import HyperVal

@dataclass
class RoutingNodePlacementState:
    """Node attributes for heuristic mapping."""

    # Number of distinct HyperVals using this routing node (NOT number of sinks)
    occupancy: int = 0

    # Base cost of the routing node
    base_cost: float = 1.0

    # Historical cost of the routing node
    historical_cost: float = 0.0

    # Values using this routing node (per-sink record for debugging)
    values: List[Tuple[HyperVal, int]] = field(default_factory=list)

    # Track number of sinks of each HyperVal using this node
    # Key: HyperVal, Value: count of sinks of that HyperVal using this node
    hyperval_usage: Dict[HyperVal, int] = field(default_factory=dict)

    # Optional packed-lane ownership map.
    # lane_index -> set of HyperVal source IDs currently using that lane.
    packed_lane_usage: Dict[int, Set[str]] = field(default_factory=dict)


@dataclass
class SinkAndLatency:
    """
    Sink destination with timing information for priority routing.

    Used to prioritize which sinks to route first within a multi-sink value.
    Routes sinks with longer timing constraints first (critical paths).
    """

    # Timing constraint (cycles to sink) - used for ordering
    cycles_to_sink: int

    # Destination index in HyperVal.destination_ids (for tie-breaking)
    dest_idx: int

    # Destination DFG node
    dest_dfg_node: 'DFGNode'

    # Sink FU node in MRRG
    sink_fu: 'MRRGNode'

    # Maximum allowed cycles to sink with timing slack
    max_cycles_to_sink: int

    # Required operand tag
    operand_tag: Optional['OperandTag'] = None

    def __lt__(self, other: 'SinkAndLatency') -> bool:
        """
        Comparison for priority queue (max-heap via reversal).

        Route sinks with longer timing constraints first.
        """
        if self.cycles_to_sink == other.cycles_to_sink:
            return self.dest_idx < other.dest_idx
        return self.cycles_to_sink > other.cycles_to_sink  # Reverse for max-heap


@dataclass
class HyperValNetInfo:
    """
    Information about a HyperVal for routing priority.

    Updated to work at HyperVal level (all destinations together)
    rather than per-destination.
    """

    # The HyperVal to route (all destinations)
    hyperval: HyperVal = None
    dest_idx: int(init=False) = -1

    hyperval_net: Tuple = None

    # Whether this net has resource overuse
    overuse: bool = False

    # Maximum timing constraint across all destinations
    max_cycles: int = 0

    # Fanout of the HyperVal (total number of destinations)
    fanout: int = 0

    def __lt__(self, other: 'HyperValNetInfo') -> bool:
        """
        Less than comparison for priority sorting.

        Priority: larger fanout and longer paths first (min-heap, so reverse).
        This helps route harder nets earlier when resources are more available.

        Returns:
            True if this net has higher priority than other
        """
        # First compare by fanout (higher fanout = higher priority = "less than" for min-heap)
        if self.fanout != other.fanout:
            return self.fanout > other.fanout
        # Then compare by max_cycles (longer paths = higher priority)
        return self.max_cycles > other.max_cycles
    
    def __post_init__(self):
        if self.hyperval_net is not None:
            self.hyperval, self.dest_idx = self.hyperval_net


@dataclass
class VertexData:
    """
    Dijkstra state data for pathfinding.

    Stores the best path found to reach a specific (node, latency) state.
    """

    # Path to reach this state (list of MRRG nodes)
    fanin: List['MRRGNode']

    # Lowest known cost to reach this state
    lowest_known_cost: float

    # Accumulated latency at this state
    num_of_cycles: int

    # Whether operand tag has been satisfied
    tag_found: bool = False


@dataclass(order=True)
class VertexAndCost:
    """
    Priority queue entry for Dijkstra exploration.

    Ordered by cost (min-heap).
    """

    # Cost to reach this node
    cost_to_here: float

    # Cycles/latency at this node (for state tracking)
    cycles: int = field(compare=False)

    # The MRRG node
    node: 'MRRGNode' = field(compare=False)

    # Whether the required tag has been satisfied on this path state
    tag_found: bool = field(default=False, compare=False)

    def __hash__(self):
        """Allow use in sets."""
        return hash(id(self.node))
