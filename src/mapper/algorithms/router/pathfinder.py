"""PathFrinder Algorithm for heuristic mapping."""

import heapq
from typing import Dict, List, Set, Tuple, Optional

from mapper.graph.dfg import DFG, DFGNode
from mapper.graph.mrrg import MRRG, MRRGNode, NodeType, OperandTag
from mapper.graph.hyperdfg import HyperDFG, HyperVal
from mapper.algorithms.router.types import (
    RoutingNodePlacementState,
    HyperValNetInfo,
    SinkAndLatency,
    VertexData,
    VertexAndCost
)


class PathFinder:
    """PathFinder Algorithm for heuristic mapping."""

    # DFG and MRRG graphs
    _dfg: DFG
    _mrrg: MRRG
    _hyperdfg: HyperDFG
    
    # Mapping from a hyper edge and value index to a routing node
    _placement: Dict[DFGNode, MRRGNode]
    _routing_nodes: Dict[MRRGNode, RoutingNodePlacementState]

    # Penalty multipliers

    # Present congestion factor
    _p_factor: float 
    _p_initial: float
    _p_growth_rate: float

    # Historical congestion factor
    _h_factor: float
    _h_initial: float
    _h_growth_rate: float

    # Routing solution
    _routing_solution: Dict[Tuple[HyperVal, int], List[MRRGNode]]

    # Global map: routing node -> set of HyperVal IDs that currently use it
    _global_routes: Dict[MRRGNode, Set[int]]

    # Dual use nodes
    _used_routing_fu_nodes: Set[MRRGNode]

    # Maximum routing iterations
    _max_iterations: int

    # Debug mode
    _debug: bool

    def __init__(
        self,
        dfg: DFG,
        mrrg: MRRG,
        placement: Dict[DFGNode, MRRGNode],
        p_growth_rate: float = 1.5,
        h_growth_rate: float = 1.5,
        p_factor_initial: float = 1.0,
        h_factor_initial: float = 1.0,
        max_iterations: int = 70,
        debug: bool = False
    ) -> None:
        """
        Initialize PathFinder router.

        Args:
            dfg: Data flow graph to route
            mrrg: Modulo routing resource graph
            placement: Initial placement of DFG operations to MRRG function nodes
            p_growth_rate: Present congestion penalty growth rate per iteration
            h_growth_rate: Historical congestion penalty growth rate per iteration
            p_factor_initial: Initial present congestion penalty multiplier
            h_factor_initial: Initial historical congestion penalty multiplier
            max_iterations: Maximum number of negotiation iterations (default 70)
            debug: Enable debug output (default False)
        """
        self._dfg = dfg
        self._mrrg = mrrg

        # Create HyperDFG representation
        self._hyperdfg = HyperDFG.from_dfg(dfg)

        # Initialize placement (will be filled from placement dict)
        self._placement = {}

        # Process placement dictionary and track dual-use nodes
        self._used_routing_fu_nodes = set()
        for dfg_node, mrrg_node in placement.items():
            self._placement[dfg_node] = mrrg_node

            # Track if this is a dual-purpose node being used as function
            if mrrg_node.node_type == NodeType.ROUTING_FUNCTION:
                self._used_routing_fu_nodes.add(mrrg_node)

        # Initialize routing nodes state
        self._routing_nodes = {}
        for mrrg_node in mrrg.get_nodes():
            if mrrg_node.node_type == NodeType.ROUTING:
                self._routing_nodes[mrrg_node] = RoutingNodePlacementState()

        # Penalty factors and growth rates
        self._p_growth_rate = p_growth_rate
        self._h_growth_rate = h_growth_rate
        self._p_factor = p_factor_initial
        self._h_factor = h_factor_initial
        self._p_initial = p_factor_initial
        self._h_initial = h_factor_initial

        # Routing solution: (HyperVal, dest_idx) -> path of MRRG nodes
        self._routing_solution = {}

        # Global HyperVal ownership per routing node
        self._global_routes = {}

        # Maximum negotiation iterations
        self._max_iterations = max_iterations

        # Debug mode
        self._debug = debug

    def _compute_cycles_source_to_sink(
        self,
        source_node: DFGNode,
        sink_node: DFGNode,
        edge_dist: int = 0,
        src_fu: Optional[MRRGNode] = None,
        sink_fu: Optional[MRRGNode] = None
    ) -> int:
        """
        Compute routing timing constraint from source to sink.

        In modulo scheduling (II > 1), uses actual placed MRRG FU cycles
        to compute how many time-expanded MRRG hops are needed. 
        Falls back to ASAP times for II=1.

        Args:
            source_node: Source DFG operation
            sink_node: Sink DFG operation
            edge_dist: Loop iteration distance (for loop-carried dependencies)
            src_fu: Placed MRRG FU node for source (modulo scheduling)
            sink_fu: Placed MRRG FU node for sink (modulo scheduling)

        Returns:
            Required latency in cycles from source to sink
        """
        II = self._mrrg.II

        # Use placement-based cycles when in modulo scheduling mode
        if src_fu is not None and sink_fu is not None and II is not None and II > 1:
            src_cycle = src_fu.cycle
            sink_cycle = sink_fu.cycle

            # Self-loop: wrap around the full II
            if source_node == sink_node:
                return II

            # Modular distance: how many cycles to go from src to sink in the II ring
            cycles_to_sink = (sink_cycle - src_cycle) % II

            # Loop-carried edges add full II iterations
            if edge_dist > 0:
                cycles_to_sink += edge_dist * II

            return cycles_to_sink

        # Fallback: use ASAP scheduling times (works for II=1)
        source_time = source_node.asap_time
        sink_time = sink_node.asap_time

        if source_time is None or sink_time is None:
            raise ValueError(
                f"Nodes must be scheduled: {source_node.id} (t={source_time}), "
                f"{sink_node.id} (t={sink_time})"
            )

        cycles_to_sink = sink_time - source_time

        if II is not None and cycles_to_sink < 0:
            cycles_to_sink = II - (abs(cycles_to_sink) % II)

        if II is not None and sink_node == source_node:
            cycles_to_sink = II

        if edge_dist > 0 and II is not None:
            cycles_to_sink += edge_dist * II

        return cycles_to_sink


    def _compute_cost(self, node: MRRGNode) -> float:
        """
        PathFinder cost function.

        Cost = base_cost + h_cost + oc_cost

        Where:
            - base_cost: Fixed wire length cost (always 1.0)
            - h_cost: Historical congestion penalty
            - oc_cost: Present overcapacity penalty

        Args:
            node: MRRG routing node

        Returns:
            Total cost for using this node
        """
        if node not in self._routing_nodes:
            # Not a routing node, return minimal cost
            return 1.0

        state = self._routing_nodes[node]

        # Get current state
        occupancy = state.occupancy
        capacity = node.capacity
        base_cost = state.base_cost
        historical_cost = state.historical_cost

        # Compute cost components
        h_cost = self._h_factor * historical_cost

        # Present overcapacity cost
        # Only apply penalty to overused nodes (occupancy > capacity)
        overuse = max(0, occupancy - capacity)
        oc_cost = 1.0 + self._p_factor * overuse

        # Total cost
        total_cost = base_cost + h_cost + oc_cost
        return total_cost

    def _commit_node(self, hyperval_net: Tuple[HyperVal, int], node: MRRGNode) -> None:
        """
        Commit a routing decision - add node to route.

        Updates occupancy (per distinct HyperVal) and tracks which hypervals use this node.
        Also maintains _global_routes for cross-HyperVal shorts prevention.

        Args:
            hyperval_net: The (HyperVal, dest_idx) identifier for this net
            node: MRRG routing node to add to the route
        """
        if node not in self._routing_nodes:
            return

        # Add node to the routing solution
        if hyperval_net not in self._routing_solution:
            self._routing_solution[hyperval_net] = []
        self._routing_solution[hyperval_net].append(node)

        hyperval, _ = hyperval_net
        hv_id = hyperval.source_id

        # Update node state
        state = self._routing_nodes[node]

        # Increment per-HyperVal usage
        prev = state.hyperval_usage.get(hyperval, 0)
        state.hyperval_usage[hyperval] = prev + 1

        # Only bump occupancy when this HyperVal first uses the node
        if prev == 0:
            state.occupancy += 1

        state.values.append(hyperval_net)

        # Update global HyperVal ownership for this node
        users = self._global_routes.get(node)
        if users is None:
            self._global_routes[node] = {hv_id}
        else:
            users.add(hv_id)

    def _rip_up_hyperval_net(self, hyperval_net: Tuple[HyperVal, int]) -> None:
        """
        Remove routing for a net and update historical costs.

        This is called during negotiated congestion resolution to remove
        existing routes before re-routing. Also maintains _global_routes.

        Args:
            hyperval_net: The (HyperVal, dest_idx) identifier for the net to rip up
        """
        if hyperval_net not in self._routing_solution:
            return

        route = self._routing_solution[hyperval_net]
        hyperval, _ = hyperval_net

        # For each node in the route
        for node in route:
            if node not in self._routing_nodes:
                continue

            state = self._routing_nodes[node]

            # Update historical cost based on overuse (distinct HyperVals)
            if state.occupancy > node.capacity:
                overuse = state.occupancy - node.capacity
                state.historical_cost += overuse

            # Remove from mapped values list (per-sink record)
            try:
                state.values.remove(hyperval_net)
            except ValueError:
                print(f"[ERROR] Cannot find {hyperval_net} in {node.get_full_name()} values")
                raise

            # Update per-HyperVal usage and occupancy
            hv_count = state.hyperval_usage.get(hyperval, 0)
            if hv_count <= 0:
                print(f"[ERROR] HyperVal {hyperval} had non-positive usage on {node.get_full_name()}")
            else:
                if hv_count == 1:
                    # This was the last sink of this HyperVal using this node
                    del state.hyperval_usage[hyperval]
                    # Decrement occupancy: one fewer distinct HyperVal
                    if state.occupancy <= 0:
                        raise ValueError(
                            f"Occupancy was already zero or less for {node.get_full_name()}: "
                            f"{state.occupancy}"
                        )
                    state.occupancy -= 1
                else:
                    # Other sinks of this HyperVal still use this node
                    state.hyperval_usage[hyperval] = hv_count - 1

            # Recompute global HyperVal ownership for this node based on remaining values
            remaining_hv_ids: Set[int] = set()
            for (hv, _) in state.values:
                remaining_hv_ids.add(hv.source_id)

            if remaining_hv_ids:
                self._global_routes[node] = remaining_hv_ids
            else:
                # No HyperVals left using this node
                if node in self._global_routes:
                    del self._global_routes[node]

        # Clear the routing
        del self._routing_solution[hyperval_net]

    def _rip_up_hyperval(self, hyperval: HyperVal) -> None:
        """
        Remove routing for ALL destinations of a HyperVal.

        Args:
            hyperval: The HyperVal whose all destinations should be ripped up
        """
        for dest_idx in range(hyperval.cardinality):
            hyperval_net = (hyperval, dest_idx)
            self._rip_up_hyperval_net(hyperval_net)

    def _check_overuse(self, verbose: bool = False) -> bool:
        """
        Check if routing is legal (no capacity violations).

        Args:
            verbose: If True, print details of each overused node

        Returns:
            True if all nodes have occupancy <= capacity, False otherwise
        """
        is_legal = True
        num_overused = 0
        overused_nodes = []

        for node, state in self._routing_nodes.items():
            if state.occupancy > node.capacity:
                is_legal = False
                num_overused += 1
                overused_nodes.append((node, state))

        if not is_legal:
            print(f"  [OVERUSE] {num_overused} nodes overused")

            if verbose or self._debug:
                for node, state in overused_nodes:
                    overuse_amount = state.occupancy - node.capacity
                    print(f"    ⚠ {node.get_full_name()}")
                    print(f"      Occupancy: {state.occupancy}/{node.capacity} (overuse: +{overuse_amount})")
                    # Show which hypervals are using this node
                    if state.values:
                        print(f"      Used by {len(state.values)} net(s):")
                        for hyperval_net in state.values[:3]:  # Show first 3 hypervals
                            hyperval, dest_idx = hyperval_net
                            dest_id = hyperval.destination_ids[dest_idx]
                            # Get operation info for better context
                            src_hyper = self._hyperdfg.get_node(hyperval.source_id)
                            dst_hyper = self._hyperdfg.get_node(dest_id)
                            src_op = ""
                            dst_op = ""
                            if src_hyper:
                                src_dfg = self._dfg.get_node(src_hyper.original_dfg_node_id)
                                if src_dfg:
                                    src_op = f" ({src_dfg.operation.value})"
                            if dst_hyper:
                                dst_dfg = self._dfg.get_node(dst_hyper.original_dfg_node_id)
                                if dst_dfg:
                                    dst_op = f" ({dst_dfg.operation.value})"
                            print(f"        • {hyperval.source_id}{src_op} → {dest_id}{dst_op} [sink {dest_idx}]")
                        if len(state.values) > 3:
                            print(f"        ... and {len(state.values) - 3} more net(s)")

        return is_legal

    def _compute_dfg_coverage(self, verbose: bool = False) -> bool:
        """
        Check if all hypervals are routed.

        Args:
            verbose: If True, print details of each uncovered hyperval

        Returns:
            True if every (HyperVal, dest_idx) has a routing, False otherwise
        """
        is_covered = True
        num_uncovered = 0
        uncovered_nets = []

        # Check every HyperVal and every destination
        for hyperval in self._hyperdfg.get_edges():
            for dest_idx in range(hyperval.cardinality):
                hyperval_net = (hyperval, dest_idx)

                # A hyperval is covered if it has a routing solution
                # Note: empty routes [] are valid for self-loops
                if hyperval_net not in self._routing_solution:
                    is_covered = False
                    num_uncovered += 1
                    uncovered_nets.append(hyperval_net)

        if not is_covered:
            print(f"  [UNCOVERED] {num_uncovered} hypervals not routed")

            if verbose or self._debug:
                for hyperval_net in uncovered_nets:
                    hyperval, dest_idx = hyperval_net
                    dest_id = hyperval.destination_ids[dest_idx]
                    # Get operation and placement info
                    src_hyper = self._hyperdfg.get_node(hyperval.source_id)
                    dst_hyper = self._hyperdfg.get_node(dest_id)
                    src_info = f"{hyperval.source_id}"
                    dst_info = f"{dest_id}"

                    if src_hyper:
                        src_dfg = self._dfg.get_node(src_hyper.original_dfg_node_id)
                        if src_dfg:
                            src_op = src_dfg.operation.value
                            src_mrrg = self._placement.get(src_dfg)
                            src_place = f" @ {src_mrrg.get_full_name()}" if src_mrrg else ""
                            src_info = f"{hyperval.source_id} ({src_op}){src_place}"

                    if dst_hyper:
                        dst_dfg = self._dfg.get_node(dst_hyper.original_dfg_node_id)
                        if dst_dfg:
                            dst_op = dst_dfg.operation.value
                            dst_mrrg = self._placement.get(dst_dfg)
                            dst_place = f" @ {dst_mrrg.get_full_name()}" if dst_mrrg else ""
                            dst_info = f"{dest_id} ({dst_op}){dst_place}"

                    print(f"    ✗ {src_info} → {dst_info} [sink {dest_idx}]")

        return is_covered

    def _print_number_of_resources_used(self) -> None:
        """Print routing statistics."""
        num_resources_used = 0

        for _, state in self._routing_nodes.items():
            if state.occupancy > 0:
                num_resources_used += 1

        print(f"\n[STATS] Number of routing resources used: {num_resources_used}")
        print(f"[STATS] Total hypervals routed: {len(self._routing_solution)}")

        # Additional statistics
        total_occupancy = sum(state.occupancy for state in self._routing_nodes.values())
        total_capacity = sum(node.capacity for node in self._routing_nodes.keys())

        print(f"[STATS] Total occupancy: {total_occupancy}")
        print(f"[STATS] Total capacity: {total_capacity}")
        print(f"[STATS] Utilization: {total_occupancy/total_capacity*100:.2f}%")

    def _dijkstra_pathfinder(
        self,
        source: MRRGNode,
        sink: MRRGNode,
        tag_to_find: Optional[OperandTag],
        hyperval: HyperVal,
        routes: Set[MRRGNode],
        cycles_to_sink: int,
        latency_of_src: int
    ) -> List[MRRGNode]:
        """
        Dijkstra pathfinding with routing constraints (cgra-me specification).

        Args:
            source: Starting routing node (FU's first fanout)
            sink: Destination FU node
            tag_to_find: Required operand tag (or None for UNTAGGED)
            hyperval: HyperVal being routed (for bitwidth checking)
            routes: Nodes already used by other sinks of this HyperVal
            cycles_to_sink: Required arrival time at sink
            latency_of_src: Starting latency

        Returns:
            Path as list of MRRG nodes, or [] if not found
        """

        # Storage: (node, latency) -> VertexData
        data: Dict[Tuple[MRRGNode, int], VertexData] = {}

        # Priority queue
        to_visit: List[VertexAndCost] = []

        # Visited states
        visited: Set[Tuple[MRRGNode, int]] = set()

        heapq.heappush(to_visit, VertexAndCost(0.0, latency_of_src, source))
        data[(source, latency_of_src)] = VertexData(
            fanin=[],  # Fanin represents path TO node (excluding node itself)
            lowest_known_cost=0.0,
            num_of_cycles=latency_of_src,
            tag_found=False
        )

        found: Optional[MRRGNode] = None
        tag_found = False

        while to_visit and not found:
            # Get lowest-cost unexplored node
            queue_top = heapq.heappop(to_visit)
            explore_curr = queue_top.node
            cycles_to_curr = queue_top.cycles

            # Skip if already visited
            if (explore_curr, cycles_to_curr) in visited:
                continue

            if not found and sink == explore_curr:
                # Check timing constraint
                if cycles_to_sink == cycles_to_curr:
                    found = explore_curr # Found the sink
                else:
                    # Wrong timing - mark visited and continue
                    visited.add((explore_curr, cycles_to_curr))
                    if (explore_curr, cycles_to_curr) in data:
                        del data[(explore_curr, cycles_to_curr)]
                continue

            for fanout_edge in self._mrrg.get_outgoing_edges(explore_curr.id):
                fanout = self._mrrg.get_node(fanout_edge.destination.id)
                if not fanout:
                    continue

                tag_found = False  # Reset for each neighbor

                if fanout == source:
                    continue

                if fanout.node_type == NodeType.FUNCTION and fanout != sink:
                    continue

                if fanout.node_type == NodeType.ROUTING_FUNCTION:
                    if fanout in self._used_routing_fu_nodes and fanout != sink:
                        continue

                # Make sure bitwidth is compatible
                if hyperval.bitwidths and hyperval.bitwidths[0]:
                    if fanout.bitwidth < hyperval.bitwidths[0]:
                        continue

                current_hv_id = hyperval.source_id
                global_users = self._global_routes.get(fanout, set())

                if global_users and current_hv_id not in global_users:
                    # Another HyperVal already owns this node: illegal short
                    continue

                data_curr = data[(explore_curr, cycles_to_curr)]
                if data_curr.tag_found and fanout != sink:
                    continue

                cost = queue_top.cost_to_here + self._compute_cost(fanout)
                cycles = data_curr.num_of_cycles

                if fanout in data_curr.fanin:
                    continue

                # Update latency accumulation (static timing)
                if fanout.node_type != NodeType.FUNCTION:
                    cycles = cycles + explore_curr.latency

                if cycles > cycles_to_sink:
                    continue

                if (fanout, cycles) in data:
                    if data[(fanout, cycles)].lowest_known_cost <= cost:
                        continue  # Existing path is better
                    else:
                        del data[(fanout, cycles)]  # Replace worse path

                if len(fanout.supported_operand_tags) > 0 and tag_to_find is not None:
                    if fanout.node_type != NodeType.ROUTING_FUNCTION:
                        # Regular routing node with tag constraint
                        if tag_to_find not in fanout.supported_operand_tags:
                            if OperandTag.BINARY_ANY not in fanout.supported_operand_tags:
                                continue  # Doesn't support required tag

                        # Tag matches! Mark as found if node is unused
                        if fanout in self._routing_nodes:
                            if len(self._routing_nodes[fanout].values) == 0 and \
                               fanout not in routes:
                                tag_found = True
                    elif tag_to_find == OperandTag.PREDICATE:
                        # Special handling for predicate tags
                        if tag_to_find not in fanout.supported_operand_tags:
                            continue
                        if fanout in self._routing_nodes:
                            if len(self._routing_nodes[fanout].values) == 0 and \
                               fanout not in routes:
                                tag_found = True

                    if not tag_found:
                        continue  # Tag requirement not met

                # Copy path from current node and append explore_curr
                new_fanin = data_curr.fanin.copy()
                new_fanin.append(explore_curr)

                data[(fanout, cycles)] = VertexData(
                    fanin=new_fanin,
                    lowest_known_cost=cost,
                    num_of_cycles=cycles,
                    tag_found=tag_found
                )

                # Add to priority queue
                heapq.heappush(to_visit, VertexAndCost(cost, cycles, fanout))

                tag_found = False  # Reset for next iteration

            # Mark current node as visited and clean up
            visited.add((explore_curr, cycles_to_curr))
            if (explore_curr, cycles_to_curr) in data:
                del data[(explore_curr, cycles_to_curr)]

        if not found:
            return []

        final_data = data[(sink, cycles_to_sink)]
        return final_data.fanin

    def _route_hyperval(self, hyperval: HyperVal) -> bool:
        """
        Route ALL destinations of a HyperVal together (multi-sink routing).

        Args:
            hyperval: The HyperVal to route (all destinations)

        Returns:
            True if routing succeeded for all destinations, False otherwise
        """
        
        # All destinations of a HyperVal must be routed together atomically. 
        # If any destination is already routed, we want to quite and return False.
        # Check if already routed (any destination has a route).
        already_routed = False
        
        for dest_idx in range(hyperval.cardinality):
            hyperval_net = (hyperval, dest_idx)
            if hyperval_net not in self._routing_solution or not self._routing_solution[hyperval_net]:
                already_routed = False
                break
            else:
                already_routed = True

        if already_routed:
            return True

        # Get source DFG node (common for all destinations)
        src_hyper_node = self._hyperdfg.get_node(hyperval.source_id)
        if not src_hyper_node:
            return False

        src_dfg_node = self._dfg.get_node(src_hyper_node.original_dfg_node_id)
        if not src_dfg_node or src_dfg_node not in self._placement:
            return False

        src_fu = self._placement[src_dfg_node]
        latency_of_src = src_fu.latency

        # Get first fanout of source FU (start routing from here)
        fanout_edges = self._mrrg.get_outgoing_edges(src_fu.id)
        if not fanout_edges:
            return False
        root_fanout = self._mrrg.get_node(fanout_edges[0].destination.id)

        # Build priority queue of sinks (longest paths first)
        sinks: List[SinkAndLatency] = []
        for dest_idx in range(hyperval.cardinality):
            dest_id = hyperval.destination_ids[dest_idx]
            dest_hyper_node = self._hyperdfg.get_node(dest_id)
            if not dest_hyper_node:
                return False

            dest_dfg_node = self._dfg.get_node(dest_hyper_node.original_dfg_node_id)
            if not dest_dfg_node or dest_dfg_node not in self._placement:
                return False

            sink_fu = self._placement[dest_dfg_node]

            # Skip routing for loop-back edges (value stays local within PE)
            if src_dfg_node == dest_dfg_node and hyperval.is_loop_backs[dest_idx]:
                # Create empty route for self-loop (no external routing needed)
                hyperval_net = (hyperval, dest_idx)
                self._routing_solution[hyperval_net] = []
                if self._debug:
                    print(f"  [SKIP] Self-loop detected for {hyperval.source_id} -> {dest_id} (loop-back edge)")
                continue

            # Get timing constraint using placement-based cycles
            edge_dist = hyperval.dists[dest_idx] if hyperval.dists[dest_idx] is not None else 0
            cycles_to_sink = self._compute_cycles_source_to_sink(
                src_dfg_node, dest_dfg_node, edge_dist, src_fu=src_fu, sink_fu=sink_fu
            )


            # Get operand tag
            operand_str = hyperval.operands[dest_idx]
            operand_tag = self._map_operand_to_tag(operand_str)

            sinks.append(SinkAndLatency(
                cycles_to_sink=cycles_to_sink,
                dest_idx=dest_idx,
                dest_dfg_node=dest_dfg_node,
                sink_fu=sink_fu,
                operand_tag=operand_tag
            ))

        # Sort sinks by priority (longest paths first)
        sinks.sort()

        # If all destinations were self-loops, we're done
        if not sinks:
            return True

        # Hyperval-aware routing ownership:
        # routes[node] = set(hyperval_ids) of HyperVals that legitimately use this node
        routes: Dict[MRRGNode, Set[int]] = {}

        # Store first-pass route for each sink
        sink_routes: Dict[int, List[MRRGNode]] = {}

        # Unique ID for this logical net (HyperVal)
        current_hv_id = hyperval.source_id

        # Route each sink in priority order
        for sink_info in sinks:
            route = self._dijkstra_pathfinder(
                source=root_fanout,
                sink=sink_info.sink_fu,
                tag_to_find=sink_info.operand_tag,
                hyperval=hyperval,
                routes=routes,
                cycles_to_sink=sink_info.cycles_to_sink,
                latency_of_src=latency_of_src
            )

            if not route:
                if self._debug:
                    dest_id = hyperval.destination_ids[sink_info.dest_idx]
                    print(f"  ✗ ROUTING FAILED")
                    print(f"    Source:      {src_dfg_node.id} ({src_dfg_node.operation.value}) @ {src_fu.get_full_name()}")
                    print(f"    Destination: {sink_info.dest_dfg_node.id} ({sink_info.dest_dfg_node.operation.value}) @ {sink_info.sink_fu.get_full_name()}")
                    print(f"    HyperVal:    {hyperval.source_id} → {dest_id} [sink {sink_info.dest_idx}]")
                    print(f"    Timing:      {sink_info.cycles_to_sink} cycles required")
                    operand_info = f" (operand: {sink_info.operand_tag.value})" if sink_info.operand_tag else ""
                    print(f"    Operand:     {hyperval.operands[sink_info.dest_idx]}{operand_info}")
                return False

            # Store this exact route so we don't rerun Dijkstra
            sink_routes[sink_info.dest_idx] = route

            # Add this route's nodes to the shared routes set
            for n in route:
                if n not in routes:
                    routes[n] = set()
                routes[n].add(current_hv_id)

            # Print successful route path when debug is enabled
            if self._debug:
                dest_id = hyperval.destination_ids[sink_info.dest_idx]
                path_str = f"{src_fu.get_full_name()}"
                for node in route:
                    path_str += f" → {node.get_full_name()}"
                path_str += f" → {sink_info.sink_fu.get_full_name()}"
                print(f"  ✓ ROUTED: {hyperval.source_id} → {dest_id} [sink {sink_info.dest_idx}]")
                print(f"    {src_dfg_node.id} ({src_dfg_node.operation.value}) → {sink_info.dest_dfg_node.id} ({sink_info.dest_dfg_node.operation.value})")
                print(f"    Path: {path_str}")
                print(f"    Stats: {sink_info.cycles_to_sink} cycles, {len(route)} hops")

        # All sinks routed successfully - commit the saved first-pass routes
        for sink_info in sinks:
            hyperval_net = (hyperval, sink_info.dest_idx)
            route = sink_routes.get(sink_info.dest_idx, [])

            for node in route:
                if node.node_type in (NodeType.ROUTING, NodeType.ROUTING_FUNCTION):
                    self._commit_node(hyperval_net, node)

        return True

    def _map_operand_to_tag(self, operand_str: Optional[str]) -> Optional[OperandTag]:
        """Map operand string to OperandTag enum."""
        if operand_str is None:
            return None  # No tag constraint

        operand_map = {
            # Frontend compatible operand strings
            "": OperandTag.UNTAGGED,  # Explicitly untagged
            "LHS": OperandTag.BINARY_LHS,
            "RHS": OperandTag.BINARY_RHS,
            "any2input": OperandTag.BINARY_ANY,
            "any3input": OperandTag.TERNARY_ANY,
            "addr": OperandTag.MEM_ADDR,
            "data": OperandTag.MEM_DATA,
            "pred": OperandTag.PREDICATE,
            "branch_true": OperandTag.BR_TRUE,
            "branch_false": OperandTag.BR_FALSE,
            # Legacy aliases for backwards compatibility
            "binary_lhs": OperandTag.BINARY_LHS,
            "binary_rhs": OperandTag.BINARY_RHS,
            "ADDR": OperandTag.MEM_ADDR,
            "DATA": OperandTag.MEM_DATA,
            "PRED": OperandTag.PREDICATE,
            "predicate": OperandTag.PREDICATE,
        }

        return operand_map.get(operand_str, None)

    def route_dfg(self) -> bool:
        """
        Main PathFinder routing algorithm with negotiated congestion resolution.

        Returns:
            True if routing succeeded, False otherwise
        """
        print("\n" + "="*60)
        print("PATHFINDER ROUTING")
        print("="*60)

        print("\n[PHASE 1] Initial routing of all HyperVals...")

        routed = True
        hypervals: List[HyperValNetInfo] = []

        # Route each HyperVal (all destinations together)
        for hyperval in self._hyperdfg.get_edges():

            # All destinations of a HyperVal will be routed together
            success = self._route_hyperval(hyperval)
            if not success:
                # Don't stop the initial routing. Just record failure.
                routed = False
                continue

            # Compute priority metrics for this HyperVal
            src_hyper_node = self._hyperdfg.get_node(hyperval.source_id)
            if src_hyper_node:
                src_dfg_node = self._dfg.get_node(src_hyper_node.original_dfg_node_id)

                if src_dfg_node:

                    # Find the maximum number of cycles across all the destinations of this HyperVal
                    # Shwet TODO: Is this the best way to do this?
                    max_cycles = 0
                    for dest_idx in range(hyperval.cardinality):
                        dest_id = hyperval.destination_ids[dest_idx]
                        dest_hyper_node = self._hyperdfg.get_node(dest_id)

                        if dest_hyper_node:
                            dest_dfg_node = self._dfg.get_node(dest_hyper_node.original_dfg_node_id)

                            if dest_dfg_node:
                                edge_dist = hyperval.dists[dest_idx] if hyperval.dists[dest_idx] is not None else 0
                                cycles = self._compute_cycles_source_to_sink(src_dfg_node, dest_dfg_node, edge_dist)
                                max_cycles = max(max_cycles, cycles)

                    hypervals.append(HyperValNetInfo(
                        hyperval=hyperval,
                        overuse=False,
                        max_cycles=max_cycles,
                        fanout=hyperval.cardinality
                    ))

        if not routed:
            print("[ERROR] Initial routing failed for some hypervals")

        # Sort hypervals by 
        hypervals.sort()

        # Count total destinations routed
        total_destinations = sum(hyperval_info.hyperval.cardinality for hyperval_info in hypervals)
        print(f"[INFO] Routed {len(hypervals)} HyperVals ({total_destinations} total destinations) in initial pass")

        print("\n[PHASE 2] Checking routing legality...")

        mapped = self._compute_dfg_coverage(verbose=True) and self._check_overuse()

        if mapped:
            print("\n[SUCCESS] Initial routing is legal!")
            self._print_number_of_resources_used()
            return True

        print("\n[PHASE 3] Negotiated congestion resolution...")

        iteration = 0

        while not mapped and iteration < self._max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration}/{self._max_iterations} ---")

            # Try ripping up and re-routing each HyperVal
            for hyperval_info in hypervals:
                hyperval = hyperval_info.hyperval

                # Rip up ALL destinations of this HyperVal
                self._rip_up_hyperval(hyperval) 

                # Re-route ALL destinations with updated costs
                routed = self._route_hyperval(hyperval)

                if not routed:
                    if iteration > 1:
                        print(f"[ERROR] HyperVal {hyperval.source_id} could not be routed in iter {iteration}")
                    break
            
            # Check solution after each iteration
            if routed:
                mrrg_overuse = self._check_overuse()
                opgraph_covered = self._compute_dfg_coverage()
                mapped = mrrg_overuse and opgraph_covered

                if mapped:
                    print(f"\n[SUCCESS] Legal routing achieved in iteration {iteration}!")

            if not routed:
                print(f"[WARN] Routing failed in iteration {iteration}")
                break

            if mapped:
                break

            # Increase congestion penalties for next iteration
            self._p_factor = self._p_factor * self._p_growth_rate
            self._h_factor = self._h_factor * self._h_growth_rate

        print("\n" + "="*60)
        coverage = self._compute_dfg_coverage(verbose=True)
        legality = self._check_overuse(verbose=True)
        print(f"ROUTING COMPLETE: covered={coverage}, legal={legality}")
        print("="*60)

        if coverage and legality:
            print("\n[SUCCESS] Routing succeeded!")
            self._print_number_of_resources_used()
            return True
        else:
            print(f"\n[PARTIAL] Routing incomplete after {iteration} iterations")
            print("[INFO] Returning partial solution with warnings")
            self._print_number_of_resources_used()
            # Rip up all routes
            for hyperval_info in hypervals:
                self._rip_up_hyperval(hyperval_info.hyperval)

            self._used_routing_fu_nodes.clear()

            # Reset the initial penalties
            self._p_factor = self._p_initial
            self._h_factor = self._h_initial

            return False