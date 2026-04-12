"""
Heuristic Mapper: ASAP Scheduling + Iterative Place-and-Route

This mapper implements a heuristic flow inspired by ClusteredMapper:
1. ASAP Scheduling - assigns operations to time slots
2. Iterative Place-and-Route Loop:
   - AnnealPlacer: spatial placement using simulated annealing
   - PathFinder: routing connections through CGRA interconnect
   - Adaptive temperature adjustment based on acceptance rate
   - Early stopping on convergence
"""

import time
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import deque

from mapper.graph.dfg import DFG, DFGNode
from mapper.graph.mrrg import MRRG, MRRGNode
from mapper.schedules.latency_spec import LatencySpecification
from mapper.algorithms.scheduler.asap_scheduler import ASAPScheduler
from mapper.algorithms.placer.anneal_placer import AnnealPlacer
from mapper.algorithms.router.pathfinder import PathFinder


class HeuristicMapper:
    """
    Heuristic mapper using ASAP scheduling followed by iterative place-and-route.

    The mapper follows this flow:
    1. ASAP Scheduling: Assigns each operation to a time slot
    2. Iterative Loop (until success or max iterations):
       a. Placement: Use simulated annealing to place operations on FUs
       b. Routing: Use PathFinder to route connections
       c. Temperature adjustment: Adapt based on acceptance rate
       d. Convergence check: Stop early if cost plateaus
    """

    def __init__(
        self,
        dfg: DFG,
        mrrg: MRRG,
        latency_spec: LatencySpecification,
        initial_temperature: float = 1000.0,
        max_iterations: int = 1000,
        max_time: Optional[float] = None,
        convergence_window: int = 10,
        convergence_threshold: float = 0.01,
        random_seed: int = 42,
        swap_factor: int = 10,
        router_max_iterations: int = 30,
        bitwidth_mismatch_penalty_weight: float = 0.5,
        p_growth_rate: float = 1.5,
        h_growth_rate: float = 1.5,
        placement_restart_patience: int = 3,
        max_placement_restarts: int = 4,
        ii_iteration_patience: int = 12,
        debug: bool = False,
    ):
        """
        Initialize the heuristic mapper.

        Args:
            dfg: Data flow graph to map
            mrrg: Modulo routing resource graph (target architecture)
            latency_spec: Operation and network latency specifications
            initial_temperature: Starting temperature for simulated annealing
            max_iterations: Maximum place-and-route iterations
            max_time: Maximum runtime in seconds (None = no limit)
            convergence_window: Number of iterations to check for convergence
            convergence_threshold: Relative cost change threshold for convergence
            random_seed: Random seed for reproducibility
            swap_factor: Number of swaps per temperature (num_ops * swap_factor)
            router_max_iterations: Max iterations for PathFinder routing
            bitwidth_mismatch_penalty_weight: Soft routing cost penalty for
                                             using wider-than-required NoC links
            p_growth_rate: PathFinder present congestion growth rate
            h_growth_rate: PathFinder history congestion growth rate
            placement_restart_patience: Failed/no-improvement routing iterations
                                      before trying a fresh placement
            max_placement_restarts: Maximum fresh placement retries per II
            ii_iteration_patience: If no routing coverage improvement for this
                                  many placement iterations, move to next II
            debug: Enable debug output
        """
        self._dfg = dfg
        self._mrrg = mrrg
        self._base_mrrg = mrrg
        self._latency_spec = latency_spec
        self._initial_temperature = initial_temperature
        self._max_iterations = max_iterations
        self._max_time = max_time
        self._convergence_window = convergence_window
        self._convergence_threshold = convergence_threshold
        self._random_seed = random_seed
        self._swap_factor = swap_factor
        self._router_max_iterations = router_max_iterations
        self._bitwidth_mismatch_penalty_weight = bitwidth_mismatch_penalty_weight
        self._p_growth_rate = p_growth_rate
        self._h_growth_rate = h_growth_rate
        self._placement_restart_patience = max(1, placement_restart_patience)
        self._max_placement_restarts = max(0, max_placement_restarts)
        self._ii_iteration_patience = max(1, ii_iteration_patience)
        self._debug = debug

        # Results
        self._placement: Optional[Dict[DFGNode, MRRGNode]] = None
        self._routing_solution = None
        self._route_metadata: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None
        self._iterations_run = 0
        self._final_cost = 0.0
        self._timed_out = False
        self._history_ii: Optional[int] = None
        self._routing_history_costs: Dict[str, float] = {}
        self._fu_failure_penalties: Dict[str, float] = {}
        self._best_placement_hint: Dict[str, str] = {}

    def _calculate_min_ii(self) -> int:
        """Calculate the theoretical minimum II based on resource constraints."""
        import math
        
        # Count operations by type in DFG
        op_counts = {}
        for node in self._dfg.get_nodes():
            op_counts[node.operation] = op_counts.get(node.operation, 0) + 1
            
        # Count FUs capable of each operation
        fu_capacity = {}
        for fu in self._mrrg.get_fu_nodes():
            for op in fu.supported_operations:
                fu_capacity[op] = fu_capacity.get(op, 0) + 1
                
        # Calculate max required II across all operation types
        min_ii = 1
        for op, count in op_counts.items():
            capacity = fu_capacity.get(op, 0)
            if capacity == 0:
                continue # Skip if unmapped or handled elsewhere
            required_ii = math.ceil(count / capacity)
            min_ii = max(min_ii, required_ii)
            
        return min_ii

    def _map_with_ii(self) -> Dict[str, Any]:
        """Run the mapping flow for the current MRRG's II."""
        start_time = time.time()
        if self._history_ii != self._mrrg.II:
            self._history_ii = self._mrrg.II
            self._routing_history_costs.clear()
            self._fu_failure_penalties.clear()
            self._best_placement_hint.clear()
        
        # Phase 1: ASAP Scheduling
        if self._debug:
            print(f"\n=== Phase 1: ASAP Scheduling (II={self._mrrg.II}) ===")

        success = self._run_asap_scheduling()
        if not success:
            return {
                'status': 'failed',
                'placement': None,
                'routes': None,
                'route_metadata': None,
                'runtime': time.time() - start_time,
                'iterations': 0,
                'final_cost': 0.0,
                'error_message': f'ASAP scheduling failed at II={self._mrrg.II}'
            }

        # Phase 2: Iterative Place-and-Route
        if self._debug:
            print(f"\n=== Phase 2: Iterative Place-and-Route (II={self._mrrg.II}) ===")

        success = self._run_iterative_place_and_route()
        runtime = time.time() - start_time

        if success:
            placement_dict = self._convert_placement_to_dict()
            routing_dict = self._convert_routing_to_dict()
            is_valid, validation_errors = self._validate_solution(placement_dict, routing_dict)

            return {
                'status': 'success',
                'placement': placement_dict,
                'routes': routing_dict,
                'route_metadata': self._route_metadata,
                'runtime': runtime,
                'iterations': self._iterations_run,
                'final_cost': self._final_cost,
                'validation': {
                    'is_valid': is_valid,
                    'errors': validation_errors
                }
            }
        else:
            if self._timed_out:
                error_msg = f'Timeout after {runtime:.2f}s ({self._iterations_run} iterations)'
            else:
                error_msg = f'Failed to route after {self._iterations_run} iterations'

            return {
                'status': 'failed',
                'placement': None,
                'routes': None,
                'route_metadata': None,
                'runtime': runtime,
                'iterations': self._iterations_run,
                'final_cost': self._final_cost,
                'error_message': error_msg
            }

    def map(self, max_ii: int = 32) -> Dict[str, Any]:
        """
        Run the heuristic mapping flow with automatic II escalation.
        
        Args:
            max_ii: Maximum Initiation Interval to attempt
            
        Returns:
            Dictionary with mapping results across all II attempts
        """
        start_time = time.time()
        
        # Calculate theoretical minimum II
        min_ii = self._calculate_min_ii()
        current_ii = max(1, min_ii)
        
        if self._debug:
            print(f"\nTheoretical minimum II: {min_ii}")
            
        # Time expand MRRG if starting II > 1.
        # Always expand from the original base MRRG to avoid compounding
        # expansion artifacts across II attempts.
        if current_ii > self._base_mrrg.II:
            if self._debug:
                print(f"Time-expanding MRRG to starting II={current_ii}...")
            self._mrrg = self._base_mrrg.time_expand(current_ii)
        else:
            self._mrrg = self._base_mrrg

        total_iterations = 0
            
        while current_ii <= max_ii:
            if (max_ii % current_ii) != 0:
                current_ii += 1
                continue
            print(f"\n{'='*50}")
            print(f"Attempting mapping with II = {current_ii}")
            print(f"{'='*50}")
            
            try:
                result = self._map_with_ii()
                total_iterations += result.get('iterations', 0)
                
                if result['status'] == 'success':
                    result['runtime'] = time.time() - start_time
                    result['iterations'] = total_iterations
                    result['final_ii'] = current_ii
                    print(f"\n[SUCCESS] Successfully mapped with II={current_ii}")
                    return result
                    
                print(f"[FAILED] Mapping failed at II={current_ii}: {result.get('error_message', 'Unknown error')}")
                
                # Check if we should escalate II based on failure
                # Escalating on any failure for now to guarantee we find a valid mapping if one exists
                
            except Exception as e:
                print(f"[ERROR] Exception during mapping at II={current_ii}: {str(e)}")
                
            # Escalate II
            current_ii += 1
            if current_ii <= max_ii:
                if self._debug:
                    print(f"\nTime-expanding MRRG from II={current_ii-1} to II={current_ii}...")
                self._mrrg = self._base_mrrg.time_expand(current_ii)
                self._timed_out = False
                
        # Failed to map even at max_ii
        return {
            'status': 'failed',
            'placement': None,
            'routes': None,
            'route_metadata': None,
            'runtime': time.time() - start_time,
            'iterations': total_iterations,
            'final_cost': 0.0,
            'error_message': f'Failed to map after exhausting all II up to {max_ii}'
        }

    def _run_asap_scheduling(self) -> bool:
        """
        Run ASAP scheduling to assign time slots to operations.

        Returns:
            True if scheduling succeeded, False otherwise
        """
        try:
            scheduler = ASAPScheduler(
                dfg=self._dfg,
                latency_spec=self._latency_spec,
                debug=self._debug
            )

            max_latency = scheduler.schedule()

            if self._debug:
                print(f"ASAP scheduling completed. Max latency: {max_latency}")
                print(f"Scheduled {len([n for n in self._dfg.nodes if hasattr(n, 'asap_time')])} operations")

            return True

        except Exception as e:
            if self._debug:
                print(f"ASAP scheduling failed: {e}")
            return False

    def _run_iterative_place_and_route(self) -> bool:
        """
        Run iterative place-and-route loop with adaptive temperature.

        Returns:
            True if routing succeeded, False otherwise
        """
        # Track start time for timeout checking
        start_time = time.time()
        self._route_metadata = None

        def _make_placer(seed: int) -> AnnealPlacer:
            return AnnealPlacer(
                num_rows=self._mrrg.rows,
                num_cols=self._mrrg.cols,
                dfg=self._dfg,
                latency_spec=self._latency_spec,
                mrrg=self._mrrg,
                fixed_placement=[],
                random_seed=seed,
                swap_factor=self._swap_factor
            )

        # Initialize placer
        current_seed = self._random_seed
        placer = _make_placer(current_seed)
        if self._best_placement_hint or self._fu_failure_penalties:
            placer.set_initial_placement_with_history(
                dfg_to_fu_hints=self._best_placement_hint,
                fu_penalties=self._fu_failure_penalties,
                explore_probability=0.35,
            )
        else:
            placer.set_initial_placement()

        # Initialize temperature and convergence tracking
        temperature = self._initial_temperature
        cost_history = deque(maxlen=self._convergence_window)
        best_uncovered = float("inf")
        stagnant_routing_iters = 0
        iterations_since_coverage_improvement = 0
        placement_restarts = 0

        # Iterative loop
        for iteration in range(self._max_iterations):
            self._iterations_run = iteration + 1

            # Check time limit
            if self._max_time is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= self._max_time:
                    if self._debug:
                        print(f"\n✗ Time limit exceeded ({elapsed_time:.2f}s >= {self._max_time}s)")
                    self._timed_out = True
                    return False

            if self._debug:
                print(f"\n--- Iteration {self._iterations_run} (T={temperature:.2f}) ---")

            # Run annealing at current temperature
            acceptance_rate = placer.anneal(initial_temperature=temperature)
            self._placement = placer._op_node_placement_map.copy()
            self._final_cost = placer.get_total_cost()

            if self._debug:
                print(f"Placement cost: {self._final_cost:.2f}")

            # Attempt to route with current placement
            router = PathFinder(
                dfg=self._dfg,
                mrrg=self._mrrg,
                placement=self._placement,
                p_growth_rate=self._p_growth_rate,
                h_growth_rate=self._h_growth_rate,
                max_iterations=self._router_max_iterations,
                historical_cost_seed=self._routing_history_costs,
                bitwidth_mismatch_cost=self._bitwidth_mismatch_penalty_weight,
                debug=self._debug
            )

            routing_success = router.route_dfg()
            self._merge_routing_history(router)

            if routing_success:
                # Success! Save routing solution and exit
                self._routing_solution = router._routing_solution
                self._route_metadata = router.get_route_metadata()

                if self._debug:
                    print(f"\n✓ Routing succeeded at iteration {self._iterations_run}")
                    print(f"Final placement cost: {self._final_cost:.2f}")

                return True

            if self._debug:
                print("✗ Routing failed, adjusting temperature and retrying")

            uncovered = getattr(router, "_last_uncovered_hypervals", None)
            if uncovered is None:
                uncovered = float("inf")

            if uncovered < best_uncovered:
                best_uncovered = uncovered
                self._best_placement_hint = {
                    dfg_node.id: fu_node.id for dfg_node, fu_node in self._placement.items()
                }
                stagnant_routing_iters = 0
                iterations_since_coverage_improvement = 0
            else:
                stagnant_routing_iters += 1
                iterations_since_coverage_improvement += 1
            self._accumulate_fu_failure_penalties(uncovered)

            # Placement appears locally impossible to route: restart from fresh placement.
            if (
                stagnant_routing_iters >= self._placement_restart_patience
                and placement_restarts < self._max_placement_restarts
            ):
                placement_restarts += 1
                current_seed = self._random_seed + (placement_restarts * 1009) + self._iterations_run
                if self._debug:
                    print(
                        f"[RESTART] Restarting placement at II={self._mrrg.II} "
                        f"(restart {placement_restarts}/{self._max_placement_restarts}, "
                        f"best_uncovered={best_uncovered}, seed={current_seed})"
                    )
                placer = _make_placer(current_seed)
                placer.set_initial_placement_with_history(
                    dfg_to_fu_hints=self._best_placement_hint,
                    fu_penalties=self._fu_failure_penalties,
                    explore_probability=0.25,
                )
                temperature = self._initial_temperature
                cost_history.clear()
                stagnant_routing_iters = 0
                continue

            # Too many placement iterations with no coverage improvement at this II:
            # hand control back to map() so it can escalate II and re-schedule.
            if iterations_since_coverage_improvement >= self._ii_iteration_patience:
                if self._debug:
                    print(
                        f"[II-ESCALATE] No routing coverage improvement for "
                        f"{iterations_since_coverage_improvement} placement iterations "
                        f"at II={self._mrrg.II}. Moving to next II."
                    )
                return False

            # Adaptive temperature adjustment based on acceptance rate
            temperature = self._adjust_temperature(temperature, acceptance_rate)

            # Early stopping if cost plateaus
            cost_history.append(self._final_cost)
            if self._check_convergence(cost_history):
                if self._debug:
                    print(f"\nConvergence detected at iteration {self._iterations_run}")
                    print("Cost has plateaued - stopping early")
                break

        # Failed to route after max iterations
        if self._debug:
            print(f"\n✗ Failed to route after {self._iterations_run} iterations")

        return False

    def _merge_routing_history(self, router: PathFinder) -> None:
        """Persist routing congestion history across placement retries."""
        current_history = router.export_historical_costs()
        for node_id, new_cost in current_history.items():
            prev = self._routing_history_costs.get(node_id, 0.0)
            # Keep memory of prior congestion while favoring newer evidence.
            self._routing_history_costs[node_id] = min(1_000_000.0, max(prev * 0.9, new_cost))

    def _accumulate_fu_failure_penalties(self, uncovered: float) -> None:
        """Penalize FUs repeatedly used in failed placements."""
        if not self._placement:
            return
        bump = max(0.5, min(5.0, float(uncovered) / max(1.0, len(self._placement))))
        for _, fu_node in self._placement.items():
            self._fu_failure_penalties[fu_node.id] = self._fu_failure_penalties.get(fu_node.id, 0.0) + bump

    def _adjust_temperature(self, current_temp: float, acceptance_rate: float) -> float:
        """
        Adjust temperature adaptively based on acceptance rate.

        Similar to ClusteredMapper's determineTemperature(10):
        - If acceptance rate is too high, decrease temperature (more selective)
        - If acceptance rate is too low, increase temperature (more exploratory)

        Args:
            current_temp: Current temperature
            acceptance_rate: Acceptance rate from last annealing run (0.0 to 1.0)

        Returns:
            Adjusted temperature
        """
        # Target acceptance rate around 0.3-0.4 (30-40%)
        target_rate = 0.35

        if acceptance_rate > target_rate + 0.1:
            # Too many acceptances - cool down faster
            return current_temp * 0.85
        elif acceptance_rate < target_rate - 0.1:
            # Too few acceptances - heat up
            return current_temp * 1.15
        else:
            # Close to target - gentle cooling
            return current_temp * 0.95

    def _check_convergence(self, cost_history: deque) -> bool:
        """
        Check if placement cost has converged (plateaued).

        Args:
            cost_history: Recent cost values (deque with max length)

        Returns:
            True if converged, False otherwise
        """
        if len(cost_history) < self._convergence_window:
            return False

        # Check if cost variation is below threshold
        costs = list(cost_history)
        min_cost = min(costs)
        max_cost = max(costs)

        if min_cost == 0:
            return False

        relative_variation = (max_cost - min_cost) / min_cost

        return relative_variation < self._convergence_threshold

    def _convert_placement_to_dict(self) -> Dict[str, str]:
        """
        Convert placement from Dict[DFGNode, MRRGNode] to Dict[str, str].

        Returns:
            Dictionary mapping DFG node IDs to MRRG node IDs
        """
        if self._placement is None:
            return {}

        return {
            dfg_node.id: mrrg_node.id
            for dfg_node, mrrg_node in self._placement.items()
        }

    def _convert_routing_to_dict(self) -> Dict[Tuple[str, int], List[str]]:
        """
        Convert routing solution to dictionary format.

        PathFinder returns: Dict[Tuple[HyperVal, int], List[MRRGNode]]
        We convert to: Dict[Tuple[str, int], List[str]]

        Returns:
            Dictionary mapping (source_id, dest_index) to routing path (list of node IDs)
        """
        if self._routing_solution is None:
            return {}

        routing_dict = {}

        for (hyperval, dest_idx), path in self._routing_solution.items():
            # Convert HyperVal to source ID (use the defining operation's ID)
            source_id = hyperval.source_id

            # Convert MRRGNode list to node ID list
            path_ids = [node.id for node in path]

            routing_dict[(source_id, dest_idx)] = path_ids

        return routing_dict

    def _validate_solution(
        self,
        placement: Dict[str, str],
        routes: Dict[Tuple[str, int], List[str]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate the heuristic mapping solution for correctness.

        Checks:
        1. All DFG nodes are placed
        2. No resource conflicts (same FU used twice in same cycle)
        3. Routing path connectivity (consecutive nodes have edges)
        4. All required routes are present
        5. Source/sink FU connectivity to routes

        Args:
            placement: Dictionary mapping DFG node IDs to MRRG node IDs
            routes: Dictionary mapping (source_id, sink_idx) to routing paths

        Returns:
            Tuple of (is_valid: bool, errors: list of error messages)
        """
        from mapper.graph.hyperdfg import HyperDFG

        errors = []

        if self._debug:
            print("\n" + "="*60)
            print("SOLUTION VALIDATION")
            print("="*60)

        # 1. Check all DFG nodes are placed
        placed_nodes = set(placement.keys())
        all_nodes = set(n.id for n in self._dfg.nodes)
        missing = all_nodes - placed_nodes

        if missing:
            error = f"Missing placements for nodes: {missing}"
            errors.append(error)
            if self._debug:
                print(f"✗ {error}")
        elif self._debug:
            print(f"✓ All {len(all_nodes)} DFG nodes placed")

        # 2. Check no resource conflicts (same FU used twice in same cycle)
        fu_usage = {}  # (fu_base_id, cycle) -> node_id
        for node_id, fu_id in placement.items():
            # Parse "cycle:fu_name" format
            if ':' in fu_id:
                cycle_str, fu_base = fu_id.split(':', 1)
                cycle = int(cycle_str)
                key = (fu_base, cycle)

                if key in fu_usage:
                    error = f"Resource conflict: {node_id} and {fu_usage[key]} both use {fu_base} at cycle {cycle}"
                    errors.append(error)
                    if self._debug:
                        print(f"✗ {error}")
                else:
                    fu_usage[key] = node_id

        if self._debug and not any("Resource conflict" in e for e in errors):
            print(f"✓ No resource conflicts ({len(fu_usage)} FU usages)")

        # 3. Check routing path connectivity
        invalid_routes = []
        for (source_id, sink_idx), path in routes.items():
            if len(path) == 0:
                # Empty path is valid for self-loops or direct connections
                continue

            if len(path) == 1:
                # Single-node path - check it connects to source and sink
                continue

            # Check consecutive edges exist in MRRG
            for i in range(len(path) - 1):
                src_node_id = path[i]
                dst_node_id = path[i + 1]

                # Check if edge exists
                outgoing = self._mrrg.get_outgoing_edges(src_node_id)
                edge_exists = any(e.destination.id == dst_node_id for e in outgoing)

                if not edge_exists:
                    error = f"Missing edge in route {source_id}→sink[{sink_idx}]: {src_node_id} → {dst_node_id}"
                    errors.append(error)
                    invalid_routes.append((source_id, sink_idx))
                    break

        if invalid_routes:
            if self._debug:
                print(f"✗ Invalid routing paths: {len(invalid_routes)} routes")
                for source_id, sink_idx in invalid_routes[:5]:
                    print(f"   - {source_id} → sink[{sink_idx}]")
                if len(invalid_routes) > 5:
                    print(f"   ... and {len(invalid_routes) - 5} more")
        elif self._debug:
            print(f"✓ All {len(routes)} routing paths have valid connectivity")

        # 4. Check all required routes are present (all HyperVal destinations)
        hyperdfg = HyperDFG.from_dfg(self._dfg)
        expected_routes = set()
        for hyperval in hyperdfg.get_edges():
            for k in range(hyperval.cardinality):
                expected_routes.add((hyperval.source_id, k))

        actual_routes = set(routes.keys())
        missing_routes = expected_routes - actual_routes

        if missing_routes:
            error = f"Missing routes for {len(missing_routes)} destinations"
            errors.append(error)
            if self._debug:
                print(f"✗ {error}")
                for source_id, sink_idx in list(missing_routes)[:5]:
                    # Find destination ID
                    dest_id = "?"
                    for hv in hyperdfg.get_edges():
                        if hv.source_id == source_id and sink_idx < len(hv.destination_ids):
                            dest_id = hv.destination_ids[sink_idx]
                            break
                    print(f"   - {source_id} → {dest_id} [sink {sink_idx}]")
                if len(missing_routes) > 5:
                    print(f"   ... and {len(missing_routes) - 5} more")
        elif self._debug:
            print(f"✓ All {len(expected_routes)} required routes present")

        # 5. Check source FU connects to route start and route end connects to sink FU
        connectivity_errors = []
        for (source_id, sink_idx), path in routes.items():
            if len(path) == 0:
                continue

            # Get source FU
            source_fu_id = placement.get(source_id)
            if not source_fu_id:
                continue

            # Check source FU has edge to first routing node
            first_routing_node = path[0]
            source_outgoing = self._mrrg.get_outgoing_edges(source_fu_id)
            source_connected = any(
                e.destination.id == first_routing_node for e in source_outgoing
            )

            if not source_connected:
                error = f"Source FU {source_fu_id} not connected to route start {first_routing_node}"
                connectivity_errors.append(error)
                errors.append(error)

            # Get sink node ID from HyperVal
            sink_node_id = None
            for hv in hyperdfg.get_edges():
                if hv.source_id == source_id and sink_idx < len(hv.destination_ids):
                    sink_node_id = hv.destination_ids[sink_idx]
                    break

            if sink_node_id:
                sink_fu_id = placement.get(sink_node_id)
                if sink_fu_id:
                    # Check last routing node has edge to sink FU
                    last_routing_node = path[-1]
                    last_outgoing = self._mrrg.get_outgoing_edges(last_routing_node)
                    sink_connected = any(
                        e.destination.id == sink_fu_id for e in last_outgoing
                    )

                    if not sink_connected:
                        error = f"Route end {last_routing_node} not connected to sink FU {sink_fu_id}"
                        connectivity_errors.append(error)
                        errors.append(error)

        if connectivity_errors:
            if self._debug:
                print(f"✗ FU connectivity errors: {len(connectivity_errors)}")
                for err in connectivity_errors[:3]:
                    print(f"   - {err}")
                if len(connectivity_errors) > 3:
                    print(f"   ... and {len(connectivity_errors) - 3} more")
        elif self._debug and routes:
            print(f"✓ All routes properly connected to source/sink FUs")

        # Shwet TODO: Is this validation correct or should we be checking for overuse by number of nets?

        routing_usage: Dict[str, Set[str]] = {}  # routing_node_id -> set of HyperVal IDs
        for (source_id, sink_idx), path in routes.items():
            for node_id in path:
                if node_id not in routing_usage:
                    routing_usage[node_id] = set()
                routing_usage[node_id].add(source_id)

        overused_nodes = []
        for node_id, hv_set in routing_usage.items():
            node = self._mrrg.get_node(node_id)
            if node and len(hv_set) > node.capacity:
                overused_nodes.append((node_id, len(hv_set), node.capacity))
                error = f"Routing node {node_id} overused: {len(hv_set)}/{node.capacity}"
                errors.append(error)

        if overused_nodes:
            if self._debug:
                print(f"✗ Routing resource overuse: {len(overused_nodes)} nodes")
                for node_id, count, capacity in overused_nodes[:5]:
                    print(f"   - {node_id}: {count}/{capacity}")
                if len(overused_nodes) > 5:
                    print(f"   ... and {len(overused_nodes) - 5} more")
        elif self._debug:
            print(f"✓ No routing resource overuse")

        if self._debug:
            print("="*60)
            if len(errors) == 0:
                print("✓✓✓ SOLUTION VALIDATION PASSED ✓✓✓")
            else:
                print(f"✗✗✗ SOLUTION VALIDATION FAILED ({len(errors)} errors) ✗✗✗")
            print("="*60)

        return len(errors) == 0, errors
