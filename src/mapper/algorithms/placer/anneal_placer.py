"""Anneal Placer for Heuristic Mapping."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
import random
import math

from mapper.graph.dfg import DFG, DFGNode, OperationType, DFGEdge
from mapper.graph.hyperdfg import HyperDFG, HyperNode, HyperVal
from mapper.schedules import LatencySpecification
from mapper.graph.mrrg import MRRG, MRRGNode, MRRGEdge, NodeType, HWEntityType, OperandTag
from mapper.algorithms.placer.types import MRRGNodePlacementState
from mapper.algorithms.archive.models.cost_models import WirelengthLatencyCostModel


class AnnealPlacer:
    """Anneal Placer for Heuristic Mapping."""

    # Placer properties
    _dfg: DFG
    _latency_spec: LatencySpecification
    _mrrg: MRRG
    _ii: int
    _num_rows: int
    _num_cols: int

    # Mapping from MRRG node to its placement state
    _fu_node_placement_state: Dict[MRRGNode, MRRGNodePlacementState]

    # Mapping from DFG node to corresponding FUs
    _op_node_placement_map: Dict[DFGNode, MRRGNode]

    # Mapping from (OperationType, cycle % II ) to MRRG nodes that can execute the operation at the given cycle
    _temporal_op_type_to_fu_nodes: Dict[Tuple[OperationType, int], List[MRRGNode]]

    # User reserved MRRG nodes
    _reserved_fu_nodes: Set[MRRGNode]

    # User reserved map from DFG node id to MRRG node id
    _reserved_dfg_node_id_to_fu_node_id: Dict[str, str]

    # Random seed
    _random_seed: int

    # Swap factor for inner loop
    _swap_factor: int

    def __init__(self, num_rows: int, num_cols: int, dfg: DFG, latency_spec: LatencySpecification, mrrg: MRRG, fixed_placement: List[Tuple[str, str]], random_seed: int = 42, swap_factor: int = 10) -> None:
        """Initialize the Anneal Placer.

        Args:
            num_rows: Number of rows in the CGRA
            num_cols: Number of columns in the CGRA
            dfg: Data Flow Graph
            latency_spec: Latency specification
            mrrg: MRRG
            fixed_placement: List of (dfg_node_id, mrrg_node_id) tuples for fixed placements
            random_seed: Random seed for reproducibility
            swap_factor: Number of swap attempts = num_operations * swap_factor
        """
        self._dfg = dfg
        self._latency_spec = latency_spec
        self._mrrg = mrrg
        self._ii = mrrg.II
        self._num_rows = num_rows
        self._num_cols = num_cols

        # Shwet TODO: Do we want to generate a random seed here or make this user driven?
        self._random_seed = random_seed
        self._swap_factor = swap_factor

        # Initialize state dictionaries
        self._fu_node_placement_state = {}
        self._op_node_placement_map = {}
        self._temporal_op_type_to_fu_nodes = {}
        self._reserved_fu_nodes = set()
        self._reserved_dfg_node_id_to_fu_node_id = {}

        # Compute the node classes
        classes = self.__compute_fu_node_classes()

        # Initialize the fixed placements
        for dfg_node_id, mrrg_node_id in fixed_placement:

            if mrrg_node_id not in self._fu_node_placement_state:
                raise ValueError(f"MRRG node {mrrg_node_id} not found in the MRRG")
            
            fu_node = mrrg.get_node(mrrg_node_id)
            self._reserved_fu_nodes.add(fu_node)
            self._reserved_dfg_node_id_to_fu_node_id[dfg_node_id] = fu_node.id

        # Initialize FU node states to zero
        for fu_node_v in classes:
            for fu_node in fu_node_v:

                self._fu_node_placement_state[fu_node] = MRRGNodePlacementState(occupancy=0)

        # Build Operation Type to FU nodes map per cycle
        for op in dfg.get_nodes():
            for i in range(self._ii):

                if (op.operation, i) in self._temporal_op_type_to_fu_nodes:
                    continue
                
                # Create an empty list of FU nodes that can execute op at cycle i (cycle % II)
                fu_nodes: List[MRRGNode] = []
                for mrrg_node in classes[i]:
                    if mrrg_node.can_execute(op.operation, op.bitwidth):
                        fu_nodes.append(mrrg_node)

                self._temporal_op_type_to_fu_nodes[(op.operation, i)] = fu_nodes

    def anneal(self, initial_temperature: float) -> None:
        """Anneal the placement."""
        temperature: float = initial_temperature

        while True:
            # Inner loop: perform swaps at current temperature
            accept_rate: float = self.inner_loop(temperature)
            
            # Check termination
            current_cost: float = self.get_total_cost()
            if current_cost == 0:
                current_cost = 1

            threshold: float = 0.05 * (current_cost / len(self._dfg.get_nodes()))
            if temperature < threshold:
                break

            temperature = self.next_temperature(temperature, accept_rate)

    def update_op_placement(self, op: DFGNode, fu_node: MRRGNode) -> None:
        """Update the placement state and mapfor the given operation and FU node."""
        self._fu_node_placement_state[fu_node].occupancy += 1
        self._op_node_placement_map[op] = fu_node

    def clear_op_placement(self, op: DFGNode) -> None:
        """Clear the placement state and map for the given operation."""
        self._fu_node_placement_state[self._op_node_placement_map[op]].occupancy -= 1
        self._op_node_placement_map.pop(op)

    def set_initial_placement(self) -> None:
        """Set the initial placement for all the DFG nodes."""
        for op in self._dfg.get_nodes():
            if op.id in self._reserved_dfg_node_id_to_fu_node_id:
                continue
            self.update_op_placement(op, self.get_random_unoccupied_fu(op))

    def get_random_fu(self, op: DFGNode) -> MRRGNode:
        """Get a random FU node that can execute the given operation, which may be occupied or not."""

        # Try all cycles, starting from the preferred cycle (asap_time % II)
        preferred_cycle: int = op.asap_time % self._ii
        cycles_to_try = [preferred_cycle] + [c for c in range(self._ii) if c != preferred_cycle]

        for cycle in cycles_to_try:
            key = (op.operation, cycle)
            if key in self._temporal_op_type_to_fu_nodes:
                candidate_fu_nodes: List[MRRGNode] = self._temporal_op_type_to_fu_nodes[key]
                
                # Filter out reserved nodes
                available_fus = [fu for fu in candidate_fu_nodes if fu.id not in self._reserved_fu_nodes]
                
                if available_fus:
                    # Return a random FU from the available ones in this cycle
                    random_index: int = random.randint(0, len(available_fus) - 1)
                    return available_fus[random_index]

        raise ValueError(f"No FU nodes found that can execute the operation {op.operation} dynamically across II={self._ii} cycles")

    def get_random_unoccupied_fu(self, op: DFGNode) -> MRRGNode:
        """Get a random FU node that is not occupied."""

        preferred_cycle: int = op.asap_time % self._ii
        cycles_to_try = [preferred_cycle] + [c for c in range(self._ii) if c != preferred_cycle]

        unoccupied_fu_nodes: List[MRRGNode] = []

        for cycle in cycles_to_try:
            key = (op.operation, cycle)
            if key in self._temporal_op_type_to_fu_nodes:
                candidate_fu_nodes: List[MRRGNode] = self._temporal_op_type_to_fu_nodes[key]

                for fu_node in candidate_fu_nodes:
                    if fu_node.id not in self._reserved_fu_nodes and self._fu_node_placement_state[fu_node].occupancy < fu_node.capacity:
                        unoccupied_fu_nodes.append(fu_node)
            
            # If we found unoccupied FUs in this cycle, we can stop searching other cycles
            # to preserve ASAP hints where possible, or we could collect all for maximum mobility.
            # Collecting all gives the annealer more flexibility to place nodes in other cycles.
            # We'll continue the loop to collect them all.

        if len(unoccupied_fu_nodes) < 1:
            raise ValueError(f"No unoccupied FU nodes found that can execute the operation {op.operation} across all II={self._ii} cycles")

        # Grab a random FU node from the unoccupied FU nodes
        random_index: int = random.randint(0, len(unoccupied_fu_nodes) - 1)
        return unoccupied_fu_nodes[random_index]

    def clear_placement(self) -> None:
        """Clear the placement state for all the FU nodes."""
        for fu_node in self._fu_node_placement_state:
            self._fu_node_placement_state[fu_node].occupancy = 0

    # Helper function to compute the FU node classes
    def __compute_fu_node_classes(self) -> List[List[MRRGNode]]:

        node_classes: List[List[MRRGNode]] = []

        # Compute the node classes
        for i in range(self._ii):
            node_classes.append([])

        for node in self._mrrg.get_fu_nodes():
            node_classes[node.cycle].append(node)

        return node_classes
        
    def get_total_cost(self) -> float:
        """
        Get the total cost of the placement.

        Uses WirelengthLatencyCostModel to compute wirelength and latency costs.

        Returns:
            Total cost of the current placement
        """
        # Create cost model
        cost_model = WirelengthLatencyCostModel()

        # Convert placement map from DFGNode -> MRRGNode to DFG node ID -> MRRGNode
        placement_map: Dict[str, MRRGNode] = {
            dfg_node.id: mrrg_node
            for dfg_node, mrrg_node in self._op_node_placement_map.items()
        }

        # Get schedule from DFG nodes (assuming they have scheduled_time attribute)
        schedule: Dict[str, int] = {}
        for dfg_node in self._dfg.get_nodes():
            if dfg_node.asap_time is not None:
                schedule[dfg_node.id] = dfg_node.asap_time
            else:
                # If not scheduled, use 0 as default
                schedule[dfg_node.id] = 0

        # Compute cost using the cost model
        # Use cost_function=1 for combined wire + latency cost
        total_cost = cost_model.compute_total_cost(
            dfg=self._dfg,
            mrrg=self._mrrg,
            placement_map=placement_map,
            schedule=schedule,
            II=self._ii,
            cost_function=1  # Combined cost mode
        )

        return total_cost

    def inner_loop(self, temperature: float) -> float:
        """
        Perform placement moves at the current temperature.

        Args:
            temperature: Current annealing temperature

        Returns:
            float: Acceptance rate (accepted_moves / total_tries)
        """
        # Calculate number of swap attempts
        num_swaps = len(self._dfg.get_nodes()) * self._swap_factor
        total_accepted = 0
        total_tries = 0

        for i in range(num_swaps):
            # Randomly select an operation
            operations = list(self._dfg.get_nodes())
            op = random.choice(operations)

            # Skip if this operation is reserved/fixed
            if op.id in self._reserved_dfg_node_id_to_fu_node_id:
                continue

            # Get a random compatible FU for this operation
            new_fu = self.get_random_fu(op)
            if new_fu is None:
                continue

            # Skip if it's the same FU (no change)
            current_fu = self._op_node_placement_map.get(op)
            if current_fu and new_fu.id == current_fu.id:
                continue

            # This is an actual attempt
            total_tries += 1

            # Save old cost and placement
            old_cost = self.get_total_cost()
            old_placement: Dict[DFGNode, MRRGNode] = {}

            # Check if new FU has capacity
            if self._fu_node_placement_state[new_fu].occupancy < new_fu.capacity:
                # FU is available, just move the operation
                old_placement[op] = current_fu
                self.clear_op_placement(op)
                self.update_op_placement(op, new_fu)
            else:
                # FU is occupied, need to swap
                # Find the operation currently on this FU
                displaced_op = None
                for dfg_op, fu_node in self._op_node_placement_map.items():
                    if fu_node.id == new_fu.id:
                        displaced_op = dfg_op
                        break

                if displaced_op is None:
                    # This shouldn't happen, but skip if it does
                    continue

                # Save old placements
                old_placement[op] = current_fu
                old_placement[displaced_op] = new_fu

                # Perform swap
                self.clear_op_placement(op)
                self.clear_op_placement(displaced_op)
                self.update_op_placement(op, new_fu)
                self.update_op_placement(displaced_op, current_fu)

            # Get new cost
            new_cost = self.get_total_cost()
            delta_cost = new_cost - old_cost

            # Accept or reject the move
            if self.accept_move(delta_cost, temperature):
                if delta_cost > 0:
                    total_accepted += 1
            else:
                # Restore old placement
                for dfg_op, fu_node in old_placement.items():
                    self.clear_op_placement(dfg_op)
                    self.update_op_placement(dfg_op, fu_node)

        # Return acceptance rate
        if total_tries == 0:
            return 0.0
        return total_accepted / total_tries

    def accept_move(self, delta_cost: float, temperature: float) -> bool:
        """Accept the move based on the delta cost and temperature."""
        if delta_cost < 0:
            return True
        probability: float = math.exp(-delta_cost / temperature)
        random_val: float = random.random()  # Returns value in [0.0, 1.0)
        return probability > random_val          

    def next_temperature(self, temperature: float, accept_rate: float) -> float:
        """Get the next temperature based on the accept rate."""
        if accept_rate > 0.96:
            return temperature * 0.5
        elif accept_rate > 0.8:
            return temperature * 0.9
        elif accept_rate > 0.15:
            return temperature * 0.95
        else:
            return temperature * 0.8

    def pretty_print_placement(self) -> str:
        """
        Generate a pretty-printed visualization of the current placement.

        Shows the mapping of DFG operations to MRRG nodes with scheduled cycles.

        Returns:
            String representation of the placement
        """
        if not self._op_node_placement_map:
            return "No placement available."

        # Build the visualization string
        lines = []
        lines.append("=" * 90)
        lines.append("PLACEMENT VISUALIZATION")
        lines.append("=" * 90)
        lines.append(f"Total operations placed: {len(self._op_node_placement_map)}")
        lines.append("=" * 90)
        lines.append("")
        lines.append(f"{'DFG Operation':<30} {'Op Type':<15} {'Cycle':<8} {'MRRG Node':<35}")
        lines.append("-" * 90)

        # Sort by scheduled cycle first, then by DFG node id
        sorted_mappings = sorted(
            self._op_node_placement_map.items(),
            key=lambda x: (x[0].scheduled_time if x[0].scheduled_time is not None else 0, x[0].id)
        )

        for dfg_node, mrrg_node in sorted_mappings:
            dfg_id = dfg_node.id
            op_type = dfg_node.operation.name
            mrrg_id = mrrg_node.id

            # Get scheduled cycle
            cycle = dfg_node.scheduled_time if dfg_node.scheduled_time is not None else "N/A"
            cycle_str = str(cycle) if cycle != "N/A" else cycle

            lines.append(f"{dfg_id:<30} {op_type:<15} {cycle_str:<8} {mrrg_id:<35}")

        lines.append("=" * 90)

        return "\n".join(lines)


    # -- Placer Properties --------------------------------------------------------
    def op_node_placement_map(self) -> Dict[DFGNode, MRRGNode]:
        """Get the operation node placement map."""
        return self._op_node_placement_map

    def fu_node_placement_state(self) -> Dict[MRRGNode, MRRGNodePlacementState]:
        """Get the FU node placement state."""
        return self._fu_node_placement_state

    def temporal_op_type_to_fu_nodes(self) -> Dict[Tuple[OperationType, int], List[MRRGNode]]:
        """Get the temporal operation type to FU nodes map."""
        return self._temporal_op_type_to_fu_nodes