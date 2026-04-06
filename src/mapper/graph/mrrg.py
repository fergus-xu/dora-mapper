"""Modulo Routing Resource Graph (MRRG) representation."""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from mapper.graph.graph_base import Graph, Node, Edge
from mapper.graph.dfg import OperationType
import mapper.graph.utils.traversal as traversal


class NodeType(Enum):
    """
    Types of nodes in an MRRG (CGRA-ME classification).

    Note: Use HWEntityType to distinguish between different hardware resources
    (registers, wires, muxes, etc.), not NodeType.
    """
    ROUTING = "routing"              # MRRG_NODE_ROUTING: Pure routing/data movement
    FUNCTION = "function"             # MRRG_NODE_FUNCTION: Computational operations
    ROUTING_FUNCTION = "route_fu"    # MRRG_NODE_ROUTING_FUNCTION: Hybrid (PHI, SELECT)


class OperandTag(Enum):
    """Operand tags for FU input binding constraints (CGRA-ME specification)."""
    UNTAGGED = ""  # Untagged/default
    PREDICATE = "pred"  # Predicate/condition
    UNARY = "unary"
    BINARY_LHS = "LHS"  # Left-hand side operand
    BINARY_RHS = "RHS"  # Right-hand side operand
    BINARY_ANY = "any2input"  # Either operand
    TERNARY_ANY = "any3input"  # Any of three operands
    MEM_ADDR = "addr"  # Memory address
    MEM_DATA = "data"  # Memory data
    BR_TRUE = "branch_true"  # Branch true
    BR_FALSE = "branch_false"  # Branch false


class HWEntityType(Enum):
    """Hardware entity types for MRRG nodes."""
    HW_WIRE = "wire"  # Combinational wire (latency=0)
    HW_REG = "reg"  # Register (latency>=1)
    HW_MUX = "mux"  # Multiplexer
    HW_COMB = "comb"  # Combinational logic (FU, memory)
    HW_UNSPECIFIED = "unspecified"


class MRRGNode(Node):
    """Unified MRRG node with all attributes."""

    def __init__(
        self,
        node_id: str,
        node_type: NodeType,
        cycle: int,
        coordinates: Optional[Tuple[int, int]] = None,
        hw_entity_type: HWEntityType = HWEntityType.HW_UNSPECIFIED,
        latency: int = 0,
        bitwidth: int = 32,
        capacity: int = 1,
        # FU-specific attributes
        supported_operations: Optional[Set[OperationType]] = None,
        supported_operand_tags: Optional[Set[OperandTag]] = None,
        num_inputs: int = 2,
        num_outputs: int = 1,
        # Routing-specific attributes
        routing_type: str = "wire",
        # Register-specific attributes
        bank_id: Optional[int] = None,
        read_ports: int = 1,
        write_ports: int = 1,
        # Mixed-precision / packing metadata
        lane_width: Optional[int] = None,
        num_lanes: Optional[int] = None,
        pack_capable: bool = False,
        unpack_capable: bool = False,
        allowed_lane_indices: Optional[Tuple[int, ...]] = None,
        pack_config: Optional[Dict[str, Any]] = None,
        unpack_config: Optional[Dict[str, Any]] = None,
        fracture_type: Optional[str] = None,
        fracture_parent_inst: Optional[str] = None,
        fracture_chunk_width: Optional[int] = None,
        fracture_parent_port: Optional[str] = None,
        fracture_chunk_index: Optional[int] = None,
        fracture_bit_offset: Optional[int] = None,
        fracture_num_chunks: Optional[int] = None,
        **attributes: Any
    ) -> None:
        """
        Initialize an MRRG node.

        Args:
            node_id: Unique identifier (format: "cycle:name" for time-expanded nodes)
            node_type: Type of MRRG node (ROUTING, FUNCTION, ROUTING_FUNCTION)
            cycle: Time slot within II (0 to II-1)
            coordinates: (x, y) position in the CGRA array
            hw_entity_type: Hardware entity type (WIRE, REG, MUX, COMB)
            latency: Cycles through this node
            bitwidth: Data width in bits
            capacity: Multi-use capacity
            supported_operations: Set of operations this node can execute (for FUNCTION nodes)
            supported_operand_tags: Set of operand tags for input binding (for FUNCTION nodes)
            num_inputs: Number of input ports (for FUNCTION nodes)
            num_outputs: Number of output ports (for FUNCTION nodes)
            routing_type: Type of routing resource - wire, mux, crossbar (for ROUTING nodes)
            bank_id: Register bank identifier (for register nodes)
            read_ports: Number of read ports (for register nodes)
            write_ports: Number of write ports (for register nodes)
            **attributes: Additional attributes
        """
        super().__init__(node_id, **attributes)
        self.node_type = node_type
        self.cycle = cycle
        self.coordinates = coordinates
        self.hw_entity_type = hw_entity_type
        self.latency = latency
        self.bitwidth = bitwidth
        self.capacity = capacity

        # FU attributes
        self.supported_operations = supported_operations or set()
        self.supported_operand_tags = supported_operand_tags or set()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # Routing attributes
        self.routing_type = routing_type

        # Register attributes
        self.bank_id = bank_id
        self.read_ports = read_ports
        self.write_ports = write_ports

        # Mixed-precision / packing metadata (optional, architecture-dependent)
        self.lane_width = lane_width
        self.num_lanes = num_lanes
        self.pack_capable = pack_capable
        self.unpack_capable = unpack_capable
        self.allowed_lane_indices = allowed_lane_indices
        self.pack_config = pack_config
        self.unpack_config = unpack_config
        self.fracture_type = fracture_type
        self.fracture_parent_inst = fracture_parent_inst
        self.fracture_chunk_width = fracture_chunk_width
        self.fracture_parent_port = fracture_parent_port
        self.fracture_chunk_index = fracture_chunk_index
        self.fracture_bit_offset = fracture_bit_offset
        self.fracture_num_chunks = fracture_num_chunks

    def can_execute(self, operation: OperationType, required_bitwidth: Optional[int] = None) -> bool:
        """
        Check if this node can execute the given operation at the required bitwidth.
        
        Args:
            operation: The operation type to check
            required_bitwidth: Optional minimum bitwidth required (in bits)
                              If None, only operation type is checked
        
        Returns:
            True if this FU can execute the operation with sufficient bitwidth
        """
        # Check operation compatibility
        if operation not in self.supported_operations:
            return False
        
        # Check bitwidth compatibility if specified
        if required_bitwidth is not None and self.bitwidth < required_bitwidth:
            return False
        
        return True

    def supports_operand_tag(self, tag: OperandTag) -> bool:
        """Check if this node supports a specific operand tag."""
        return tag in self.supported_operand_tags

    def get_base_name(self) -> str:
        """Get the base name without cycle prefix."""
        if ':' in self.id:
            return self.id.split(':', 1)[1]
        return self.id

    def get_full_name(self) -> str:
        """Get full name in CGRA-ME format: 'cycle:name'."""
        base_name = self.get_base_name()
        return f"{self.cycle}:{base_name}"

    def to_string(self) -> str:
        """
        Return detailed string representation of the node.

        Returns:
            Multi-line string with all node attributes
        """
        lines = [
            f"Node: {self.id}",
            f"  Node Type: {self.node_type.value}",
            f"  HW Entity Type: {self.hw_entity_type.value}",
            f"  Cycle: {self.cycle}",
            f"  Coordinates: {self.coordinates}",
            f"  Latency: {self.latency}",
            f"  Bitwidth: {self.bitwidth}",
            f"  Capacity: {self.capacity}",
        ]

        # Add FUNCTION-specific attributes
        if self.supported_operations:
            ops = ', '.join(sorted([op.value for op in self.supported_operations]))
            lines.append(f"  Operations: {ops}")

        if self.supported_operand_tags:
            tags = ', '.join(sorted([tag.value for tag in self.supported_operand_tags]))
            lines.append(f"  Operand Tags: {tags}")

        if self.num_inputs != 2:  # Only show if non-default
            lines.append(f"  Num Inputs: {self.num_inputs}")

        if self.num_outputs != 1:  # Only show if non-default
            lines.append(f"  Num Outputs: {self.num_outputs}")

        # Add ROUTING-specific attributes
        if self.routing_type != "wire":
            lines.append(f"  Routing Type: {self.routing_type}")

        # Add REGISTER-specific attributes
        if self.bank_id is not None:
            lines.append(f"  Bank ID: {self.bank_id}")
            if self.read_ports != 1:
                lines.append(f"  Read Ports: {self.read_ports}")
            if self.write_ports != 1:
                lines.append(f"  Write Ports: {self.write_ports}")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        coord_str = f", pos={self.coordinates}" if self.coordinates else ""
        type_str = self.node_type.value
        if self.node_type == NodeType.FUNCTION and self.supported_operations:
            type_str += f"({len(self.supported_operations)} ops)"
        elif self.hw_entity_type == HWEntityType.HW_REG:
            type_str += "(reg)"
        return f"MRRGNode(id={self.id}, cycle={self.cycle}, type={type_str}{coord_str})"


class MRRGEdge(Edge):
    """Edge in an MRRG representing routing connections."""

    def __init__(
        self,
        edge_id: str,
        source: MRRGNode,
        destination: MRRGNode,
        latency: int = 0,
        **attributes: Any
    ) -> None:
        """
        Initialize an MRRG edge.

        Args:
            edge_id: Unique identifier
            source: Source node
            destination: Destination node
            latency: Routing latency
            **attributes: Additional attributes
        """
        super().__init__(edge_id, source, destination, **attributes)
        self.latency = latency
        self.capacity: int = attributes.get('capacity', 1)
        self.conflict_free: bool = attributes.get('conflict_free', True)

    def to_string(self) -> str:
        """
        Return detailed string representation of the edge.

        Returns:
            Multi-line string with all edge attributes
        """
        lines = [
            f"Edge: {self.id}",
            f"  Source: {self.source.id}",
            f"  Destination: {self.destination.id}",
            f"  Latency: {self.latency}",
            f"  Capacity: {self.capacity}",
            f"  Conflict Free: {self.conflict_free}",
        ]

        # Add cross-cycle information if applicable
        if self.source.cycle != self.destination.cycle:
            lines.append(f"  Cross-Cycle: {self.source.cycle} -> {self.destination.cycle}")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f"MRRGEdge({self.source.id} -> {self.destination.id}, lat={self.latency})"


class MRRG(Graph[MRRGNode, MRRGEdge]):
    """Modulo Routing Resource Graph representation."""

    def __init__(
        self,
        name: str = "MRRG",
        II: int = 1,
        rows: int = 0,
        cols: int = 0
    ) -> None:
        """
        Initialize an MRRG.

        Args:
            name: Name of the MRRG
            II: Initiation Interval (number of time slots for time expansion)
            rows: Number of rows in the CGRA
            cols: Number of columns in the CGRA
        """
        super().__init__(name)
        self.II = II
        self.rows = rows
        self.cols = cols
        self.is_from_json = False

        # Cycle-indexed storage (CGRA-ME style)
        # nodes_by_cycle[cycle][base_name] -> MRRGNode
        self.nodes_by_cycle: List[Dict[str, MRRGNode]] = [
            {} for _ in range(II)
        ]

        # Spatial indexing for quick lookup
        self._fu_nodes: Dict[Tuple[int, int], List[str]] = {}  # (x,y) -> list of FU node IDs
        self._routing_nodes: Dict[Tuple[int, int], List[str]] = {}  # (x,y) -> list of route node IDs
        self._register_nodes: Dict[Tuple[int, int], List[str]] = {}  # (x,y) -> list of reg node IDs

    def get_node_by_cycle(self, cycle: int, base_name: str) -> Optional[MRRGNode]:
        """
        Get node by cycle and base name (CGRA-ME style accessor).

        Args:
            cycle: Time slot (0 to II-1)
            base_name: Base name without cycle prefix

        Returns:
            MRRGNode if found, None otherwise
        """
        if 0 <= cycle < self.II:
            return self.nodes_by_cycle[cycle].get(base_name)
        return None

    def get_nodes_at_cycle(self, cycle: int) -> List[MRRGNode]:
        """Get all nodes at a specific cycle."""
        if 0 <= cycle < self.II:
            return list(self.nodes_by_cycle[cycle].values())
        return []

    def add_node(self, node: MRRGNode) -> None:
        """Override to add cycle-indexed and spatial storage."""
        super().add_node(node)

        # Add to cycle-indexed storage
        if 0 <= node.cycle < self.II:
            base_name = node.get_base_name()
            self.nodes_by_cycle[node.cycle][base_name] = node

        # Add to spatial indexing based on node type
        if node.coordinates:
            if node.node_type == NodeType.FUNCTION:
                if node.coordinates not in self._fu_nodes:
                    self._fu_nodes[node.coordinates] = []
                self._fu_nodes[node.coordinates].append(node.id)
            elif node.hw_entity_type == HWEntityType.HW_REG:
                if node.coordinates not in self._register_nodes:
                    self._register_nodes[node.coordinates] = []
                self._register_nodes[node.coordinates].append(node.id)
            elif node.node_type == NodeType.ROUTING:
                if node.coordinates not in self._routing_nodes:
                    self._routing_nodes[node.coordinates] = []
                self._routing_nodes[node.coordinates].append(node.id)

    def get_fu_nodes(self) -> List[MRRGNode]:
        """Get all functional unit nodes."""
        return [n for n in self.get_nodes() if n.node_type == NodeType.FUNCTION]

    def get_fus_at_position(self, x: int, y: int) -> List[MRRGNode]:
        """Get all FUs at a specific position."""
        node_ids = self._fu_nodes.get((x, y), [])
        return [self.get_node(nid) for nid in node_ids if self.get_node(nid)]

    def get_routing_nodes(self) -> List[MRRGNode]:
        """Get all routing nodes (excluding registers)."""
        return [n for n in self.get_nodes()
                if n.node_type == NodeType.ROUTING and n.hw_entity_type != HWEntityType.HW_REG]

    def get_register_nodes(self) -> List[MRRGNode]:
        """Get all register nodes."""
        return [n for n in self.get_nodes() if n.hw_entity_type == HWEntityType.HW_REG]

    def get_compatible_fus(self, operation: OperationType) -> List[MRRGNode]:
        """Get all FUs that can execute a specific operation."""
        return [fu for fu in self.get_fu_nodes() if fu.can_execute(operation)]

    def get_nodes_at_position(self, x: int, y: int) -> List[MRRGNode]:
        """Get all nodes at a specific position."""
        nodes = []
        for node_ids in [
            self._fu_nodes.get((x, y), []),
            self._routing_nodes.get((x, y), []),
            self._register_nodes.get((x, y), [])
        ]:
            nodes.extend([self.get_node(nid) for nid in node_ids if self.get_node(nid)])
        return nodes

    def validate(self) -> bool:
        """
        Validate the MRRG structure.

        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check that all FU nodes have valid coordinates
        for fu in self.get_fu_nodes():
            if fu.coordinates is None:
                raise ValueError(f"FU node {fu.id} has no coordinates")
            x, y = fu.coordinates
            if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
                raise ValueError(f"FU node {fu.id} has invalid coordinates {fu.coordinates}")

        # Check that edges connect valid nodes
        for edge in self.get_edges():
            if not self.has_node(edge.source.id) or not self.has_node(edge.destination.id):
                raise ValueError(f"Edge {edge.id} connects non-existent nodes")

        return True

    def get_array_dimensions(self) -> Tuple[int, int]:
        """Get the dimensions of the CGRA array."""
        return (self.rows, self.cols)

    def _parse_coordinates(self, attrs: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """
        Parse coordinates from node attributes.

        Args:
            attrs: Dictionary of node attributes

        Returns:
            (x, y) tuple if coordinates found, None otherwise

        Notes:
            - Tries 'x'/'y' attributes first
            - Falls back to 'col'/'row' attributes
            - Returns None if coordinates cannot be parsed
        """
        x_str = attrs.get('x', attrs.get('col', ''))
        y_str = attrs.get('y', attrs.get('row', ''))

        if x_str and y_str:
            try:
                return (int(x_str), int(y_str))
            except ValueError:
                pass

        return None

    def _parse_coordinates_from_name(self, node_name: str, add_edge_offset: bool = True) -> Optional[Tuple[int, int]]:
        """
        Extract coordinates from node name patterns.

        Args:
            node_name: Node name or base name (without cycle prefix)
            add_edge_offset: If True, add 1 to coordinates to account for edge PEs

        Returns:
            (x, y) tuple if pattern matches, None otherwise

        Notes:
            Supports common naming patterns:
            - pe_cX_rY (e.g., pe_c2_r1 -> (3, 2) with edge offset)
            - pe_X_Y (e.g., pe_2_1 -> (3, 2) with edge offset)
            - cXrY (e.g., c2r1 -> (3, 2) with edge offset)

            The add_edge_offset parameter accounts for architectures where
            edge PEs are numbered separately (e.g., CGRA-ME convention where
            pe_c0_r0 is at position (1,1) in the actual array).
        """
        import re

        # Remove cycle prefix if present
        if ':' in node_name:
            node_name = node_name.split(':', 1)[1]

        # Remove port suffixes (.in, .out, .in_a, etc.)
        base = node_name.split('.')[0]

        col, row = None, None

        # Pattern 1: pe_cX_rY or fu_cX_rY or reg_cX_rY
        match = re.search(r'[_]c(\d+)_r(\d+)', base)
        if match:
            col = int(match.group(1))
            row = int(match.group(2))

        # Pattern 2: pe_X_Y or fu_X_Y (generic underscore-separated)
        if col is None:
            match = re.search(r'[_](\d+)_(\d+)', base)
            if match:
                col = int(match.group(1))
                row = int(match.group(2))

        # Pattern 3: cXrY
        if col is None:
            match = re.search(r'c(\d+)r(\d+)', base)
            if match:
                col = int(match.group(1))
                row = int(match.group(2))

        # Apply edge offset if requested and coordinates found
        if col is not None and row is not None:
            if add_edge_offset:
                return (col + 1, row + 1)
            return (col, row)

        return None

    def create_time_expanded_fu(
        self,
        base_name: str,
        coordinates: Tuple[int, int],
        supported_operations: Set[OperationType],
        fu_latency: int = 1,
        bitwidth: int = 32,
        fu_ii: int = 1
    ) -> None:
        """
        Create time-expanded FU nodes following CGRA-ME formulation.

        Creates input operand pins, FU node, and output node for each time slot,
        with latency edges connecting FU to output at (cycle + latency) % II.

        Args:
            base_name: Base name for the FU (e.g., "fu_0_0")
            coordinates: (x, y) position in CGRA
            supported_operations: Set of operations this FU can execute
            fu_latency: Execution latency
            bitwidth: Data width in bits
            fu_ii: FU initiation interval (for pipelined FUs)
        """
        # Create nodes at each time slot (stepping by FU II)
        for cycle in range(0, self.II, fu_ii):
            # Input operand pins
            in_a_id = f"{cycle}:{base_name}.in_a"
            in_a = MRRGNode(
                in_a_id,
                NodeType.ROUTING,  # Input pins are routing nodes
                cycle,
                coordinates,
                hw_entity_type=HWEntityType.HW_WIRE,
                latency=0,
                bitwidth=bitwidth,
                supported_operand_tags={OperandTag.BINARY_LHS, OperandTag.BINARY_ANY}
            )

            in_b_id = f"{cycle}:{base_name}.in_b"
            in_b = MRRGNode(
                in_b_id,
                NodeType.ROUTING,
                cycle,
                coordinates,
                hw_entity_type=HWEntityType.HW_WIRE,
                latency=0,
                bitwidth=bitwidth,
                supported_operand_tags={OperandTag.BINARY_RHS, OperandTag.BINARY_ANY}
            )

            # Function node
            fu_id = f"{cycle}:{base_name}"
            fu = MRRGNode(
                fu_id,
                NodeType.FUNCTION,
                cycle,
                coordinates,
                hw_entity_type=HWEntityType.HW_COMB,
                latency=fu_latency,
                bitwidth=bitwidth,
                supported_operations=supported_operations
            )

            # Output node (will connect to future cycle)
            out_id = f"{cycle}:{base_name}.out"
            out = MRRGNode(
                out_id,
                NodeType.ROUTING,
                cycle,
                coordinates,
                hw_entity_type=HWEntityType.HW_WIRE,
                latency=0,
                bitwidth=bitwidth,
                routing_type="wire"
            )

            # Add nodes
            self.add_node(in_a)
            self.add_node(in_b)
            self.add_node(fu)
            self.add_node(out)

            # Connect inputs to FU (within same cycle)
            self.add_edge(MRRGEdge(f"{in_a_id}_to_{fu_id}", in_a, fu, latency=0))
            self.add_edge(MRRGEdge(f"{in_b_id}_to_{fu_id}", in_b, fu, latency=0))

        # Connect FU to output across cycles (latency edges)
        for cycle in range(0, self.II, fu_ii):
            fu_id = f"{cycle}:{base_name}"
            fu = self.get_node(fu_id)

            out_cycle = (cycle + fu_latency) % self.II  # Modulo wrap
            out_id = f"{out_cycle}:{base_name}.out"
            out = self.get_node(out_id)

            if fu and out:
                self.add_edge(MRRGEdge(f"{fu_id}_to_{out_id}", fu, out, latency=fu_latency))

    def create_time_expanded_register(
        self,
        base_name: str,
        coordinates: Tuple[int, int],
        bank_id: Optional[int] = None,
        reg_latency: int = 1,
        bitwidth: int = 32
    ) -> None:
        """
        Create time-expanded register nodes following CGRA-ME formulation.

        Creates register feedback loop across II boundary for pipelined operation.
        Structure per cycle:
            in -> m_in -> reg -[next_cycle]-> m_out -> out -> m_enable -|
                                 ^                                      |
                                 |--------------------------------------|

        Args:
            base_name: Base name for the register (e.g., "reg_0_0")
            coordinates: (x, y) position in CGRA
            bank_id: Register bank identifier
            reg_latency: Register latency (typically 1)
            bitwidth: Data width in bits
        """
        # Create nodes for all cycles
        for cycle in range(self.II):
            # Input
            in_id = f"{cycle}:{base_name}.in"
            in_node = MRRGNode(in_id, NodeType.ROUTING, cycle, coordinates,
                              HWEntityType.HW_WIRE, 0, bitwidth, routing_type="wire")

            # Multiplexer input
            m_in_id = f"{cycle}:{base_name}.m_in"
            m_in = MRRGNode(m_in_id, NodeType.ROUTING, cycle, coordinates,
                           HWEntityType.HW_MUX, 0, bitwidth, routing_type="mux")

            # Register (with latency)
            reg_id = f"{cycle}:{base_name}.reg"
            reg = MRRGNode(reg_id, NodeType.ROUTING, cycle, coordinates,
                          HWEntityType.HW_REG, reg_latency, bitwidth, bank_id=bank_id)

            # Multiplexer output
            m_out_id = f"{cycle}:{base_name}.m_out"
            m_out = MRRGNode(m_out_id, NodeType.ROUTING, cycle, coordinates,
                            HWEntityType.HW_WIRE, 0, bitwidth, routing_type="wire")

            # Enable multiplexer
            m_enable_id = f"{cycle}:{base_name}.m_enable"
            m_enable = MRRGNode(m_enable_id, NodeType.ROUTING, cycle, coordinates,
                               HWEntityType.HW_MUX, 0, bitwidth, routing_type="mux")

            # Output
            out_id = f"{cycle}:{base_name}.out"
            out_node = MRRGNode(out_id, NodeType.ROUTING, cycle, coordinates,
                               HWEntityType.HW_WIRE, 0, bitwidth, routing_type="wire")

            # Add nodes
            self.add_node(in_node)
            self.add_node(m_in)
            self.add_node(reg)
            self.add_node(m_out)
            self.add_node(m_enable)
            self.add_node(out_node)

        # Connect across cycles
        for cycle in range(self.II):
            next_cycle = (cycle + reg_latency) % self.II  # Modulo wrapping

            # Get nodes for this cycle
            in_node = self.get_node(f"{cycle}:{base_name}.in")
            m_in = self.get_node(f"{cycle}:{base_name}.m_in")
            reg = self.get_node(f"{cycle}:{base_name}.reg")
            m_enable = self.get_node(f"{cycle}:{base_name}.m_enable")
            out_node = self.get_node(f"{cycle}:{base_name}.out")

            # Get nodes for next cycle
            m_out_next = self.get_node(f"{next_cycle}:{base_name}.m_out")

            # Within same cycle
            if in_node and m_in:
                self.add_edge(MRRGEdge(f"{in_node.id}_to_{m_in.id}", in_node, m_in, 0))
            if m_in and reg:
                self.add_edge(MRRGEdge(f"{m_in.id}_to_{reg.id}", m_in, reg, 0))
            if m_enable and reg:
                self.add_edge(MRRGEdge(f"{m_enable.id}_to_{reg.id}", m_enable, reg, 0))
            if out_node and m_enable:
                self.add_edge(MRRGEdge(f"{out_node.id}_to_{m_enable.id}", out_node, m_enable, 0))

            # Cross-cycle for latency (register to next cycle's output mux)
            if reg and m_out_next:
                self.add_edge(MRRGEdge(f"{reg.id}_to_{m_out_next.id}", reg, m_out_next, reg_latency))

            # Connect m_out to out within next cycle
            if m_out_next:
                out_next = self.get_node(f"{next_cycle}:{base_name}.out")
                if out_next:
                    self.add_edge(MRRGEdge(f"{m_out_next.id}_to_{out_next.id}", m_out_next, out_next, 0))

    def create_time_expanded_mux(
        self,
        base_name: str,
        coordinates: Tuple[int, int],
        num_inputs: int,
        bitwidth: int = 32
    ) -> None:
        """
        Create time-expanded multiplexer nodes following CGRA-ME formulation.

        Args:
            base_name: Base name for the mux (e.g., "mux_0_0")
            coordinates: (x, y) position in CGRA
            num_inputs: Number of input ports
            bitwidth: Data width in bits
        """
        for cycle in range(self.II):
            # Output
            out_id = f"{cycle}:{base_name}.out"
            out = MRRGNode(out_id, NodeType.ROUTING, cycle, coordinates,
                          HWEntityType.HW_WIRE, 0, bitwidth, routing_type="wire")
            self.add_node(out)

            # Mux node
            mux_id = f"{cycle}:{base_name}.mux"
            mux = MRRGNode(mux_id, NodeType.ROUTING, cycle, coordinates,
                          HWEntityType.HW_MUX, 0, bitwidth, routing_type="mux")
            self.add_node(mux)

            # Connect mux to output
            self.add_edge(MRRGEdge(f"{mux_id}_to_{out_id}", mux, out, 0))

            # Input ports
            for i in range(num_inputs):
                in_id = f"{cycle}:{base_name}.in{i}"
                in_node = MRRGNode(in_id, NodeType.ROUTING, cycle, coordinates,
                                  HWEntityType.HW_WIRE, 0, bitwidth, routing_type="wire")
                self.add_node(in_node)

                # Connect input to mux
                self.add_edge(MRRGEdge(f"{in_id}_to_{mux_id}", in_node, mux, 0))

    def create_time_expanded_routing_wire(
        self,
        base_name: str,
        coordinates: Tuple[int, int],
        wire_latency: int = 0,
        bitwidth: int = 32
    ) -> None:
        """
        Create time-expanded routing wire nodes.

        Args:
            base_name: Base name for the wire (e.g., "wire_n_0_0")
            coordinates: (x, y) position in CGRA
            wire_latency: Wire latency (0 for combinational)
            bitwidth: Data width in bits
        """
        for cycle in range(self.II):
            wire_id = f"{cycle}:{base_name}"
            wire = MRRGNode(wire_id, NodeType.ROUTING, cycle, coordinates,
                           HWEntityType.HW_WIRE, wire_latency, bitwidth, routing_type="wire")
            self.add_node(wire)

    def to_string(self) -> str:
        """
        Return detailed string representation of the MRRG.

        Returns:
            Multi-line string with MRRG statistics and breakdown
        """
        # Count nodes by type
        fu_nodes = self.get_fu_nodes()
        routing_nodes = self.get_routing_nodes()
        register_nodes = self.get_register_nodes()

        # Count by hardware entity type
        hw_types = {}
        for node in self.get_nodes():
            hw_type = node.hw_entity_type.value
            hw_types[hw_type] = hw_types.get(hw_type, 0) + 1

        # Build detailed string
        lines = [
            f"MRRG: {self.name}",
            f"  Initiation Interval (II): {self.II}",
            f"  Array Dimensions: {self.rows} x {self.cols}",
            f"  Total Nodes: {self.num_nodes()}",
            f"  Total Edges: {self.num_edges()}",
            "",
            "  Node Breakdown by NodeType:",
            f"    FUNCTION nodes:  {len(fu_nodes):5d}",
            f"    ROUTING nodes:   {len(routing_nodes):5d}",
            f"    REGISTER nodes:  {len(register_nodes):5d}",
            "",
            "  Node Breakdown by HWEntityType:",
        ]

        for hw_type, count in sorted(hw_types.items()):
            lines.append(f"    {hw_type.upper():15s}: {count:5d}")

        # Add cycle information
        if self.II > 0:
            lines.append("")
            lines.append("  Nodes per Cycle:")
            for cycle in range(self.II):
                nodes_at_cycle = self.get_nodes_at_cycle(cycle)
                fu_at_cycle = [n for n in nodes_at_cycle if n.node_type == NodeType.FUNCTION]
                reg_at_cycle = [n for n in nodes_at_cycle if n.hw_entity_type == HWEntityType.HW_REG]
                lines.append(f"    Cycle {cycle}: {len(nodes_at_cycle):4d} nodes "
                           f"({len(fu_at_cycle)} FUs, {len(reg_at_cycle)} regs)")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        return (f"MRRG(name={self.name}, II={self.II}, dim={self.rows}x{self.cols}, "
                f"nodes={self.num_nodes()}, edges={self.num_edges()})")

    # -- Helper methods for Version 1 ---------------------------------------------
    def get_fu_nodes(self) -> List[MRRGNode]:
        """Get all FU nodes."""
        return [n for n in self.get_nodes() if n.node_type == NodeType.FUNCTION]

    def get_routing_nodes(self) -> List[MRRGNode]:
        """Get all routing nodes."""
        return [n for n in self.get_nodes() if n.node_type == NodeType.ROUTING]

    def create_routing_fanout_map(self) -> Dict[int, List[int]]:
        """Create a mapping from routing node index to list of destination node indices."""
        routing_nodes = self.get_routing_nodes()
        fanout_map = {i: [] for i in range(len(routing_nodes))}
        routing_node_id_reverse_map = {node.id: i for i, node in enumerate(routing_nodes)}

        for node in routing_nodes:
            # get_outgoing_edges returns EDGE OBJECTS, not IDs!
            for edge_obj in self.get_outgoing_edges(node.id):
                if edge_obj:  # Changed from get_edge(edge_id)
                    dest_id = edge_obj.destination.id
                    if dest_id in routing_node_id_reverse_map:
                        src_idx = routing_node_id_reverse_map[node.id]
                        dst_idx = routing_node_id_reverse_map[dest_id]
                        fanout_map[src_idx].append(dst_idx)
        
        return fanout_map

    def create_routing_fanin_map(self) -> Dict[int, List[int]]:
        """Create a mapping from routing node index to list of source (predecessor) node indices.

        This is the inverse of create_routing_fanout_map - for each routing node,
        it returns the list of routing nodes that feed into it.
        """
        routing_nodes = self.get_routing_nodes()
        fanin_map = {i: [] for i in range(len(routing_nodes))}
        routing_node_id_reverse_map = {node.id: i for i, node in enumerate(routing_nodes)}

        for node in routing_nodes:
            # Get incoming edges to this node
            for edge_obj in self.get_incoming_edges(node.id):
                if edge_obj:
                    src_id = edge_obj.source.id
                    if src_id in routing_node_id_reverse_map:
                        dst_idx = routing_node_id_reverse_map[node.id]
                        src_idx = routing_node_id_reverse_map[src_id]
                        fanin_map[dst_idx].append(src_idx)

        return fanin_map

    def get_sink_fus_connected_to_src_routing_node(self, routing_node_id: str) -> List[MRRGNode]:
        """Get all FUs connected to a routing node."""

        # Make sure the entered routing node id is actually a routing node.
        if self.get_node(routing_node_id).node_type != NodeType.ROUTING:
            raise ValueError(f"The entered routing node id {routing_node_id} is not a routing node.")

        return [edge.destination for edge in self.get_edges() if edge.destination.node_type == NodeType.FUNCTION and edge.source.id == routing_node_id]

    def get_sink_routing_nodes_connected_to_src_fu(self, fu_node_id: str) -> List[MRRGNode]:
        """Get all routing nodes (sinks) connected to a FU node."""

        # Make sure the entered FU node id is actually a FU node.
        if self.get_node(fu_node_id).node_type != NodeType.FUNCTION:
            raise ValueError(f"The entered FU node id {fu_node_id} is not a FU node.")

        return [edge.destination for edge in self.get_edges() if edge.destination.node_type == NodeType.ROUTING and edge.source.id == fu_node_id]

    def get_fan_in_nodes(self, node_id: str) -> List[MRRGNode]:
        """Get all nodes that are fan-in to this node."""

        # Make sure the entered node id is actually a node.
        if self.get_node(node_id) is None:
            raise ValueError(f"The entered node id {node_id} is not a node.")

        return [edge.source for edge in self.get_edges() if edge.destination.id == node_id]

    # -- Helper methods for Version 2 ---------------------------------------------
    def get_sink_fu_neighbors(self, fu_node_id: str, neighbor_count: int = 20) -> List[MRRGNode]:
        """Get the specified number of FU nodes that are neighbors to the given FU node."""

        # Make sure the entered FU node id is actually a FU node.
        if self.get_node(fu_node_id).node_type != NodeType.FUNCTION:
            raise ValueError(f"The entered FU node id {fu_node_id} is not a FU node.")

        # Perform a breadth-first search to find the specified number of FU nodes that are neighbors to the given FU node.
        all_neighbors = traversal.bfs(graph=self, start_node_id=fu_node_id)

        # Prune out the nodes that are not FU nodes.
        fu_neighbors = [n for n in all_neighbors if n.node_type == NodeType.FUNCTION]

        # Return the specified number of FU nodes.
        return fu_neighbors[:neighbor_count]

    def get_all_paths_between_fu_nodes(self, start_node_id: str, end_node_id: str, max_paths: int = 20) -> List[List[MRRGNode]]:
        """Get the shortest path between two nodes."""

        # Make sure the entered start node id is actually a node.
        if self.get_node(start_node_id) is None or self.get_node(start_node_id).node_type != NodeType.FUNCTION:
            raise ValueError(f"The entered start node id {start_node_id} is not a node.")

        # Make sure the entered end node id is actually a node.
        if self.get_node(end_node_id) is None or self.get_node(end_node_id).node_type != NodeType.FUNCTION:
            raise ValueError(f"The entered end node id {end_node_id} is not a node.")

        return traversal.get_all_paths(graph=self, start_node_id=start_node_id, end_node_id=end_node_id, max_paths=20)

    def get_all_paths_in_mrrg_fu_space(self) -> Dict[Tuple[int, int], List[List[MRRGNode]]]:
        """Get a map of all paths in the MRRG FU space."""

        # Get all FU nodes.
        fu_nodes = self.get_fu_nodes()

        # Create a map of all paths between all FU nodes both ways.
        path_map = {}
        for i in range(len(fu_nodes)):
            for j in range(len(fu_nodes)):
                path_map[(i, j)] = self.get_all_paths_between_fu_nodes(fu_nodes[i].id, fu_nodes[j].id)
        return path_map

    def time_expand(self, new_ii: int) -> 'MRRG':
        """Time-expand the graph to a new Initiation Interval."""
        # Create a new MRRG with updated II
        expanded = MRRG(name=f"{self.name}_expanded", II=new_ii, rows=self.rows, cols=self.cols)
        expanded.is_from_json = self.is_from_json

        # If loaded from JSON (Dora), use generic duplication instead of smart renaming
        if self.is_from_json:
            import copy
            # 1. Duplicate all nodes across cycles
            processed_bases = set()
            for node in self.get_nodes():
                # Extract base ID (remove cycle prefix if it exists)
                base_id = node.id.split(':', 1)[1] if ':' in node.id else node.id
                if base_id in processed_bases:
                    continue
                processed_bases.add(base_id)
                
                for cycle in range(new_ii):
                    new_id = f"{cycle}:{base_id}"
                    new_node = MRRGNode(
                        node_id=new_id,
                        node_type=node.node_type,
                        cycle=cycle,
                        coordinates=node.coordinates,
                        hw_entity_type=node.hw_entity_type,
                        latency=node.latency,
                        bitwidth=node.bitwidth,
                        supported_operations=node.supported_operations.copy()
                    )
                    # Copy extra attributes if they exist
                    for attr in ['routing_type', 'bank_id', 'read_ports', 'write_ports']:
                        if hasattr(node, attr):
                            setattr(new_node, attr, getattr(node, attr))
                    expanded.add_node(new_node)
            
            # 2. Duplicate all edges across cycles
            processed_edges = set()
            for edge in self.get_edges():
                src_base = edge.source.id.split(':', 1)[1] if ':' in edge.source.id else edge.source.id
                dst_base = edge.destination.id.split(':', 1)[1] if ':' in edge.destination.id else edge.destination.id
                
                # Dora MRRGs often encode pipeline delay on nodes (e.g., registers/LSUs)
                # while edge time deltas are all 0 in II=1.
                # Preserve cycle reachability by advancing destination cycle by node latency
                # when edge latency is not explicitly positive.
                effective_latency = edge.latency if edge.latency > 0 else max(0, edge.source.latency)

                edge_key = (src_base, dst_base, effective_latency)
                if edge_key in processed_edges:
                    continue
                processed_edges.add(edge_key)
                
                for cycle in range(new_ii):
                    # Edge latency determines the cycle jump (modulo scheduled)
                    dst_cycle = (cycle + effective_latency) % new_ii
                    
                    src_new = f"{cycle}:{src_base}"
                    dst_new = f"{dst_cycle}:{dst_base}"
                    
                    if expanded.has_node(src_new) and expanded.has_node(dst_new):
                        new_edge = MRRGEdge(
                            edge_id=f"{src_new}_to_{dst_new}",
                            source=expanded.get_node(src_new),
                            destination=expanded.get_node(dst_new),
                            latency=effective_latency
                        )
                        new_edge.capacity = edge.capacity
                        expanded.add_edge(new_edge)
                        
            return expanded

        # --- Legacy 'Smart' Expansion ---
        if new_ii <= 0:
            raise ValueError(f"Invalid II: {new_ii}")

        # Create new empty MRRG with the same basic parameters
        expanded = MRRG(f"{self.name}_II{new_ii}", new_ii, self.rows, self.cols)
        debug = False
        if debug:
            print(f"Time-expanding MRRG from II={self.II} to II={new_ii}...")

        # 1. Expand Functional Units
        for fu in self.get_fu_nodes():
            base_name = fu.id
            if ':' in base_name: # Handle case where base MRRG already has cycle prefixes
                base_name = base_name.split(':', 1)[1]
                
            expanded.create_time_expanded_fu(
                base_name=base_name,
                coordinates=fu.coordinates,
                supported_operations=fu.supported_operations.copy(),
                fu_latency=fu.latency,
                bitwidth=fu.bitwidth,
                fu_ii=1 # Default assumption for Dora
            )

        # 2. Expand Registers
        for reg in self.get_register_nodes():
            base_name = reg.id
            if ':' in base_name:
                base_name = base_name.split(':', 1)[1]
            
            expanded.create_time_expanded_register(
                base_name=base_name,
                coordinates=reg.coordinates,
                bank_id=reg.bank_id,
                reg_latency=reg.latency, # Typically 1
                bitwidth=reg.bitwidth
            )

        # 3. Expand Pure Routing/Wire/Mux Nodes
        for route_node in self.get_routing_nodes():
            base_name = route_node.id
            if ':' in base_name:
                base_name = base_name.split(':', 1)[1]
                
            # specifically handle pure routing nodes (switches, network wires)
            # Skip CGRA-ME generated internal pins to prevent duplicated re-expansion
            if any(base_name.endswith(ext) for ext in ['.in', '.out', '.in_a', '.in_b', '.m_in', '.m_out', '.m_enable', '.reg']):
                continue
                
            for cycle in range(new_ii):
                new_id = f"{cycle}:{base_name}"
                new_node = MRRGNode(
                    node_id=new_id,
                    node_type=route_node.node_type,
                    cycle=cycle,
                    coordinates=route_node.coordinates,
                    hw_entity_type=route_node.hw_entity_type,
                    latency=route_node.latency,
                    bitwidth=route_node.bitwidth,
                    bank_id=route_node.bank_id,
                    supported_operations=route_node.supported_operations.copy() if route_node.supported_operations else None,
                    supported_operand_tags=route_node.supported_operand_tags.copy() if route_node.supported_operand_tags else None,
                    routing_type=route_node.routing_type
                )
                if not expanded.has_node(new_id):
                    expanded.add_node(new_node)

        # 4. Connect original topology across cycles
        edge_count = 0
        for edge in self.get_edges():
            src_base = edge.source.id
            dst_base = edge.destination.id
            
            # Remove any prefix from base graph
            if ':' in src_base: src_base = src_base.split(':', 1)[1]
            if ':' in dst_base: dst_base = dst_base.split(':', 1)[1]
                
            for cycle in range(new_ii):
                # If source is an FU or Register, its outputs are exposed on its .out pin
                src_probe = f"{cycle}:{src_base}.out"
                if not expanded.has_node(src_probe):
                    src_probe = f"{cycle}:{src_base}"
                    
                actual_src = expanded.get_node(src_probe)
                if not actual_src:
                    continue

                # Destination cycle is source cycle + edge latency
                dst_cycle = (cycle + edge.latency) % new_ii
                
                # If destination is an FU, its explicit inputs are .in_a and .in_b. 
                # If it is a Register, its input is .in.
                dst_probes = []
                if expanded.has_node(f"{dst_cycle}:{dst_base}.in_a") and expanded.has_node(f"{dst_cycle}:{dst_base}.in_b"):
                    dst_probes = [f"{dst_cycle}:{dst_base}.in_a", f"{dst_cycle}:{dst_base}.in_b"]
                elif expanded.has_node(f"{dst_cycle}:{dst_base}.in"):
                    dst_probes = [f"{dst_cycle}:{dst_base}.in"]
                else:
                    dst_probes = [f"{dst_cycle}:{dst_base}"]

                for dst_probe in dst_probes:
                    actual_dst = expanded.get_node(dst_probe)
                    if not actual_dst:
                        continue
                        
                    edge_id = f"{actual_src.id}_to_{actual_dst.id}"
                    if not expanded.has_edge(edge_id):
                        new_edge = MRRGEdge(
                            edge_id=edge_id,
                            source=actual_src,
                            destination=actual_dst,
                            latency=edge.latency
                        )
                        new_edge.capacity = edge.capacity
                        new_edge.conflict_free = edge.conflict_free
                        expanded.add_edge(new_edge)
                        edge_count += 1
                            
        if debug:
            print(f"Time expansion complete. New graph has {len(expanded.get_nodes())} nodes, {len(expanded.get_edges())} edges.")
            
        return expanded


    def get_k_shortest_paths_between_fu_nodes(self, start_node_id: str, end_node_id: str, k: int) -> List[List[str]]:
        """Get the k shortest paths between two FU nodes.

        Only returns paths where all intermediate nodes are routing nodes.
        The start and end nodes are FU nodes, but internal nodes must be routing nodes.

        Returns:
            List of paths, where each path is a list of node IDs (strings), not MRRGNode objects.
        """

        # Make sure the entered start node id is actually a node.
        if self.get_node(start_node_id) is None or self.get_node(start_node_id).node_type != NodeType.FUNCTION:
            raise ValueError(f"The entered start node id {start_node_id} is not a node.")

        # Make sure the entered end node id is actually a node.
        if self.get_node(end_node_id) is None or self.get_node(end_node_id).node_type != NodeType.FUNCTION:
            raise ValueError(f"The entered end node id {end_node_id} is not a node.")

        # Get more paths than requested since we'll filter some out
        # Use a multiplier to account for paths that go through intermediate FUs
        candidate_paths = traversal.get_k_shortest_paths(
            graph=self, start_node_id=start_node_id, end_node_id=end_node_id, k=k * 5
        )

        # Filter to only keep paths where intermediate nodes are routing nodes
        valid_paths = []
        for path in candidate_paths:
            # Check intermediate nodes (exclude first and last which are the FU endpoints)
            is_valid = True
            for node_id in path[1:-1]:
                node = self.get_node(node_id)
                if node.node_type == NodeType.FUNCTION:
                    is_valid = False
                    break
            if is_valid:
                valid_paths.append(path)
                if len(valid_paths) >= k:
                    break

        return valid_paths

    def get_k_shortest_paths_between_fu_nodes_optimized(
        self, start_node_id: str, end_node_id: str, k: int
    ) -> List[List[str]]:
        """Optimized version using subgraph filtering before path enumeration.

        Converts the MRRG to a NetworkX DiGraph, excludes intermediate FU nodes,
        then runs Yen's algorithm. This eliminates the need for the k*5 multiplier
        and post-hoc filtering.

        Args:
            start_node_id: ID of the source FU node
            end_node_id: ID of the destination FU node
            k: Number of shortest paths to find

        Returns:
            List of paths, where each path is a list of node IDs (strings).
        """
        import itertools
        import networkx as nx

        # Validate inputs
        start_node = self.get_node(start_node_id)
        end_node = self.get_node(end_node_id)

        if start_node is None or start_node.node_type != NodeType.FUNCTION:
            raise ValueError(f"The entered start node id {start_node_id} is not a valid FU node.")
        if end_node is None or end_node.node_type != NodeType.FUNCTION:
            raise ValueError(f"The entered end node id {end_node_id} is not a valid FU node.")

        # Get all FU nodes and build exclusion set
        fu_node_ids = {n.id for n in self.get_fu_nodes()}
        nodes_to_exclude = fu_node_ids - {start_node_id, end_node_id}

        # Build a NetworkX DiGraph with only non-excluded nodes
        # This is more compatible than using subgraph_view on our custom graph
        nx_graph = nx.DiGraph()

        # Add nodes (excluding intermediate FUs)
        for node_id in self:
            if node_id not in nodes_to_exclude:
                nx_graph.add_node(node_id)

        # Add edges (only between included nodes)
        for src_id in self:
            if src_id in nodes_to_exclude:
                continue
            for dst_id, edge_data in self[src_id].items():
                if dst_id not in nodes_to_exclude:
                    nx_graph.add_edge(src_id, dst_id, weight=edge_data.get('weight', 1))

        # Request exactly k paths - no multiplier needed since we pre-filtered
        try:
            gen = nx.shortest_simple_paths(nx_graph, start_node_id, end_node_id, weight='weight')
            return list(itertools.islice(gen, k))
        except nx.NetworkXNoPath:
            return []

    def get_k_shortest_paths_in_mrrg_fu_space(
        self, k: int, max_manhattan_dist: Optional[int] = None
    ) -> Dict[Tuple[int, int], List[List[str]]]:
        """Get a map of the k shortest paths in the MRRG FU space.

        Uses spatial pruning to skip FU pairs that are too far apart,
        and uses the optimized path computation that pre-filters intermediate FUs.

        Args:
            k: Number of shortest paths to find per FU pair
            max_manhattan_dist: Maximum Manhattan distance between FU pairs.
                               If None, defaults to max(rows, cols).
                               Pairs beyond this distance return empty paths.

        Returns:
            Dictionary mapping (FU_index_i, FU_index_j) to list of paths,
            where each path is a list of node IDs (strings), not MRRGNode objects.
        """
        fu_nodes = self.get_fu_nodes()

        # Use max(rows, cols) as default - safer than sum (avoids over-pruning)
        if max_manhattan_dist is None:
            max_manhattan_dist = max(self.rows, self.cols)

        path_map = {}
        for i, fu_i in enumerate(fu_nodes):
            for j, fu_j in enumerate(fu_nodes):
                # Spatial pruning with safe fallback for missing coordinates
                if fu_i.coordinates is not None and fu_j.coordinates is not None:
                    x1, y1 = fu_i.coordinates
                    x2, y2 = fu_j.coordinates
                    if abs(x1 - x2) + abs(y1 - y2) > max_manhattan_dist:
                        path_map[(i, j)] = []
                        continue

                # Use optimized version with subgraph filtering
                path_map[(i, j)] = self.get_k_shortest_paths_between_fu_nodes_optimized(
                    fu_i.id, fu_j.id, k
                )

        return path_map

    @classmethod
    def from_json(
        cls, 
        json_file_path: str, 
        name: str = "MRRG",
        compiler_arch_path: Optional[str] = None
    ) -> "MRRG":
        import json
        import re

        # -- Op Code Mapping between Dora Architecture and Mapper IR -----------------
        # Maps Dora architecture "optype" to the mapper's internal OperationType
        DORA_OPTYPE_MAP = {
            "ADD": OperationType.ADD,
            "SUB": OperationType.SUB,
            "MUL": OperationType.MUL,
            "DIV": OperationType.DIV,
            "AND": OperationType.AND,
            "OR": OperationType.OR,
            "XOR": OperationType.XOR,
            "LSL": OperationType.SHL,
            "LSR": OperationType.LSHR,
            "ASR": OperationType.ASHR,
            "LD": OperationType.LOAD,
            "ST": OperationType.STORE,
            "ICMP": OperationType.ICMP,
            "SELECT": OperationType.SELECT,
            "PHI": OperationType.PHI,
            "INPUT": OperationType.INPUT,
            "OUTPUT": OperationType.OUTPUT,
            "CONST": OperationType.CONST,
            "NOP": OperationType.NOP,
            # Handle mixed case if needed
            "add": OperationType.ADD,
            "sub": OperationType.SUB,
            "mul": OperationType.MUL,
            "div": OperationType.DIV,
            "and": OperationType.AND,
            "or": OperationType.OR,
            "xor": OperationType.XOR,
            "lsl": OperationType.SHL,
            "lsr": OperationType.LSHR,
            "asr": OperationType.ASHR,
            "ld": OperationType.LOAD,
            "st": OperationType.STORE,
            "icmp": OperationType.ICMP,
            "select": OperationType.SELECT,
            "phi": OperationType.PHI,
            "input": OperationType.INPUT,
            "output": OperationType.OUTPUT,
            "const": OperationType.CONST,
            "nop": OperationType.NOP,
        }

        DORA_OP_ALIASES = {
            "LOAD": "LD",
            "STORE": "ST",
            "SHL": "LSL",
            "SHR": "LSR",
            "SAR": "ASR",
            "CMP": "ICMP",
        }

        def map_dora_operation(optype: str, operation_id: str = "") -> Optional[OperationType]:
            """Map Dora operation strings to mapper OperationType with aliases/fallbacks."""
            if not optype:
                optype = ""

            # Fast path: existing direct map (supports mixed case keys already).
            if optype in DORA_OPTYPE_MAP:
                return DORA_OPTYPE_MAP[optype]

            norm = optype.strip().upper()
            if norm in DORA_OP_ALIASES:
                norm = DORA_OP_ALIASES[norm]

            if norm in DORA_OPTYPE_MAP:
                return DORA_OPTYPE_MAP[norm]

            # Enum-name fallback
            try:
                return OperationType[norm]
            except (KeyError, ValueError):
                pass

            # Enum-value fallback
            for op_enum in OperationType:
                if op_enum.value.lower() == norm.lower():
                    return op_enum

            # Last-resort fallback from operation_id tokens (e.g., hycube.mem.load.i32)
            if operation_id:
                for token in operation_id.replace('-', '.').split('.'):
                    token_norm = token.strip().upper()
                    if not token_norm:
                        continue
                    if token_norm in DORA_OP_ALIASES:
                        token_norm = DORA_OP_ALIASES[token_norm]
                    if token_norm in DORA_OPTYPE_MAP:
                        return DORA_OPTYPE_MAP[token_norm]
                    try:
                        return OperationType[token_norm]
                    except (KeyError, ValueError):
                        continue

            return None

        # Load JSON file
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        # Load compiler_arch.json if provided
        # module_name -> Set[OperationType]
        module_operations_map: Dict[str, Set[OperationType]] = {}
        
        if compiler_arch_path:
            with open(compiler_arch_path, 'r') as f:
                compiler_arch = json.load(f)
            
            # Parse module_operation_capabilities (Dora Version)
            # The new structure has module_name and a list of bindings, 
            # where each binding has an "optype" string.
            capabilities = compiler_arch.get("module_operation_capabilities", [])
            for cap in capabilities:
                module_name = cap.get("module_name")
                bindings = cap.get("bindings", [])
                
                if module_name and bindings:
                    op_set = set()
                    for binding in bindings:
                        op_str = binding.get("optype", "")
                        operation_id = binding.get("operation_id", "")
                        mapped_op = map_dora_operation(op_str, operation_id)
                        if mapped_op is not None:
                            op_set.add(mapped_op)
                    
                    if op_set:
                        module_operations_map[module_name] = op_set

        # Extract nodes and edges
        nodes_data = data.get("nodes", [])
        edges_data = data.get("edges", [])

        # Determine II from the maximum time value
        max_time = max([n.get("time", 0) for n in nodes_data], default=0)
        II = max_time + 1

        # Create MRRG instance
        mrrg = cls(name=name, II=II)
        mrrg.is_from_json = True

        # Helper to classify nodes from Dora properties
        def classify_node(node_data: Dict[str, Any]) -> Tuple[NodeType, HWEntityType, Optional[Tuple[int, int]]]:
            """Determine NodeType and HWEntityType from Dora JSON fields."""
            kind = node_data.get("kind", "net")
            node_class = node_data.get("node_class", "routing")
            model = node_data.get("model", "")
            
            # Determine mapping position (global_coordinates)
            # Dora uses [x, y] list or null
            coords = node_data.get("global_coordinates")
            coordinates = None
            if isinstance(coords, list) and len(coords) >= 2:
                coordinates = (coords[0], coords[1])
            elif not coords and "node_id" in node_data:
                # Fallback to name parsing if coordinates are missing (unlikely in Dora, but safe)
                coordinates = mrrg._parse_coordinates_from_name(node_data["node_id"], add_edge_offset=False)

            # Assign types based on node_class and kind
            if node_class in ("functional_unit", "alu", "io"):
                return NodeType.FUNCTION, HWEntityType.HW_COMB, coordinates
            elif node_class == "register":
                return NodeType.ROUTING, HWEntityType.HW_REG, coordinates
            elif node_class == "routing" or kind == "net":
                # Check for multiplexers in models or names
                if model and ("mux" in model.lower() or "xbar" in model.lower()):
                    return NodeType.ROUTING, HWEntityType.HW_MUX, coordinates
                return NodeType.ROUTING, HWEntityType.HW_WIRE, coordinates
                
            return NodeType.ROUTING, HWEntityType.HW_UNSPECIFIED, coordinates

        # Use a map to track nodes by their raw IDs for edge connection
        node_map: Dict[str, MRRGNode] = {}
        
        # Create nodes
        for node_data in nodes_data:
            node_id = node_data["node_id"]
            time = node_data.get("time", 0)
            datatype = node_data.get("datatype")
            model = node_data.get("model")
            
            # Use explicit classification
            node_type, hw_entity_type, coordinates = classify_node(node_data)
            
            # Bitwidth detection
            bitwidth = node_data.get("bitwidth")
            if bitwidth is None:
                # Fallback to datatype parsing
                if datatype and "int" in datatype:
                    match = re.search(r'int(\d+)', datatype)
                    bitwidth = int(match.group(1)) if match else 32
                else:
                    bitwidth = 32
            
            # Check explicit "operations" field first (new Dora feature)
            json_ops = node_data.get("operations", [])

            # Latency detection:
            # 1) explicit top-level node latency
            # 2) per-operation scheduling latency in Dora API
            # 3) sane default (registers=1, others=0)
            latency = node_data.get("latency")
            if latency is None:
                latency = 0

            if latency == 0 and json_ops:
                sched_lats = []
                for op_entry in json_ops:
                    scheduling = op_entry.get("scheduling", {})
                    lat_cycles = scheduling.get("latency_cycles")
                    if isinstance(lat_cycles, int):
                        sched_lats.append(lat_cycles)
                if sched_lats:
                    # Be conservative for timing correctness.
                    latency = max(sched_lats)

            if latency == 0 and hw_entity_type == HWEntityType.HW_REG:
                latency = 1 # Default for registers if unspecified
            
            # Gather supported operations
            supported_operations = set()
            
            for op_entry in json_ops:
                op_str = op_entry.get("optype", "")
                operation_id = op_entry.get("operation_id", "")
                mapped_op = map_dora_operation(op_str, operation_id)
                if mapped_op is not None:
                    supported_operations.add(mapped_op)
            
            # If no explicit ops, fall back to module_operations_map from compiler_arch
            if not supported_operations and model:
                if model in module_operations_map:
                    supported_operations = module_operations_map[model]
                else:
                    # Partial match for module variants (e.g., hycube_alu_32b)
                    for m_name, ops in module_operations_map.items():
                        if m_name in model or model in m_name:
                            supported_operations = ops
                            break

            # Compatibility shim:
            # DFGs often retain PHI/SELECT pseudo-ops even when HyCUBE API does not
            # list them explicitly. Restrict this augmentation to ALU-like compute units
            # only, instead of the previous blanket add-to-all-function-units behavior.
            if node_type == NodeType.FUNCTION:
                model_l = (model or "").lower()
                node_class_l = str(node_data.get("node_class", "")).lower()
                is_alu_like = (
                    ("alu" in model_l)
                    or (
                        node_class_l == "functional_unit"
                        and not any(tok in model_l for tok in ("const", "lsu", "mem", "iopad", "io"))
                    )
                )
                if is_alu_like:
                    supported_operations.add(OperationType.PHI)
                    supported_operations.add(OperationType.SELECT)
            
            # Create full node ID with time prefix: "cycle:base_id"
            # This is critical for the mapper's expectation of unique time-expanded IDs
            full_node_id = f"{time}:{node_id}"

            node_attrs = node_data.get("extensions", {}).get("node_attrs", {})
            packing = node_attrs.get("sub_word_packing", {})
            fracture = node_attrs.get("fracture", {})

            allowed_lane_indices = packing.get("allowed_lane_indices")
            if isinstance(allowed_lane_indices, list):
                allowed_lane_indices = tuple(
                    lane for lane in allowed_lane_indices if isinstance(lane, int)
                )
            else:
                allowed_lane_indices = None
            
            mrrg_node = MRRGNode(
                node_id=full_node_id,
                node_type=node_type,
                cycle=time,
                coordinates=coordinates,
                hw_entity_type=hw_entity_type,
                latency=latency,
                bitwidth=bitwidth,
                supported_operations=supported_operations,
                model=model,
                datatype=datatype,
                dora_kind=node_data.get("kind"),
                dora_node_class=node_data.get("node_class"),
                lane_width=packing.get("lane_width"),
                num_lanes=packing.get("num_lanes"),
                pack_capable=bool(packing.get("pack_capable", False)),
                unpack_capable=bool(packing.get("unpack_capable", False)),
                allowed_lane_indices=allowed_lane_indices,
                pack_config=packing.get("pack_config"),
                unpack_config=packing.get("unpack_config"),
                fracture_type=fracture.get("type"),
                fracture_parent_inst=fracture.get("parent_inst"),
                fracture_chunk_width=fracture.get("chunk_width"),
                fracture_parent_port=fracture.get("parent_port"),
                fracture_chunk_index=fracture.get("chunk_index"),
                fracture_bit_offset=fracture.get("bit_offset"),
                fracture_num_chunks=fracture.get("num_chunks"),
            )
            
            mrrg.add_node(mrrg_node)
            # Map by original node_id (the one encountered in the edges section of the JSON)
            node_map[f"{time}:{node_id}"] = mrrg_node

        # Create edges
        for edge_data in edges_data:
            # Dora MRRG JSON uses relative or unrolled edge definitions
            src_id = edge_data.get("source_node")
            src_time = edge_data.get("source_time", edge_data.get("time", 0))
            dst_id = edge_data.get("target_node")
            dst_time = edge_data.get("target_time", edge_data.get("time", 0))
            
            # Build keys using the same pattern as full_node_id
            # Using modulo II ensures that cross-cycle edges in unrolled descriptions 
            # connect back to the base node version for the mapper's expansion.
            src_key = f"{src_time % II}:{src_id}"
            dst_key = f"{dst_time % II}:{dst_id}"
            
            src_node = node_map.get(src_key)
            dst_node = node_map.get(dst_key)
            
            if src_node and dst_node:
                # Calculate latency from time difference
                # For base architectures (II=1), the direct difference is the latency.
                # For modulo architectures (II>1), we use the modulo difference.
                if II > 1:
                    edge_latency = (dst_time - src_time) % II
                else:
                    edge_latency = dst_time - src_time
                
                edge_id = f"{src_node.id}_to_{dst_node.id}"
                
                mrrg_edge = MRRGEdge(
                    edge_id=edge_id,
                    source=src_node,
                    destination=dst_node,
                    latency=edge_latency
                )
                
                # Check for capacity or other attributes
                if "capacity" in edge_data:
                    mrrg_edge.capacity = edge_data["capacity"]
                
                mrrg.add_edge(mrrg_edge)

        # Finalize array dimensions from coordinates
        if mrrg.get_nodes():
            max_x = max_y = 0
            for node in mrrg.get_nodes():
                if node.coordinates:
                    x, y = node.coordinates
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
            mrrg.cols = max_x + 1
            mrrg.rows = max_y + 1

        return mrrg