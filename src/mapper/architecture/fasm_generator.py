"""FASM generator for Dora bitstream integration.

Translates mapper mapping results (placement + routing) into FASM files
that Dora can ingest for bitstream generation. Requires compiler_arch.json
exported by Dora, which provides feature definitions, routing-edge
programming tables, and operation encoding hints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from mapper.architecture.compiler_arch import load_and_validate_compiler_arch
from mapper.graph.dfg import DFG, DFGNode, OperationType
from mapper.graph.hyperdfg import HyperDFG

_OP_TO_DORA: Dict[OperationType, str] = {
    OperationType.ADD: "ADD",
    OperationType.SUB: "SUB",
    OperationType.MUL: "MUL",
    OperationType.DIV: "DIV",
    OperationType.AND: "AND",
    OperationType.OR: "OR",
    OperationType.XOR: "XOR",
    OperationType.SHL: "LSL",
    OperationType.LSHR: "LSR",
    OperationType.ASHR: "ASR",
    OperationType.LOAD: "LD",
    OperationType.STORE: "ST",
    OperationType.ICMP: "ICMP",
    OperationType.SELECT: "SELECT",
    OperationType.PHI: "PHI",
    OperationType.INPUT: "INPUT",
    OperationType.OUTPUT: "OUTPUT",
    OperationType.CONST: "CONST",
    OperationType.NOP: "NOP",
}


@dataclass(frozen=True)
class _PackedFeatureDescriptor:
    """Dynamic pack/unpack feature encoding attached to one MRRG node."""

    feature_name: str
    bypass_bit: Optional[int] = None
    enable_bit: Optional[int] = None
    lane_lsb: Optional[int] = None
    lane_width_bits: int = 0
    sign_extend_bit: Optional[int] = None
    sign_extend_value: int = 1

    def encode(self, *, lane_index: int = 0, active: bool) -> int:
        """Encode one feature value for the requested lane state."""
        value = 0

        if self.bypass_bit is not None and not active:
            value |= 1 << self.bypass_bit
        if self.enable_bit is not None and active:
            value |= 1 << self.enable_bit

        if active and self.lane_lsb is not None and self.lane_width_bits > 0:
            max_lane = 1 << self.lane_width_bits
            if lane_index < 0 or lane_index >= max_lane:
                raise ValueError(
                    f"lane_index={lane_index} does not fit in {self.lane_width_bits} bits"
                )
            value |= lane_index << self.lane_lsb

        if active and self.sign_extend_bit is not None and self.sign_extend_value:
            value |= 1 << self.sign_extend_bit

        return value


@dataclass(frozen=True)
class _PackedNodeDescriptor:
    """Pack/unpack metadata for one MRRG routing node."""

    node_id: str
    pack_capable: bool
    unpack_capable: bool
    allowed_lane_indices: Tuple[int, ...] = ()
    pack_config: Optional[_PackedFeatureDescriptor] = None
    unpack_config: Optional[_PackedFeatureDescriptor] = None


@dataclass
class FasmAssignment:
    """A single FASM feature assignment."""

    feature_name: str
    width: int
    value: int
    context: int = 0


@dataclass
class FasmResult:
    """Result of FASM generation."""

    fasm_path: str
    assignment_count: int
    expanded_compiler_arch_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


class FasmGenerator:
    """Generate FASM from mapper mapping results + compiler_arch.json."""

    def __init__(self, compiler_arch_path: str) -> None:
        self._compiler_arch_path = compiler_arch_path
        self._payload = load_and_validate_compiler_arch(compiler_arch_path)

        self.features: Dict[str, int] = {}
        for feat in self._payload["features"]:
            self.features[feat["name"]] = feat["width"]

        self.routing_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for edge in self._payload["routing_programming"]["edges"]:
            key = (edge["source"], edge["sink"])
            self.routing_map[key] = edge["required_assignments"]

        self.optype_encoding_by_module: Dict[Tuple[str, str], Tuple[int, int]] = {}
        self.optype_encoding: Dict[str, Tuple[int, int]] = {}
        for module in self._payload.get("module_operation_capabilities", []):
            module_name = module.get("module_name")
            if not isinstance(module_name, str) or module_name == "":
                continue
            for binding in module.get("bindings", []):
                hint = binding.get("encoding_hint")
                if hint is None:
                    continue
                optype = binding["optype"]
                encoding = (
                    hint["control_word_value"],
                    hint["control_word_width"],
                )
                self.optype_encoding_by_module[(module_name, optype)] = encoding
                self.optype_encoding[optype] = encoding

        self.layout_hash: str = self._payload["layout_hash"]
        self.fabric_contexts: int = self._payload["fabric_contexts"]
        self._node_model_by_id: Dict[str, str] = {}
        self._packed_node_by_id: Dict[str, _PackedNodeDescriptor] = {}
        self._packed_nodes: List[_PackedNodeDescriptor] = []
        self._load_mrrg_sidecar()

    def generate(
        self,
        dfg: DFG,
        placement: Dict[str, str],
        routes: Dict[Tuple[str, int], Union[List[str], List[List[str]]]],
        output_path: str,
        ii: int = 1,
        route_metadata: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
        expanded_compiler_arch_output_path: Optional[str] = None,
    ) -> FasmResult:
        """Generate a FASM file from a mapping result."""
        errors: List[str] = []
        warnings: List[str] = []

        expanded_compiler_arch_path: Optional[str] = None
        if ii > self.fabric_contexts:
            warnings.append(
                f"II={ii} exceeds fabric_contexts={self.fabric_contexts}; "
                "emitting FASM for expanded virtual contexts."
            )
            if expanded_compiler_arch_output_path is not None:
                expanded_compiler_arch_path = self._write_expanded_compiler_arch(
                    expanded_compiler_arch_output_path,
                    ii,
                )

        dfg_nodes: Dict[str, DFGNode] = {n.id: n for n in dfg.get_nodes()}
        hyperdfg = HyperDFG.from_dfg(dfg)

        fu_assignments = self._generate_fu_assignments(dfg_nodes, placement, errors, warnings)
        routing_assignments = self._generate_routing_assignments(
            dfg_nodes, placement, routes, hyperdfg, errors, warnings
        )
        packed_defaults = self._generate_packed_default_assignments(errors)
        packed_assignments = self._generate_packed_route_assignments(
            route_metadata=route_metadata,
            errors=errors,
            warnings=warnings,
        )

        all_assignments: Dict[Tuple[str, int], FasmAssignment] = {}
        default_keys: Set[Tuple[str, int]] = set()

        for assignment in packed_defaults:
            key = (assignment.feature_name, assignment.context)
            if not self._validate_assignment(assignment, errors):
                continue
            if key in all_assignments and all_assignments[key].value != assignment.value:
                errors.append(
                    f"Conflict: feature '{assignment.feature_name}' context {assignment.context} "
                    f"has multiple packed-default values ({all_assignments[key].value}, {assignment.value})"
                )
                continue
            all_assignments[key] = assignment
            default_keys.add(key)

        for assignment in fu_assignments + routing_assignments + packed_assignments:
            key = (assignment.feature_name, assignment.context)
            if key in all_assignments:
                existing = all_assignments[key]
                if existing.value == assignment.value:
                    continue
                if key in default_keys:
                    if not self._validate_assignment(assignment, errors):
                        continue
                    all_assignments[key] = assignment
                    default_keys.discard(key)
                    continue
                errors.append(
                    f"Conflict: feature '{assignment.feature_name}' context {assignment.context} "
                    f"assigned both {existing.value} and {assignment.value}"
                )
            else:
                if not self._validate_assignment(assignment, errors):
                    continue
                all_assignments[key] = assignment

        if errors:
            return FasmResult(
                fasm_path=output_path,
                assignment_count=0,
                expanded_compiler_arch_path=expanded_compiler_arch_path,
                warnings=warnings,
                errors=errors,
            )

        sorted_assignments = sorted(
            all_assignments.values(), key=lambda a: (a.context, a.feature_name)
        )
        self._write_fasm(sorted_assignments, output_path)

        return FasmResult(
            fasm_path=output_path,
            assignment_count=len(sorted_assignments),
            expanded_compiler_arch_path=expanded_compiler_arch_path,
            warnings=warnings,
        )

    def _write_expanded_compiler_arch(
        self,
        output_path: str,
        ii: int,
    ) -> str:
        """Write a compiler_arch clone with fabric_contexts expanded to II."""
        expanded_payload = dict(self._payload)
        expanded_payload["fabric_contexts"] = int(ii)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expanded_payload, indent=2, sort_keys=True), encoding="utf-8")
        return str(path)

    def _normalize_route_paths(
        self,
        route_payload: Union[List[str], List[List[str]]],
    ) -> List[List[str]]:
        """Normalize route payload into a list-of-subpaths shape."""
        if not route_payload:
            return []
        first = route_payload[0]
        if isinstance(first, str):
            return [list(route_payload)]  # type: ignore[arg-type]
        return route_payload  # type: ignore[return-value]

    def _load_mrrg_sidecar(self) -> None:
        """Load sibling mrrg.json to recover module and packing metadata."""
        mrrg_path = Path(self._compiler_arch_path).with_name("mrrg.json")
        if not mrrg_path.exists():
            return

        payload = json.loads(mrrg_path.read_text(encoding="utf-8"))
        nodes = payload.get("nodes", [])
        ii = max((node.get("time", 0) for node in nodes), default=0) + 1
        prefix_cycles = ii > 1

        for node in nodes:
            raw_id = node.get("node_id")
            if not isinstance(raw_id, str) or raw_id == "":
                continue
            time = int(node.get("time", 0))
            full_id = f"{time}:{raw_id}" if prefix_cycles else raw_id
            prefixed_id = f"{time}:{raw_id}"

            model = node.get("model")
            if isinstance(model, str) and model != "":
                self._node_model_by_id[full_id] = model
                self._node_model_by_id.setdefault(raw_id, model)
                self._node_model_by_id.setdefault(prefixed_id, model)

            packing = (
                node.get("extensions", {})
                .get("node_attrs", {})
                .get("sub_word_packing")
            )
            if not isinstance(packing, dict):
                continue

            descriptor = _PackedNodeDescriptor(
                node_id=full_id,
                pack_capable=bool(packing.get("pack_capable", False)),
                unpack_capable=bool(packing.get("unpack_capable", False)),
                allowed_lane_indices=self._parse_allowed_lane_indices(
                    packing.get("allowed_lane_indices"),
                    num_lanes=int(packing.get("num_lanes", 1) or 1),
                    node_id=full_id,
                ),
                pack_config=self._parse_packed_feature_descriptor(packing.get("pack_config")),
                unpack_config=self._parse_packed_feature_descriptor(packing.get("unpack_config")),
            )
            self._packed_nodes.append(descriptor)
            self._packed_node_by_id[full_id] = descriptor
            self._packed_node_by_id.setdefault(raw_id, descriptor)
            self._packed_node_by_id.setdefault(prefixed_id, descriptor)

    def _generate_fu_assignments(
        self,
        dfg_nodes: Dict[str, DFGNode],
        placement: Dict[str, str],
        errors: List[str],
        warnings: List[str],
    ) -> List[FasmAssignment]:
        assignments: List[FasmAssignment] = []

        for dfg_id, mrrg_id in placement.items():
            dfg_node = dfg_nodes.get(dfg_id)
            if dfg_node is None:
                errors.append(f"DFG node '{dfg_id}' not found in DFG")
                continue

            bare = self._strip_cycle_prefix(mrrg_id)
            ctx = self._get_context(mrrg_id)
            op = dfg_node.operation

            dora_optype = _OP_TO_DORA.get(op)
            if dora_optype is None:
                warnings.append(
                    f"No Dora optype mapping for {op.value} "
                    f"(DFG node '{dfg_id}' placed at '{mrrg_id}')"
                )
                continue

            module_name = self._resolve_module_name(mrrg_id)
            encoding = None
            if module_name is not None:
                encoding = self.optype_encoding_by_module.get((module_name, dora_optype))
            if encoding is None:
                encoding = self.optype_encoding.get(dora_optype)
            if encoding is None:
                warnings.append(
                    f"No encoding_hint for Dora optype '{dora_optype}' "
                    f"(DFG node '{dfg_id}', module={module_name})"
                )
                continue

            cw_value, cw_width = encoding

            if op == OperationType.CONST:
                const_val = dfg_node.get_attribute("const_value")
                if const_val is None:
                    const_val = dfg_node.get_attribute("value")
                if const_val is None:
                    errors.append(f"CONST node '{dfg_id}' has no const_value attribute")
                    continue
                const_int = int(const_val)
                if const_int < 0:
                    const_int = const_int & ((1 << cw_width) - 1)
                assignments.append(FasmAssignment(bare, cw_width, const_int, ctx))
            else:
                if bare not in self.features:
                    warnings.append(
                        f"FU feature '{bare}' not found in compiler_arch.json "
                        f"(DFG node '{dfg_id}'); skipping opcode assignment"
                    )
                    continue
                assignments.append(FasmAssignment(bare, cw_width, cw_value, ctx))

        return assignments

    def _generate_packed_default_assignments(
        self,
        errors: List[str],
    ) -> List[FasmAssignment]:
        """Emit deterministic disabled defaults for all pack/unpack features."""
        assignments: List[FasmAssignment] = []
        seen: Set[Tuple[str, int]] = set()

        for node in self._packed_nodes:
            ctx = self._get_context(node.node_id)
            for descriptor in (node.pack_config, node.unpack_config):
                if descriptor is None:
                    continue
                key = (descriptor.feature_name, ctx)
                if key in seen:
                    continue
                width = self.features.get(descriptor.feature_name)
                if width is None:
                    errors.append(
                        f"Packed feature '{descriptor.feature_name}' from mrrg.json "
                        f"is missing from compiler_arch.json"
                    )
                    continue
                assignments.append(
                    FasmAssignment(
                        descriptor.feature_name,
                        width,
                        descriptor.encode(active=False),
                        ctx,
                    )
                )
                seen.add(key)

        return assignments

    def _generate_packed_route_assignments(
        self,
        *,
        route_metadata: Optional[Dict[Tuple[str, int], Dict[str, Any]]],
        errors: List[str],
        warnings: List[str],
    ) -> List[FasmAssignment]:
        """Generate dynamic pack/unpack assignments for packed routes."""
        if not route_metadata:
            return []

        assignments: List[FasmAssignment] = []
        for route_key, metadata in route_metadata.items():
            if not isinstance(metadata, dict):
                errors.append(f"route_metadata[{route_key!r}] must be a dict")
                continue
            if metadata.get("transport_mode") != "packed":
                continue

            lane_assignments = metadata.get("lane_assignments", {})
            if not isinstance(lane_assignments, dict) or not lane_assignments:
                errors.append(
                    f"Packed route {route_key!r} is missing lane_assignments metadata"
                )
                continue

            active_pack_nodes = metadata.get("active_pack_nodes")
            active_unpack_nodes = metadata.get("active_unpack_nodes")

            def _normalize_active_nodes(payload: Any, field_name: str) -> Optional[Set[str]]:
                if payload is None:
                    return None
                if not isinstance(payload, list) or not all(
                    isinstance(node_id, str) and node_id != "" for node_id in payload
                ):
                    errors.append(
                        f"Packed route {route_key!r} has invalid {field_name}: {payload!r}"
                    )
                    return set()
                return set(payload)

            active_pack_set = _normalize_active_nodes(active_pack_nodes, "active_pack_nodes")
            active_unpack_set = _normalize_active_nodes(
                active_unpack_nodes,
                "active_unpack_nodes",
            )
            if active_pack_set is None:
                active_pack_set = {
                    node_id
                    for node_id in lane_assignments
                    if (
                        (self._packed_node_by_id.get(node_id) or self._packed_node_by_id.get(
                            self._strip_cycle_prefix(node_id)
                        ))
                        and (
                            self._packed_node_by_id.get(node_id)
                            or self._packed_node_by_id.get(self._strip_cycle_prefix(node_id))
                        ).pack_config is not None
                    )
                }
            if active_unpack_set is None:
                active_unpack_set = {
                    node_id
                    for node_id in lane_assignments
                    if (
                        (self._packed_node_by_id.get(node_id) or self._packed_node_by_id.get(
                            self._strip_cycle_prefix(node_id)
                        ))
                        and (
                            self._packed_node_by_id.get(node_id)
                            or self._packed_node_by_id.get(self._strip_cycle_prefix(node_id))
                        ).unpack_config is not None
                    )
                }

            for field_name, node_ids in (
                ("active_pack_nodes", active_pack_set),
                ("active_unpack_nodes", active_unpack_set),
            ):
                for node_id in node_ids:
                    if node_id not in lane_assignments:
                        errors.append(
                            f"Packed route {route_key!r} marks node '{node_id}' in "
                            f"{field_name} but it has no lane assignment"
                        )

            for node_id, lane_index in lane_assignments.items():
                descriptor = self._packed_node_by_id.get(node_id)
                if descriptor is None:
                    descriptor = self._packed_node_by_id.get(
                        self._strip_cycle_prefix(node_id)
                    )
                if descriptor is None:
                    errors.append(
                        f"Packed route {route_key!r} references node '{node_id}' "
                        f"with no sub_word_packing metadata in mrrg.json"
                    )
                    continue

                if not isinstance(lane_index, int):
                    errors.append(
                        f"Packed route {route_key!r} has non-integer lane index "
                        f"for node '{node_id}': {lane_index!r}"
                    )
                    continue
                if descriptor.allowed_lane_indices and lane_index not in descriptor.allowed_lane_indices:
                    errors.append(
                        f"Packed route {route_key!r} assigns disallowed lane {lane_index} "
                        f"to node '{node_id}'; allowed lanes are "
                        f"{list(descriptor.allowed_lane_indices)}"
                    )
                    continue

                ctx = self._get_context(node_id)
                if descriptor.pack_config is not None and node_id in active_pack_set:
                    width = self.features.get(descriptor.pack_config.feature_name)
                    if width is None:
                        errors.append(
                            f"Packed feature '{descriptor.pack_config.feature_name}' "
                            f"is missing from compiler_arch.json"
                        )
                    else:
                        try:
                            value = descriptor.pack_config.encode(
                                lane_index=lane_index,
                                active=True,
                            )
                        except ValueError as exc:
                            errors.append(
                                f"Packed route {route_key!r} has invalid lane assignment "
                                f"for node '{node_id}': {exc}"
                            )
                        else:
                            assignments.append(
                                FasmAssignment(
                                    descriptor.pack_config.feature_name,
                                    width,
                                    value,
                                    ctx,
                                )
                            )

                if descriptor.unpack_config is not None and node_id in active_unpack_set:
                    width = self.features.get(descriptor.unpack_config.feature_name)
                    if width is None:
                        errors.append(
                            f"Packed feature '{descriptor.unpack_config.feature_name}' "
                            f"is missing from compiler_arch.json"
                        )
                    else:
                        try:
                            value = descriptor.unpack_config.encode(
                                lane_index=lane_index,
                                active=True,
                            )
                        except ValueError as exc:
                            errors.append(
                                f"Packed route {route_key!r} has invalid lane assignment "
                                f"for node '{node_id}': {exc}"
                            )
                        else:
                            assignments.append(
                                FasmAssignment(
                                    descriptor.unpack_config.feature_name,
                                    width,
                                    value,
                                    ctx,
                                )
                            )

                if descriptor.pack_config is None and descriptor.unpack_config is None:
                    warnings.append(
                        f"Packed carrier node '{node_id}' in route {route_key!r} "
                        f"has no programmable pack/unpack feature; skipping FASM emission"
                    )

        return assignments

    def _generate_routing_assignments(
        self,
        dfg_nodes: Dict[str, DFGNode],
        placement: Dict[str, str],
        routes: Dict[Tuple[str, int], Union[List[str], List[List[str]]]],
        hyperdfg: HyperDFG,
        errors: List[str],
        warnings: List[str],
    ) -> List[FasmAssignment]:
        assignments: List[FasmAssignment] = []
        warned_load_addr_data_i: Set[Tuple[str, int]] = set()

        hyperval_by_source: Dict[str, Any] = {}
        for hv in hyperdfg.get_edges():
            hyperval_by_source[hv.source_id] = hv

        for (source_dfg_id, dest_idx), sub_paths_payload in routes.items():
            sub_paths = self._normalize_route_paths(sub_paths_payload)
            if not sub_paths:
                continue

            src_fu_id = placement.get(source_dfg_id)
            if src_fu_id is None:
                errors.append(
                    f"Source DFG node '{source_dfg_id}' not in placement "
                    f"(route key ({source_dfg_id}, {dest_idx}))"
                )
                continue

            hv = hyperval_by_source.get(source_dfg_id)
            if hv is None:
                errors.append(f"No HyperVal for source '{source_dfg_id}'")
                continue
            if dest_idx >= len(hv.destination_ids):
                errors.append(
                    f"dest_idx={dest_idx} out of range for HyperVal "
                    f"from '{source_dfg_id}' (cardinality={hv.cardinality})"
                )
                continue
            dest_dfg_id = hv.destination_ids[dest_idx]
            sink_fu_id = placement.get(dest_dfg_id)
            if sink_fu_id is None:
                errors.append(
                    f"Dest DFG node '{dest_dfg_id}' not in placement "
                    f"(route from '{source_dfg_id}' dest_idx={dest_idx})"
                )
                continue
            sink_fu_bare = self._strip_cycle_prefix(sink_fu_id)

            for path in sub_paths:
                if not path:
                    continue

                dest_dfg_node = dfg_nodes.get(dest_dfg_id)
                if dest_dfg_node and dest_dfg_node.operation == OperationType.LOAD:
                    operand = hv.operands[dest_idx] if dest_idx < len(hv.operands) else None
                    warn_key = (source_dfg_id, dest_idx)
                    if operand in ("addr", "ADDR") and warn_key not in warned_load_addr_data_i:
                        for node_id in path:
                            bare_node = self._strip_cycle_prefix(node_id)
                            if bare_node.startswith(f"{sink_fu_bare}.") and "data_i" in bare_node:
                                warnings.append(
                                    f"LOAD address operand routed through data_i "
                                    f"(route {source_dfg_id}->{dest_dfg_id}, "
                                    f"node '{node_id}'). This may indicate a "
                                    f"router pin-binding bug."
                                )
                                warned_load_addr_data_i.add(warn_key)
                                break

                chain = [src_fu_id] + list(path) + [sink_fu_id]

                for i in range(len(chain) - 1):
                    a_bare = self._strip_cycle_prefix(chain[i])
                    b_bare = self._strip_cycle_prefix(chain[i + 1])
                    ctx = self._get_context(chain[i])

                    rp = self.routing_map.get((a_bare, b_bare))
                    if rp is None:
                        continue

                    for entry in rp:
                        fname = entry["feature_name"]
                        fvalue = entry["value"]
                        fwidth = self.features.get(fname)
                        if fwidth is None:
                            errors.append(
                                f"routing_programming references unknown feature "
                                f"'{fname}' (edge {a_bare} -> {b_bare})"
                            )
                            continue
                        assignments.append(FasmAssignment(fname, fwidth, fvalue, ctx))

        return assignments

    def _validate_assignment(
        self,
        assignment: FasmAssignment,
        errors: List[str],
    ) -> bool:
        if assignment.feature_name not in self.features:
            errors.append(
                f"Unknown feature '{assignment.feature_name}' (not in compiler_arch.json)"
            )
            return False
        expected_w = self.features[assignment.feature_name]
        if assignment.width != expected_w:
            errors.append(
                f"Width mismatch for '{assignment.feature_name}': "
                f"expected {expected_w}, got {assignment.width}"
            )
            return False
        return True

    def _resolve_module_name(self, node_id: str) -> Optional[str]:
        model = self._node_model_by_id.get(node_id)
        if model is not None:
            return model
        return self._node_model_by_id.get(self._strip_cycle_prefix(node_id))

    @staticmethod
    def _parse_packed_feature_descriptor(
        payload: Any,
    ) -> Optional[_PackedFeatureDescriptor]:
        if not isinstance(payload, dict):
            return None
        feature_name = payload.get("feature_name")
        if not isinstance(feature_name, str) or feature_name == "":
            return None

        def _optional_int(key: str) -> Optional[int]:
            value = payload.get(key)
            return value if isinstance(value, int) else None

        sign_extend_value = payload.get("sign_extend_value", 1)
        if not isinstance(sign_extend_value, int):
            sign_extend_value = 1

        return _PackedFeatureDescriptor(
            feature_name=feature_name,
            bypass_bit=_optional_int("bypass_bit"),
            enable_bit=_optional_int("enable_bit"),
            lane_lsb=_optional_int("lane_lsb"),
            lane_width_bits=int(payload.get("lane_width_bits", 0) or 0),
            sign_extend_bit=_optional_int("sign_extend_bit"),
            sign_extend_value=sign_extend_value,
        )

    @staticmethod
    def _parse_allowed_lane_indices(
        payload: Any,
        *,
        num_lanes: int,
        node_id: str,
    ) -> Tuple[int, ...]:
        if payload is None:
            if num_lanes <= 1:
                return ()
            return tuple(range(num_lanes))
        if not isinstance(payload, list):
            raise ValueError(
                f"Packed node '{node_id}' has non-list allowed_lane_indices: {payload!r}"
            )
        normalized = sorted({lane_idx for lane_idx in payload if isinstance(lane_idx, int)})
        if len(normalized) != len(payload):
            raise ValueError(
                f"Packed node '{node_id}' has invalid allowed_lane_indices: {payload!r}"
            )
        if not normalized:
            raise ValueError(
                f"Packed node '{node_id}' declares empty allowed_lane_indices"
            )
        if any(lane_idx < 0 or lane_idx >= num_lanes for lane_idx in normalized):
            raise ValueError(
                f"Packed node '{node_id}' has allowed_lane_indices outside "
                f"[0, {num_lanes}): {payload!r}"
            )
        return tuple(normalized)

    @staticmethod
    def _strip_cycle_prefix(node_id: str) -> str:
        if ":" in node_id:
            return node_id.split(":", 1)[1]
        return node_id

    @staticmethod
    def _get_context(node_id: str) -> int:
        if ":" in node_id:
            return int(node_id.split(":", 1)[0])
        return 0

    def _write_fasm(self, assignments: List[FasmAssignment], output_path: str) -> None:
        lines = [
            "# Dora CGRA FASM Configuration",
            f"# dora_layout_hash: {self.layout_hash}",
            "# Source: local mapper mapping",
            f"# Compiler architecture: {self._compiler_arch_path}",
            "",
        ]

        for assignment in assignments:
            prefix = f"{{ctx{assignment.context}}}" if assignment.context > 0 else ""
            bits = format(assignment.value, f"0{assignment.width}b")
            lines.append(f"{prefix}{assignment.feature_name}[{assignment.width - 1}:0] = {assignment.width}'b{bits}")

        lines.append("")
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
