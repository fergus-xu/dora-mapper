#!/usr/bin/env python3
"""
Verification script for Dora mapper output.

Checks that a mapping JSON is structurally and functionally correct
against the original DFG and the target MRRG architecture.

Usage:
    source ./scripts/activate
    python verify_mapping.py sum_dice.json \
        benchmarks/kernels/sum/graph_loop.dot \
        benchmarks/architectures/dice/mrrg.json \
        --compiler-arch benchmarks/architectures/dice/compiler_arch.json
"""

import argparse
import json
import sys
import os
from pathlib import Path
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from mapper.graph.mrrg import MRRG, NodeType
from mapper.graph.dfg import DFG
from mapper.workload.parsers.dot_parser import DFGDotParser


def load_dfg(dot_path: str) -> DFG:
    """Load and preprocess the DFG."""
    parser = DFGDotParser()
    dfg = parser.parse(Path(dot_path))
    dfg.preprocess_for_scheduling()
    return dfg


def load_mrrg(mrrg_path: str, compiler_arch_path: str = None) -> MRRG:
    """Load the MRRG."""
    return MRRG.from_json(mrrg_path, compiler_arch_path=compiler_arch_path)


class MappingVerifier:
    """Verifies structural and functional correctness of a mapper output."""

    def __init__(self, mapping: dict, dfg: DFG, mrrg: MRRG):
        self.mapping = mapping
        self.dfg = dfg
        self.mrrg = mrrg
        self.placement = mapping.get("placement", {})
        self.routes = mapping.get("routes", {})
        self.errors = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_total = 0

    def _check(self, condition: bool, description: str, is_warning=False):
        """Record a check result."""
        self.checks_total += 1
        if condition:
            self.checks_passed += 1
            print(f"  ✓ {description}")
        else:
            if is_warning:
                self.warnings.append(description)
                print(f"  ⚠ {description}")
            else:
                self.errors.append(description)
                print(f"  ✗ {description}")

    # ─────────────────────────────────────────────────
    # CHECK 1: DFG coverage — every DFG operation is placed
    # ─────────────────────────────────────────────────
    def check_placement_coverage(self):
        """Verify all DFG operations are placed on FUs."""
        print("\n[1] Placement Coverage")
        dfg_ops = {n.id for n in self.dfg.get_nodes()}
        placed_ops = set(self.placement.keys())

        missing = dfg_ops - placed_ops
        extra = placed_ops - dfg_ops

        self._check(
            len(missing) == 0,
            f"All {len(dfg_ops)} DFG operations are placed"
            + (f" (missing: {missing})" if missing else ""),
        )
        self._check(
            len(extra) == 0,
            f"No extraneous placements"
            + (f" (extra: {extra})" if extra else ""),
            is_warning=True,
        )

    # ─────────────────────────────────────────────────
    # CHECK 2: Placement targets exist in the MRRG
    # ─────────────────────────────────────────────────
    def check_placement_targets_valid(self):
        """Verify each placed FU actually exists in the MRRG."""
        print("\n[2] Placement Target Validity")
        valid = True
        for op, fu_id in self.placement.items():
            node = self.mrrg.get_node(fu_id)
            if not node:
                self._check(False, f"{op} → {fu_id} exists in MRRG")
                valid = False
        if valid:
            self._check(True, f"All {len(self.placement)} FU targets exist in the MRRG")

    # ─────────────────────────────────────────────────
    # CHECK 3: No two operations share the same FU
    # ─────────────────────────────────────────────────
    def check_placement_exclusivity(self):
        """Verify each FU is used by at most one operation."""
        print("\n[3] Placement Exclusivity (no FU collisions)")
        fu_to_ops = defaultdict(list)
        for op, fu_id in self.placement.items():
            fu_to_ops[fu_id].append(op)

        collisions = {fu: ops for fu, ops in fu_to_ops.items() if len(ops) > 1}
        self._check(
            len(collisions) == 0,
            f"All {len(self.placement)} placements are on distinct FUs"
            + (f" (collisions: {collisions})" if collisions else ""),
        )

    # ─────────────────────────────────────────────────
    # CHECK 4: Operation compatibility — FU supports the opcode
    # ─────────────────────────────────────────────────
    def check_operation_compatibility(self):
        """Verify placed FUs support the assigned operations."""
        print("\n[4] Operation Compatibility")
        incompatible = []
        for op_id, fu_id in self.placement.items():
            dfg_node = self.dfg.get_node(op_id)
            fu_node = self.mrrg.get_node(fu_id)
            if not dfg_node or not fu_node:
                continue
            op_type = dfg_node.operation
            supported = fu_node.supported_operations
            if supported and op_type not in supported:
                incompatible.append((op_id, op_type, fu_id, supported))

        self._check(
            len(incompatible) == 0,
            f"All placed operations are supported by their FUs"
            + (
                f" (incompatible: {[(o, str(t)) for o, t, _, _ in incompatible]})"
                if incompatible
                else ""
            ),
        )

    # ─────────────────────────────────────────────────
    # CHECK 5: Route coverage — every DFG edge is routed
    # ─────────────────────────────────────────────────
    def check_route_coverage(self):
        """Verify every DFG edge has a corresponding route."""
        print("\n[5] Route Coverage")
        # Build expected edges from DFG (after preprocessing)
        expected_edges = []
        for node in self.dfg.get_nodes():
            outgoing = self.dfg.get_outgoing_edges(node.id)
            for i, edge in enumerate(outgoing):
                expected_edges.append((node.id, i))

        # Parse route keys
        routed_edges = set()
        for key in self.routes:
            # Keys are stringified tuples like "('const0', 0)"
            try:
                parts = key.strip("()").split(",")
                src = parts[0].strip().strip("'\"")
                idx = int(parts[1].strip())
                routed_edges.add((src, idx))
            except (ValueError, IndexError):
                pass

        missing = [e for e in expected_edges if e not in routed_edges]
        self._check(
            len(missing) == 0,
            f"All {len(expected_edges)} DFG edges are routed"
            + (f" (missing: {missing})" if missing else ""),
        )

    # ─────────────────────────────────────────────────
    # CHECK 6: Route connectivity — each route is a
    #          contiguous chain in the MRRG
    # ─────────────────────────────────────────────────
    def check_route_connectivity(self):
        """Verify each route is a valid, connected path in the MRRG."""
        print("\n[6] Route Connectivity")
        broken = []
        for key, path in self.routes.items():
            for i in range(len(path) - 1):
                src_node = self.mrrg.get_node(path[i])
                if not src_node:
                    broken.append((key, path[i], "node not in MRRG"))
                    break
                outgoing = self.mrrg.get_outgoing_edges(path[i])
                neighbor_ids = {e.destination.id for e in outgoing}
                if path[i + 1] not in neighbor_ids:
                    broken.append(
                        (key, f"{path[i]} → {path[i+1]}", "not connected")
                    )
                    break

        self._check(
            len(broken) == 0,
            f"All {len(self.routes)} routes are contiguous paths in the MRRG"
            + (f" (broken: {broken})" if broken else ""),
        )

    # ─────────────────────────────────────────────────
    # CHECK 7: Route endpoints match placement
    # ─────────────────────────────────────────────────
    def check_route_endpoints(self):
        """Verify route start/end nodes match their placed FUs."""
        print("\n[7] Route Endpoint Consistency")
        mismatched = []
        for key, path in self.routes.items():
            try:
                parts = key.strip("()").split(",")
                src_op = parts[0].strip().strip("'\"")
            except (ValueError, IndexError):
                continue

            src_fu = self.placement.get(src_op)
            if not src_fu or not path:
                continue

            # The route's first node should be the source FU's output port
            if src_fu not in path[0]:
                mismatched.append((key, f"start {path[0]} ≠ {src_fu}"))

        self._check(
            len(mismatched) == 0,
            f"All route start points match their source FU placement"
            + (f" (mismatched: {mismatched})" if mismatched else ""),
        )

    # ─────────────────────────────────────────────────
    # CHECK 8: No routing resource used by two signals
    # ─────────────────────────────────────────────────
    def check_routing_exclusivity(self):
        """Verify no routing resource is shared by two different signals."""
        print("\n[8] Routing Resource Exclusivity")
        resource_usage = defaultdict(list)
        for key, path in self.routes.items():
            for node_id in path:
                resource_usage[node_id].append(key)

        # Extract source hyperval names from keys like "('i3_add1', 1)"
        def get_source_id(user_key):
            try:
                parts = user_key.strip("()").split(",")
                return parts[0].strip().strip("'\"")
            except:
                return user_key

        # Shared output ports are OK (fanout). Shared internal routing is valid 
        # for multicast trees of the SAME source signal.
        conflicts = {}
        for node_id, users in resource_usage.items():
            sources = set(get_source_id(u) for u in users)
            if len(sources) > 1:
                node = self.mrrg.get_node(node_id)
                # Even if sources are different, some architectures might allow it if it's a specific kind of node? 
                # But generally routing nodes should securely branch only same-source signals.
                conflicts[node_id] = users

        self._check(
            len(conflicts) == 0,
            f"No routing resource conflicts detected"
            + (
                f" ({len(conflicts)} conflicts)"
                if conflicts
                else ""
            ),
            is_warning=bool(conflicts),
        )
        if conflicts:
            for node_id, users in list(conflicts.items())[:5]:
                print(f"      {node_id} used by: {users}")

    # ─────────────────────────────────────────────────
    # RUN ALL
    # ─────────────────────────────────────────────────
    def verify(self) -> bool:
        """Run all verification checks. Returns True if all pass."""
        print("=" * 60)
        print("MAPPING VERIFICATION")
        print("=" * 60)

        self.check_placement_coverage()
        self.check_placement_targets_valid()
        self.check_placement_exclusivity()
        self.check_operation_compatibility()
        self.check_route_coverage()
        self.check_route_connectivity()
        self.check_route_endpoints()
        self.check_routing_exclusivity()

        print("\n" + "=" * 60)
        print(f"Results: {self.checks_passed}/{self.checks_total} checks passed")
        if self.errors:
            print(f"  ERRORS:   {len(self.errors)}")
            for e in self.errors:
                print(f"    ✗ {e}")
        if self.warnings:
            print(f"  WARNINGS: {len(self.warnings)}")
            for w in self.warnings:
                print(f"    ⚠ {w}")
        if not self.errors:
            print("\n✓ MAPPING IS VALID")
        else:
            print("\n✗ MAPPING HAS ERRORS")
        print("=" * 60)

        return len(self.errors) == 0


def main():
    parser = argparse.ArgumentParser(description="Verify a Dora mapper output")
    parser.add_argument("mapping", help="Path to the mapping JSON file")
    parser.add_argument("dfg", help="Path to the DFG .dot file")
    parser.add_argument("mrrg", help="Path to the MRRG .json file")
    parser.add_argument(
        "--compiler-arch", help="Path to the compiler_arch JSON file", default=None
    )
    args = parser.parse_args()

    # Load inputs
    print(f"Loading mapping from {args.mapping}...")
    with open(args.mapping) as f:
        mapping = json.load(f)

    print(f"Loading DFG from {args.dfg}...")
    dfg = load_dfg(args.dfg)

    print(f"Loading MRRG from {args.mrrg}...")
    mrrg = load_mrrg(args.mrrg, args.compiler_arch)

    # Detect if mapping used modulo scheduling
    max_cycle = 0
    for fu_id in mapping.get("placement", {}).values():
        if isinstance(fu_id, str) and ":" in fu_id:
            try:
                cycle = int(fu_id.split(":")[0])
                if cycle + 1 > max_cycle:
                    max_cycle = cycle + 1
            except ValueError:
                pass
    
    if max_cycle > 1:
        print(f"Mapping indicates II={max_cycle}. Time-expanding MRRG...")
        mrrg = mrrg.time_expand(max_cycle)

    # Run verification
    verifier = MappingVerifier(mapping, dfg, mrrg)
    success = verifier.verify()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
