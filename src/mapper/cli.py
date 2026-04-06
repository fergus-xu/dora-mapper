from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mapper.tools.irgen import IRGenError, c_to_llvm_ir


def cmd_irgen(args: argparse.Namespace) -> int:
    """Generate LLVM IR from C file."""
    try:
        res = c_to_llvm_ir(
            args.c_file,
            out_ll=args.out,
            run_passes=not args.no_passes,
            include_dirs=args.include_dirs,
            defines=args.defines,
            std=args.std,
            opt_level=args.opt_level,
            extra_flags=args.flag,
        )
    except (IRGenError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"Wrote: {res.output_ll}")
    print(f"Cmd:   {' '.join(res.cmd)}")
    if res.output_dfg:
        print(f"Wrote: {res.output_dfg}")
        print(f"Cmd:   {' '.join(res.dfg_cmd)}")
    return 0


def cmd_map(args: argparse.Namespace) -> int:
    """Run heuristic mapper on DFG."""
    from mapper.graph.mrrg import MRRG
    from mapper.schedules.latency_spec import LatencySpecification
    from mapper.algorithms.mapper.heuristic_mapper import HeuristicMapper
    from mapper.workload.parsers.dot_parser import DFGDotParser
    import json

    # Load DFG from DOT file
    if not args.dfg.exists():
        print(f"error: DFG file not found: {args.dfg}", file=sys.stderr)
        return 1

    print(f"Loading DFG from {args.dfg}...")
    parser = DFGDotParser()
    dfg = parser.parse(args.dfg)
    dfg.preprocess_for_mapping()
    print(f"  Processed {dfg.num_nodes()} nodes, {dfg.num_edges()} edges")

    # Load MRRG from JSON file
    if not args.mrrg.exists():
        print(f"error: MRRG file not found: {args.mrrg}", file=sys.stderr)
        return 1

    print(f"Loading MRRG from {args.mrrg}...")
    
    # Check for optional compiler_arch
    compiler_arch_path = None
    if args.compiler_arch:
        if not args.compiler_arch.exists():
            print(f"error: compiler_arch file not found: {args.compiler_arch}", file=sys.stderr)
            return 1
        compiler_arch_path = str(args.compiler_arch)
        print(f"  Using compiler_arch: {args.compiler_arch}")
    
    mrrg = MRRG.from_json(str(args.mrrg), compiler_arch_path=compiler_arch_path)
    print(f"  Loaded {mrrg.num_nodes()} nodes, {mrrg.num_edges()} edges")
    print(f"  Array: {mrrg.rows}x{mrrg.cols}, II={mrrg.II}")
    
    # Print FU capabilities if available
    fu_nodes = mrrg.get_fu_nodes()
    if fu_nodes:
        fus_with_ops = [fu for fu in fu_nodes if fu.supported_operations]
        print(f"  FU nodes: {len(fu_nodes)} ({len(fus_with_ops)} with operations)")

    # Create default latency specification
    print("Creating latency specification...")
    from mapper.graph.dfg import OperationType
    from mapper.schedules.latency_spec import OperationLatencyEdge
    
    # OPERATION LATENCIES
    # Infer from MRRG FUs to handle heterogeneous FUs.
    op_latencies = {}
    for fu in mrrg.get_fu_nodes():
        for op in fu.supported_operations:
            # Take the optimistic minimum latency for ASAP scheduling.
            if op not in op_latencies:
                op_latencies[op] = fu.latency
            else:
                op_latencies[op] = min(op_latencies[op], fu.latency)
                
    # Fallback to compiler_arch if provided and MRRG lacked latency specs
    if compiler_arch_path:
        with open(compiler_arch_path, 'r') as f:
            compiler_arch = json.load(f)
        # Dora API structure: list of module capability objects
        for cap in compiler_arch.get("module_operation_capabilities", []):
            # Check for Dora "bindings" structure
            bindings = cap.get("bindings", [])
            for binding in bindings:
                op_str = binding.get("optype", "")
                op_latency = binding.get("latency", 1)
                
                for op_enum in OperationType:
                    # Match by name or Dora-style optype string
                    if op_enum.name == op_str.upper() or op_enum.value.lower() == op_str.lower():
                        if op_enum not in op_latencies:
                            op_latencies[op_enum] = op_latency
                        break
            
            # Legacy support (Dice)
            if not bindings:
                module_latency = cap.get("latency", 0)
                for op_str in cap.get("operations", []):
                    for op_enum in OperationType:
                        if op_enum.name == op_str.upper() or op_enum.value.lower() == op_str.lower():
                            if op_enum not in op_latencies:
                                op_latencies[op_enum] = module_latency
                            break

    # Final fallback for missing operations to prevent crashes
    for op_enum in OperationType:
        if op_enum not in op_latencies:
            op_latencies[op_enum] = 1

    # NETWORK LATENCIES
    # Calculate routing latencies using topological shortest paths on the MRRG.
    # Runtime optimization: only compute latencies for operation pairs that
    # actually appear on DFG edges (instead of all OperationType x OperationType).
    network_latencies = {}
    required_op_pairs = {
        (edge.source.operation, edge.destination.operation)
        for edge in dfg.get_edges()
    }

    if args.debug:
        print(f"  Computing network latencies for {len(required_op_pairs)} DFG op-pairs")

    for src, sink in required_op_pairs:
        src_fus = mrrg.get_compatible_fus(src)
        sink_fus = mrrg.get_compatible_fus(sink)

        # If we don't have compatible FUs for either, fallback to default (1, 2)
        if not src_fus or not sink_fus:
            network_latencies[OperationLatencyEdge(src, sink)] = (1, 2)
            continue

        # Calculate cycle routing latency using MRRG shortest paths
        path_latencies = []
        if not hasattr(mrrg, '_path_cache'):
            mrrg._path_cache = {}

        for s_fu in src_fus:
            for d_fu in sink_fus:
                cache_key = (s_fu.id, d_fu.id)
                if cache_key not in mrrg._path_cache:
                    # k=1 gives the absolute shortest path between this specific src/sink pair
                    paths = mrrg.get_k_shortest_paths_between_fu_nodes_optimized(s_fu.id, d_fu.id, k=1)
                    mrrg._path_cache[cache_key] = paths

                paths = mrrg._path_cache[cache_key]

                if paths and len(paths[0]) > 0:
                    shortest_path = paths[0]
                    # Combinational wire latency=0, register latency=1.
                    # Sum the true pipeline stage latency of the intermediate path.
                    latency = sum(
                        mrrg.get_node(node_id).latency
                        for node_id in shortest_path[1:-1]
                        if mrrg.get_node(node_id)
                    )
                    path_latencies.append(latency)

        if path_latencies:
            # Bound between the fastest path we found and a much larger upper bound
            # to allow the scheduler flexibility (especially for loop-backs).
            min_lat = min(path_latencies)
            # Using a large upper bound (e.g., II + 10 or just a large constant)
            # ensures ASAP doesn't fail prematurely.
            max_lat = max(path_latencies) + 20
            network_latencies[OperationLatencyEdge(src, sink)] = (min_lat, max_lat)
        else:
            # Fall back on (0, 20) if no paths found
            network_latencies[OperationLatencyEdge(src, sink)] = (0, 20)
    
    latency_spec = LatencySpecification(
        op_latencies=op_latencies,
        network_latencies=network_latencies
    )

    def _build_mapper() -> HeuristicMapper:
        return HeuristicMapper(
            dfg=dfg,
            mrrg=mrrg,
            latency_spec=latency_spec,
            initial_temperature=args.temperature,
            max_iterations=args.max_iterations,
            max_time=args.max_time,
            random_seed=args.seed,
            router_max_iterations=args.router_max_iterations,
            bitwidth_mismatch_penalty_weight=args.bitwidth_mismatch_penalty_weight,
            placement_restart_patience=args.placement_restart_patience,
            max_placement_restarts=args.max_placement_restarts,
            ii_iteration_patience=args.ii_iteration_patience,
            debug=args.debug,
        )

    def _run_mapping(start_ii: int | None, max_ii: int):
        mapper = _build_mapper()
        if start_ii is not None:
            mapper._calculate_min_ii = lambda ii=start_ii: ii  # type: ignore[attr-defined]
        return mapper.map(max_ii=max_ii)

    # Create and run mapper
    print(f"\nRunning heuristic mapper (max_iterations={args.max_iterations})...")
    result = _run_mapping(start_ii=None, max_ii=args.max_ii)

    # Print results
    print("\n" + "="*60)
    print(f"Status: {result['status']}")
    print(f"Runtime: {result['runtime']:.2f}s")
    print(f"Iterations: {result['iterations']}")
    print(f"Final Cost: {result.get('final_cost', 0):.2f}")
    
    if result['status'] == 'success':
        print(f"Placement: {len(result['placement'])} operations placed")
        print(f"Routes: {len(result['routes'])} connections routed")
    else:
        print(f"Error: {result.get('error_message', 'Unknown error')}")
    print("="*60)

    if result["status"] == "success" and args.fasm_output:
        if not args.compiler_arch:
            print(
                "error: --fasm-output requires --compiler-arch (compiler_arch.json)",
                file=sys.stderr,
            )
            return 1
        validation = result.get("validation", {})
        if validation and not validation.get("is_valid", True):
            print("error: FASM generation blocked because mapping validation failed", file=sys.stderr)
            for err in validation.get("errors", []):
                print(f"  validation error: {err}", file=sys.stderr)
            return 1

        from mapper.architecture.fasm_generator import FasmGenerator

        def _is_feature_conflict_errors(errors: list[str]) -> bool:
            return any(err.startswith("Conflict: feature ") for err in errors)

        def _attempt_fasm(mapping_result):
            print(f"\nGenerating FASM: {args.fasm_output}")
            args.fasm_output.parent.mkdir(parents=True, exist_ok=True)
            effective = int(mapping_result.get("final_ii", mrrg.II))
            expanded_out = args.fasm_output.with_suffix(".expanded_compiler_arch.json")
            generated = fasm_gen.generate(
                dfg=dfg,
                placement=mapping_result["placement"],
                routes=mapping_result["routes"],
                route_metadata=mapping_result.get("route_metadata"),
                output_path=str(args.fasm_output),
                ii=effective,
                expanded_compiler_arch_output_path=str(expanded_out),
            )
            if generated.expanded_compiler_arch_path:
                print(
                    "Expanded compiler_arch written to "
                    f"{generated.expanded_compiler_arch_path}"
                )
            for warning in generated.warnings:
                print(f"FASM warning: {warning}")
            return generated, effective

        fasm_gen = FasmGenerator(str(args.compiler_arch))
        fasm_result, effective_ii = _attempt_fasm(result)

        # If mapping is legal but FASM has switch-feature conflicts, continue II search
        # and pick the minimum II where both mapping and FASM succeed.
        if (
            not fasm_result.ok
            and _is_feature_conflict_errors(fasm_result.errors)
            and effective_ii < args.max_ii
        ):
            next_ii = effective_ii + 1
            print(
                f"FASM conflicts at II={effective_ii}; retrying mapper from II={next_ii}..."
            )
            solved = False
            for target_ii in range(next_ii, args.max_ii + 1):
                candidate = _run_mapping(start_ii=target_ii, max_ii=target_ii)
                if candidate["status"] != "success":
                    continue
                candidate_fasm, candidate_ii = _attempt_fasm(candidate)
                if candidate_fasm.ok:
                    result = candidate
                    fasm_result = candidate_fasm
                    effective_ii = candidate_ii
                    solved = True
                    break
            if not solved and not fasm_result.ok:
                print("error: FASM generation failed", file=sys.stderr)
                for err in fasm_result.errors:
                    print(f"  {err}", file=sys.stderr)
                return 1

        if not fasm_result.ok:
            print("error: FASM generation failed", file=sys.stderr)
            for err in fasm_result.errors:
                print(f"  {err}", file=sys.stderr)
            return 1
        print(f"FASM written to {fasm_result.fasm_path}")
        print(f"FASM assignments: {fasm_result.assignment_count}")

    # Save results if requested
    if args.output and result['status'] == 'success':
        def serialize_keys(d):
            if isinstance(d, dict):
                return {str(k): serialize_keys(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [serialize_keys(i) for i in d]
            elif isinstance(d, tuple):
                return tuple(serialize_keys(i) for i in d)
            return d

        output_data = {
            'status': result['status'],
            'placement': result['placement'],
            'routes': result['routes'],
            'route_metadata': result.get('route_metadata'),
            'runtime': result['runtime'],
            'iterations': result['iterations'],
            'final_cost': result.get('final_cost', 0),
        }
        output_data = serialize_keys(output_data)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved results to {args.output}")

    return 0 if result['status'] == 'success' else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mapper", description="CGRA mapping tools")
    subparsers = p.add_subparsers(dest='command', help='Command to run')

    # IRGen command
    p_irgen = subparsers.add_parser('irgen', help='Generate LLVM IR from C file')
    p_irgen.add_argument("c_file", type=Path)
    p_irgen.add_argument("-o", "--out", type=Path, default=None)
    p_irgen.add_argument("--no-passes", action="store_true", help="Skip LLVM passes (DFG extraction)")
    p_irgen.add_argument("-I", dest="include_dirs", action="append", default=[])
    p_irgen.add_argument("-D", dest="defines", action="append", default=[])
    p_irgen.add_argument("--std", default="c11")
    p_irgen.add_argument("--O", dest="opt_level", default="1")
    p_irgen.add_argument("--flag", action="append", default=[], help="Extra clang flag (repeatable)")

    # Map command
    p_map = subparsers.add_parser('map', help='Run heuristic mapper on DFG')
    p_map.add_argument("dfg", type=Path, help="DFG file in DOT format")
    p_map.add_argument("mrrg", type=Path, help="MRRG file in JSON format")
    p_map.add_argument("-o", "--output", type=Path, default=None, help="Output JSON file for results")
    p_map.add_argument("--compiler-arch", type=Path, default=None, help="compiler_arch.json for operation capabilities")
    p_map.add_argument("--max-iterations", type=int, default=1000, help="Max place-and-route iterations")
    p_map.add_argument("--temperature", type=float, default=1000.0, help="Initial annealing temperature")
    p_map.add_argument("--max-time", type=float, default=None, help="Max runtime in seconds")
    p_map.add_argument("--max-ii", type=int, default=32, help="Maximum Initiation Interval to attempt")
    p_map.add_argument("--router-max-iterations", type=int, default=30, help="PathFinder negotiated-congestion iterations")
    p_map.add_argument("--bitwidth-mismatch-penalty-weight", type=float, default=0.5, help="Soft routing penalty for using wider-than-needed NoC bitwidth")
    p_map.add_argument("--placement-restart-patience", type=int, default=3, help="Placement retries before fresh restart")
    p_map.add_argument("--max-placement-restarts", type=int, default=4, help="Maximum fresh placement restarts per II")
    p_map.add_argument("--ii-iteration-patience", type=int, default=12, help="No-coverage-improvement placement iterations before escalating II")
    p_map.add_argument("--fasm-output", type=Path, default=None, help="Output FASM file for successful mapping")
    p_map.add_argument("--seed", type=int, default=42, help="Random seed")
    p_map.add_argument("--debug", action="store_true", help="Enable debug output")

    args = p.parse_args(argv)

    if not args.command:
        p.print_help()
        return 1

    if args.command == 'irgen':
        return cmd_irgen(args)
    elif args.command == 'map':
        return cmd_map(args)
    else:
        print(f"error: unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
