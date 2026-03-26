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
    print(f"  Loaded {dfg.num_nodes()} nodes, {dfg.num_edges()} edges")

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
        for cap in compiler_arch.get("module_operation_capabilities", []):
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
    network_latencies = {}
    for src in OperationType:
        if src == OperationType.OUTPUT:
            continue
        src_fus = mrrg.get_compatible_fus(src)

        for sink in OperationType:
            if sink == OperationType.INPUT:
                continue
                
            sink_fus = mrrg.get_compatible_fus(sink)
            
            # If we don't have compatible FUs for either, fallback to default (1, 2)
            if not src_fus or not sink_fus:
                network_latencies[OperationLatencyEdge(src, sink)] = (1, 2)
                continue
                
            # Calculate cycle routing latency using MRRG shortest paths
            path_latencies = []
            for s_fu in src_fus:
                for d_fu in sink_fus:
                    # k=1 gives the absolute shortest path between this specific src/sink pair
                    paths = mrrg.get_k_shortest_paths_between_fu_nodes_optimized(s_fu.id, d_fu.id, k=1)
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
                # Bound between the fastest path we found and the slowest "fastest" path + 1
                min_lat = max(1, min(path_latencies))
                max_lat = max(2, max(path_latencies) + 1)
                network_latencies[OperationLatencyEdge(src, sink)] = (min_lat, max_lat)
            else:
                # Fall back on (1, 2) if no paths found
                network_latencies[OperationLatencyEdge(src, sink)] = (1, 2)
    
    latency_spec = LatencySpecification(
        op_latencies=op_latencies,
        network_latencies=network_latencies
    )

    # Create and run mapper
    print(f"\nRunning heuristic mapper (max_iterations={args.max_iterations})...")
    mapper = HeuristicMapper(
        dfg=dfg,
        mrrg=mrrg,
        latency_spec=latency_spec,
        initial_temperature=args.temperature,
        max_iterations=args.max_iterations,
        max_time=args.max_time,
        random_seed=args.seed,
        debug=args.debug,
    )

    result = mapper.map()

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

    # Save results if requested
    if args.output and result['status'] == 'success':
        output_data = {
            'status': result['status'],
            'placement': result['placement'],
            'routes': result['routes'],
            'runtime': result['runtime'],
            'iterations': result['iterations'],
            'final_cost': result.get('final_cost', 0),
        }
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
