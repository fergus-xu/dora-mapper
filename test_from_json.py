#!/usr/bin/env python3
"""Test script for MRRG.from_json() method."""

import sys
sys.path.insert(0, '/root/dora-mapper/src')

from mapper.graph.mrrg import MRRG

def main():
    print("Testing MRRG.from_json()...")
    
    # Load MRRG from JSON
    json_path = "/root/dora-mapper/benchmarks/mrrg.json"
    print(f"Loading MRRG from {json_path}")
    
    mrrg = MRRG.from_json(json_path, name="Test_MRRG")
    
    # Print basic statistics
    print("\n" + "="*60)
    print(mrrg.to_string())
    print("="*60)
    
    # Print some sample nodes
    print("\n\nSample Nodes:")
    print("-" * 60)
    nodes = mrrg.get_nodes()[:5]
    for node in nodes:
        print(node.to_string())
        print("-" * 60)
    
    # Print some sample edges
    print("\n\nSample Edges:")
    print("-" * 60)
    edges = mrrg.get_edges()[:5]
    for edge in edges:
        print(edge.to_string())
        print("-" * 60)
    
    # Print node type breakdown
    print("\n\nNode Type Breakdown:")
    fu_nodes = mrrg.get_fu_nodes()
    routing_nodes = mrrg.get_routing_nodes()
    register_nodes = mrrg.get_register_nodes()
    
    print(f"  Function nodes: {len(fu_nodes)}")
    print(f"  Routing nodes: {len(routing_nodes)}")
    print(f"  Register nodes: {len(register_nodes)}")
    
    # Print some examples of each type
    if fu_nodes:
        print(f"\n  Example FU node: {fu_nodes[0]}")
    if routing_nodes:
        print(f"  Example routing node: {routing_nodes[0]}")
    if register_nodes:
        print(f"  Example register node: {register_nodes[0]}")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    main()
