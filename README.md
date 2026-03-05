# dora-mapper

CGRA mapping tool for DFG to MRRG mapping.

## Install

```bash
cd dora-mapper
pip install -e .
```

## Usage

### Generate LLVM IR from C

```bash
mapper irgen input.c -o output.ll
```

### Map DFG kernel to MRRG

```bash
mapper map kernel.dot mrrg.json -o results.json
```

Options:
- `--max-iterations N` - Max place-and-route iterations (default: 1000)
- `--temperature T` - Initial annealing temperature (default: 1000.0)
- `--max-time SECS` - Max runtime in seconds
- `--seed N` - Random seed (default: 42)
- `--debug` - Enable debug output

### Example

```bash
mapper map benchmarks/kernels/simple/graph_loop.dot benchmarks/mrrg.json --max-iterations 100 -o result.json
```

## Input Formats

**DFG (kernel.dot)**: DOT format with nodes having `opcode`, `bitwidth` attributes

**MRRG (mrrg.json)**: JSON format with `nodes` and `edges` arrays containing:
- Nodes: `node_id`, `time`, `kind`, `datatype`, `model`
- Edges: `source_node`, `source_time`, `target_node`, `target_time`

## Output

JSON with `status`, `placement`, `routes`, `runtime`, `iterations`, `final_cost`



## Operation Nmaes (LLVM Naming Conventions)

### Integer Arithmetic
- add
- sub
- mul


### Bitwise
- and
- or
- xor
- shl
- lshr
- ashr


### Comparison
- icmp
- fcmp

### Floating Point Arithmetic
- fadd
- fsub
- fmul
- fma

### Memory
- load
- store

### Address Computation
- getelementptr

