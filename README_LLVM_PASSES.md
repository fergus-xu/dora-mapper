# DORA Mapper - LLVM IR to DFG Conversion

This system automatically converts C code to LLVM IR and extracts dataflow graphs (DFGs) for CGRA mapping.

## Overview

The system performs a complete C-to-JSON pipeline:
1. **C Source** → Compile to LLVM IR with Clang
2. **LLVM IR** → Apply transformation passes (GEP lowering)
3. **LLVM IR** → Extract DFG from kernel regions
4. **JSON DFG** → Ready for CGRA mapping

## Quick Start

### Building the LLVM Passes

```bash
# From repository root
make passes
```

This compiles the LLVM passes using CMake and generates `llvm/llvm_passes.so`.

### Generating IR and DFG

```bash
python src/mapper/cli.py irgen benchmarks/test.c -I include
```

**Output:**
- `benchmarks/test.ll` - LLVM IR file
- `benchmarks/test_dfg.json` - Dataflow graph in JSON format

### Options

- `--O <level>` - Optimization level (default: O1)
- `--no-passes` - Skip DFG extraction (IR only)
- `-o <file>` - Output filename (placed in same directory as input)

## LLVM Passes

### 1. LowerGEPPass (`lower-gep`)

Lowers `getelementptr` (GEP) instructions into explicit arithmetic operations.

**Before:**
```llvm
%ptr = getelementptr float, ptr %array, i64 %index
```

**After:**
```llvm
%1 = ptrtoint ptr %array to i64
%2 = mul i64 %index, 4        ; element size constant
%3 = add i64 %1, %2
%ptr = inttoptr i64 %3 to ptr
```

**Key features:**
- Creates explicit constant nodes for element sizes
- Converts pointer arithmetic to integer operations
- NOP casts (ptrtoint/inttoptr) are removed in DFG extraction

### 2. DFGExtractionPass (`extract-dfg`)

Extracts dataflow graph from code regions marked with kernel region markers.

**Kernel Region Markers:**
```c
__kernel_region_start();
// ... computation to extract ...
__kernel_region_end();
```

**Features:**
- Extracts instructions between region markers
- Creates nodes for operations, inputs, and constants
- Builds dataflow edges based on dependencies
- Removes NOP operations (casts, type conversions)
- Assigns functional units to operations
- Tracks predecessors and successors for each node

## JSON DFG Format

```json
{
  "function": "function_name",
  "nodes": [
    {
      "id": "0",
      "opcode": "const",
      "fu": "const",
      "bitwidth": 32,
      "value": 4,
      "pred": [],
      "succ": ["1", "2"]
    },
    {
      "id": "1",
      "opcode": "mul",
      "fu": "alu",
      "bitwidth": 32,
      "pred": ["0", "3"],
      "succ": ["4"]
    }
  ],
  "edges": [
    {
      "source": "0",
      "dest": "1",
      "bitwidth": 32
    }
  ]
}
```

### Node Fields

- **id** - Unique node identifier (numeric string)
- **opcode** - Operation type (add, mul, load, const, input, etc.)
- **fu** - Functional unit assignment:
  - `alu` - Integer arithmetic (add, sub, mul, div, shifts, logic)
  - `fpu` - Floating-point operations (fadd, fmul, fdiv, fma)
  - `mem` - Memory operations (load, store)
  - `const` - Constant values
  - `input` - External inputs
- **bitwidth** - Data width in bits (capped at 32)
- **pred** - Array of predecessor node IDs
- **succ** - Array of successor node IDs
- **value** - Constant value (only for const nodes)

### Edge Fields

- **source** - Source node ID
- **dest** - Destination node ID
- **bitwidth** - Data width for this edge

## Operation Mapping

### Supported Operations

| LLVM Instruction | DFG Opcode | Functional Unit |
|-----------------|------------|-----------------|
| add, sub, mul, div | add, sub, mul, div | alu |
| and, or, xor | and, or, xor | alu |
| shl, lshr, ashr | shl, lshr, ashr | alu |
| icmp | icmp | alu |
| fadd, fsub, fmul, fdiv | fadd, fsub, fmul, fdiv | fpu |
| llvm.fmuladd | fma | fpu |
| load, store | load, store | mem |
| Constants | const | const |
| External values | input | input |

### Removed Operations (NOPs)

These operations are transparently wired through:
- Type casts: `sext`, `zext`, `trunc`
- Type conversions: `ptrtoint`, `inttoptr`, `bitcast`
- FP conversions: `fptosi`, `fptoui`, `sitofp`, `uitofp`

## Key Design Decisions

1. **Bitwidth capped at 32 bits** - CGRA target uses 32-bit datapaths
2. **Constants as nodes** - Enables reuse across multiple operations
3. **Inputs for external values** - PHI nodes, function arguments, loop-carried dependencies
4. **GEP lowering** - Makes address arithmetic explicit for CGRA mapping
5. **NOP removal** - Simplifies DFG by removing type-only operations
6. **Functional unit assignment** - Pre-assigns operations to FU types for mapping

## Example Workflow

**Input C code:**
```c
#include "qualifiers.h"

__global__
float kernel(float *a, int i) {
    __kernel_region_start();
    float val = a[i];
    float result = val * 2.0f + 1.0f;
    __kernel_region_end();
    return result;
}
```

**Command:**
```bash
python src/mapper/cli.py irgen kernel.c -I include
```

**Generated Files:**
- `kernel.ll` - LLVM IR
- `kernel_dfg.json` - Dataflow graph

**Resulting DFG includes:**
- Input nodes for `a` (pointer) and `i` (index)
- Const nodes for `2.0` and `1.0`
- Load node for `a[i]`
- Mul and add nodes for computation
- FMA node if optimized to fused multiply-add

## Integration with CGRA-ME

The generated JSON DFG format is compatible with CGRA-ME's dataflow graph representation. The functional unit assignments map directly to CGRA resource types, and the predecessor/successor information enables efficient scheduling and routing.

## Files Modified/Created

- `llvm/llvm_passes.cpp` - LLVM transformation and extraction passes
- `llvm/CMakeLists.txt` - CMake build configuration
- `llvm/README.md` - LLVM passes documentation
- `Makefile` - Top-level build wrapper
- `src/mapper/tools/irgen.py` - IR generation with automatic pass execution
- `src/mapper/cli.py` - CLI with pass integration
- `.gitignore` - Ignore build artifacts

## Requirements

- LLVM 20
- CMake 3.13.4+
- C++17 compiler
- Python 3.x
