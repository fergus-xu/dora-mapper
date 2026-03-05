# LLVM Passes for DORA Mapper

This directory contains LLVM passes for extracting dataflow graphs from C code.

## Building

### Using Make (recommended)
From the repository root:
```bash
make passes
```

### Using CMake directly
```bash
cd llvm
cmake -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm .
make
```

This will generate `llvm_passes.so` in this directory.

## Passes

### 1. LowerGEPPass (`lower-gep`)
Lowers `getelementptr` instructions to explicit arithmetic operations (add, mul, ptrtoint, inttoptr).

### 2. DFGExtractionPass (`extract-dfg`)
Extracts a JSON-formatted dataflow graph from code regions marked with:
- `__kernel_region_start()` - marks the beginning of the region
- `__kernel_region_end()` - marks the end of the region

The pass outputs `<function_name>_dfg.json` containing:
- **nodes**: Instructions with operation type, bitwidth, and constants
- **edges**: Dataflow dependencies between instructions

## Usage

### Through Python CLI (automatic)
```bash
python src/mapper/cli.py irgen benchmarks/test.c -I include
```

This automatically runs the DFG extraction pass and generates JSON output.

### Manual opt invocation
```bash
opt-20 -load-pass-plugin=llvm/llvm_passes.so \
  -passes=extract-dfg \
  -disable-output benchmarks/test.ll
```

## Requirements

- LLVM 20
- CMake 3.13.4+
- C++17 compiler
