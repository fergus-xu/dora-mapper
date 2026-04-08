# dora-mapper


#Overview 

This system takes a dot file and generates a mapping onto CGRA architectures. 

# Instructions

### Running the mapper 

Always run the mapper using the Makefile and not directly using the Python scripts. 

``` bash
# From repository root
./scripts/activate
```

This enters into the virtual environment with all Python dependencies 

``` bash 
cd /benchmarks/kernels
```

This enters into the kernels directory 

``` bash
make clean KERNEL=mibench/fft
make map KERNEL=mibench/fft ARCH=../architectures/hycube_mem/II_1 MAX_ITERATIONS=1
```

This runs the mapper

``` bash 
make test KERNEL=mibench/fft ARCH=../architectures/hycube_mem/II_1 
```

This tests the mapper using the verify_mapping.py script

### Running the mapper on different kernels

When testing different kernels, replace with the relative path to the graph_lop.dot directory. When testing different architectures, replace with the relative path to the architecture json files. When using more iterations per II, change the number of max iterations. 

Example: Running the adpcm_dec kernel on the mp_hycube_mem architecture with 3 iterations per II

``` bash
make clean KERNEL=mibench/adpcm_dec
make map KERNEL=mibench/adpcm_dec ARCH=../architectures/mp_hycube_mem/II_1 MAX_ITERATIONS=3
make test KERNEL=mibench/adpcm_dec ARCH=../architectures/mp_hycube_mem/II_1 
```
