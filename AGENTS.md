# dora-mapper


#Overview 

This system takes a dot file and generates a mapping onto CGRA architectures. 

# Instructions

### Running the mapper 

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
make clean
make map KERNEL=fft ARCH=../architectures/hycube_mem/II_1 MAX_ITERATIONS=1
```

This runs the mapper

``` bash 
make test KERNEL=mibench/fft ARCH=../architectures/hycube_mem/II_1 
```

This tests the mapper using the verify_mapping.py script

When testing different kernels, replace with the relative path to the graph_lop.dot directory. When testing different architectures, replace with the relative path to the architecture json files. 