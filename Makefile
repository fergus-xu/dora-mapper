.PHONY: all clean passes

all: passes

passes:
	@echo "Building LLVM passes..."
	@cd llvm && \
	if [ ! -f Makefile ]; then \
		cmake -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm . ; \
	fi && \
	make

clean:
	@echo "Cleaning build artifacts..."
	@cd llvm && \
	rm -rf CMakeCache.txt CMakeFiles/ cmake_install.cmake Makefile *.so
