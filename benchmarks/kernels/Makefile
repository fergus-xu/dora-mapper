# ──────────────────────────────────────────────────────────
# Dora Mapper — Kernel Benchmark Makefile
# ──────────────────────────────────────────────────────────
#
# Usage:
#   make map KERNEL=sum                      # map sum/graph_loop.dot
#   make map KERNEL=sum DOT=graph_loop.dot   # explicit .dot file
#   make test KERNEL=sum                     # verify the mapping
#   make clean KERNEL=sum                    # remove results
#
# All outputs go to: <kernel>/results/
# ──────────────────────────────────────────────────────────

# ── User-configurable ─────────────────────────────────────
KERNEL       ?= sum
DOT          ?= graph_loop.dot
# ARCH is the relative path from benchmarks/kernels to the architecture directory
ARCH         ?= ../architectures/vanilla_hycube/II_1
MAX_ITERATIONS ?= 4
MAX_II         ?= 32
PYTHON         ?= python3

# ── Paths (relative to repo root) ─────────────────────────
SHELL        := /bin/bash
.ONESHELL:

ROOT         := $(shell git -C $(dir $(lastword $(MAKEFILE_LIST))) rev-parse --show-toplevel)
# Use realpath to resolve ARCH relative to this Makefile's directory
ARCH_DIR     := $(shell realpath -m $(CURDIR)/$(ARCH))
MRRG         := $(ARCH_DIR)/mrrg.json
COMPILER_ARCH:= $(ARCH_DIR)/compiler_arch.json
MAPPER       := $(ROOT)/src/mapper/cli.py
VERIFIER     := $(ROOT)/benchmarks/kernels/verify_mapping.py

# ── Derived ───────────────────────────────────────────────
KERNEL_DIR   := $(CURDIR)/$(KERNEL)
DOT_FILE     := $(KERNEL_DIR)/$(DOT)
RESULTS_DIR  := $(KERNEL_DIR)/results
RESULTS_LOG   := $(RESULTS_DIR)/results.log

# Extract architecture name relative to benchmarks/architectures
ARCH_NAME    := $(shell realpath -m --relative-to=$(ROOT)/benchmarks/architectures $(ARCH_DIR))
ARCH_ID      := $(subst /,,$(ARCH_NAME))
KERNEL_JSON  := $(RESULTS_DIR)/$(KERNEL)_$(ARCH_ID).json
KERNEL_FASM  := $(RESULTS_DIR)/$(KERNEL)_$(ARCH_ID).fasm

# ══════════════════════════════════════════════════════════
.PHONY: map test clean help

help: ## Show this help
	@echo ""
	@echo "Dora Mapper — Kernel Benchmark Targets"
	@echo "───────────────────────────────────────"
	@echo "  make map   KERNEL=<name> [DOT=<file>] [ARCH=<path>]  Map a kernel"
	@echo "  make test  KERNEL=<name>                 Verify the mapping"
	@echo "  make clean KERNEL=<name>                 Delete results"
	@echo ""
	@echo "Available kernels:"
	@ls -d */ | sed 's|/||' | sed 's/^/  /'
	@echo ""

# ── map ───────────────────────────────────────────────────
map: $(KERNEL_JSON) ## Run the mapper

$(KERNEL_JSON): $(DOT_FILE)
	@echo "═══════════════════════════════════════════"
	@echo " Mapping kernel: $(KERNEL)"
	@echo " DFG:            $(DOT_FILE)"
	@echo " Architecture:   $(ARCH_NAME)"
	@echo "═══════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)
	set -o pipefail
	PYTHONPATH="$(ROOT)/src:$${PYTHONPATH}" PYTHONUNBUFFERED=1 $(PYTHON) -u $(MAPPER) map \
		$(DOT_FILE) \
		$(MRRG) \
		--compiler-arch $(COMPILER_ARCH) \
		--max-iterations $(MAX_ITERATIONS) \
		--max-ii $(MAX_II) \
		--output $(KERNEL_JSON) \
		--fasm-output $(KERNEL_FASM) \
		| tee $(RESULTS_LOG) && \
	(echo "" && echo "✓ Outputs written to $(RESULTS_DIR)/")

# ── test ──────────────────────────────────────────────────
test: ## Verify mapping correctness
	@if [ ! -f "$(KERNEL_JSON)" ]; then \
		echo "✗ No mapping found at $(KERNEL_JSON). Run 'make map KERNEL=$(KERNEL)' first."; \
		exit 1; \
	fi
	@echo "═══════════════════════════════════════════"
	@echo " Verifying mapping: $(KERNEL)"
	@echo "═══════════════════════════════════════════"
	PYTHONPATH="$(ROOT)/src:$${PYTHONPATH}" $(PYTHON) $(VERIFIER) \
		$(KERNEL_JSON) \
		$(DOT_FILE) \
		$(MRRG) \
		--compiler-arch $(COMPILER_ARCH)

# ── clean ─────────────────────────────────────────────────
clean: ## Remove results directory
	@if [ -d "$(RESULTS_DIR)" ]; then \
		rm -rf $(RESULTS_DIR); \
		echo "✓ Removed $(RESULTS_DIR)"; \
	else \
		echo "Nothing to clean for $(KERNEL)."; \
	fi
