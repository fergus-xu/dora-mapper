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

# ── Paths (relative to repo root) ─────────────────────────
SHELL        := /bin/bash
.ONESHELL:

ROOT         := $(shell git -C $(dir $(lastword $(MAKEFILE_LIST))) rev-parse --show-toplevel)
ARCH_DIR     := $(ROOT)/benchmarks/architectures/dice
MRRG         := $(ARCH_DIR)/mrrg.json
COMPILER_ARCH:= $(ARCH_DIR)/compiler_arch.json
MAPPER       := $(ROOT)/src/mapper/cli.py
VERIFIER     := $(ROOT)/benchmarks/kernels/verify_mapping.py
ACTIVATE     := source $(ROOT)/scripts/config.sh && source $(ROOT)/.venv/bin/activate

# ── User-configurable ─────────────────────────────────────
KERNEL       ?= sum
DOT          ?= graph_loop.dot

# ── Derived ───────────────────────────────────────────────
KERNEL_DIR   := $(CURDIR)/$(KERNEL)
DOT_FILE     := $(KERNEL_DIR)/$(DOT)
RESULTS_DIR  := $(KERNEL_DIR)/results
RESULTS_JSON := $(RESULTS_DIR)/results.json
KERNEL_JSON  := $(RESULTS_DIR)/$(KERNEL)_dice.json

# ══════════════════════════════════════════════════════════
.PHONY: map test clean help

help: ## Show this help
	@echo ""
	@echo "Dora Mapper — Kernel Benchmark Targets"
	@echo "───────────────────────────────────────"
	@echo "  make map   KERNEL=<name> [DOT=<file>]   Map a kernel onto dice"
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
	@echo " Architecture:   dice"
	@echo "═══════════════════════════════════════════"
	@mkdir -p $(RESULTS_DIR)
	$(ACTIVATE) && python $(MAPPER) map \
		$(DOT_FILE) \
		$(MRRG) \
		--compiler-arch $(COMPILER_ARCH) \
		--output $(KERNEL_JSON) \
		| tee $(RESULTS_JSON)
	@echo ""
	@echo "✓ Outputs written to $(RESULTS_DIR)/"

# ── test ──────────────────────────────────────────────────
test: ## Verify mapping correctness
	@if [ ! -f "$(KERNEL_JSON)" ]; then \
		echo "✗ No mapping found at $(KERNEL_JSON). Run 'make map KERNEL=$(KERNEL)' first."; \
		exit 1; \
	fi
	@echo "═══════════════════════════════════════════"
	@echo " Verifying mapping: $(KERNEL)"
	@echo "═══════════════════════════════════════════"
	$(ACTIVATE) && python $(VERIFIER) \
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
