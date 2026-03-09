#!/usr/bin/env bash
set -e  # Exit on error
trap 'rm -f llvm.sh' EXIT

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
DEV_MODE=0
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dev)
            DEV_MODE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-d|--dev] [--llvm-version N]"
            echo "  -d, --dev            Install with development dependencies"
            echo "  --llvm-version N     Install LLVM/Clang major version N using apt.llvm.org llvm.sh (e.g. 18)"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Mapper Installation Script"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/config.sh"

cd "$PROJECT_ROOT"

# Check for Python 3.12+
echo -e "${YELLOW}Checking Python version...${NC}"
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD=python3.12
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 12 ]; then
        PYTHON_CMD=python3
    else
        echo -e "${RED}Error: Python 3.12+ required, found Python $PYTHON_VERSION${NC}"
        echo "Please install Python 3.12 or higher"
        exit 1
    fi
else
    echo -e "${RED}Error: Python 3 not found${NC}"
    echo "Please install Python 3.12 or higher"
    exit 1
fi

echo -e "${GREEN}✓ Found $($PYTHON_CMD --version)${NC}"
echo ""

# Check for uv, install if not present
echo -e "${YELLOW}Checking for uv package manager...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv not found, installing...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    export PATH="$HOME/.local/bin:$PATH"

    # Source the cargo env to make uv available in current shell
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi

    if ! command -v uv &> /dev/null; then
        echo -e "${RED}Error: Failed to install uv${NC}"
        echo "Please install uv manually: https://github.com/astral-sh/uv"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Found $(uv --version)${NC}"
echo ""

# ------------------------------------------------------------
# LLVM/Clang install (apt.llvm.org llvm.sh) — EXACT METHOD
# ------------------------------------------------------------
echo -e "${YELLOW}Checking LLVM/Clang (apt.llvm.org) ...${NC}"
if command -v "clang-$LLVM_VERSION" &> /dev/null; then
    echo -e "${GREEN}✓ Found clang-$LLVM_VERSION${NC}"
    echo "  $(clang-$LLVM_VERSION --version | head -n 1)"
else
    echo -e "${YELLOW}clang-$LLVM_VERSION not found; installing LLVM $LLVM_VERSION (all packages)...${NC}"

    # EXACT code you pasted:
    wget https://apt.llvm.org/llvm.sh
    chmod +x llvm.sh
    sudo ./llvm.sh "$LLVM_VERSION" all

    # Cleanup
    rm -f llvm.sh

    # Verify
    if ! command -v "clang-$LLVM_VERSION" &> /dev/null; then
        echo -e "${RED}Error: LLVM install completed but clang-$LLVM_VERSION is still not on PATH${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Installed clang-$LLVM_VERSION${NC}"
    echo "  $(clang-$LLVM_VERSION --version | head -n 1)"
fi
echo ""

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo "Virtual environment already exists, skipping creation"
else
    uv venv --python "$PYTHON_CMD"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${YELLOW}Installing mapper...${NC}"
source .venv/bin/activate

# Install the package
if [ $DEV_MODE -eq 1 ]; then
    echo "Installing with development dependencies..."
    uv pip install -e ".[dev,viz]"
else
    echo "Installing core dependencies..."
    uv pip install -e .
fi

echo -e "${GREEN}✓ mapper installed successfully${NC}"
echo ""

# Build LLVM passes
echo -e "${YELLOW}Building LLVM passes...${NC}"
if command -v cmake &> /dev/null; then
    if make passes; then
        echo -e "${GREEN}✓ LLVM passes built successfully${NC}"
    else
        echo -e "${YELLOW}Warning: Failed to build LLVM passes${NC}"
        echo "You can build them later with: make passes"
    fi
else
    echo -e "${YELLOW}Warning: cmake not found, skipping LLVM pass build${NC}"
    echo "Install cmake and run 'make passes' to build LLVM passes"
fi
echo ""

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
if python -c "import mapper; print('mapper import OK:', mapper.__file__)" 2>/dev/null; then
    echo -e "${GREEN}✓ Installation verified${NC}"
else
    echo -e "${RED}✗ Installation verification failed${NC}"
    python -c "import sys; print(sys.executable); import site; print(site.getsitepackages())" || true
    exit 1
fi
echo ""

# Verify LLVM toolchain availability (versioned tools)
echo -e "${YELLOW}Verifying LLVM tools...${NC}"
clang-$LLVM_VERSION --version >/dev/null
if command -v "opt-$LLVM_VERSION" &> /dev/null; then
    opt-$LLVM_VERSION --version >/dev/null
fi
if command -v "llvm-dis-$LLVM_VERSION" &> /dev/null; then
    llvm-dis-$LLVM_VERSION --version >/dev/null
fi
echo -e "${GREEN}✓ LLVM $LLVM_VERSION tools accessible (at least clang-$LLVM_VERSION)${NC}"
echo ""

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To enter the mapper development environment, run:"
echo -e "  ${GREEN}./scripts/activate${NC}"
echo ""
echo "LLVM notes:"
echo "  - Installed via apt.llvm.org into this Ubuntu/WSL distro"
echo "  - Use versioned binaries for determinism:"
echo -e "    ${GREEN}clang-$LLVM_VERSION${NC}  (and opt-$LLVM_VERSION, llvm-dis-$LLVM_VERSION, etc.)"
echo "  - LLVM passes built in llvm/llvm_passes.so"
echo "  - Run 'make passes' to rebuild if needed"
echo ""
if [ $DEV_MODE -eq 1 ]; then
    echo "Development tools installed:"
    echo "  - pytest: Run tests with 'pytest'"
    echo "  - black: Format code with 'black src/ tests/'"
    echo "  - mypy: Type check with 'mypy src/'"
    echo "  - flake8: Lint code with 'flake8 src/ tests/'"
    echo ""
fi
echo "Happy mapping!"
