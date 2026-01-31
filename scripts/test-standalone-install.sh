#!/usr/bin/env bash
# Test that a package can be installed standalone from its subdirectory
# Usage: ./scripts/test-standalone-install.sh [package_name]
# Example: ./scripts/test-standalone-install.sh railroad

set -euo pipefail

PACKAGE_NAME="${1:-railroad}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_DIR="$REPO_ROOT/packages/$PACKAGE_NAME"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Verify package exists
if [[ ! -d "$PACKAGE_DIR" ]]; then
    log_error "Package directory not found: $PACKAGE_DIR"
    exit 1
fi

# Create temporary directory for isolated test
TEMP_DIR=$(mktemp -d)
log_info "Created temp directory: $TEMP_DIR"

# Cleanup on exit
cleanup() {
    log_info "Cleaning up temp directory..."
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Copy package to temp directory (simulating a standalone checkout)
log_info "Copying $PACKAGE_NAME package to isolated directory..."
cp -r "$PACKAGE_DIR"/* "$TEMP_DIR/"

cd "$TEMP_DIR"

# Create isolated virtual environment
log_info "Creating isolated virtual environment..."
uv venv .venv

# Install the package
log_info "Installing $PACKAGE_NAME package (standalone)..."
uv pip install --python .venv/bin/python .

# Install test dependencies
log_info "Installing test dependencies..."
uv pip install --python .venv/bin/python pytest pytest-timeout

# Run tests
log_info "Running tests..."
.venv/bin/python -m pytest -v tests/

# Run an example to verify all runtime dependencies are installed
log_info "Running example to verify dependencies..."
.venv/bin/railroad example clear-table

# Test benchmarks discovery (dry-run to avoid running actual benchmarks)
log_info "Testing benchmark discovery..."
.venv/bin/railroad benchmarks run --dry-run

log_info "Standalone install test completed successfully!"
