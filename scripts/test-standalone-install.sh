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
cp -r "$PACKAGE_DIR" "$TEMP_DIR/$PACKAGE_NAME"

cd "$TEMP_DIR"

# Create a wrapper project that depends on the package (replicates monorepo setup)
log_info "Creating wrapper project..."
uv init --name test-standalone --no-readme

# Run an example to verify all runtime dependencies are installed
log_info "Intstall base package and running example..."
uv add "./$PACKAGE_NAME"
uv run railroad example clear-table

# Install the package with test dependencies + run tests
log_info "Installing $PACKAGE_NAME package with test dependencies and running tests..."
uv add "./$PACKAGE_NAME[test]"
uv run pytest "$PACKAGE_NAME/tests"

# Test benchmarks discovery (dry-run to avoid running actual benchmarks)
log_info "Testing benchmark discovery..."
uv add "./$PACKAGE_NAME[bench]"
uv run railroad benchmarks run --dry-run

log_info "Standalone install test completed successfully!"
