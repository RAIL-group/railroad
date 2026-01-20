.DEFAULT_GOAL := help
.PHONY: help
PYTEST_FILTER ?= test

help:  ## Show this help message
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: build
build:
	@echo "IMPORTANT UPDATE:"
	@echo "  Build is no longer needed!"
	@echo "  'uv sync' or 'uv run ...' should trigger all building."
	@echo "  If 'make clean' has been done, 'make rebuild-cpp' may be necessary, but the system should prompt that as needed.\n"
	@exit 1

rebuild-cpp: clean-cpp  ## Rebuild C++ Modules
	@echo "Rebuilding C++"
	@uv sync --reinstall-package mrppddl

.PHONY: clean clean-cpp clean-python clean-venv
clean-cpp:  ## Remove C++-specific build artifacts
	@rm -rf packages/mrppddl/build packages/mrppddl/src/mrppddl/*cpython*
	@touch packages/mrppddl/src/mrppddl/_bindings.cpp  # Trigger rebuild when needed

clean-python:  ## Remove Python temporary artifacts
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .mypy_cache .pytype .coverage htmlcov .eggs dist build

clean-venv:  ## Remove virtualenv directory 'venv'
	@rm -rf .venv

clean: clean-cpp clean-python clean-venv  ## Remove build artifacts, Python cache, and the venv
	@find . -type f -name "*.DS_Store" -delete 2>/dev/null || true
	@rm -rf uv.lock

.PHONY: typecheck test
typecheck:  ## Runs the typechecker via pyright
	@uv run ty check packages

test:  ## Runs tests (limit scope via PYTEST_FILTER=filter)
	@uv run pytest -vk $(PYTEST_FILTER)
