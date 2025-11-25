.DEFAULT_GOAL := help
.PHONY: help

help:  ## Show this help message
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: build
build:  ## Build the environment
	@uv sync
	@uv pip install -r requirements.txt
	@uv pip install -e "mrppddl @ ./mrppddl" "mrppddl_env @ ./mrppddl_env"
	@uv pip install -e ./procthor -e ./common -e ./gridmap -e ./environments

build-cpp:  ## Specifically build the C++ code
	@uv sync
	@rm -rf ./mrppddl/build && uv pip install --force-reinstall ./mrppddl

clean:  ## Remove build artifacts and the venv
	@rm -rf .venv mrppddl/build mrppddl/src/mrppddl/*cpython*

typecheck:  ## Runs the typechecker via pyright
	@uv run pyright -w mrppddl/src/mrppddl mrppddl/tests

test:
	@uv run pytest -vsk $(PYTEST_FILTER)

include procthor/Makefile.mk
