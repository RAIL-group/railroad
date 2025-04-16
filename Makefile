.DEFAULT_GOAL := help
.PHONY: help
help:  ## Show this help message
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: build
build:  ## Build the environment
	@uv sync
	@uv pip install -e ./mrppddl

build-cpp:  ## Specifically build the C++ code
	@uv sync
	@rm -rf ./mrppddl/build && uv pip install --force-reinstall ./mrppddl

clean:  ## Remove build artifacts and the venv
	@rm -rf .venv mrppddl/build mrppddl/src/mrppddl/*cpython*
