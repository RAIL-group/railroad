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

.PHONY: clean clean-cpp clean-venv
clean-cpp:  ## Remove C++-specific build artifacts
	@rm -rf mrppddl/build mrppddl/src/mrppddl/*cpython*

clean-venv:  ## Remove virtualenv directory 'venv'
	@rm -rf .venv

clean: clean-cpp clean-venv  ## Remove build artifacts and the venv
	@rm -rf uv.lock

.PHONY: typecheck test
typecheck:  ## Runs the typechecker via pyright
	@uv run pyright -w mrppddl/src/mrppddl mrppddl/tests

test: download-procthor-all  ## Runs tests (limit scope via PYTEST_FILTER=filter)
	@uv run pytest -vk $(PYTEST_FILTER)

include procthor/Makefile.mk
