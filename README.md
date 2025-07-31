
# Quickstart

Running `make` will output the available targets. 

If not using C++, `make build` is sufficient to start running code. Running `make build` again is *not* necessary, since the `mrppddl` package is installed in "edit" mode and will auto detect changes.

If C++ is being used, `make build-cpp` is required to build those bindings and then use them. If the C++ code is changed, `make build-cpp` must be rerun.

Everything else is run via `uv run`: e.g., `uv run mrppddl_astar.py` will run the relevant script.

# TODO Items

- [ ] I can use the FF heuristic (or similar) to determine which actions I never execute and prune those from the set of available actions.
- [ ] I still need to look over the other hashing functions to avoid using a "bare" XOR, which makes the likelihood of a collision problematically high.
