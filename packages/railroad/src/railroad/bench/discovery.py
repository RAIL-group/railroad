"""Entry-point based benchmark discovery."""
import importlib.metadata
import importlib.util
from pathlib import Path
from .registry import get_all_benchmarks

ENTRY_POINT_GROUP = "railroad.benchmarks"


def load_benchmark_files(paths: list[str | Path]) -> None:
    """Import Python files to trigger @benchmark decorator registration.

    Args:
        paths: List of file paths to Python files containing benchmarks
    """
    for path in paths:
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")
        if not path.suffix == ".py":
            raise ValueError(f"Benchmark file must be a .py file: {path}")

        # Create a unique module name based on the file path
        module_name = f"_benchmark_file_{path.stem}_{id(path)}"

        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load benchmark file: {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def discover_benchmarks(include_files: list[str | Path] | None = None):
    """Load all benchmarks registered via entry points and included files.

    Scans all installed packages for entry points in the "railroad.benchmarks"
    group and imports them, which triggers @benchmark decorator registration.
    Also loads any explicitly included benchmark files.

    Args:
        include_files: Optional list of paths to Python files containing benchmarks

    Returns:
        List of all registered Benchmark objects
    """
    # Load explicitly included files first
    if include_files:
        load_benchmark_files(include_files)

    # Then load entry points
    eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    for ep in eps:
        try:
            ep.load()  # Imports module, triggering @benchmark decorators
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load benchmark entry point {ep.name}: {e}")
    return get_all_benchmarks()
