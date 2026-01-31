"""Entry-point based benchmark discovery."""
import importlib.metadata
from .registry import get_all_benchmarks

ENTRY_POINT_GROUP = "railroad.benchmarks"


def discover_benchmarks():
    """Load all benchmarks registered via entry points.

    Scans all installed packages for entry points in the "railroad.benchmarks"
    group and imports them, which triggers @benchmark decorator registration.

    Returns:
        List of all registered Benchmark objects
    """
    eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    for ep in eps:
        try:
            ep.load()  # Imports module, triggering @benchmark decorators
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load benchmark entry point {ep.name}: {e}")
    return get_all_benchmarks()
