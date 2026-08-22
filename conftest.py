# conftest.py
import collections
import pathlib

DATA_DIR = pathlib.Path(__file__).parent / "resources"

# (category, message, filename, lineno) -> count, collected across xdist workers.
_warning_tally: "collections.Counter[tuple[str, str, str, int]]" = collections.Counter()


def pytest_configure(config):
    # This hook runs in both controller and workers, but only
    # the controller process has no 'workerinput' attribute.
    if not hasattr(config, "workerinput"):
        # controller process
        if not (DATA_DIR / ".download_complete").exists():
            from railroad.environment.procthor import ensure_all_resources
            ensure_all_resources()
            (DATA_DIR / ".download_complete").touch()


def pytest_warning_recorded(warning_message, when, nodeid, location):
    """Tally warnings so they survive pytest-richer.

    pytest-richer's reporter implements this hook as a no-op ("Currently we are
    just dropping warnings") and nops out the standard reporter's
    pytest_terminal_summary, so on a tty nothing would report warnings at all.
    This is a separate plugin, so neither patch reaches it.
    """
    _warning_tally[(
        warning_message.category.__name__,
        str(warning_message.message),
        warning_message.filename,
        warning_message.lineno,
    )] += 1


def pytest_unconfigure(config):
    """Print the tally the rich reporter would otherwise swallow.

    Only when its reporter is actually installed -- otherwise the standard
    terminal reporter has already printed a warnings summary and this would
    duplicate it. Runs after the reporter has torn down, so the summary is the
    last thing on screen rather than being overwritten by the progress display.
    """
    if hasattr(config, "workerinput") or not _warning_tally:
        return
    if config.pluginmanager.get_plugin("richer-terminal-reporter") is None:
        return

    total = sum(_warning_tally.values())
    print(f"\n=== warnings summary ({total}) ===")
    for (category, message, filename, lineno), count in _warning_tally.most_common():
        suffix = f" (x{count})" if count > 1 else ""
        print(f"{filename}:{lineno}: {category}: {message}{suffix}")
    print("-- restored by conftest.py; the --rich reporter discards warnings --")
