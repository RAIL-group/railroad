"""Convert IPC PDDL/PPDDL problems into railroad planning problems.

See README.md in this package for supported/unsupported PDDL features and the
mapping semantics.

Typical usage::

    from railroad import pddl_converter as pc

    paths = pc.fetch_domain("ipc-2000", "blocks-strips-typed", max_instances=1)
    problem = pc.load_problem(paths.domain_for(paths.instances[0]), paths.instances[0])
    result = pc.solve(problem, seed=0)
"""

from pathlib import Path
from typing import Union

from .converter import CompiledOperator, ConvertedProblem, convert
from .download import COLLECTIONS, FetchedDomain, fetch_domain, list_domains
from .errors import PDDLParseError, UnsupportedPDDLError
from .parser import PDDLDomain, PDDLProblem, parse_domain, parse_problem
from .runner import RunResult, solve

__all__ = [
    "COLLECTIONS",
    "CompiledOperator",
    "ConvertedProblem",
    "FetchedDomain",
    "PDDLDomain",
    "PDDLParseError",
    "PDDLProblem",
    "RunResult",
    "UnsupportedPDDLError",
    "convert",
    "convert_texts",
    "fetch_domain",
    "list_domains",
    "load_problem",
    "parse_domain",
    "parse_problem",
    "solve",
]


def convert_texts(domain_text: str, problem_text: str) -> ConvertedProblem:
    """Parse and convert a domain/problem pair given as PDDL text."""
    return convert(parse_domain(domain_text), parse_problem(problem_text))


def load_problem(
    domain_path: Union[str, Path], problem_path: Union[str, Path]
) -> ConvertedProblem:
    """Parse and convert a domain/problem pair given as file paths."""
    return convert_texts(
        Path(domain_path).read_text(), Path(problem_path).read_text()
    )
