"""Errors raised while parsing or converting PDDL problems."""


class PDDLParseError(Exception):
    """The input is not syntactically valid PDDL (or is truncated)."""


class UnsupportedPDDLError(Exception):
    """The input uses a PDDL feature railroad cannot represent.

    Args:
        reason: machine-readable slug (e.g. ``"durative-actions"``,
            ``"conditional-effects"``, ``"metric:maximize reward"``) used by
            the compatibility scanner to aggregate failures.
        message: optional human-readable elaboration.
    """

    def __init__(self, reason: str, message: str = ""):
        self.reason = reason
        super().__init__(f"{reason}: {message}" if message else reason)
