from .procthor_new import ThorInterface  # noqa: F401
from . import simulators  # noqa: F401
from . import plotting  # noqa: F401
from . import utils  # noqa: F401
from . import scenegraph  # noqa: F401
from . import procthor  # noqa: F401

import os

from .resources import ensure_procthor_10k 

# Allow opt-out, e.g. for docs builds or constrained CI:
#   PROCTHOR_AUTO_DOWNLOAD=0  → skip automatic downloads.
if os.environ.get("PROCTHOR_AUTO_DOWNLOAD", "1") != "0":
    ensure_procthor_10k()
