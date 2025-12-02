from .procthor import ThorInterface  # noqa: F401
from . import simulators  # noqa: F401
from . import plotting  # noqa: F401
from . import utils  # noqa: F401
from . import scenegraph  # noqa: F401
from . import procthor  # noqa: F401

import os

from .resources import ensure_all_resources

if os.environ.get("PROCTHOR_AUTO_DOWNLOAD", "1") != "0":
    ensure_all_resources()
