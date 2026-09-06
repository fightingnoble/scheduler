"""Compatibility alias for approach.approach_sched."""

import sys
from approach import approach_sched as _implementation

if __name__ != "__main__":
    sys.modules[__name__] = _implementation
