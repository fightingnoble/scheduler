"""Compatibility alias for approach.approach_def."""

import sys
from approach import approach_def as _implementation

if __name__ != "__main__":
    sys.modules[__name__] = _implementation
