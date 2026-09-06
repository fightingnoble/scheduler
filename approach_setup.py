"""Compatibility alias for approach.approach_setup."""

import sys
from approach import approach_setup as _implementation

if __name__ != "__main__":
    sys.modules[__name__] = _implementation
