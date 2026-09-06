"""Compatibility alias for approach.approach_Eq."""

import sys
from approach import approach_Eq as _implementation

if __name__ != "__main__":
    sys.modules[__name__] = _implementation
