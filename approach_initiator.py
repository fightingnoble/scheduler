"""Compatibility alias for approach.approach_initiator."""

import sys
from approach import approach_initiator as _implementation

if __name__ != "__main__":
    sys.modules[__name__] = _implementation
