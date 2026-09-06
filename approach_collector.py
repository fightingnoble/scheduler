"""Compatibility entry point for approach.approach_collector."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("approach.approach_collector", run_name="__main__", alter_sys=True)
else:
    import sys
    from approach import approach_collector as _implementation

    sys.modules[__name__] = _implementation
