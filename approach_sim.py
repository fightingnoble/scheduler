"""Compatibility entry point for approach.approach_sim."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("approach.approach_sim", run_name="__main__", alter_sys=True)
else:
    import sys
    from approach import approach_sim as _implementation

    sys.modules[__name__] = _implementation
