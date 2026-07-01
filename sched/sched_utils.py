"""Re-export shell — deprecated runtime-scheduling cluster moved to sched/runtime_legacy/ (B7).
Kept so external `from sched.sched_utils import ...` needs zero change."""
from sched.runtime_legacy.sched_utils import *  # noqa: F401,F403
