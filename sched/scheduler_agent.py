"""Re-export shell — deprecated runtime-scheduling cluster moved to sched/runtime_legacy/ (B7).
Kept so external `from sched.scheduler_agent import ...` needs zero change."""
from sched.runtime_legacy.scheduler_agent import *  # noqa: F401,F403
