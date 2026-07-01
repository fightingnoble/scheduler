"""Re-export shell — deprecated runtime-scheduling cluster moved to sched/runtime_legacy/ (B7).
Kept so external `from sched.state_trans import ...` needs zero change."""
from sched.runtime_legacy.state_trans import *  # noqa: F401,F403
