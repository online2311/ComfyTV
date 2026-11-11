"""Read-only MCP view of the shot x stage scheduler.

Enqueue and cancel stay on the HTTP API. This tool does not submit a
prompt and does not stop a host.
"""

from __future__ import annotations

from ..scheduler import scheduler_status
from ._shared import _no_args_schema


async def _scheduler(_args: dict) -> dict:
    return scheduler_status()


TOOLS: dict[str, dict] = {
    "scheduler": {
        "description": (
            "Read the shot x stage scheduler: the queue of units, current "
            "assignments, shot affinity (project, shot, gpu_class, preferred "
            "executor), and per-executor load. A unit waits when no executor "
            "with accepts_work has its gpu class; it is not moved to a "
            "different class. This tool does not dispatch, cancel, or stop "
            "a host. With only the seeded local executor, headless runs stay "
            "on the in-process queue."
        ),
        "inputSchema": _no_args_schema(),
        "handler": _scheduler,
    },
}
