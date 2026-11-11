"""Whether a ComfyUI prompt queue is empty.

Both the headless worker and the executor health probe use this. A queue is
empty only when the running list and the pending list are both empty. Prompts
submitted by other users count. ``None`` is an empty list, matching the
in-process reader. Any other shape was not observed, so it is not empty.
"""

from __future__ import annotations


def queue_is_empty(running, pending) -> bool:
    return _empty(running) and _empty(pending)


def _empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False
