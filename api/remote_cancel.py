"""Cancel one owned prompt on a ComfyUI host.

This follows the decision in ``feature/remote-cancel-safe`` (PR #10):

- a prompt still in ``queue_pending`` is removed with POST /queue
  ``{"delete": [prompt_id]}``
- a prompt already in ``queue_running`` is left running
  (``cancel_requested``); its files are not downloaded
- an unreadable queue gets one pending-delete attempt, then the same
  leave-running result

``runners/remote_comfy.py`` on this branch still posts the host stop route
for the older per-server runner. That file is the unmerged PR #10 change.
This helper is what the scheduler calls so the two paths agree on the
return values ``deleted``, ``cancel_requested``, and ``absent``. It does
not request a host stop or a memory free.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

_log = logging.getLogger(__name__)

_STOPPED = frozenset({"interrupt", "free"})


def assert_not_stopped(url: str) -> None:
    parts = [part.lower() for part in urlsplit(url).path.split("/") if part]
    if _STOPPED.intersection(parts):
        raise RuntimeError("scheduler will not stop a host")


def _ids(rows) -> set[str]:
    found: set[str] = set()
    if not isinstance(rows, list):
        return found
    for item in rows:
        if isinstance(item, (list, tuple)) and len(item) > 1 and item[1]:
            found.add(str(item[1]))
    return found


async def _queue_state(transport, base_url: str, prompt_id: str) -> str:
    """pending, running, absent, or unknown."""
    url = base_url.rstrip("/") + "/queue"
    assert_not_stopped(url)
    try:
        status, payload = await transport("GET", url)
    except Exception:
        return "unknown"
    if status != 200 or not isinstance(payload, dict):
        return "unknown"
    pid = str(prompt_id)
    if pid in _ids(payload.get("queue_pending")):
        return "pending"
    if pid in _ids(payload.get("queue_running")):
        return "running"
    return "absent"


async def _delete_pending(transport, base_url: str, prompt_id: str) -> None:
    url = base_url.rstrip("/") + "/queue"
    assert_not_stopped(url)
    try:
        status, _body = await transport("POST", url, {"delete": [str(prompt_id)]})
    except Exception:
        _log.warning("[ComfyTV/scheduler] pending delete failed")
        return
    if status != 200:
        _log.warning("[ComfyTV/scheduler] pending delete returned HTTP %s", status)


async def cancel_owned_prompt(transport, base_url: str, prompt_id: str) -> str:
    """Return deleted, cancel_requested, or absent.

    ``transport(method, url, json_body=None)`` returns ``(status, body)``.
    """
    if not prompt_id:
        return "absent"
    try:
        state = await _queue_state(transport, base_url, prompt_id)
        if state == "pending":
            await _delete_pending(transport, base_url, prompt_id)
            # A follow-up read that still says pending is the queue view
            # lagging the delete. Only a prompt that has since started
            # must be left running.
            if await _queue_state(transport, base_url, prompt_id) == "running":
                return "cancel_requested"
            return "deleted"
        if state == "running":
            return "cancel_requested"
        if state == "unknown":
            await _delete_pending(transport, base_url, prompt_id)
            return "cancel_requested"
        return "absent"
    except Exception:
        _log.warning("[ComfyTV/scheduler] cancel on executor failed")
        return "cancel_requested"
