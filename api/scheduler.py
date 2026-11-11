"""Dispatch shot x stage work to registered executors.

With only the seeded local executor, headless jobs stay on the in-process
queue. Any other executor makes the scheduler the sender: it submits one
prompt per executor, and only after a fresh read shows that executor's
queue empty, including prompts from other clients. A required ``gpu_class``
is never replaced with a different class. The same shot keeps the class of
its first placement and prefers that executor.

Timeouts use the remote-connect, remote-read, remote-poll, and remote-job
settings. ``scheduler-max-retries`` is how many submit attempts an
unreachable host may consume before the unit errors. A lost response does
not send the prompt again when the attempt id is already on that host.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlencode, urlsplit

import aiohttp
import folder_paths
from aiohttp import web

from .. import storage
from .queue_state import queue_is_empty
from .remote_cancel import assert_not_stopped, cancel_owned_prompt
from ._common import routes

_log = logging.getLogger(__name__)

_RESULT_SUBFOLDER = "comfytv/remote"
_FILE_KEYS = ("images", "audio", "videos", "gifs", "video", "3d")
_ACTIVE = ("submitting", "running", "cancel-requested")
_QUEUE_STATUSES = ("queued", "submitting", "running", "cancel-requested")


class _Unreachable(Exception):
    pass


class _Deadline(_Unreachable):
    """The unit's job deadline ran out while waiting on one host call."""


def scheduler_limits() -> dict[str, Any]:
    """Connect, read, poll, job timeout, and max submit attempts.

    The first four keys match ``remote_timeouts`` on the remote-cancel
    branch so both sides share one setting once that branch lands.
    """

    def read(key: str, default: int, lo: int) -> int:
        try:
            value = int(storage.get_setting(key))
        except Exception:
            value = default
        if value < lo:
            return default
        return value

    read_timeout = read("remote-read-timeout-s", 0, 0)
    return {
        "connect": float(read("remote-connect-timeout-s", 10, 1)),
        "read": None if read_timeout == 0 else float(read_timeout),
        "poll": float(read("remote-poll-interval-s", 2, 1)),
        "job": float(read("remote-job-timeout-s", 3600, 1)),
        "max_attempts": int(read("scheduler-max-retries", 3, 1)),
    }


def _client_timeout() -> aiohttp.ClientTimeout:
    limits = scheduler_limits()
    return aiohttp.ClientTimeout(
        total=None,
        sock_connect=limits["connect"],
        sock_read=limits["read"],
    )


def _loads(raw, fallback):
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def _safe_detail(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    if "://" in text or "token" in lowered or "authorization" in lowered:
        return ""
    return text[:500]


def _parse_body(raw: bytes):
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError):
        return None


def _is_view(url: str) -> bool:
    return urlsplit_path(url).endswith("/view")


def urlsplit_path(url: str) -> str:
    return urlsplit(url).path.rstrip("/")


def _unreachable(exc: BaseException) -> bool:
    if isinstance(exc, _Unreachable):
        return True
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError)):
        return True
    return isinstance(exc, aiohttp.ClientError)


async def _http(session: aiohttp.ClientSession, method: str, url: str, json_body=None):
    assert_not_stopped(url)
    try:
        if method == "GET":
            async with session.get(url) as resp:
                raw = await resp.read()
                if _is_view(url):
                    return resp.status, raw
                return resp.status, _parse_body(raw)
        if method == "POST":
            async with session.post(url, json=json_body) as resp:
                raw = await resp.read()
                return resp.status, _parse_body(raw)
    except Exception as exc:
        if _unreachable(exc):
            raise _Unreachable() from None
        raise
    raise RuntimeError("unsupported method")


def _headless_api():
    # headless_queue imports this module at the bottom of that file.
    from .headless_queue import ComfyQueueClient, QueueBusy
    return ComfyQueueClient, QueueBusy


def _view_url(filename: str, subfolder: str, type_: str) -> str:
    return "/view?" + urlencode({
        "filename": filename,
        "subfolder": subfolder,
        "type": type_,
    })


def _safe_component(text: str) -> str:
    cleaned = re.sub(r"[^\w\-.]+", "_", text or "").strip("_")
    return cleaned or "executor"


def _required_class(unit: dict, affinity: Optional[dict]) -> tuple[Optional[str], Optional[str]]:
    requested = unit.get("gpu_class") or ""
    if affinity is None:
        if requested:
            return requested, None
        return None, None
    locked = affinity.get("gpu_class") or ""
    if requested and requested != locked:
        return None, "gpu_class"
    return locked, None


def _candidates(executors, required, affinity, avoid_id, loads) -> list[dict]:
    preferred = (affinity or {}).get("executor_id") or ""

    def eligible(row: dict) -> bool:
        if not row.get("accepts_work"):
            return False
        if required is not None and (row.get("gpu_class") or "") != required:
            return False
        cap = int(row.get("max_concurrency") or 1)
        if int(loads.get(row["id"], 0)) >= cap:
            return False
        return True

    rows = [row for row in executors if eligible(row)]
    if avoid_id and any(row["id"] != avoid_id for row in rows):
        rows = [row for row in rows if row["id"] != avoid_id]
    # No class yet: keep the seeded local executor when it can take the unit.
    rows.sort(key=lambda row: (
        0 if row["id"] == preferred else 1,
        0 if required is None and row.get("is_default") else 1,
        row.get("name") or "",
    ))
    return rows


def _wait_reason(executors, required, loads) -> str:
    if required is None:
        matching = list(executors)
    else:
        matching = [row for row in executors if (row.get("gpu_class") or "") == required]
    if not matching:
        return "gpu_class"
    for row in matching:
        cap = int(row.get("max_concurrency") or 1)
        if row.get("accepts_work") and int(loads.get(row["id"], 0)) < cap:
            return "busy"
        if int(loads.get(row["id"], 0)) >= cap or row.get("admission_reason") == "at_capacity":
            return "busy"
    if any(row.get("accepts_work") for row in matching):
        return "busy"
    return "gpu_class"


def _match_rows(rows, client_id: Optional[str], prompt_id: Optional[str]) -> Optional[str]:
    if not isinstance(rows, list):
        return None
    for item in rows:
        if not isinstance(item, (list, tuple)) or len(item) < 2 or not item[1]:
            continue
        pid = str(item[1])
        extra = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
        if prompt_id and pid == str(prompt_id):
            return pid
        if client_id and extra.get("client_id") == client_id:
            return pid
    return None


def _history_client(entry: dict) -> Optional[str]:
    prompt = entry.get("prompt")
    if isinstance(prompt, (list, tuple)) and len(prompt) > 3 and isinstance(prompt[3], dict):
        cid = prompt[3].get("client_id")
        if isinstance(cid, str):
            return cid
    extra = entry.get("extra_data")
    if isinstance(extra, dict) and isinstance(extra.get("client_id"), str):
        return extra["client_id"]
    return None


def _history_done(entry) -> bool:
    if not isinstance(entry, dict):
        return False
    status = entry.get("status") or {}
    if status.get("completed"):
        return True
    return status.get("status_str") in ("success", "error")


def _history_failed(entry: dict) -> bool:
    return (entry.get("status") or {}).get("status_str") == "error"


def _history_error(entry: dict) -> str:
    for msg in (entry.get("status") or {}).get("messages") or []:
        if isinstance(msg, (list, tuple)) and len(msg) == 2 and msg[0] == "execution_error":
            data = msg[1] or {}
            detail = _safe_detail(str(data.get("exception_message") or ""))
            return detail or "execution failed"
    return "execution failed"


def _file_items(entry: dict) -> list[dict]:
    found: list[dict] = []
    outputs = entry.get("outputs") if isinstance(entry, dict) else None
    if not isinstance(outputs, dict):
        return found
    for node in outputs.values():
        if not isinstance(node, dict):
            continue
        for key in _FILE_KEYS:
            for item in node.get(key) or []:
                if isinstance(item, dict) and item.get("filename"):
                    found.append(item)
    return found


def _public_unit(unit: dict, names: dict[str, str]) -> dict:
    outputs = _loads(unit.get("outputs_json"), [])
    if not isinstance(outputs, list):
        outputs = []
    return {
        "id": unit["id"],
        "project_id": unit["project_id"],
        "shot_id": unit["shot_id"],
        "stage": unit["stage"],
        "gpu_class": unit.get("gpu_class") or "",
        "status": unit["status"],
        "executor_id": unit.get("executor_id"),
        "executor_name": names.get(unit.get("executor_id") or "", ""),
        "prompt_id": unit.get("prompt_id"),
        "attempt": unit.get("attempt") or 0,
        "attempt_id": unit.get("attempt_id"),
        "waiting_reason": unit.get("waiting_reason"),
        "error": unit.get("error_text"),
        "outputs": outputs,
        "discard_outputs": bool(unit.get("discard_outputs")),
        "headless_job_id": unit.get("headless_job_id"),
        "submitted_at": unit["submitted_at"].isoformat() if unit.get("submitted_at") else None,
        "created_at": unit["created_at"].isoformat() if unit.get("created_at") else None,
    }


def scheduler_status() -> dict:
    names = {row["id"]: row["name"] for row in storage.list_executors(ensure_default=True)}
    active = [
        _public_unit(row, names)
        for row in storage.each_schedule_unit(statuses=_QUEUE_STATUSES)
    ]
    seen = {row["id"] for row in active}
    recent = [
        _public_unit(row, names)
        for row in storage.recent_terminal_units()
        if row["id"] not in seen
    ]
    units = active + recent
    queue = active
    assignments = [
        row for row in units
        if row["executor_id"] and row["status"] in _ACTIVE
    ]
    breakdown = storage.load_breakdown()
    executors = []
    for row in storage.list_executors(ensure_default=True):
        load = breakdown.get(row["id"]) or {
            "submitting": 0, "running": 0, "cancel_requested": 0,
        }
        executors.append({
            "id": row["id"],
            "name": row["name"],
            "gpu_class": row.get("gpu_class") or "",
            "is_default": bool(row.get("is_default")),
            "accepts_work": bool(row.get("accepts_work")),
            "max_concurrency": int(row.get("max_concurrency") or 1),
            "health": row.get("health"),
            "running_count": int(row.get("running_count") or 0),
            "pending_count": int(row.get("pending_count") or 0),
            "load": load,
        })
    return {
        "queue": queue,
        "units": units,
        "assignments": assignments,
        "executors": executors,
        "affinity": storage.list_affinity(),
    }


def enqueue_unit(
    *,
    project_id: str,
    shot_id: str,
    stage: str,
    prompt: dict,
    gpu_class: str = "",
    partial: Optional[list] = None,
    headless_job_id: Optional[str] = None,
) -> dict:
    row = storage.create_schedule_unit(
        project_id=project_id,
        shot_id=shot_id,
        stage=stage,
        prompt=prompt,
        gpu_class=gpu_class,
        partial=partial,
        headless_job_id=headless_job_id,
    )
    get_scheduler().ensure_started()
    _log.info(
        "[ComfyTV/scheduler] queued %s shot %s stage %s",
        row["id"], row["shot_id"], row["stage"],
    )
    names = {item["id"]: item["name"] for item in storage.list_executors(ensure_default=False)}
    return _public_unit(row, names)


class Scheduler:
    def __init__(self, transport=None, clock=None):
        self.transport = transport
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    def ensure_started(self) -> None:
        if os.environ.get("COMFYTV_TESTING") == "1":
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._run(), name="comfytv-scheduler")

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log.error("[ComfyTV/scheduler] tick failed: %s", type(exc).__name__)
            await asyncio.sleep(max(1.0, float(scheduler_limits()["poll"])))

    async def tick(self, local_client=None) -> None:
        async with self._lock:
            if self.transport is not None:
                await self._tick(self.transport, local_client)
                return
            async with aiohttp.ClientSession(timeout=_client_timeout()) as session:
                async def transport(method, url, json_body=None):
                    return await _http(session, method, url, json_body)
                await self._tick(transport, local_client)

    async def _tick(self, transport, local_client) -> None:
        if storage.foreign_executors_registered():
            storage.adopt_headless_jobs()
        await self._reconcile(transport, local_client)
        for unit in storage.each_schedule_unit(status="queued"):
            await self._place(transport, local_client, unit["id"])
        for unit in storage.each_schedule_unit(statuses=("running", "cancel-requested", "submitting")):
            await self._poll(transport, local_client, unit["id"])

    async def cancel_unit(self, unit_id: str) -> dict:
        # The flag is stored before taking the tick lock, so a poll that is
        # already inside a history read can see it and skip the download.
        marked = storage.request_cancel(unit_id)
        if marked is None:
            raise ValueError("unit not found")
        if marked["status"] in ("done", "error"):
            raise ValueError(f"unit is {marked['status']}")
        if marked["status"] == "cancelled" and not marked.get("post_started"):
            self._sync_headless(marked)
            return marked
        async with self._lock:
            if self.transport is not None:
                return await self._apply_cancel(self.transport, None, unit_id)
            async with aiohttp.ClientSession(timeout=_client_timeout()) as session:
                async def transport(method, url, json_body=None):
                    return await _http(session, method, url, json_body)
                return await self._apply_cancel(transport, None, unit_id)

    def _ensure_deadline(self, unit: dict) -> dict:
        if unit.get("deadline_at"):
            return unit
        limit = float(scheduler_limits()["job"])
        updated = storage.update_schedule_unit(
            unit["id"],
            deadline_at=self.clock() + timedelta(seconds=limit),
        )
        return updated or unit

    def _remaining(self, unit: dict) -> float:
        deadline = unit.get("deadline_at")
        if deadline is None:
            return float(scheduler_limits()["job"])
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        return (deadline - self.clock()).total_seconds()

    def _bound(self, transport, unit: dict):
        """Stop one host call at the connect cap or the unit's job deadline."""

        async def wrapped(method, url, json_body=None):
            path = urlsplit_path(url)
            transfer = path.endswith("/view") or "/history" in path
            remaining = self._remaining(unit)
            if transfer:
                cap = remaining
            else:
                cap = min(remaining, float(scheduler_limits()["connect"]))
            if cap <= 0:
                raise _Deadline()
            # A TimeoutError raised by the host call itself is a failed
            # attempt, not the job deadline. Only the wait limit is.
            task = asyncio.ensure_future(transport(method, url, json_body))
            try:
                done, _pending = await asyncio.wait({task}, timeout=cap)
            except asyncio.CancelledError:
                task.cancel()
                raise
            if not done:
                task.cancel()
                raise _Deadline()
            return task.result()

        return wrapped

    def _wants_discard(self, unit: Optional[dict]) -> bool:
        if not unit:
            return False
        return bool(
            unit.get("cancel_intent")
            or unit.get("discard_outputs")
            or unit.get("status") == "cancel-requested"
        )

    def _sync_headless(self, unit: Optional[dict]) -> None:
        if not unit or not unit.get("headless_job_id"):
            return
        mapped = {
            "queued": "delegated",
            "submitting": "delegated",
            "running": "delegated",
            "cancel-requested": "delegated",
            "done": "done",
            "error": "error",
            "cancelled": "cancelled",
        }.get(unit["status"])
        if mapped is None:
            return
        fields: dict[str, Any] = {"status": mapped}
        if unit.get("prompt_id"):
            fields["prompt_id"] = unit["prompt_id"]
        if unit["status"] == "done":
            fields["outputs_json"] = unit.get("outputs_json") or "[]"
            fields["error_text"] = None
        elif unit["status"] in ("error", "cancelled"):
            fields["error_text"] = unit.get("error_text")
            fields["outputs_json"] = unit.get("outputs_json") or "[]"
        storage.update_headless_job(unit["headless_job_id"], **fields)

    def _bind_running(self, unit: dict, executor: dict, prompt_id: str, required) -> dict:
        locked = required if required is not None else (executor.get("gpu_class") or "")
        fresh = storage.get_schedule_unit(unit["id"]) or unit
        intent = bool(fresh.get("cancel_intent"))
        updated = storage.update_schedule_unit(
            unit["id"],
            status="cancel-requested" if intent else "running",
            executor_id=executor["id"],
            prompt_id=prompt_id,
            post_started=True,
            discard_outputs=True if intent else bool(fresh.get("discard_outputs")),
            waiting_reason=None,
            error_text="cancel requested; remote prompt left running" if intent else None,
            submitted_at=fresh.get("submitted_at") or unit.get("submitted_at") or self.clock(),
        )
        storage.lock_affinity(unit["project_id"], unit["shot_id"], locked, executor["id"])
        self._sync_headless(updated)
        _log.info(
            "[ComfyTV/scheduler] %s running on %s attempt %s",
            unit["id"], executor.get("name"), updated.get("attempt"),
        )
        return updated

    def _rollback_mint(self, unit: dict) -> None:
        storage.update_schedule_unit(
            unit["id"],
            post_started=False,
            attempt=max(0, int(unit.get("attempt") or 0) - 1),
            clear_attempt=True,
            clear_executor=True,
        )

    def _exhaust(self, unit: dict, executor: dict) -> str:
        if int(unit.get("attempt") or 0) >= scheduler_limits()["max_attempts"]:
            updated = storage.update_schedule_unit(
                unit["id"],
                status="error",
                post_started=False,
                waiting_reason=None,
                error_text="executor unreachable",
            )
            self._sync_headless(updated)
            return "failed"
        storage.update_schedule_unit(
            unit["id"],
            status="queued",
            post_started=False,
            clear_attempt=True,
            clear_prompt=True,
            clear_executor=True,
            avoid_executor_id=executor["id"],
            waiting_reason="unreachable",
            error_text="executor unreachable",
        )
        return "failed"

    def _reject(self, unit: dict, detail: str) -> str:
        updated = storage.update_schedule_unit(
            unit["id"],
            status="error",
            post_started=False,
            waiting_reason=None,
            error_text=detail or "prompt rejected",
        )
        self._sync_headless(updated)
        return "rejected"

    async def _inspect(self, transport, local_client, executor, client_id, prompt_id, need_history: bool):
        if executor.get("is_default"):
            return await self._inspect_local(local_client, client_id, prompt_id, need_history)
        return await self._inspect_remote(
            transport, executor["base_url"], client_id, prompt_id, need_history,
        )

    async def _inspect_local(self, local_client, client_id, prompt_id, need_history: bool):
        client = local_client or _headless_api()[0]()
        try:
            if client_id and hasattr(client, "find_client"):
                found = client.find_client(client_id)
                if found:
                    return "found", found
            running, pending = client.queue_ids()
        except Exception:
            return "unknown", None
        if prompt_id and (str(prompt_id) in set(map(str, running)) or str(prompt_id) in set(map(str, pending))):
            return "found", str(prompt_id)
        if need_history and prompt_id and hasattr(client, "history_for"):
            try:
                if client.history_for(prompt_id):
                    return "found", str(prompt_id)
            except Exception:
                return "unknown", None
        if not queue_is_empty(running, pending):
            return "busy", None
        return "empty", None

    async def _inspect_remote(self, transport, base_url, client_id, prompt_id, need_history: bool):
        url = base_url.rstrip("/") + "/queue"
        try:
            status, body = await transport("GET", url)
        except _Deadline:
            return "unknown", None
        except Exception as exc:
            if _unreachable(exc):
                return "unknown", None
            raise
        if status != 200 or not isinstance(body, dict):
            return "unknown", None
        running = body.get("queue_running")
        pending = body.get("queue_pending")
        if not isinstance(running, list) or not isinstance(pending, list):
            return "unknown", None
        found = _match_rows(running, client_id, prompt_id) or _match_rows(pending, client_id, prompt_id)
        if found:
            return "found", found
        if need_history:
            located = await self._find_history(transport, base_url, client_id, prompt_id)
            if located == "unknown":
                return "unknown", None
            if located:
                return "found", located
        if not queue_is_empty(running, pending):
            return "busy", None
        return "empty", None

    async def _find_history(self, transport, base_url, client_id, prompt_id):
        url = base_url.rstrip("/") + "/history"
        try:
            status, body = await transport("GET", url)
        except _Deadline:
            return "unknown"
        except Exception as exc:
            if _unreachable(exc):
                return "unknown"
            raise
        if status != 200 or not isinstance(body, dict):
            return "unknown"
        for pid, entry in body.items():
            if not isinstance(entry, dict):
                continue
            if prompt_id and str(pid) == str(prompt_id):
                return str(pid)
            if client_id and _history_client(entry) == client_id:
                return str(pid)
        if prompt_id:
            one = base_url.rstrip("/") + "/history/" + quote(str(prompt_id), safe="")
            try:
                status, body = await transport("GET", one)
            except Exception as exc:
                if _unreachable(exc):
                    return "unknown"
                raise
            if status == 200 and isinstance(body, dict) and isinstance(body.get(str(prompt_id)), dict):
                return str(prompt_id)
        return None

    async def _fetch_entry(self, transport, local_client, executor, prompt_id):
        if executor.get("is_default"):
            client = local_client or _headless_api()[0]()
            try:
                return client.history_for(prompt_id)
            except Exception:
                return None
        url = executor["base_url"].rstrip("/") + "/history/" + quote(str(prompt_id), safe="")
        try:
            status, body = await transport("GET", url)
        except Exception as exc:
            if _unreachable(exc):
                return None
            raise
        if status != 200 or not isinstance(body, dict):
            return None
        entry = body.get(str(prompt_id))
        return entry if isinstance(entry, dict) else None

    def _mint(self, unit: dict, executor: dict) -> dict:
        """Reserve an attempt id. The send itself sets post_started just before POST."""
        attempt_id = uuid.uuid4().hex
        updated = storage.update_schedule_unit(
            unit["id"],
            attempt_id=attempt_id,
            client_id=f"comfytv-sched-{attempt_id}",
            executor_id=executor["id"],
            post_started=False,
            attempt=int(unit.get("attempt") or 0) + 1,
            waiting_reason=None,
        )
        return updated

    async def _submit(self, transport, local_client, executor, unit: dict) -> str:
        prompt = _loads(unit.get("prompt_json"), {})
        partial = _loads(unit.get("partial_json"), None)
        if not isinstance(partial, list):
            partial = None
        if executor.get("is_default"):
            return await self._submit_local(local_client, prompt, partial, unit["client_id"])
        url = executor["base_url"].rstrip("/") + "/prompt"
        body: dict[str, Any] = {"prompt": prompt, "client_id": unit["client_id"]}
        if partial:
            body["partial_execution_targets"] = [str(node) for node in partial]
        try:
            status, payload = await transport("POST", url, body)
        except _Deadline:
            raise
        except Exception as exc:
            if _unreachable(exc):
                raise _Unreachable() from None
            raise
        body = payload
        if status >= 500:
            raise _Unreachable()
        if status != 200 or not isinstance(body, dict) or not body.get("prompt_id"):
            detail = ""
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict):
                    detail = _safe_detail(str(err.get("message") or ""))
            raise _Rejected(detail or "prompt rejected")
        return str(body["prompt_id"])

    async def _submit_local(self, local_client, prompt, partial, client_id: str) -> str:
        Client, QueueBusy = _headless_api()
        client = local_client or Client()
        try:
            return await client.submit(prompt, partial, client_id)
        except QueueBusy as exc:
            raise _Busy() from exc
        except ValueError as exc:
            raise _Rejected(_safe_detail(str(exc)) or "prompt rejected") from None

    async def _attempt(self, transport, local_client, unit, executor, required) -> str:
        same = unit.get("executor_id") == executor["id"]
        client_id = unit.get("client_id") if same else None
        need_history = bool(unit.get("post_started") and same)
        state, found = await self._inspect(
            transport, local_client, executor, client_id, unit.get("prompt_id") if same else None,
            need_history,
        )
        if state == "unknown":
            if unit.get("post_started") and same:
                return "held"
            return "skip"
        if state == "found" and found:
            self._bind_running(unit, executor, found, required)
            return "placed"
        if state == "busy":
            return "skip"
        if unit.get("post_started") and same:
            self._exhaust(unit, executor)
            return "failed"
        current = storage.get_schedule_unit(unit["id"]) or unit
        if current.get("cancel_intent"):
            return "rejected" if current["status"] in ("done", "error", "cancelled") else "held"
        minted = self._mint(unit, executor)
        state, found = await self._inspect(
            transport, local_client, executor, minted["client_id"], None, False,
        )
        if state == "unknown":
            self._rollback_mint(minted)
            return "skip"
        if state == "found" and found:
            self._bind_running(minted, executor, found, required)
            return "placed"
        if state != "empty":
            self._rollback_mint(minted)
            return "skip"
        minted = storage.update_schedule_unit(minted["id"], post_started=True) or minted
        try:
            prompt_id = await self._submit(transport, local_client, executor, minted)
        except _Busy:
            self._rollback_mint(minted)
            return "skip"
        except _Rejected as exc:
            self._reject(minted, str(exc) or "prompt rejected")
            return "rejected"
        except _Deadline:
            fresh = storage.get_schedule_unit(minted["id"]) or minted
            state, found = await self._inspect(
                transport, local_client, executor, fresh.get("client_id"), None, True,
            )
            if state == "found" and found:
                self._bind_running(fresh, executor, found, required)
                return "placed"
            return "held"
        except _Unreachable:
            state, found = await self._inspect(
                transport, local_client, executor, minted["client_id"], None, True,
            )
            fresh = storage.get_schedule_unit(minted["id"]) or minted
            if state == "found" and found:
                self._bind_running(fresh, executor, found, required)
                return "placed"
            if state == "unknown":
                return "held"
            self._exhaust(fresh, executor)
            return "failed"
        self._bind_running(minted, executor, prompt_id, required)
        return "placed"

    async def _place(self, transport, local_client, unit_id: str) -> None:
        unit = storage.get_schedule_unit(unit_id)
        if unit is None or unit["status"] != "queued":
            return
        if unit.get("cancel_intent"):
            return
        unit = self._ensure_deadline(unit)
        transport = self._bound(transport, unit)
        executors = storage.list_executors(ensure_default=False)
        affinity = storage.get_affinity(unit["project_id"], unit["shot_id"])
        required, blocked = _required_class(unit, affinity)
        loads = storage.executor_loads()
        if blocked:
            storage.update_schedule_unit(unit_id, waiting_reason=blocked)
            return
        chosen = _candidates(
            executors, required, affinity, unit.get("avoid_executor_id"), loads,
        )
        if not chosen:
            storage.update_schedule_unit(
                unit_id, waiting_reason=_wait_reason(executors, required, loads),
            )
            return
        claimed = storage.claim_schedule_unit(unit_id, expect="queued", become="submitting")
        if claimed is None:
            return
        for executor in chosen:
            current = storage.get_schedule_unit(unit_id)
            if current is None:
                return
            if current["status"] == "queued":
                again = storage.claim_schedule_unit(unit_id, expect="queued", become="submitting")
                if again is None:
                    return
                current = again
            if current["status"] != "submitting":
                return
            outcome = await self._attempt(transport, local_client, current, executor, required)
            if outcome in ("placed", "held", "rejected"):
                return
            if outcome == "failed":
                fresh = storage.get_schedule_unit(unit_id)
                if fresh is None or fresh["status"] != "queued":
                    return
                continue
        fresh = storage.get_schedule_unit(unit_id)
        if fresh and fresh["status"] == "submitting":
            storage.update_schedule_unit(
                unit_id, status="queued", waiting_reason="busy", clear_executor=True,
            )

    async def _reconcile(self, transport, local_client) -> None:
        for unit in storage.each_schedule_unit(status="submitting"):
            if unit.get("cancel_intent") and not unit.get("post_started"):
                if unit["status"] != "cancelled":
                    updated = storage.update_schedule_unit(
                        unit["id"],
                        status="cancelled",
                        error_text="cancelled before submit",
                        waiting_reason=None,
                        clear_executor=True,
                    )
                    self._sync_headless(updated)
                continue
            unit = self._ensure_deadline(unit)
            transport_one = self._bound(transport, unit)
            if not unit.get("post_started"):
                storage.update_schedule_unit(
                    unit["id"],
                    status="queued",
                    clear_executor=True,
                    clear_attempt=True,
                    attempt=max(0, int(unit.get("attempt") or 0) - 1),
                )
                continue
            if unit.get("cancel_intent"):
                await self._resolve_uncertain(transport_one, local_client, unit)
                continue
            executor = storage.get_executor(unit["executor_id"]) if unit.get("executor_id") else None
            if executor is None:
                self._exhaust(unit, {"id": unit.get("executor_id") or "", "name": ""})
                continue
            affinity = storage.get_affinity(unit["project_id"], unit["shot_id"])
            required, _blocked = _required_class(unit, affinity)
            state, found = await self._inspect(
                transport_one, local_client, executor, unit.get("client_id"), unit.get("prompt_id"), True,
            )
            if state == "found" and found:
                self._bind_running(unit, executor, found, required)
            elif state == "unknown":
                continue
            else:
                self._exhaust(unit, executor)

    def _expired(self, unit: dict) -> bool:
        if unit.get("deadline_at") and self._remaining(unit) <= 0:
            return True
        started = unit.get("submitted_at")
        if started is None:
            return False
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        limit = float(scheduler_limits()["job"])
        return (self.clock() - started).total_seconds() >= limit

    async def _resolve_uncertain(self, transport, local_client, unit: dict) -> dict:
        """Find a prompt that was sent but whose id never came back."""
        executor = storage.get_executor(unit["executor_id"]) if unit.get("executor_id") else None
        if executor is None or not unit.get("client_id"):
            return unit
        state, found = await self._inspect(
            transport, local_client, executor, unit.get("client_id"), None, True,
        )
        if state == "found" and found:
            unit = storage.update_schedule_unit(unit["id"], prompt_id=found) or unit
            outcome = await cancel_owned_prompt(transport, executor["base_url"], found)
            if outcome == "deleted":
                updated = storage.update_schedule_unit(
                    unit["id"],
                    status="cancelled",
                    prompt_id=found,
                    outputs_json="[]",
                    error_text="removed from the remote queue while pending",
                    waiting_reason=None,
                )
            else:
                updated = storage.update_schedule_unit(
                    unit["id"],
                    status="cancel-requested",
                    prompt_id=found,
                    discard_outputs=True,
                    error_text="cancel requested; remote prompt left running",
                    waiting_reason=None,
                )
            self._sync_headless(updated)
            return updated or unit
        if state == "empty":
            updated = storage.update_schedule_unit(
                unit["id"],
                status="cancelled",
                post_started=False,
                outputs_json="[]",
                error_text="cancelled before the prompt landed",
                waiting_reason=None,
            )
            self._sync_headless(updated)
            return updated or unit
        return unit

    async def _poll(self, transport, local_client, unit_id: str) -> None:
        unit = storage.get_schedule_unit(unit_id)
        if unit is None or unit["status"] not in ("running", "cancel-requested"):
            return
        unit = self._ensure_deadline(unit)
        transport = self._bound(transport, unit)
        if self._wants_discard(unit) and not unit.get("prompt_id") and unit.get("client_id"):
            unit = await self._resolve_uncertain(transport, local_client, unit)
            if unit["status"] not in ("running", "cancel-requested"):
                return
        if unit["status"] == "running" and self._expired(unit):
            await self._apply_cancel(transport, local_client, unit_id)
            unit = storage.get_schedule_unit(unit_id)
            if unit is None or unit["status"] not in ("running", "cancel-requested"):
                return
        executor = storage.get_executor(unit["executor_id"]) if unit.get("executor_id") else None
        if executor is None or not unit.get("prompt_id"):
            return
        entry = await self._fetch_entry(transport, local_client, executor, unit["prompt_id"])
        unit = storage.get_schedule_unit(unit_id) or unit
        if not _history_done(entry):
            return
        if self._wants_discard(unit):
            self._discard(unit)
            return
        try:
            urls = await self._collect(transport, executor, unit, entry)
        except _Unreachable:
            fresh = storage.get_schedule_unit(unit_id) or unit
            if self._wants_discard(fresh) or self._remaining(fresh) <= 0:
                self._discard(fresh)
            else:
                storage.update_schedule_unit(unit["id"], error_text="download failed")
            return
        unit = storage.get_schedule_unit(unit_id) or unit
        if self._wants_discard(unit):
            self._discard(unit)
            return
        if _history_failed(entry):
            updated = storage.update_schedule_unit(
                unit["id"],
                status="error",
                error_text=_history_error(entry),
                outputs_json=json.dumps(urls),
                waiting_reason=None,
            )
            self._sync_headless(updated)
            return
        output_id = self._persist(unit, urls)
        updated = storage.update_schedule_unit(
            unit["id"],
            status="done",
            error_text=None,
            outputs_json=json.dumps(urls),
            output_id=output_id,
            waiting_reason=None,
        )
        self._sync_headless(updated)
        _log.info("[ComfyTV/scheduler] %s done (%d outputs)", unit["id"], len(urls))

    def _discard(self, unit: dict) -> None:
        updated = storage.update_schedule_unit(
            unit["id"],
            status="cancelled",
            discard_outputs=True,
            outputs_json="[]",
            error_text="cancel requested; remote outputs discarded",
            waiting_reason=None,
        )
        self._sync_headless(updated)

    async def _collect(self, transport, executor, unit, entry) -> list[str]:
        if executor.get("is_default"):
            return [
                _view_url(item.get("filename") or "", item.get("subfolder") or "", item.get("type") or "output")
                for item in _file_items(entry)
            ]
        return await self._download(transport, executor, unit, entry)

    async def _download(self, transport, executor, unit, entry) -> list[str]:
        root = Path(folder_paths.get_output_directory())
        dest = root / _RESULT_SUBFOLDER / _safe_component(executor.get("name") or "") / unit["id"][:8]
        urls: list[str] = []
        for item in _file_items(entry):
            name = Path(str(item.get("filename") or "")).name
            if not name or name in {".", ".."}:
                continue
            kind = _safe_component(str(item.get("type") or "output"))
            parts = []
            for part in str(item.get("subfolder") or "").replace("\\", "/").split("/"):
                if part in ("", ".", ".."):
                    continue
                cleaned = _safe_component(part)
                if cleaned:
                    parts.append(cleaned)
            folder = dest / kind
            for part in parts:
                folder = folder / part
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise _Unreachable() from exc
            query = urlencode({
                "filename": item.get("filename") or name,
                "subfolder": item.get("subfolder") or "",
                "type": item.get("type") or "output",
            })
            url = executor["base_url"].rstrip("/") + "/view?" + query
            try:
                status, body = await transport("GET", url)
            except Exception as exc:
                if _unreachable(exc):
                    raise _Unreachable() from None
                raise
            if status != 200 or not isinstance(body, (bytes, bytearray)):
                raise _Unreachable()
            target = folder / name
            target.write_bytes(bytes(body))
            subfolder = target.parent.relative_to(root).as_posix()
            urls.append(_view_url(name, subfolder, "output"))
        return urls

    def _persist(self, unit: dict, urls: list[str]) -> Optional[int]:
        if not urls:
            return None
        try:
            saved = storage.persist_output(
                project_id=unit["project_id"],
                stage_class=unit["stage"],
                stage_node_id=unit["shot_id"],
                output_type="scheduler",
                payload_url=urls[0],
                payload_json={"urls": urls} if len(urls) > 1 else None,
                stage_uid=unit["id"],
            )
        except Exception as exc:
            _log.warning("[ComfyTV/scheduler] output record failed: %s", type(exc).__name__)
            return None
        if not saved:
            return None
        return saved.get("id")

    async def _apply_cancel(self, transport, local_client, unit_id: str) -> dict:
        unit = storage.get_schedule_unit(unit_id)
        if unit is None:
            raise ValueError("unit not found")
        if unit["status"] in ("done", "error"):
            raise ValueError(f"unit is {unit['status']}")
        if unit["status"] == "cancelled":
            return unit
        if unit["status"] == "cancel-requested" and unit.get("prompt_id"):
            return unit
        unit = self._ensure_deadline(unit)
        transport = self._bound(transport, unit)
        if (
            self._wants_discard(unit)
            and not unit.get("prompt_id")
            and unit.get("client_id")
            and unit.get("post_started")
        ):
            return await self._resolve_uncertain(transport, local_client, unit)
        if unit["status"] in ("queued", "submitting") and not unit.get("post_started"):
            updated = storage.update_schedule_unit(
                unit_id,
                status="cancelled",
                waiting_reason=None,
                error_text="cancelled before submit",
                clear_executor=True,
            )
            self._sync_headless(updated)
            return updated
        executor = storage.get_executor(unit["executor_id"]) if unit.get("executor_id") else None
        prompt_id = unit.get("prompt_id")
        if executor is not None and not prompt_id and unit.get("client_id"):
            state, found = await self._inspect(
                transport, local_client, executor, unit.get("client_id"), None, True,
            )
            if state == "found" and found:
                prompt_id = found
                unit = storage.update_schedule_unit(unit_id, prompt_id=found) or unit
            elif state == "empty":
                updated = storage.update_schedule_unit(
                    unit_id,
                    status="cancelled",
                    post_started=False,
                    error_text="cancelled before the prompt landed",
                    waiting_reason=None,
                )
                self._sync_headless(updated)
                return updated
        if executor is None or not prompt_id:
            updated = storage.update_schedule_unit(
                unit_id,
                status="cancel-requested",
                discard_outputs=True,
                error_text="cancel requested; remote prompt left running",
                waiting_reason=None,
            )
            self._sync_headless(updated)
            return updated
        outcome = await cancel_owned_prompt(transport, executor["base_url"], prompt_id)
        if outcome == "deleted":
            updated = storage.update_schedule_unit(
                unit_id,
                status="cancelled",
                discard_outputs=True,
                outputs_json="[]",
                prompt_id=prompt_id,
                error_text="removed from the remote queue while pending",
                waiting_reason=None,
            )
        elif outcome == "absent":
            entry = await self._fetch_entry(transport, local_client, executor, prompt_id)
            if entry is None:
                updated = storage.update_schedule_unit(
                    unit_id,
                    status="cancelled",
                    outputs_json="[]",
                    prompt_id=prompt_id,
                    error_text="left the remote queue while pending",
                    waiting_reason=None,
                )
            else:
                updated = storage.update_schedule_unit(
                    unit_id,
                    status="cancel-requested",
                    discard_outputs=True,
                    prompt_id=prompt_id,
                    error_text="cancel requested; remote prompt left running",
                    waiting_reason=None,
                )
        else:
            updated = storage.update_schedule_unit(
                unit_id,
                status="cancel-requested",
                discard_outputs=True,
                prompt_id=prompt_id,
                error_text="cancel requested; remote prompt left running",
                waiting_reason=None,
            )
        self._sync_headless(updated)
        return updated


class _Busy(Exception):
    pass


class _Rejected(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)


_SCHEDULER: Scheduler | None = None


def get_scheduler() -> Scheduler:
    global _SCHEDULER
    if _SCHEDULER is None:
        _SCHEDULER = Scheduler()
    return _SCHEDULER


def set_scheduler(scheduler: Scheduler | None) -> None:
    global _SCHEDULER
    if _SCHEDULER is not None and scheduler is not _SCHEDULER:
        _SCHEDULER.stop()
    _SCHEDULER = scheduler


def ensure_started() -> None:
    get_scheduler().ensure_started()


async def drive_scheduler(local_client=None) -> None:
    """One scheduler step. Used by the headless worker when other hosts exist."""
    if not storage.foreign_executors_registered():
        return
    await get_scheduler().tick(local_client=local_client)


async def cancel_headless(job_id: str) -> dict:
    unit = storage.unit_for_headless(job_id)
    if unit is None:
        raise ValueError("this headless job has no schedule unit")
    updated = await get_scheduler().cancel_unit(unit["id"])
    fresh = storage.get_headless_job(job_id) or {}
    return {
        "cancelled": updated["status"] == "cancelled",
        "cancel_requested": updated["status"] == "cancel-requested",
        "job_id": job_id,
        "status": fresh.get("status") or updated["status"],
        "headless": True,
        "unit_id": updated["id"],
    }


def _body_error(message: str, status: int) -> web.Response:
    return web.json_response({"error": message}, status=status)


async def _json_body(request: web.Request) -> dict | None:
    try:
        body = await request.json()
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    return body


@routes.get("/comfytv/scheduler")
async def get_scheduler_status(_request: web.Request) -> web.Response:
    return web.json_response(scheduler_status())


@routes.post("/comfytv/scheduler/units")
async def post_schedule_unit(request: web.Request) -> web.Response:
    body = await _json_body(request)
    if body is None:
        return _body_error("invalid JSON body", 400)
    prompt = body.get("prompt")
    try:
        unit = enqueue_unit(
            project_id=str(body.get("project_id") or ""),
            shot_id=str(body.get("shot_id") or ""),
            stage=str(body.get("stage") or ""),
            prompt=prompt,
            gpu_class=body.get("gpu_class") or "",
        )
    except ValueError as exc:
        return _body_error(str(exc), 400)
    return web.json_response({"unit": unit}, status=201)


@routes.post("/comfytv/scheduler/units/{unit_id}/cancel")
async def post_cancel_unit(request: web.Request) -> web.Response:
    unit_id = request.match_info["unit_id"]
    try:
        unit = await get_scheduler().cancel_unit(unit_id)
    except ValueError as exc:
        message = str(exc)
        status = 404 if message == "unit not found" else 409
        return _body_error(message, status)
    names = {row["id"]: row["name"] for row in storage.list_executors(ensure_default=False)}
    return web.json_response({"unit": _public_unit(unit, names)})
