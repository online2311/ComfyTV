"""One-at-a-time queue of MCP runs onto ComfyUI's own prompt queue.

Jobs are stored in sqlite so a restart can submit unfinished work again.
Submission checks the in-process prompt queue — the same list ``GET /queue``
reads — and refuses to put a second ComfyTV job up while anything is already
running or pending. This module never calls interrupt or frees memory.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
import uuid
from urllib.parse import urlencode

import execution

from .. import storage
from ..nodes.stages import STAGE_META
from .stages import NOT_RUNNABLE_VARIANTS
from . import headless_graph as graph

_log = logging.getLogger(__name__)

_ABSENT_POLLS = 3
_POLL_S = 0.4
HEADLESS_CLIENT_ID = "comfytv-headless"

_PROGRESS: dict[str, dict] = {}
_COMPLETED: dict[str, dict] = {}


class QueueBusy(RuntimeError):
    """ComfyUI's queue was not empty at the moment of submit."""


def _loads(raw, fallback):
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def _is_link(value, node_id: str, slot: int) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    if isinstance(value[1], bool) or not isinstance(value[1], int):
        return False
    return str(value[0]) == str(node_id) and int(value[1]) == int(slot)


def _still_linked(prompt: dict, node_id: str) -> bool:
    for node in prompt.values():
        for value in (node.get("inputs") or {}).values():
            if (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and not isinstance(value[1], bool)
                and isinstance(value[1], int)
                and str(value[0]) == str(node_id)
            ):
                return True
    return False


def _slot_values(info: dict) -> dict[int, object]:
    raw = info.get("slots")
    out: dict[int, object] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                out[int(key)] = value
            except (TypeError, ValueError):
                continue
    if out:
        return out
    if "value" not in info:
        return {}
    try:
        slot = int(info.get("slot") or 0)
    except (TypeError, ValueError):
        slot = 0
    return {slot: info["value"]}


def apply_completed(prompt: dict, completed: dict | None, partial: list | None):
    """Drop finished nodes whose outputs were saved and rewire downstream inputs.

    A completed node is removed only when every link that pointed at it can be
    replaced with the saved value. Nodes we finished but could not snapshot
    stay in the prompt and run again.
    """
    prompt = copy.deepcopy(prompt or {})
    completed = completed or {}
    for nid, info in list(completed.items()):
        if not isinstance(info, dict):
            continue
        slot_map = _slot_values(info)
        if not slot_map:
            continue
        for node in prompt.values():
            inputs = node.get("inputs") or {}
            for key, current in list(inputs.items()):
                for slot, value in slot_map.items():
                    if _is_link(current, str(nid), slot):
                        inputs[key] = value
                        break
        if str(nid) in prompt and not _still_linked(prompt, str(nid)):
            prompt.pop(str(nid), None)
    if partial is not None:
        partial = [str(p) for p in partial if str(p) in prompt]
    return prompt, partial


def _collect_urls(value, seen: set[str] | None = None) -> list[str]:
    """Pull /view and http URLs out of history, including ComfyTV stage payloads."""
    if seen is None:
        seen = set()
    found: list[str] = []

    def add(url: str) -> None:
        if url and url not in seen:
            seen.add(url)
            found.append(url)

    if isinstance(value, str):
        text = value.strip()
        if text.startswith("/view?") or text.startswith("http://") or text.startswith("https://"):
            add(text)
        elif text[:1] in "{[":
            try:
                found.extend(_collect_urls(json.loads(text), seen))
            except (TypeError, ValueError):
                pass
    elif isinstance(value, dict):
        if value.get("filename"):
            add("/view?" + urlencode({
                "filename": value["filename"],
                "subfolder": value.get("subfolder") or "",
                "type": value.get("type") or "output",
            }))
        for child in value.values():
            if isinstance(child, (dict, list, str)):
                found.extend(_collect_urls(child, seen))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_urls(item, seen))
    return found


def view_urls(entry: dict | None) -> list[str]:
    if not isinstance(entry, dict):
        return []
    return _collect_urls(entry.get("outputs"))


def _history_node_ids(entry: dict | None) -> list[str]:
    """Node ids ComfyUI reported as executed or cached, plus history output keys."""
    if not isinstance(entry, dict):
        return []
    found: list[str] = []
    status = entry.get("status") or {}
    for msg in status.get("messages") or []:
        if not isinstance(msg, (list, tuple)) or len(msg) != 2:
            continue
        kind, data = msg
        if not isinstance(data, dict):
            continue
        if kind == "execution_cached":
            found.extend(str(node) for node in (data.get("nodes") or []))
        elif kind == "executed" and data.get("node") is not None:
            found.append(str(data["node"]))
    outputs = entry.get("outputs")
    if isinstance(outputs, dict):
        found.extend(str(node) for node in outputs)
    return found


def merge_completed_nodes(row: dict, entry: dict | None, live: dict | None, *,
                          success: bool) -> dict:
    """Nodes to report as completed.

    Progress snapshots win over a bare id. On success the prompt's own node
    list is included, so a fast job that never emitted progress is not left
    at ``[]``.
    """
    current = _loads(row.get("completed_json"), {})
    if not isinstance(current, dict):
        current = {}
    merged: dict = {}
    for nid, info in current.items():
        if isinstance(info, dict):
            merged[str(nid)] = info
    for nid, info in (live or {}).items():
        if isinstance(info, dict):
            merged[str(nid)] = info
    for nid in _history_node_ids(entry):
        merged.setdefault(nid, {"node": nid})
    if success:
        prompt = _loads(row.get("prompt_json"), {})
        if isinstance(prompt, dict):
            for nid in prompt:
                merged.setdefault(str(nid), {"node": str(nid)})
    return merged


def history_error(entry: dict) -> str:
    status = entry.get("status") or {}
    for msg in status.get("messages") or []:
        if (
            isinstance(msg, (list, tuple))
            and len(msg) == 2
            and msg[0] == "execution_error"
        ):
            data = msg[1] or {}
            return (
                f"{data.get('node_type')} (node {data.get('node_id')}): "
                f"{data.get('exception_message')}"
            )
    return "execution failed"


def _as_payload(value):
    if isinstance(value, str):
        return value
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _summarize_output(output) -> dict | None:
    """Checkpoint one node's outputs, including ComfyTV ``output``/``picked`` lists."""
    if not isinstance(output, dict):
        if isinstance(output, str) and output:
            return {"value": output, "slot": 0, "slots": {"0": output}}
        return None
    slots: dict[str, str] = {}
    urls = _collect_urls({
        key: output.get(key)
        for key in ("images", "gifs", "videos", "audio", "video")
        if key in output
    })
    if urls:
        slots["0"] = urls[0]
    listed = output.get("output")
    if isinstance(listed, list) and listed:
        payload = _as_payload(listed[0])
        if payload:
            slots["0"] = payload
    picked = output.get("picked")
    if isinstance(picked, list) and picked:
        payload = _as_payload(picked[0])
        if payload:
            slots["1"] = payload
    text = output.get("text")
    if "0" not in slots and isinstance(text, list) and text:
        slots["0"] = _as_payload(text[0]) or ""
    elif "0" not in slots and isinstance(text, str):
        slots["0"] = text
    if not slots:
        return {"slot": 0}
    return {"value": slots.get("0"), "slot": 0, "slots": slots}


def _observe(event, data) -> None:
    if not isinstance(data, dict):
        return
    prompt_id = data.get("prompt_id")
    if event == "progress" and prompt_id:
        _PROGRESS[str(prompt_id)] = {
            "value": data.get("value"),
            "max": data.get("max"),
            "node": data.get("node"),
        }
        return
    if event == "executing" and prompt_id and data.get("node"):
        prev = dict(_PROGRESS.get(str(prompt_id)) or {})
        prev["node"] = data.get("node")
        _PROGRESS[str(prompt_id)] = prev
        return
    if event == "executed" and prompt_id and data.get("node") is not None:
        summary = _summarize_output(data.get("output"))
        if summary is not None:
            _COMPLETED.setdefault(str(prompt_id), {})[str(data.get("node"))] = summary


def install_progress_tap(server=None) -> None:
    if server is None:
        from server import PromptServer
        server = PromptServer.instance
    current = getattr(server, "send_sync", None)
    if current is None or getattr(current, "_comfytv_headless_tap", False):
        return

    def wrapped(event, data, sid=None):
        try:
            _observe(event, data)
        except Exception:
            _log.debug("[ComfyTV/headless] progress tap failed", exc_info=True)
        return current(event, data, sid)

    wrapped._comfytv_headless_tap = True
    server.send_sync = wrapped


def _queue_item_id(item) -> str | None:
    if isinstance(item, (list, tuple)) and len(item) > 1:
        return str(item[1])
    return None


class ComfyQueueClient:
    """In-process client for ComfyUI's PromptQueue. No interrupt, no free."""

    def __init__(self):
        self._number = 0

    def _queue(self):
        from server import PromptServer
        queue = getattr(PromptServer.instance, "prompt_queue", None)
        if queue is None:
            raise ValueError(
                "ComfyUI prompt queue is not available on this server"
            )
        return queue

    def queue_ids(self) -> tuple[list[str], list[str]]:
        queue = self._queue()
        getter = getattr(queue, "get_current_queue_volatile", None)
        if getter is None:
            getter = getattr(queue, "get_current_queue", None)
        if getter is None:
            raise ValueError("ComfyUI prompt queue cannot be read")
        running, pending = getter()
        return (
            [i for i in (_queue_item_id(item) for item in running or []) if i],
            [i for i in (_queue_item_id(item) for item in pending or []) if i],
        )

    def busy(self) -> bool:
        running, pending = self.queue_ids()
        return bool(running or pending)

    def history_for(self, prompt_id: str) -> dict | None:
        if not prompt_id:
            return None
        hist = self._queue().get_history(prompt_id=prompt_id) or {}
        entry = hist.get(prompt_id)
        return entry if isinstance(entry, dict) else None

    def progress_for(self, prompt_id: str) -> dict | None:
        return _PROGRESS.get(str(prompt_id))

    def completed_for(self, prompt_id: str) -> dict:
        return dict(_COMPLETED.get(str(prompt_id)) or {})

    async def submit(self, prompt: dict, partial: list | None) -> str:
        if self.busy():
            raise QueueBusy("ComfyUI queue is not empty")
        install_progress_tap()
        prompt_id = str(uuid.uuid4())
        outputs = [str(n) for n in partial] if partial else None
        validate = getattr(execution, "validate_prompt", None)
        if validate is not None:
            valid = await validate(prompt_id, prompt, outputs)
            if not valid[0]:
                err = valid[1] if len(valid) > 1 else {}
                message = err.get("message") if isinstance(err, dict) else str(err)
                details = err.get("details") if isinstance(err, dict) else ""
                raise ValueError(
                    f"ComfyUI rejected the prompt: {message}"
                    + (f" — {details}" if details else "")
                )
            outputs = [str(n) for n in (valid[2] or [])]
        if self.busy():
            raise QueueBusy("ComfyUI queue filled while the prompt was checked")
        self._number += 1
        extra = {
            "client_id": HEADLESS_CLIENT_ID,
            "create_time": int(time.time() * 1000),
        }
        item = (self._number, prompt_id, prompt, extra, outputs or [], {})
        self._queue().put(item)
        _log.info(
            "[ComfyTV/headless] submitted prompt %s (%d nodes)",
            prompt_id, len(prompt or {}),
        )
        return prompt_id


def public_job(row: dict) -> dict:
    completed = _loads(row.get("completed_json"), {})
    if not isinstance(completed, dict):
        completed = {}
    return {
        "job_id": row["id"],
        "status": row["status"],
        "kind": row["kind"],
        "project_id": row["project_id"],
        "prompt_id": row.get("prompt_id"),
        "stage_ref": row.get("stage_ref"),
        "stage_uid": row.get("stage_uid"),
        "progress": _loads(row.get("progress_json"), None),
        "outputs": _loads(row.get("outputs_json"), []),
        "error": row.get("error_text"),
        "completed_nodes": sorted(str(k) for k in completed.keys()),
        "attempt": row.get("attempt") or 1,
        "created_at": storage.format_utc_timestamp(row.get("created_at")),
        "updated_at": storage.format_utc_timestamp(row.get("updated_at")),
        "started_at": storage.format_utc_timestamp(row.get("started_at")),
        "finished_at": storage.format_utc_timestamp(row.get("finished_at")),
        "headless": True,
    }


def _reject_not_runnable(stage_class: str) -> None:
    name = stage_class.removeprefix("ComfyTV.")
    variant = (STAGE_META.get(name) or {}).get("variant")
    if variant not in NOT_RUNNABLE_VARIANTS:
        return
    what = (
        "a loader stage — it only holds media"
        if variant == "loader"
        else "an editor stage — its result is applied live downstream"
    )
    raise ValueError(
        f"{stage_class} is {what}, so there is nothing to queue. "
        "Set its widgets with set_stage and run the downstream stage."
    )


class HeadlessQueue:
    def __init__(self, client=None, autostart: bool = True, poll_s: float = _POLL_S):
        self.client = client if client is not None else ComfyQueueClient()
        self.autostart = autostart
        self.poll_s = poll_s
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._absent: dict[str, int] = {}

    def ensure_started(self) -> None:
        if not self.autostart:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._run())

    async def _run(self) -> None:
        # Re-attach or collect. Resubmit only when headless-auto-resume is on.
        await self.recover(resubmit=None)
        while True:
            try:
                async with self._lock:
                    await self._watch()
                    await self._submit_one()
            except asyncio.CancelledError:
                raise
            except Exception:
                _log.exception("[ComfyTV/headless] queue step failed")
            await asyncio.sleep(self.poll_s)

    async def tick(self) -> None:
        async with self._lock:
            await self._watch()
            await self._submit_one()

    async def recover(self, *, resubmit: bool | None = None,
                      only_id: str | None = None) -> list[str]:
        """Match running jobs to ComfyUI before any resubmit.

        ``resubmit`` None uses the headless-auto-resume setting (default off).
        A prompt still in GET /queue is re-attached. One in GET /history is
        collected. A prompt ComfyUI has no record of is resubmitted only when
        ``resubmit`` is true.
        """
        if resubmit is None:
            resubmit = auto_resume_enabled()
        async with self._lock:
            return await self._recover(resubmit=resubmit, only_id=only_id)

    def _prompt_state(self, prompt_id: str) -> str:
        """running, pending, history, absent, or unknown.

        Queue is read before history, the same data as GET /queue and
        GET /history/{prompt_id}. unknown means the read failed; the caller
        must not submit another prompt.
        """
        if not prompt_id:
            return "absent"
        try:
            running, pending = self.client.queue_ids()
        except Exception:
            _log.warning("[ComfyTV/headless] could not read ComfyUI queue")
            return "unknown"
        if prompt_id in running:
            return "running"
        if prompt_id in pending:
            return "pending"
        try:
            entry = self.client.history_for(prompt_id)
        except Exception:
            _log.warning("[ComfyTV/headless] could not read ComfyUI history")
            return "unknown"
        if entry:
            return "history"
        return "absent"

    def _reconcile_job(self, row: dict, *, resubmit: bool) -> str:
        prompt_id = row.get("prompt_id") or ""
        state = self._prompt_state(prompt_id)
        if state == "unknown":
            return "skipped"
        if state in ("running", "pending"):
            fields: dict = {}
            if row.get("status") != "running" or row.get("prompt_id") != prompt_id:
                fields.update(status="running", prompt_id=prompt_id, error_text=None)
            if not row.get("started_at"):
                fields["started_at"] = storage.utcnow()
            if fields:
                storage.update_headless_job(row["id"], **fields)
            _log.info(
                "[ComfyTV/headless] job %s still %s as prompt %s",
                row["id"], state, prompt_id,
            )
            return "attached"
        if state == "history":
            entry = self.client.history_for(prompt_id)
            if entry:
                self._finish_history(row, entry)
            return "finished"
        prompt = _loads(row.get("prompt_json"), {})
        partial = _loads(row.get("partial_json"), None)
        completed = _loads(row.get("completed_json"), {})
        if not isinstance(completed, dict):
            completed = {}
        reduced, partial_left = apply_completed(prompt, completed, partial)
        if not reduced or (partial_left is not None and len(partial_left) == 0):
            self._nothing_left(row, completed)
            return "finished"
        if resubmit:
            self._requeue(row)
            _log.info(
                "[ComfyTV/headless] ComfyUI has no record of prompt %s; "
                "job %s queued again",
                prompt_id or "(none)", row["id"],
            )
            return "resubmitted"
        storage.update_headless_job(
            row["id"],
            status="interrupted",
            error_text=(
                "ComfyUI has no record of this prompt. "
                "Call resume_queue to submit the unfinished stages again."
            ),
        )
        _log.info(
            "[ComfyTV/headless] job %s interrupted; auto-resume is off",
            row["id"],
        )
        return "interrupted"

    async def _recover(self, *, resubmit: bool, only_id: str | None = None) -> list[str]:
        resumed: list[str] = []
        if only_id:
            one = storage.get_headless_job(only_id)
            rows = [
                one for one in ([one] if one else [])
                if one["status"] in ("running", "submitting", "interrupted")
            ]
        else:
            rows = (
                storage.list_headless_jobs(status="running", limit=200)
                + storage.list_headless_jobs(status="submitting", limit=200)
            )
            if resubmit:
                rows = rows + storage.list_headless_jobs(status="interrupted", limit=200)
        seen: set[str] = set()
        for row in rows:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            if self._reconcile_job(row, resubmit=resubmit) == "resubmitted":
                resumed.append(row["id"])
        return resumed

    def _requeue(self, row: dict) -> None:
        storage.update_headless_job(
            row["id"],
            status="queued",
            clear_prompt_id=True,
            clear_started_at=True,
            clear_finished_at=True,
            attempt=(row.get("attempt") or 1) + 1,
            error_text=None,
        )

    def _merge_completed(self, row: dict, extra: dict) -> dict:
        current = _loads(row.get("completed_json"), {})
        if not isinstance(current, dict):
            current = {}
        for nid, info in (extra or {}).items():
            if isinstance(info, dict):
                current[str(nid)] = info
        return current

    async def _watch(self) -> None:
        for row in storage.list_headless_jobs(status="running", limit=200):
            prompt_id = row.get("prompt_id") or ""
            state = self._prompt_state(prompt_id)
            if state == "unknown":
                return
            if state in ("running", "pending"):
                self._absent.pop(row["id"], None)
                fresh = self._merge_completed(row, self.client.completed_for(prompt_id))
                if fresh != _loads(row.get("completed_json"), {}):
                    storage.update_headless_job(
                        row["id"], completed_json=json.dumps(fresh),
                    )
                progress = self.client.progress_for(prompt_id)
                if progress:
                    storage.update_headless_job(
                        row["id"], progress_json=json.dumps(progress),
                    )
                continue
            if state == "history":
                self._absent.pop(row["id"], None)
                entry = self.client.history_for(prompt_id)
                if entry:
                    self._finish_history(row, entry)
                continue
            self._absent[row["id"]] = self._absent.get(row["id"], 0) + 1
            if self._absent[row["id"]] >= _ABSENT_POLLS:
                self._absent.pop(row["id"], None)
                self._reconcile_job(row, resubmit=auto_resume_enabled())

    def _finish_history(self, row: dict, entry: dict) -> None:
        status = (entry.get("status") or {}).get("status_str")
        outputs = view_urls(entry)
        success = status != "error"
        completed = merge_completed_nodes(
            row,
            entry,
            self.client.completed_for(row.get("prompt_id") or ""),
            success=success,
        )
        finished = storage.utcnow()
        if not success:
            storage.update_headless_job(
                row["id"],
                status="error",
                error_text=history_error(entry)[:4000],
                outputs_json=json.dumps(outputs),
                completed_json=json.dumps(completed),
                finished_at=finished,
            )
            _log.info("[ComfyTV/headless] job %s failed", row["id"])
            return
        storage.update_headless_job(
            row["id"],
            status="done",
            error_text=None,
            outputs_json=json.dumps(outputs),
            completed_json=json.dumps(completed),
            finished_at=finished,
        )
        _log.info("[ComfyTV/headless] job %s done (%d outputs)", row["id"], len(outputs))

    def _nothing_left(self, row: dict, completed: dict) -> None:
        urls = []
        for info in completed.values():
            value = info.get("value") if isinstance(info, dict) else None
            if isinstance(value, str) and (
                value.startswith("/view?") or value.startswith("http")
            ):
                urls.append(value)
        recorded = dict(row)
        recorded["completed_json"] = json.dumps(completed)
        merged = merge_completed_nodes(recorded, None, None, success=True)
        storage.update_headless_job(
            row["id"],
            status="done",
            error_text=None,
            outputs_json=json.dumps(urls),
            completed_json=json.dumps(merged),
            finished_at=storage.utcnow(),
            clear_prompt_id=True,
        )
        _log.info(
            "[ComfyTV/headless] job %s already finished its stages; not resubmitting",
            row["id"],
        )

    async def _submit_one(self) -> None:
        try:
            if self.client.busy():
                return
        except Exception:
            _log.warning("[ComfyTV/headless] queue check failed; not submitting")
            return
        row = storage.next_queued_headless_job()
        if row is None:
            return
        prompt = _loads(row.get("prompt_json"), {})
        partial = _loads(row.get("partial_json"), None)
        completed = _loads(row.get("completed_json"), {})
        if not isinstance(completed, dict):
            completed = {}
        prompt, partial = apply_completed(prompt, completed, partial)
        if not prompt or (partial is not None and len(partial) == 0):
            self._nothing_left(row, completed)
            return
        claimed = storage.claim_headless_job(
            row["id"], expect="queued", become="submitting",
        )
        if claimed is None:
            return
        try:
            prompt_id = await self.client.submit(prompt, partial)
        except QueueBusy:
            storage.claim_headless_job(
                row["id"], expect="submitting", become="queued",
            )
            return
        except Exception as e:
            storage.claim_headless_job(
                row["id"], expect="submitting", become="error",
                error_text=str(e)[:4000],
                finished_at=storage.utcnow(),
            )
            _log.info("[ComfyTV/headless] job %s rejected: %s", row["id"], type(e).__name__)
            return
        storage.claim_headless_job(
            row["id"], expect="submitting", become="running", prompt_id=prompt_id,
            started_at=storage.utcnow(), clear_finished_at=True,
        )

    async def wait_for_stage(self, project_id: str, node_ref: str, timeout_s: float) -> dict | None:
        row = storage.latest_headless_job_for_stage(project_id, node_ref)
        if row is None:
            return None
        started = time.monotonic()
        while True:
            await self.tick()
            fresh = storage.get_headless_job(row["id"]) or row
            if fresh["status"] == "done":
                return {
                    "status": "done",
                    "job_id": fresh["id"],
                    "prompt_id": fresh.get("prompt_id"),
                    "output": {
                        "prompt_id": fresh.get("prompt_id"),
                        "urls": _loads(fresh.get("outputs_json"), []),
                    },
                    "outputs": _loads(fresh.get("outputs_json"), []),
                    "waited_s": round(time.monotonic() - started, 1),
                    "headless": True,
                }
            if fresh["status"] == "error":
                return {
                    "status": "error",
                    "job_id": fresh["id"],
                    "error": fresh.get("error_text") or "stage run failed",
                    "waited_s": round(time.monotonic() - started, 1),
                    "headless": True,
                }
            if fresh["status"] == "cancelled":
                return {
                    "status": "error",
                    "job_id": fresh["id"],
                    "error": "job was cancelled before it started",
                    "waited_s": round(time.monotonic() - started, 1),
                    "headless": True,
                }
            if time.monotonic() - started >= timeout_s:
                return {
                    "status": "running" if fresh["status"] == "running" else fresh["status"],
                    "job_id": fresh["id"],
                    "prompt_id": fresh.get("prompt_id"),
                    "after_output_id": 0,
                    "waited_s": round(time.monotonic() - started, 1),
                    "hint": "still running — call queue_status with this job_id, "
                            "or wait_stage again",
                    "headless": True,
                }
            await asyncio.sleep(min(0.05, self.poll_s))


_QUEUE: HeadlessQueue | None = None


def get_queue() -> HeadlessQueue:
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = HeadlessQueue()
    _QUEUE.ensure_started()
    return _QUEUE


def set_queue(queue: HeadlessQueue | None) -> None:
    global _QUEUE
    _QUEUE = queue


def auto_resume_enabled() -> bool:
    """True only when the headless-auto-resume setting is on. Default is off."""
    try:
        return bool(storage.get_setting("headless-auto-resume"))
    except Exception:
        return False


def reset_for_tests() -> None:
    global _QUEUE, _PROGRESS, _COMPLETED
    if _QUEUE is not None and _QUEUE._task is not None:
        _QUEUE._task.cancel()
    _QUEUE = None
    _PROGRESS = {}
    _COMPLETED = {}


def _prompt_from_args(args: dict) -> tuple[str, dict, list | None]:
    project_id = graph.project_id_of(args)
    prompt = args.get("prompt")
    if prompt is not None:
        if not isinstance(prompt, dict) or not prompt:
            raise ValueError("prompt must be a non-empty API-format object")
        return project_id, prompt, None
    workflow = args.get("workflow")
    if isinstance(workflow, dict) and workflow:
        if "nodes" in workflow and "links" in workflow:
            graph.adopt_workflow(project_id, workflow)
        elif all(isinstance(v, dict) and "class_type" in v for v in workflow.values()):
            return project_id, workflow, None
        else:
            raise ValueError(
                "workflow must be a UI graph (nodes and links) or an API prompt"
            )
    doc = graph.load_document(project_id)
    if not doc["nodes"]:
        raise ValueError(
            "no headless graph for this project — pass prompt (API format), "
            "workflow (UI format), or build one with add_stage / graph_edit"
        )
    return project_id, graph.graph_to_prompt(doc), None


async def enqueue_graph(args: dict) -> dict:
    project_id, prompt, partial = _prompt_from_args(args)
    row = storage.create_headless_job(
        project_id=project_id,
        kind="graph",
        prompt=prompt,
        partial=partial,
    )
    get_queue().ensure_started()
    _log.info("[ComfyTV/headless] queued graph job %s", row["id"])
    return {
        "job_id": row["id"],
        "status": "queued",
        "kind": "graph",
        "project_id": project_id,
        "headless": True,
    }


async def enqueue_stage(payload: dict) -> dict:
    project_id = graph.project_id_of(payload)
    ref = str(payload.get("node") or "")
    if not ref:
        raise ValueError("node is required (stage uid or graph node id)")
    doc = graph.load_document(project_id)
    node = graph.find_node(doc, ref)
    if node is None:
        raise ValueError(f"stage {ref!r} not found on the headless graph")
    stage_class = str(node.get("type") or "")
    _reject_not_runnable(stage_class)
    graph.local_server_or_raise(node)
    widgets = node.setdefault("widgets", {})
    widgets["force_run_token"] = int(time.time() * 1000) & 0x7FFFFFFF
    graph.save_document(project_id, doc)
    doc = graph.load_document(project_id)
    node = graph.find_node(doc, ref)
    api = graph.graph_to_prompt(doc)
    keep = graph.ancestor_ids(doc, str(node["id"]))
    api = {k: v for k, v in api.items() if k in keep}
    if str(node["id"]) not in api:
        raise ValueError(
            f"stage {ref!r} is muted, so the headless runner has nothing to queue"
        )
    partial = [str(node["id"])]
    uid = graph._uid_of(node)
    row = storage.create_headless_job(
        project_id=project_id,
        kind="stage",
        prompt=api,
        partial=partial,
        stage_ref=str(node["id"]),
        stage_uid=str(uid) if uid else None,
    )
    get_queue().ensure_started()
    _log.info("[ComfyTV/headless] queued stage job %s node %s", row["id"], node["id"])
    return {
        "job_id": row["id"],
        "status": "queued",
        "kind": "stage",
        "graph_node_id": str(node["id"]),
        "uid": uid,
        "project_id": project_id,
        "headless": True,
    }


def cancel_queued_stage(payload: dict) -> dict:
    project_id = graph.project_id_of(payload)
    ref = str(payload.get("node") or "")
    row = storage.latest_headless_job_for_stage(project_id, ref)
    if row is None:
        raise ValueError(
            f"no headless job for stage {ref!r}. cancel_stage on a page-owned "
            "run still needs the open ComfyTV page."
        )
    if row["status"] == "queued":
        claimed = storage.claim_headless_job(
            row["id"], expect="queued", become="cancelled",
            error_text="cancelled before submit",
            finished_at=storage.utcnow(),
        )
        if claimed is None:
            raise ValueError(
                "this job is already being submitted to ComfyUI. The headless "
                "queue does not interrupt a prompt that has been claimed."
            )
        return {
            "cancelled": True,
            "job_id": row["id"],
            "status": "cancelled",
            "headless": True,
        }
    if row["status"] in ("submitting", "running"):
        raise ValueError(
            "this job is already in ComfyUI's queue. The headless queue does "
            "not interrupt a running prompt; it will finish on its own."
        )
    raise ValueError(f"job {row['id']} is {row['status']} and cannot be cancelled")


async def queue_status(args: dict) -> dict:
    job_id = str(args.get("job_id") or "")
    prompt_id = str(args.get("prompt_id") or "")
    row = storage.get_headless_job(job_id) if job_id else None
    if row is None and prompt_id:
        row = storage.headless_job_by_prompt(prompt_id)
    if row is None and prompt_id:
        queue = get_queue()
        state = "absent"
        try:
            state = queue._prompt_state(prompt_id)
        except Exception:
            state = "unknown"
        if state in ("running", "pending"):
            return {
                "prompt_id": prompt_id,
                "status": "running",
                "headless": True,
            }
        try:
            entry = queue.client.history_for(prompt_id)
        except Exception:
            entry = None
        if entry:
            status = (entry.get("status") or {}).get("status_str")
            return {
                "prompt_id": prompt_id,
                "status": "error" if status == "error" else "done",
                "outputs": view_urls(entry),
                "error": history_error(entry) if status == "error" else None,
                "headless": True,
            }
        return {
            "prompt_id": prompt_id,
            "status": "missing",
            "headless": True,
        }
    if row is None:
        raise ValueError("job_id or prompt_id is required")
    await get_queue().tick()
    fresh = storage.get_headless_job(row["id"]) or row
    return {"job": public_job(fresh)}


async def list_queue(args: dict) -> dict:
    status = args.get("status")
    if status and status not in (
        "queued", "running", "done", "error", "cancelled", "interrupted",
    ):
        raise ValueError(f"unknown status {status!r}")
    try:
        limit = int(args.get("limit", 50))
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer")
    project_id = args.get("project_id")
    rows = storage.list_headless_jobs(
        status=str(status) if status else None,
        project_id=str(project_id) if project_id else None,
        limit=limit,
    )
    return {"jobs": [public_job(row) for row in rows], "headless": True}


async def resume_queue(args: dict) -> dict:
    """Reconcile against ComfyUI, then resubmit only prompts it has no record of."""
    job_id = str(args.get("job_id") or "")
    queue = get_queue()
    if job_id:
        row = storage.get_headless_job(job_id)
        if row is None:
            raise ValueError(f"job {job_id!r} not found")
        if row["status"] == "error":
            state = queue._prompt_state(row.get("prompt_id") or "")
            if state == "unknown":
                raise ValueError(
                    "could not read ComfyUI's queue or history; not submitting again"
                )
            if state in ("running", "pending"):
                queue._reconcile_job(row, resubmit=False)
            elif state == "history":
                entry = queue.client.history_for(row.get("prompt_id") or "")
                status = ((entry or {}).get("status") or {}).get("status_str")
                if status == "error":
                    queue._requeue(row)
                else:
                    queue._reconcile_job(row, resubmit=False)
            else:
                queue._requeue(row)
            fresh = storage.get_headless_job(row["id"]) or row
            return {
                "resumed": [row["id"]] if fresh["status"] == "queued" else [],
                "job": public_job(fresh),
            }
        resumed = await queue.recover(resubmit=True, only_id=job_id)
        queue.ensure_started()
        fresh = storage.get_headless_job(row["id"]) or row
        return {"resumed": resumed, "job": public_job(fresh)}
    resumed = await queue.recover(resubmit=True)
    queue.ensure_started()
    jobs = []
    for jid in resumed:
        fresh = storage.get_headless_job(jid)
        if fresh is not None:
            jobs.append(public_job(fresh))
    return {"resumed": resumed, "jobs": jobs}
