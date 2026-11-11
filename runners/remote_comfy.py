"""Run a ComfyUI workflow on another machine.

Cancelling that job never calls the remote interrupt or free routes.
A still-pending prompt is removed with POST /queue {"delete": [prompt_id]}.
A prompt that has already started is marked cancel-requested, allowed to
finish, and its files are not downloaded.

An open ComfyTV page can still interrupt the ComfyUI serving that page.
That local path is src/composables/stages/queueControl.ts (cancelPrompt)
and the Comfy.Interrupt fallback in src/composables/stages/mcpGraphCommands.ts.
"""

import asyncio
import contextvars
import hashlib
import json
import logging
import re
import urllib.parse
import uuid
from pathlib import Path

import aiohttp

from .. import storage
from .base import Runner, RunnerContext
from .local_comfy import (
    _save_files_from,
    _view_url,
    prepare_workflow,
)


_log = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 10
# 0 keeps the previous behavior: no per-read socket limit. The job timeout
# still bounds the whole call.
READ_TIMEOUT_S = 0
JOB_TIMEOUT_S = 3600
POLL_INTERVAL_S = 2
_POLL_INTERVAL_S = POLL_INTERVAL_S

_TIMEOUT_DEFAULTS = {
    "remote-connect-timeout-s": CONNECT_TIMEOUT_S,
    "remote-read-timeout-s": READ_TIMEOUT_S,
    "remote-poll-interval-s": POLL_INTERVAL_S,
    "remote-job-timeout-s": JOB_TIMEOUT_S,
}

_ANNOTATED_RE = re.compile(r"^(.+?) \[(input|output|temp)\]$")

_REMOTE_UPLOAD_SUBFOLDER = "comfytv/upload"
_REMOTE_RESULT_SUBFOLDER = "comfytv/remote"


class JobCancelled(RuntimeError):
    pass


def _setting_int(key: str) -> int:
    default = _TIMEOUT_DEFAULTS[key]
    try:
        value = storage.get_setting(key)
    except Exception:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def remote_timeouts() -> dict[str, float | None]:
    connect = _setting_int("remote-connect-timeout-s")
    read = _setting_int("remote-read-timeout-s")
    poll = _setting_int("remote-poll-interval-s")
    job = _setting_int("remote-job-timeout-s")
    if connect < 1:
        connect = CONNECT_TIMEOUT_S
    if poll < 1:
        poll = POLL_INTERVAL_S
    if job < 1:
        job = JOB_TIMEOUT_S
    if read < 0:
        read = 0
    return {
        "connect": float(connect),
        "read": None if read == 0 else float(read),
        "poll": float(poll),
        "job": float(job),
    }


def remote_client_timeout() -> aiohttp.ClientTimeout:
    limits = remote_timeouts()
    return aiohttp.ClientTimeout(
        total=limits["job"],
        sock_connect=limits["connect"],
        sock_read=limits["read"],
    )


def remote_poll_interval_s() -> float:
    return float(remote_timeouts()["poll"])


CURRENT_JOB: 'contextvars.ContextVar[RemoteJobHandle | None]' = \
    contextvars.ContextVar("comfytv_current_remote_job", default=None)


class RemoteJobHandle:

    def __init__(self, job_id: str, stage_node_id: str):
        self.job_id = job_id
        self.stage_node_id = str(stage_node_id)
        self.cancel_event = asyncio.Event()
        self.remote_prompt_id: str | None = None
        self.discard_outputs = False
        self.cancel_marked = False


def _safe_path_component(s: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", s or "").strip("_") or "server"


def emit_remote_progress(node_id: str, value: float, max_: float,
                         text: str = "") -> None:
    try:
        from server import PromptServer
        payload: dict = {"node": str(node_id), "value": value, "max": max_}
        if text:
            payload["text"] = text
        PromptServer.instance.send_sync("comfytv-remote-progress", payload)
    except Exception as e:
        _log.debug("[ComfyTV/remote] progress emit failed: %s", e)


def _collect_annotated_refs(workflow: dict) -> dict[str, list[tuple[str, str]]]:
    refs: dict[str, list[tuple[str, str]]] = {}
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        for input_name, value in (node.get("inputs") or {}).items():
            if isinstance(value, str) and _ANNOTATED_RE.match(value):
                refs.setdefault(value, []).append((node_id, input_name))
    return refs


def _upload_name_for(local_path: Path) -> str:
    try:
        stamp = f"{local_path}:{local_path.stat().st_mtime_ns}"
    except OSError:
        stamp = str(local_path)
    digest = hashlib.sha1(stamp.encode("utf-8", "replace")).hexdigest()[:12]
    return f"{digest}-{local_path.name}"


def _aggregate_progress(nodes_dict: dict, total_nodes: float) -> tuple[float, float]:
    finished = 0.0
    running_frac = 0.0
    for st in nodes_dict.values():
        if not isinstance(st, dict):
            continue
        stt = st.get("state")
        if stt in ("finished", "cached"):
            finished += 1.0
        elif stt == "running":
            try:
                v = float(st.get("value") or 0)
                m = float(st.get("max") or 1)
                if m > 0:
                    running_frac += min(1.0, v / m)
            except (TypeError, ValueError):
                pass
    return finished + running_frac, total_nodes


class RemoteComfyUIRunner(Runner):

    def __init__(self, base: Runner, server: dict, job: RemoteJobHandle | None = None):
        super().__init__(base.id, base.label, set(base.kinds))
        self.server = server
        self.job = job

    @property
    def base_url(self) -> str:
        return f"http://{self.server['host']}:{self.server['port']}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.server['host']}:{self.server['port']}"

    def _check_cancel(self) -> None:
        if self.job is not None and self.job.cancel_event.is_set():
            raise JobCancelled(f"job {self.job.job_id} cancelled")

    def _poll_s(self) -> float:
        cached = getattr(self, "_timeouts", None)
        if isinstance(cached, dict) and cached.get("poll"):
            return float(cached["poll"])
        return remote_poll_interval_s()

    def _emit_local_progress(self, value: float, max_: float) -> None:
        if self.job is None:
            return
        emit_remote_progress(self.job.stage_node_id, value, max_)

    async def _upload_local_files(self, session: aiohttp.ClientSession,
                                  workflow: dict) -> None:
        import folder_paths

        refs = _collect_annotated_refs(workflow)
        if not refs:
            return
        uploaded: dict[str, str] = {}
        for annotated, sites in refs.items():
            try:
                local_path = Path(folder_paths.get_annotated_filepath(annotated))
            except Exception:
                continue
            if not local_path.is_file():
                _log.info(
                    "[ComfyTV/remote] %r not found locally — leaving the "
                    "reference for the remote to resolve", annotated,
                )
                continue
            if annotated not in uploaded:
                self._check_cancel()
                name = _upload_name_for(local_path)
                fh = open(local_path, "rb")
                try:
                    form = aiohttp.FormData()
                    form.add_field("image", fh, filename=name,
                                   content_type="application/octet-stream")
                    form.add_field("subfolder", _REMOTE_UPLOAD_SUBFOLDER)
                    form.add_field("type", "input")
                    form.add_field("overwrite", "true")
                    async with session.post(f"{self.base_url}/upload/image",
                                            data=form) as resp:
                        if resp.status != 200:
                            body = (await resp.text())[:300]
                            raise RuntimeError(
                                f"uploading {local_path.name!r} to "
                                f"{self.server['label']!r} failed "
                                f"(HTTP {resp.status}): {body}"
                            )
                        info = await resp.json()
                finally:
                    fh.close()
                sub = info.get("subfolder") or _REMOTE_UPLOAD_SUBFOLDER
                rname = info.get("name") or name
                path = f"{sub}/{rname}" if sub else rname
                uploaded[annotated] = f"{path} [input]"
                _log.info("[ComfyTV/remote] uploaded %s -> %s",
                          annotated, uploaded[annotated])
            for node_id, input_name in sites:
                workflow[node_id]["inputs"][input_name] = uploaded[annotated]

    async def _queue_prompt(self, session: aiohttp.ClientSession,
                            workflow: dict, client_id: str) -> str:
        body = {"prompt": workflow, "client_id": client_id}
        async with session.post(f"{self.base_url}/prompt", json=body) as resp:
            text = await resp.text()
            if resp.status != 200:
                detail = text[:500]
                try:
                    err = json.loads(text)
                    detail = (err.get("error") or {}).get("message") or detail
                    node_errors = err.get("node_errors") or {}
                    if node_errors:
                        first = next(iter(node_errors.values()))
                        msgs = [e.get("message", "") for e in first.get("errors", [])]
                        detail += " — " + "; ".join(m for m in msgs if m)
                except (ValueError, AttributeError, StopIteration):
                    pass
                raise RuntimeError(
                    f"remote {self.server['label']!r} rejected the workflow: {detail}"
                )
            data = json.loads(text)
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(
                f"remote {self.server['label']!r} returned no prompt_id"
            )
        return prompt_id

    @staticmethod
    def _prompt_ids(rows) -> set[str]:
        found: set[str] = set()
        if not isinstance(rows, list):
            return found
        for item in rows:
            if isinstance(item, (list, tuple)) and len(item) > 1 and item[1]:
                found.add(str(item[1]))
        return found

    async def _queue_state(self, session: aiohttp.ClientSession,
                           prompt_id: str) -> str:
        """pending, running, absent, or unknown. Never touches other prompts."""
        try:
            async with session.get(f"{self.base_url}/queue") as resp:
                if resp.status != 200:
                    return "unknown"
                payload = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError):
            return "unknown"
        if not isinstance(payload, dict):
            return "unknown"
        pid = str(prompt_id)
        if pid in self._prompt_ids(payload.get("queue_pending")):
            return "pending"
        if pid in self._prompt_ids(payload.get("queue_running")):
            return "running"
        return "absent"

    async def _delete_pending(self, session: aiohttp.ClientSession,
                              prompt_id: str) -> None:
        # ComfyUI v0.45 removes only pending items here. A running prompt stays.
        async with session.post(
            f"{self.base_url}/queue",
            json={"delete": [prompt_id]},
        ) as resp:
            if resp.status != 200:
                body = (await resp.text())[:300]
                _log.warning(
                    "[ComfyTV/remote] pending delete for %s returned HTTP %s: %s",
                    prompt_id, resp.status, body,
                )

    def _mark_cancel_requested(self) -> None:
        if self.job is None:
            return
        self.job.discard_outputs = True
        if self.job.cancel_marked:
            return
        self.job.cancel_marked = True
        try:
            storage.update_remote_job(
                self.job.job_id,
                status="cancel-requested",
                remote_prompt_id=self.job.remote_prompt_id,
                error_text="cancel requested; remote prompt left running",
            )
        except Exception as e:
            _log.warning("[ComfyTV/remote] could not record cancel-requested: %s", e)

    async def _cancel_on_remote(self, session: aiohttp.ClientSession,
                                prompt_id: str) -> str:
        """Release one remote prompt without stopping the host.

        Returns deleted, cancel_requested, or absent.
        """
        try:
            state = await self._queue_state(session, prompt_id)
            if state == "pending":
                await self._delete_pending(session, prompt_id)
                # A follow-up read that still says pending is the queue view
                # lagging the delete. Only a prompt that has since started
                # must be left running.
                if await self._queue_state(session, prompt_id) == "running":
                    self._mark_cancel_requested()
                    return "cancel_requested"
                return "deleted"
            if state == "running":
                self._mark_cancel_requested()
                return "cancel_requested"
            if state == "unknown":
                await self._delete_pending(session, prompt_id)
                self._mark_cancel_requested()
                return "cancel_requested"
            return "absent"
        except aiohttp.ClientError as e:
            _log.warning("[ComfyTV/remote] cancel on remote failed: %s", e)
            self._mark_cancel_requested()
            return "cancel_requested"

    async def _on_cancel_during_wait(self, session: aiohttp.ClientSession,
                                     prompt_id: str) -> None:
        if self.job is None or not self.job.cancel_event.is_set():
            return
        if self.job.discard_outputs:
            return
        outcome = await self._cancel_on_remote(session, prompt_id)
        if outcome == "deleted":
            raise JobCancelled(
                f"job {self.job.job_id} removed from the remote queue while pending"
            )
        if outcome == "absent":
            entry = await self._fetch_history(session, prompt_id)
            if entry is None:
                raise JobCancelled(
                    f"job {self.job.job_id} left the remote queue while pending"
                )
            self._mark_cancel_requested()

    async def _fetch_history(self, session: aiohttp.ClientSession,
                             prompt_id: str) -> dict | None:
        async with session.get(f"{self.base_url}/history/{prompt_id}") as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
        return data.get(prompt_id)

    @staticmethod
    def _raise_history_error(entry: dict, server_label: str) -> None:
        status = entry.get("status") or {}
        if status.get("status_str") != "error":
            return
        detail = ""
        for msg in status.get("messages") or []:
            if isinstance(msg, (list, tuple)) and len(msg) == 2 \
                    and msg[0] == "execution_error":
                d = msg[1] or {}
                detail = (
                    f"{d.get('node_type', '?')} (node {d.get('node_id', '?')}): "
                    f"{d.get('exception_message', 'unknown error')}"
                )
                break
        raise RuntimeError(
            f"remote {server_label!r} execution failed"
            + (f" — {detail}" if detail else "")
        )

    async def _wait_via_ws(self, session: aiohttp.ClientSession,
                           prompt_id: str, client_id: str,
                           total_nodes: float) -> bool:
        try:
            ws = await session.ws_connect(
                f"{self.ws_url}/ws?clientId={client_id}",
                heartbeat=15.0,
            )
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            _log.info("[ComfyTV/remote] ws connect failed (%s); polling instead", e)
            return False

        try:
            while True:
                await self._on_cancel_during_wait(session, prompt_id)
                try:
                    msg = await ws.receive(timeout=self._poll_s())
                except asyncio.TimeoutError:
                    continue
                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING,
                                aiohttp.WSMsgType.ERROR):
                    return False
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    event = json.loads(msg.data)
                except ValueError:
                    continue
                etype = event.get("type")
                data = event.get("data") or {}
                if data.get("prompt_id") not in (None, prompt_id):
                    continue
                if etype == "progress_state":
                    nodes = data.get("nodes") or {}
                    if nodes:
                        v, m = _aggregate_progress(nodes, total_nodes)
                        self._emit_local_progress(v, m)
                elif etype == "execution_success":
                    self._emit_local_progress(total_nodes, total_nodes)
                    return True
                elif etype == "executing" and data.get("node") is None \
                        and data.get("prompt_id") == prompt_id:
                    return True
                elif etype == "execution_error":
                    raise RuntimeError(
                        f"remote {self.server['label']!r} execution failed — "
                        f"{data.get('node_type', '?')} "
                        f"(node {data.get('node_id', '?')}): "
                        f"{data.get('exception_message', 'unknown error')}"
                    )
                elif etype == "execution_interrupted":
                    raise JobCancelled(
                        f"remote {self.server['label']!r} interrupted the job"
                    )
        finally:
            await ws.close()

    async def _wait_via_poll(self, session: aiohttp.ClientSession,
                             prompt_id: str) -> None:
        while True:
            await self._on_cancel_during_wait(session, prompt_id)
            entry = await self._fetch_history(session, prompt_id)
            if entry is not None:
                status = entry.get("status") or {}
                if (status.get("completed")
                        or status.get("status_str") == "error"
                        or entry.get("outputs")):
                    return
            await asyncio.sleep(self._poll_s())

    async def _download_file(self, session: aiohttp.ClientSession,
                             item: dict, dest_dir: Path) -> str:
        import folder_paths

        filename = item.get("filename", "")
        if not filename:
            raise RuntimeError("remote result item has no filename")
        qs = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        })
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / Path(filename).name
        async with session.get(f"{self.base_url}/view?{qs}") as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"downloading {filename!r} from {self.server['label']!r} "
                    f"failed (HTTP {resp.status})"
                )
            with open(dest, "wb") as fh:
                async for chunk in resp.content.iter_chunked(1 << 20):
                    fh.write(chunk)
        out_root = Path(folder_paths.get_output_directory())
        subfolder = dest.parent.relative_to(out_root).as_posix()
        return _view_url(dest.name, subfolder, "output")

    async def _extract_result(self, session: aiohttp.ClientSession,
                              entry: dict, result_meta: dict) -> str:
        rtype = result_meta.get("type")
        node_id = result_meta.get("node")
        outputs = entry.get("outputs") or {}

        import folder_paths
        dest_dir = (
            Path(folder_paths.get_output_directory())
            / _REMOTE_RESULT_SUBFOLDER
            / _safe_path_component(self.server["label"])
            / uuid.uuid4().hex[:8]
        )

        if rtype == "ui_save_url":
            items = _save_files_from(outputs.get(node_id) or {})
            if not items:
                raise RuntimeError(
                    f"remote save node {node_id!r} produced no files"
                )
            return await self._download_file(session, items[0], dest_dir)

        if rtype == "ui_save_batch":
            files: list[dict] = []
            ordered_ids = [node_id] if result_meta.get("only_node") else                 [node_id] + [k for k in outputs.keys() if k != node_id]
            for nid in ordered_ids:
                files.extend(_save_files_from(outputs.get(nid) or {}))
            if not files:
                raise RuntimeError("remote workflow produced no image files")
            images = []
            for i, it in enumerate(files):
                self._check_cancel()
                url = await self._download_file(session, it, dest_dir)
                images.append({
                    "index": str(i + 1),
                    "label": f"#{i + 1}",
                    "image_url": url,
                })
            return json.dumps({"images": images})

        if rtype == "multi":
            multi: dict[str, str] = {}
            for sub in result_meta.get("outputs") or []:
                key = str(sub.get("id") or sub.get("kind") or sub.get("node"))
                multi[key] = await self._extract_result(session, entry, sub)
            return json.dumps({"multi": multi})

        if rtype == "graph_output_first":
            out = outputs.get(node_id) or {}
            for key in ("text", "string", "value"):
                vals = out.get(key)
                if isinstance(vals, list) and vals:
                    return str(vals[0])
            raise RuntimeError(
                f"remote node {node_id!r} exposed no text output in history — "
                f"text-result workflows need a UI text node (ShowText / "
                f"PreviewAny) to run remotely."
            )

        raise RuntimeError(f"unsupported result.type for remote run: {rtype!r}")

    async def invoke(self, ctx: RunnerContext):
        workflow, result_meta = prepare_workflow(self.id, self.kinds, ctx)
        client_id = f"comfytv-remote-{uuid.uuid4().hex[:12]}"

        self._timeouts = remote_timeouts()
        timeout = aiohttp.ClientTimeout(
            total=self._timeouts["job"],
            sock_connect=self._timeouts["connect"],
            sock_read=self._timeouts["read"],
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                await self._upload_local_files(session, workflow)
                self._check_cancel()
                prompt_id = await self._queue_prompt(session, workflow, client_id)
            except aiohttp.ClientError as e:
                raise RuntimeError(
                    f"can't reach remote {self.server['label']!r} "
                    f"({self.server['host']}:{self.server['port']}): {e}"
                ) from e

            if self.job is not None:
                self.job.remote_prompt_id = prompt_id
            _log.info("[ComfyTV/remote] %s queued on %s as %s",
                      self.id, self.server["label"], prompt_id)

            try:
                await self._on_cancel_during_wait(session, prompt_id)
                finished = await self._wait_via_ws(
                    session, prompt_id, client_id, float(len(workflow) or 1))
                if not finished:
                    await self._wait_via_poll(session, prompt_id)
            except JobCancelled:
                if self.job is None or not self.job.discard_outputs:
                    if self.job is not None and self.job.cancel_event.is_set():
                        outcome = await self._cancel_on_remote(session, prompt_id)
                        if outcome == "deleted":
                            raise JobCancelled(
                                f"job {self.job.job_id} removed from the remote "
                                "queue while pending"
                            ) from None
                raise

            if self.job is not None and self.job.discard_outputs:
                raise JobCancelled(
                    f"job {self.job.job_id} cancel requested; remote outputs ignored"
                )

            entry = await self._fetch_history(session, prompt_id)
            if entry is None:
                raise RuntimeError(
                    f"remote {self.server['label']!r} finished but has no "
                    f"history for {prompt_id} — was it restarted mid-run?"
                )
            self._raise_history_error(entry, self.server["label"])
            return await self._extract_result(session, entry, result_meta)
