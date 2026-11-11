"""Remote executors must never be interrupted or freed.

Cancelling a job on another ComfyUI host may only POST /queue
{"delete": [that prompt_id]} while the prompt is still pending.
A prompt that is already running is marked cancel-requested, left
to finish, and its outputs are discarded.

The open page may still interrupt its own ComfyUI. That local path
is src/composables/stages/queueControl.ts (cancelPrompt) and the
canvas command fallback in src/composables/stages/mcpGraphCommands.ts.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import pytest

from ComfyTV.runners import remote_comfy
from ComfyTV.runners.base import Runner, RunnerContext
from ComfyTV.runners.remote_comfy import (
    CONNECT_TIMEOUT_S,
    JOB_TIMEOUT_S,
    JobCancelled,
    RemoteComfyUIRunner,
    RemoteJobHandle,
)

POLL_INTERVAL_S = getattr(remote_comfy, "POLL_INTERVAL_S", remote_comfy._POLL_INTERVAL_S)
READ_TIMEOUT_S = getattr(remote_comfy, "READ_TIMEOUT_S", 0)


def remote_client_timeout():
    return remote_comfy.remote_client_timeout()


def remote_poll_interval_s():
    return remote_comfy.remote_poll_interval_s()


_REPO = Path(__file__).resolve().parents[1]
_REMOTE_HTTP = (
    "runners/remote_comfy.py",
    "api/servers.py",
    "api/mcp_tools/info.py",
)
_FORBIDDEN = ("/interrupt", "/free")
_PROMPT_ID = "pid-9"


def _assert_no_interrupt_or_free(calls: list[tuple]) -> None:
    for method, path, _body in calls:
        assert path not in _FORBIDDEN, f"{method} {path}"
        assert not path.startswith("/interrupt"), path
        assert not path.startswith("/free"), path
        assert "interrupt" not in path
        assert path.rstrip("/").endswith("free") is False


class _Resp:
    def __init__(self, status: int, payload):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    async def text(self):
        if isinstance(self._payload, str):
            return self._payload
        return json.dumps(self._payload)

    @property
    def content(self):
        return self

    async def iter_chunked(self, _n):
        yield b"nope"


class ScriptedRemote:
    def __init__(self, mode: str, *, history_misses: int = 0, queue_status: int = 200):
        self.mode = mode
        self.history_misses = history_misses
        self.queue_status = queue_status
        self.calls: list[tuple] = []
        self.timeouts = []
        self.history_gets = 0
        self.arm_cancel = None
        if mode == "pending":
            self.running = [[0, "someone-else"]]
            self.pending = [[1, "other-pending"], [2, _PROMPT_ID]]
        elif mode == "running":
            self.running = [[0, _PROMPT_ID]]
            self.pending = [[1, "other-pending"]]
        else:
            self.running = []
            self.pending = []

    def request(self, method: str, url: str, body):
        path = urlparse(url).path
        self.calls.append((method, path, body))
        if path == "/prompt":
            if self.arm_cancel is not None:
                self.arm_cancel()
            return _Resp(200, {"prompt_id": _PROMPT_ID})
        if path == "/queue" and method == "GET":
            if self.queue_status != 200:
                return _Resp(self.queue_status, {"error": "nope"})
            return _Resp(200, {
                "queue_running": self.running,
                "queue_pending": self.pending,
            })
        if path == "/queue" and method == "POST":
            drop = set((body or {}).get("delete") or [])
            self.pending = [item for item in self.pending if item[1] not in drop]
            return _Resp(200, {})
        if path.startswith("/history/"):
            self.history_gets += 1
            if self.history_gets <= self.history_misses:
                return _Resp(200, {})
            return _Resp(200, {
                _PROMPT_ID: {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {
                        "9": {"images": [{
                            "filename": "a.png",
                            "subfolder": "",
                            "type": "output",
                        }]},
                    },
                },
            })
        if path == "/view":
            return _Resp(200, {})
        return _Resp(404, {})


def _install_session(monkeypatch, script: ScriptedRemote) -> None:
    from ComfyTV.runners import remote_comfy

    class FakeSession:
        def __init__(self, *args, timeout=None, **kwargs):
            script.timeouts.append(timeout)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def get(self, url, **kwargs):
            return script.request("GET", url, None)

        def post(self, url, **kwargs):
            return script.request("POST", url, kwargs.get("json"))

        async def ws_connect(self, url, **kwargs):
            script.calls.append(("WS", urlparse(url).path, None))
            raise aiohttp.ClientConnectionError("ws disabled in test")

    monkeypatch.setattr(remote_comfy.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(
        remote_comfy,
        "prepare_workflow",
        lambda runner_id, kinds, ctx: (
            {"1": {"class_type": "Note", "inputs": {}}},
            {"type": "ui_save_url", "node": "9"},
        ),
    )


def _runner(reset_job: bool = True):
    from ComfyTV import storage

    server = storage.create_server(label="spark", host="spark.example", port=8188)
    job_id = "job-1"
    storage.create_remote_job(
        job_id=job_id,
        server_id=server["id"],
        server_label="spark",
        project_id="p",
        stage_node_id="7",
        stage_uid="uid-7",
    )
    handle = RemoteJobHandle(job_id, "7")
    base = Runner("image/X", "X", {"image"})
    return RemoteComfyUIRunner(base, server, handle), handle


def _arm(script: ScriptedRemote, handle: RemoteJobHandle) -> None:
    script.arm_cancel = handle.cancel_event.set


def _posts(script: ScriptedRemote, path: str) -> list:
    return [body for method, got, body in script.calls if method == "POST" and got == path]


class TestRemoteCancelDoesNotInterrupt:
    def test_remote_http_modules_do_not_name_interrupt_or_free(self):
        for rel in _REMOTE_HTTP:
            text = (_REPO / rel).read_text(encoding="utf-8")
            for token in _FORBIDDEN:
                assert token not in text, f"{rel} still contains {token}"

    def test_open_page_cancel_is_the_only_documented_local_interrupt(self):
        page = (_REPO / "src/composables/stages/queueControl.ts").read_text(encoding="utf-8")
        canvas = (_REPO / "src/composables/stages/mcpGraphCommands.ts").read_text(encoding="utf-8")
        assert "/interrupt" in page
        assert "open page" in page
        assert "remote executor" in page
        assert "app.api.interrupt" in canvas or "api.interrupt" in canvas
        assert "open page" in canvas

    async def test_pending_prompt_is_deleted_and_nothing_else_is_touched(self, reset_db, monkeypatch):
        script = ScriptedRemote("pending")
        _install_session(monkeypatch, script)
        runner, handle = _runner()
        _arm(script, handle)
        with pytest.raises(JobCancelled, match="pending"):
            await runner.invoke(RunnerContext(kind="image"))
        _assert_no_interrupt_or_free(script.calls)
        assert _posts(script, "/queue") == [{"delete": [_PROMPT_ID]}]
        assert not any(path == "/view" for _m, path, _b in script.calls)
        assert script.history_gets == 0

    async def test_running_prompt_is_cancel_requested_and_outputs_are_ignored(
        self, reset_db, monkeypatch,
    ):
        from ComfyTV import storage

        script = ScriptedRemote("running")
        _install_session(monkeypatch, script)
        runner, handle = _runner()
        _arm(script, handle)
        with pytest.raises(JobCancelled, match="ignored"):
            await runner.invoke(RunnerContext(kind="image"))
        _assert_no_interrupt_or_free(script.calls)
        assert _posts(script, "/queue") == []
        assert _posts(script, "/interrupt") == []
        assert _posts(script, "/free") == []
        assert script.history_gets >= 1
        assert not any(path == "/view" for _m, path, _b in script.calls)
        row = storage.get_remote_job(handle.job_id)
        assert row["status"] == "cancel-requested"
        assert row["remote_prompt_id"] == _PROMPT_ID
        assert row["output_id"] is None
        assert handle.discard_outputs is True

    async def test_unreadable_queue_does_not_interrupt(self, reset_db, monkeypatch):
        from ComfyTV import storage

        script = ScriptedRemote("running", queue_status=500)
        _install_session(monkeypatch, script)
        runner, handle = _runner()
        _arm(script, handle)
        with pytest.raises(JobCancelled, match="ignored"):
            await runner.invoke(RunnerContext(kind="image"))
        _assert_no_interrupt_or_free(script.calls)
        assert _posts(script, "/queue") == [{"delete": [_PROMPT_ID]}]
        assert storage.get_remote_job(handle.job_id)["status"] == "cancel-requested"
        assert not any(path == "/view" for _m, path, _b in script.calls)


class TestRemoteTimeoutSettings:
    def test_defaults_match_the_previous_constants(self):
        from ComfyTV.settings import SETTINGS_SPEC, effective_settings

        cfg = effective_settings({})
        assert SETTINGS_SPEC["remote-connect-timeout-s"]["default"] == CONNECT_TIMEOUT_S == 10
        assert SETTINGS_SPEC["remote-read-timeout-s"]["default"] == READ_TIMEOUT_S == 0
        assert SETTINGS_SPEC["remote-poll-interval-s"]["default"] == int(POLL_INTERVAL_S) == 2
        assert SETTINGS_SPEC["remote-job-timeout-s"]["default"] == JOB_TIMEOUT_S == 3600
        assert cfg["remote-connect-timeout-s"] == 10
        assert cfg["remote-read-timeout-s"] == 0
        assert cfg["remote-poll-interval-s"] == 2
        assert cfg["remote-job-timeout-s"] == 3600

    def test_client_timeout_reads_settings(self, monkeypatch):
        from ComfyTV.runners import remote_comfy

        values = {
            "remote-connect-timeout-s": 7,
            "remote-read-timeout-s": 15,
            "remote-poll-interval-s": 4,
            "remote-job-timeout-s": 90,
        }
        monkeypatch.setattr(remote_comfy.storage, "get_setting", lambda key: values[key])
        timeout = remote_client_timeout()
        assert timeout.total == 90
        assert timeout.sock_connect == 7
        assert timeout.sock_read == 15
        assert remote_poll_interval_s() == 4

    def test_zero_read_timeout_leaves_sock_read_unset(self, monkeypatch):
        from ComfyTV.runners import remote_comfy

        values = {
            "remote-connect-timeout-s": 10,
            "remote-read-timeout-s": 0,
            "remote-poll-interval-s": 2,
            "remote-job-timeout-s": 3600,
        }
        monkeypatch.setattr(remote_comfy.storage, "get_setting", lambda key: values[key])
        timeout = remote_client_timeout()
        assert timeout.sock_connect == 10
        assert timeout.sock_read is None
        assert timeout.total == 3600

    async def test_poll_interval_comes_from_settings(self, reset_db, monkeypatch):
        from ComfyTV.runners import remote_comfy

        script = ScriptedRemote("running", history_misses=1)
        _install_session(monkeypatch, script)
        sleeps: list[float] = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr(remote_comfy.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(
            remote_comfy.storage,
            "get_setting",
            lambda key: {
                "remote-connect-timeout-s": 10,
                "remote-read-timeout-s": 0,
                "remote-poll-interval-s": 6,
                "remote-job-timeout-s": 3600,
            }[key],
        )
        runner, _handle = _runner()
        await runner.invoke(RunnerContext(kind="image"))
        assert sleeps == [6]


def _stub_hidden_holder():
    name = "comfy_api.latest._io"
    mod = types.ModuleType(name)
    mod.Hidden = types.SimpleNamespace(unique_id="unique_id")

    class HiddenHolder:
        @staticmethod
        def from_dict(data):
            return data

    mod.HiddenHolder = HiddenHolder
    sys.modules[name] = mod


class TestRunJobAfterCancelRequested:
    async def test_remote_error_after_cancel_requested_is_cancelled(self, reset_db):
        from ComfyTV import storage
        from ComfyTV.api.servers import _run_job

        _stub_hidden_holder()
        server = storage.create_server(label="spark2", host="spark.example", port=8188)
        storage.create_remote_job(
            job_id="job-err",
            server_id=server["id"],
            server_label="spark2",
            project_id="p",
            stage_node_id="3",
        )
        handle = RemoteJobHandle("job-err", "3")
        handle.discard_outputs = True
        handle.remote_prompt_id = _PROMPT_ID

        class Boom:
            @staticmethod
            def PREPARE_CLASS_CLONE(_prompt):
                return Boom()

            async def EXECUTE_NORMALIZED_ASYNC(self, **kwargs):
                raise RuntimeError("remote blew up")

        await _run_job("job-err", handle, Boom, {}, "3", "p")
        row = storage.get_remote_job("job-err")
        assert row["status"] == "cancelled"
        assert row["output_id"] is None
        assert "remote blew up" in (row["error_text"] or "")


class TestInvokeUsesDefaultTimeout:
    async def test_session_timeout_defaults(self, reset_db, monkeypatch):
        script = ScriptedRemote("pending")
        _install_session(monkeypatch, script)
        runner, handle = _runner()
        _arm(script, handle)
        with pytest.raises(JobCancelled):
            await runner.invoke(RunnerContext(kind="image"))
        timeout = script.timeouts[0]
        assert timeout.total == JOB_TIMEOUT_S
        assert timeout.sock_connect == CONNECT_TIMEOUT_S
        assert timeout.sock_read is None
