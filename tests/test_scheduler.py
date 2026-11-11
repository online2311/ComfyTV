"""Scheduler: shot affinity, class gate, empty queue, retry, cancel.

ComfyUI is mocked. These tests do not open a socket and do not ask a host
to stop or free memory.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from ComfyTV.api.headless_queue import (
    ComfyQueueClient,
    HeadlessQueue,
    public_job,
    reset_for_tests,
    set_queue,
)
from ComfyTV.api.scheduler import Scheduler, set_scheduler

PROMPT = {
    "1": {"class_type": "ComfyTV.ImageStage", "inputs": {"main_prompt": "a cat"}},
}

PUBLIC_JOB_KEYS = {
    "job_id", "status", "kind", "project_id", "prompt_id", "stage_ref",
    "stage_uid", "progress", "outputs", "error", "completed_nodes",
    "attempt", "created_at", "updated_at", "headless",
}


def _entry(item, filename="frame.png"):
    return {
        "prompt": item,
        "outputs": {
            "9": {"images": [{"filename": filename, "subfolder": "", "type": "output"}]},
        },
        "status": {"status_str": "success", "completed": True, "messages": []},
    }


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


class Host:
    def __init__(self) -> None:
        self.running: list = []
        self.pending: list = []
        self.history: dict = {}
        self.files = {"frame.png": b"frame-bytes", "cancel-me.png": b"nope"}
        self.fail_post = None
        self.fail_get_queue = False
        self.autocomplete = False
        self.posts = 0
        self._n = 0

    def promote(self) -> None:
        self.running.extend(self.pending)
        self.pending = []

    def finish(self, filename="frame.png") -> None:
        items = list(self.running) + list(self.pending)
        self.running = []
        self.pending = []
        for item in items:
            self.history[str(item[1])] = _entry(item, filename)

    async def handle(self, method: str, url: str, json_body):
        path = urlsplit(url).path.rstrip("/")
        parts = [part.lower() for part in path.split("/") if part]
        if "interrupt" in parts or "free" in parts:
            raise AssertionError(url)
        if path.endswith("/queue") and method == "GET":
            if self.fail_get_queue:
                return 500, None
            return 200, {"queue_running": self.running, "queue_pending": self.pending}
        if path.endswith("/queue") and method == "POST":
            delete = list((json_body or {}).get("delete") or [])
            self.pending = [item for item in self.pending if str(item[1]) not in set(map(str, delete))]
            return 200, {"delete": delete}
        if path.endswith("/prompt") and method == "POST":
            self.posts += 1
            if self.fail_post == "before":
                raise TimeoutError("http://secret.example/should-not-be-logged")
            self._n += 1
            prompt_id = f"p-{self._n}"
            client_id = (json_body or {}).get("client_id")
            item = [self._n, prompt_id, (json_body or {}).get("prompt"), {"client_id": client_id}, []]
            if self.fail_post == "after":
                self.pending.append(item)
                raise TimeoutError("http://secret.example/should-not-be-logged")
            if self.autocomplete:
                self.history[prompt_id] = _entry(item)
            else:
                self.pending.append(item)
            return 200, {"prompt_id": prompt_id}
        if "/history/" in path and method == "GET":
            prompt_id = path.rsplit("/", 1)[-1]
            entry = self.history.get(prompt_id)
            return 200, ({prompt_id: entry} if entry else {})
        if path.endswith("/history") and method == "GET":
            return 200, dict(self.history)
        if path.endswith("/view") and method == "GET":
            name = parse_qs(urlsplit(url).query).get("filename", [""])[0]
            body = self.files.get(name)
            if body is None:
                return 404, b""
            return 200, body
        raise AssertionError(url)


class Fleet:
    def __init__(self) -> None:
        self.hosts: dict[str, Host] = {}
        self.calls: list[tuple] = []

    def add(self, base_url: str) -> Host:
        host = Host()
        self.hosts[base_url.rstrip("/")] = host
        return host

    def _host(self, url: str) -> Host:
        base = urlsplit(url)
        origin = f"{base.scheme}://{base.netloc}"
        host = self.hosts.get(origin)
        if host is None:
            raise AssertionError(url)
        return host

    async def __call__(self, method, url, json_body=None):
        self.calls.append((method, url, json_body))
        return await self._host(url).handle(method, url, json_body)

    def posts_to(self, origin: str) -> int:
        return self.hosts[origin.rstrip("/")].posts


def _stopped(calls) -> list:
    bad = []
    for method, url, _body in calls:
        parts = [part.lower() for part in urlsplit(url).path.split("/") if part]
        if "interrupt" in parts or "free" in parts:
            bad.append((method, url))
    return bad


def _admit(storage, row, running=0, pending=0):
    return storage.record_observation(
        row["id"],
        health="reachable",
        running_count=running,
        pending_count=pending,
        queue_known=True,
        queue_empty=running == 0 and pending == 0,
        gpu=None,
        error=None,
        now=datetime.now(timezone.utc),
    )


def _register(storage, fleet, name, gpu, base, max_concurrency=1):
    row = storage.register_executor(
        name=name,
        base_url=base,
        kind="permanent",
        gpu_class=gpu,
        max_concurrency=max_concurrency,
    )
    host = fleet.add(base)
    _admit(storage, row)
    return row, host


@pytest.fixture()
def fleet(reset_db):
    from ComfyTV.api.scheduler import Scheduler, set_scheduler

    board = Fleet()
    clock = Clock()
    scheduler = Scheduler(transport=board, clock=clock)
    set_scheduler(scheduler)
    yield {"fleet": board, "scheduler": scheduler, "clock": clock}
    set_scheduler(None)
    reset_for_tests()


def _enqueue(project="default", shot="ep01_sc01_sh01", stage="first_frame", gpu="", prompt=None):
    from ComfyTV.api.scheduler import enqueue_unit

    return enqueue_unit(
        project_id=project,
        shot_id=shot,
        stage=stage,
        gpu_class=gpu,
        prompt=prompt or PROMPT,
    )


class TestAffinityAndClass:
    async def test_same_shot_stays_on_one_class_and_prefers_the_executor(self, fleet):
        from ComfyTV import storage

        board = fleet["fleet"]
        _register(storage, board, "a1", "gb10-128g", "http://10.0.0.1:8188")
        _register(storage, board, "a2", "gb10-128g", "http://10.0.0.2:8188")
        _register(storage, board, "b", "rtx5090-32g", "http://10.0.0.3:8188")

        first = _enqueue(stage="first_frame", gpu="gb10-128g")
        video = _enqueue(stage="video")
        lips = _enqueue(stage="lipsync")
        other = _enqueue(shot="ep01_sc01_sh02", stage="video", gpu="rtx5090-32g")
        await fleet["scheduler"].tick()

        assert board.posts_to("http://10.0.0.1:8188") == 1
        assert board.posts_to("http://10.0.0.2:8188") == 1
        assert board.posts_to("http://10.0.0.3:8188") == 1
        by_id = {row["id"]: row for row in storage.list_schedule_units()}
        assert by_id[first["id"]]["executor_id"]
        assert by_id[video["id"]]["executor_id"] != by_id[first["id"]]["executor_id"]
        assert by_id[lips["id"]]["status"] == "queued"
        assert by_id[lips["id"]]["waiting_reason"] == "busy"
        assert by_id[lips["id"]]["prompt_id"] is None
        assert by_id[other["id"]]["status"] == "running"
        affinity = storage.get_affinity("default", "ep01_sc01_sh01")
        assert affinity["gpu_class"] == "gb10-128g"
        assert affinity["executor_id"] == by_id[first["id"]]["executor_id"]
        b = storage.get_executor(by_id[other["id"]]["executor_id"])
        assert b["gpu_class"] == "rtx5090-32g"
        assert _stopped(board.calls) == []

    async def test_wrong_class_waits(self, fleet):
        from ComfyTV import storage

        _register(storage, fleet["fleet"], "b", "rtx5090-32g", "http://10.0.0.3:8188")
        unit = _enqueue(gpu="gb10-128g")
        await fleet["scheduler"].tick()
        fresh = storage.get_schedule_unit(unit["id"])
        assert fresh["status"] == "queued"
        assert fresh["waiting_reason"] == "gpu_class"
        assert fresh["prompt_id"] is None
        assert fleet["fleet"].posts_to("http://10.0.0.3:8188") == 0

    async def test_foreign_prompt_blocks_submit(self, fleet):
        from ComfyTV import storage

        _row, host = _register(storage, fleet["fleet"], "a1", "gb10-128g", "http://10.0.0.1:8188")
        host.pending.append([0, "foreign-prompt", {}, {"client_id": "browser"}, []])
        unit = _enqueue(gpu="gb10-128g")
        await fleet["scheduler"].tick()
        fresh = storage.get_schedule_unit(unit["id"])
        assert fresh["status"] == "queued"
        assert fresh["waiting_reason"] == "busy"
        assert host.posts == 0
        assert not any(call[0] == "POST" and call[1].endswith("/prompt") for call in fleet["fleet"].calls)

    async def test_max_concurrency_and_nonempty_queue(self, fleet):
        from ComfyTV import storage

        full, host = _register(
            storage, fleet["fleet"], "a1", "gb10-128g", "http://10.0.0.1:8188", max_concurrency=1,
        )
        _admit(storage, full, running=1, pending=0)
        waiting = _enqueue(gpu="gb10-128g")
        await fleet["scheduler"].tick()
        assert storage.get_schedule_unit(waiting["id"])["status"] == "queued"
        assert host.posts == 0

        _admit(storage, full, running=0, pending=0)
        first = waiting
        second = _enqueue(shot="ep01_sc01_sh09", stage="video", gpu="gb10-128g")
        await fleet["scheduler"].tick()
        assert host.posts == 1
        assert storage.get_schedule_unit(first["id"])["status"] == "running"
        assert storage.get_schedule_unit(second["id"])["status"] == "queued"

        room, other = _register(
            storage, fleet["fleet"], "wide", "gb10-128g", "http://10.0.0.4:8188", max_concurrency=2,
        )
        _admit(storage, room, running=1, pending=0)
        other.pending.append([0, "someone-else", {}, {"client_id": "browser"}, []])
        third = _enqueue(shot="ep01_sc02_sh01", stage="first_frame", gpu="gb10-128g")
        await fleet["scheduler"].tick()
        assert other.posts == 0
        assert storage.get_schedule_unit(third["id"])["status"] == "queued"


class TestRetryAndCancel:
    async def test_unreachable_retries_once_on_another_executor(self, fleet, caplog):
        from ComfyTV import storage

        storage.set_settings({"scheduler-max-retries": 2})
        _a, first = _register(storage, fleet["fleet"], "a1", "gb10-128g", "http://10.0.0.1:8188")
        _b, second = _register(storage, fleet["fleet"], "a2", "gb10-128g", "http://10.0.0.2:8188")
        first.fail_post = "before"
        unit = _enqueue(gpu="gb10-128g")
        await fleet["scheduler"].tick()
        fresh = storage.get_schedule_unit(unit["id"])
        assert first.posts == 1
        assert second.posts == 1
        assert fresh["status"] == "running"
        assert fresh["executor_id"] == _b["id"]
        assert "secret.example" not in caplog.text
        assert "secret.example" not in (fresh.get("error_text") or "")

    async def test_attempt_budget_stops_before_a_second_host(self, fleet):
        from ComfyTV import storage

        storage.set_settings({"scheduler-max-retries": 1})
        _a, first = _register(storage, fleet["fleet"], "a1", "gb10-128g", "http://10.0.0.1:8188")
        _b, second = _register(storage, fleet["fleet"], "a2", "gb10-128g", "http://10.0.0.2:8188")
        first.fail_post = "before"
        unit = _enqueue(gpu="gb10-128g")
        await fleet["scheduler"].tick()
        fresh = storage.get_schedule_unit(unit["id"])
        assert first.posts == 1
        assert second.posts == 0
        assert fresh["status"] == "error"

    async def test_lost_response_does_not_submit_twice(self, fleet):
        from ComfyTV import storage

        _row, host = _register(storage, fleet["fleet"], "a1", "gb10-128g", "http://10.0.0.1:8188")
        host.fail_post = "after"
        unit = _enqueue(gpu="gb10-128g")
        await fleet["scheduler"].tick()
        once = storage.get_schedule_unit(unit["id"])
        assert host.posts == 1
        assert once["status"] == "running"
        assert once["prompt_id"] == "p-1"
        attempt_id = once["attempt_id"]
        host.fail_post = None
        await fleet["scheduler"].tick()
        again = storage.get_schedule_unit(unit["id"])
        assert host.posts == 1
        assert again["attempt_id"] == attempt_id
        assert again["prompt_id"] == "p-1"
        assert len(host.pending) == 1

    async def test_cancel_pending_deletes_only_that_prompt(self, fleet):
        from ComfyTV import storage

        _row, host = _register(storage, fleet["fleet"], "a1", "gb10-128g", "http://10.0.0.1:8188")
        unit = _enqueue(gpu="gb10-128g")
        await fleet["scheduler"].tick()
        prompt_id = storage.get_schedule_unit(unit["id"])["prompt_id"]
        await fleet["scheduler"].cancel_unit(unit["id"])
        fresh = storage.get_schedule_unit(unit["id"])
        assert fresh["status"] == "cancelled"
        deletes = [
            call for call in fleet["fleet"].calls
            if call[0] == "POST" and urlsplit(call[1]).path.rstrip("/").endswith("/queue")
        ]
        assert deletes == [("POST", "http://10.0.0.1:8188/queue", {"delete": [prompt_id]})]
        assert _stopped(fleet["fleet"].calls) == []
        host.finish()
        await fleet["scheduler"].tick()
        assert json.loads(storage.get_schedule_unit(unit["id"])["outputs_json"] or "[]") == []
        import folder_paths
        root = Path(folder_paths.get_output_directory()) / "comfytv" / "remote"
        own = [path for path in root.rglob("*") if unit["id"][:8] in path.as_posix() and path.is_file()]
        assert own == []

    async def test_cancel_running_discards_outputs(self, fleet):
        import folder_paths
        from ComfyTV import storage

        _row, host = _register(storage, fleet["fleet"], "a1", "gb10-128g", "http://10.0.0.1:8188")
        unit = _enqueue(gpu="gb10-128g")
        await fleet["scheduler"].tick()
        host.promote()
        await fleet["scheduler"].cancel_unit(unit["id"])
        assert storage.get_schedule_unit(unit["id"])["status"] == "cancel-requested"
        deletes = [
            call for call in fleet["fleet"].calls
            if call[0] == "POST" and urlsplit(call[1]).path.rstrip("/").endswith("/queue")
        ]
        assert deletes == []
        host.finish(filename="cancel-me.png")
        await fleet["scheduler"].tick()
        fresh = storage.get_schedule_unit(unit["id"])
        assert fresh["status"] == "cancelled"
        assert json.loads(fresh["outputs_json"]) == []
        assert list(Path(folder_paths.get_output_directory()).rglob("cancel-me.png")) == []
        assert _stopped(fleet["fleet"].calls) == []

    async def test_unreadable_queue_deletes_once_then_waits(self, fleet):
        from ComfyTV import storage

        _row, host = _register(storage, fleet["fleet"], "a1", "gb10-128g", "http://10.0.0.1:8188")
        unit = _enqueue(gpu="gb10-128g")
        await fleet["scheduler"].tick()
        host.fail_get_queue = True
        await fleet["scheduler"].cancel_unit(unit["id"])
        fresh = storage.get_schedule_unit(unit["id"])
        assert fresh["status"] == "cancel-requested"
        deletes = [
            call for call in fleet["fleet"].calls
            if call[0] == "POST" and urlsplit(call[1]).path.rstrip("/").endswith("/queue")
        ]
        assert len(deletes) == 1
        assert _stopped(fleet["fleet"].calls) == []

    async def test_download_uses_the_remote_output_layout(self, fleet):
        import folder_paths
        from ComfyTV import storage

        _row, host = _register(storage, fleet["fleet"], "a1", "gb10-128g", "http://10.0.0.1:8188")
        host.autocomplete = True
        unit = _enqueue(gpu="gb10-128g")
        await fleet["scheduler"].tick()
        fresh = storage.get_schedule_unit(unit["id"])
        assert fresh["status"] == "done"
        urls = json.loads(fresh["outputs_json"])
        assert urls and urls[0].startswith("/view?")
        assert "frame.png" in urls[0]
        root = Path(folder_paths.get_output_directory()) / "comfytv" / "remote"
        saved = [
            path for path in root.rglob("frame.png")
            if unit["id"][:8] in path.as_posix()
        ]
        assert len(saved) == 1
        assert saved[0].read_bytes() == b"frame-bytes"
        assert "comfytv" in saved[0].as_posix() and "remote" in saved[0].as_posix()
        assert fresh["output_id"]


class TestSingleHost:
    async def test_headless_path_is_unchanged(self, reset_db, monkeypatch):
        import execution
        import server
        from ComfyTV import storage

        storage.ensure_default_executor()

        async def boom(*_args, **_kwargs):
            raise AssertionError("single-host path called the scheduler transport")

        set_scheduler(Scheduler(transport=boom))

        class _Queue:
            def __init__(self):
                self.items = []
                self.running = []
                self.pending = []

            def put(self, item):
                self.items.append(item)
                self.pending.append(item)

            def get_current_queue_volatile(self):
                return (list(self.running), list(self.pending))

            def get_history(self, prompt_id=None, **kwargs):
                return {}

        prompt_queue = _Queue()
        monkeypatch.setattr(server.PromptServer.instance, "prompt_queue", prompt_queue, raising=False)

        async def validate(prompt_id, prompt, partial):
            return (True, None, partial or list(prompt), {})

        monkeypatch.setattr(execution, "validate_prompt", validate, raising=False)
        queue = HeadlessQueue(client=ComfyQueueClient(), autostart=False)
        set_queue(queue)
        row = storage.create_headless_job(project_id="default", kind="graph", prompt=PROMPT)
        await queue.tick()
        assert len(prompt_queue.items) == 1
        item = prompt_queue.items[0]
        assert item[2] == PROMPT
        assert item[3]["client_id"] == "comfytv-headless"
        assert item[5] == {}
        assert item[1] == storage.get_headless_job(row["id"])["prompt_id"]
        fresh = storage.get_headless_job(row["id"])
        assert fresh["status"] == "running"
        assert set(public_job(fresh)) == PUBLIC_JOB_KEYS
        assert storage.list_schedule_units() == []
        set_scheduler(None)
        reset_for_tests()

    async def test_scheduler_uses_in_process_client_for_the_seeded_local(self, fleet):
        from ComfyTV import storage

        row = storage.ensure_default_executor()
        assert row["is_default"] is True
        _admit(storage, row)

        class Local:
            def __init__(self):
                self.running = []
                self.pending = []
                self.submitted = []

            def queue_ids(self):
                return list(self.running), list(self.pending)

            def find_client(self, _client_id):
                return None

            def history_for(self, _prompt_id):
                return None

            async def submit(self, prompt, partial=None, client_id=None):
                self.submitted.append(prompt)
                self.running.append("local-1")
                return "local-1"

        local = Local()
        unit = _enqueue(stage="first_frame")
        await fleet["scheduler"].tick(local_client=local)
        assert local.submitted == [PROMPT]
        assert fleet["fleet"].calls == []
        assert storage.get_schedule_unit(unit["id"])["status"] == "running"
        assert storage.get_schedule_unit(unit["id"])["prompt_id"] == "local-1"

    async def test_unclassified_unit_stays_on_the_local_executor(self, fleet):
        from ComfyTV import storage

        local_row = storage.ensure_default_executor()
        _admit(storage, local_row)
        _register(storage, fleet["fleet"], "a1", "gb10-128g", "http://10.0.0.1:8188")

        class Local:
            def __init__(self):
                self.running = []
                self.submitted = []

            def queue_ids(self):
                return list(self.running), []

            def find_client(self, _client_id):
                return None

            def history_for(self, _prompt_id):
                return None

            async def submit(self, prompt, partial=None, client_id=None):
                self.submitted.append(prompt)
                self.running.append("local-1")
                return "local-1"

        local = Local()
        unit = _enqueue(stage="first_frame")
        await fleet["scheduler"].tick(local_client=local)
        assert local.submitted == [PROMPT]
        assert fleet["fleet"].posts_to("http://10.0.0.1:8188") == 0
        fresh = storage.get_schedule_unit(unit["id"])
        assert fresh["status"] == "running"
        assert fresh["executor_id"] == local_row["id"]

    async def test_headless_job_is_adopted_when_another_host_exists(self, fleet):
        from ComfyTV import storage

        _register(storage, fleet["fleet"], "a1", "gb10-128g", "http://10.0.0.1:8188")
        host = fleet["fleet"].hosts["http://10.0.0.1:8188"]
        host.autocomplete = True

        class Local:
            def __init__(self):
                self.submitted = []

            def queue_ids(self):
                return [], []

            def busy(self):
                return False

            def history_for(self, _prompt_id):
                return None

            def progress_for(self, _prompt_id):
                return None

            def completed_for(self, _prompt_id):
                return {}

            async def submit(self, prompt, partial=None, client_id=None):
                self.submitted.append(prompt)
                return "should-not"

        local = Local()
        queue = HeadlessQueue(client=local, autostart=False)
        set_queue(queue)
        prompt = {
            "1": {
                "class_type": "ComfyTV.ImageStage",
                "inputs": {"shot_id": "ep01_sc01_sh01", "gpu_class": "gb10-128g"},
            },
        }
        job = storage.create_headless_job(
            project_id="default", kind="stage", prompt=prompt, stage_ref="9",
        )
        await queue.tick()
        assert local.submitted == []
        assert host.posts == 1
        fresh = storage.get_headless_job(job["id"])
        assert fresh["status"] == "done"
        unit = storage.unit_for_headless(job["id"])
        assert unit["shot_id"] == "ep01_sc01_sh01"
        assert unit["gpu_class"] == "gb10-128g"
        assert unit["status"] == "done"


class TestStatusApi:
    async def test_http_and_mcp_read_the_queue(self, reset_db):
        from ComfyTV import storage
        from ComfyTV.api.scheduler import enqueue_unit, set_scheduler

        set_scheduler(Scheduler(transport=Fleet()))
        storage.set_settings({"enable-mcp": True})
        from ComfyTV import api  # noqa: F401
        import server
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        created = enqueue_unit(
            project_id="default",
            shot_id="ep01_sc01_sh01",
            stage="first_frame",
            gpu_class="gb10-128g",
            prompt=PROMPT,
        )
        app = web.Application()
        app.router.add_routes(server.PromptServer.instance.routes)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            listed = await client.get("/comfytv/scheduler")
            assert listed.status == 200
            body = await listed.json()
            assert body["queue"][0]["id"] == created["id"]
            assert body["queue"][0]["shot_id"] == "ep01_sc01_sh01"
            assert "prompt" not in body["queue"][0]
            assert any(row["name"] == "local" for row in body["executors"])
            assert "load" in body["executors"][0]

            resp = await client.post("/comfytv/mcp", json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "scheduler", "arguments": {}},
            })
            payload = await resp.json()
            viewed = json.loads(payload["result"]["content"][0]["text"])
            assert viewed["queue"][0]["id"] == created["id"]
            assert storage.list_schedule_units()[0]["id"] == created["id"]

            cancelled = await client.post(f"/comfytv/scheduler/units/{created['id']}/cancel")
            assert cancelled.status == 200
            assert (await cancelled.json())["unit"]["status"] == "cancelled"
        finally:
            await client.close()
            set_scheduler(None)

    def test_sources_do_not_stop_a_host(self):
        root = Path(__file__).resolve().parents[1]
        for name in (
            "api/scheduler.py",
            "api/remote_cancel.py",
            "api/mcp_tools/scheduler.py",
            "storage/scheduler.py",
        ):
            text = (root / name).read_text(encoding="utf-8")
            assert "/interrupt" not in text
            assert "/free" not in text
            assert "interrupt_processing" not in text
            assert "wipe_queue" not in text

    def test_limits_follow_settings(self, reset_db):
        from ComfyTV import storage
        from ComfyTV.api.scheduler import scheduler_limits

        storage.set_settings({
            "remote-connect-timeout-s": 4,
            "remote-read-timeout-s": 0,
            "remote-poll-interval-s": 3,
            "remote-job-timeout-s": 9,
            "scheduler-max-retries": 2,
        })
        limits = scheduler_limits()
        assert limits["connect"] == 4
        assert limits["read"] is None
        assert limits["poll"] == 3
        assert limits["job"] == 9
        assert limits["max_attempts"] == 2

    async def test_job_timeout_leaves_a_running_prompt(self, fleet):
        from ComfyTV import storage

        storage.set_settings({"remote-job-timeout-s": 5})
        _row, host = _register(storage, fleet["fleet"], "a1", "gb10-128g", "http://10.0.0.1:8188")
        unit = _enqueue(gpu="gb10-128g")
        await fleet["scheduler"].tick()
        host.promote()
        fleet["clock"].advance(seconds=5)
        await fleet["scheduler"].tick()
        assert storage.get_schedule_unit(unit["id"])["status"] == "cancel-requested"
        assert _stopped(fleet["fleet"].calls) == []
        assert not any(
            call[0] == "POST" and urlsplit(call[1]).path.rstrip("/").endswith("/queue")
            for call in fleet["fleet"].calls
        )
