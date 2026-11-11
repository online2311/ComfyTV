"""Executor registry: health, idle detection, join tokens, and draining.

ComfyUI is mocked. Nothing in these tests opens a socket or calls
/interrupt or /free.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import select

from sqlalchemy import inspect

from ComfyTV.api.queue_state import queue_is_empty
from ComfyTV.db import ExecutorJoinToken

TOKEN = "one-time-join-token-value"
OTHER = "different-join-token-value"
FOREIGN = [0, "prompt-from-another-user", {}, {"client_id": "browser"}, []]


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


class FakeComfy:
    """Stand-in for one executor's GET /system_stats and GET /queue."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail = False
        self.stats_status = 200
        self.queue_status = 200
        self.stats = {
            "system": {"comfyui_version": "0.45.0", "os": "linux"},
            "devices": [{
                "name": "cuda:0 NVIDIA GeForce RTX 5090 : cudaMallocAsync",
                "type": "cuda",
                "index": 0,
                "vram_total": 34359738368,
                "vram_free": 31100000000,
            }],
        }
        self.queue = {"queue_running": [], "queue_pending": []}

    async def __call__(self, method: str, url: str):
        self.calls.append((method, url))
        if method != "GET":
            raise AssertionError(method)
        path = url.split("?", 1)[0].rstrip("/")
        if path.endswith("/interrupt") or path.endswith("/free"):
            raise AssertionError(url)
        if self.fail:
            raise RuntimeError("http://secret.example/should-not-be-logged")
        if path.endswith("/system_stats"):
            return self.stats_status, self.stats if self.stats_status == 200 else None
        if path.endswith("/queue"):
            return self.queue_status, self.queue if self.queue_status == 200 else None
        raise AssertionError(url)


def _db_bytes(reset_db) -> bytes:
    path = reset_db._engine.url.database
    assert path
    return Path(path).read_bytes()


def _register(storage, **overrides):
    fields = dict(
        name="spark",
        kind="permanent",
        base_url="http://100.64.0.2:8188",
        gpu_class="gb10-128g",
        labels={"arch": "arm64"},
        capabilities={"video": True},
        max_concurrency=1,
        provider={"site": "home"},
    )
    fields.update(overrides)
    return storage.register_executor(**fields)


class TestQueueEmpty:
    def test_empty_only_when_both_lists_are_empty(self):
        assert queue_is_empty([], []) is True
        assert queue_is_empty(None, None) is True
        assert queue_is_empty([FOREIGN], []) is False
        assert queue_is_empty([], [FOREIGN]) is False
        assert queue_is_empty("not-a-list", []) is False

    async def test_headless_client_uses_the_shared_empty_check(self, monkeypatch):
        import server
        from ComfyTV.api.headless_queue import ComfyQueueClient

        class _Queue:
            def __init__(self):
                self.running = []
                self.pending = []

            def get_current_queue_volatile(self):
                return (list(self.running), list(self.pending))

        queue = _Queue()
        monkeypatch.setattr(
            server.PromptServer.instance, "prompt_queue", queue, raising=False,
        )
        client = ComfyQueueClient()
        assert client.busy() is False

        queue.pending.append((1, "foreign-prompt", {}, {}, [], {}))
        assert client.busy() is True

        queue.pending.clear()
        queue.running.append((0, "also-foreign", {}, {}, [], {}))
        assert client.busy() is True


class TestJoinToken:
    def test_configured_token_is_hashed_and_absent_from_the_database(
        self, reset_db, caplog,
    ):
        from ComfyTV import storage

        caplog.set_level(logging.DEBUG)
        storage.configure_join_token(TOKEN)
        assert TOKEN not in caplog.text

        with reset_db.get_session() as s:
            rows = s.execute(select(ExecutorJoinToken)).scalars().all()
            assert len(rows) == 1
            assert rows[0].origin == "configured"
            assert rows[0].token_hash != TOKEN
            assert len(rows[0].token_hash) == 64
        assert TOKEN.encode() not in _db_bytes(reset_db)

        columns = {c["name"] for c in inspect(reset_db._engine).get_columns(
            ExecutorJoinToken.__tablename__,
        )}
        assert "token" not in columns
        assert columns == {"token_hash", "origin", "created_at"}

    def test_environment_token_is_hashed_then_removed(self, reset_db, monkeypatch, caplog):
        from ComfyTV import storage

        monkeypatch.setenv("COMFYTV_EXECUTOR_JOIN_TOKEN", TOKEN)
        caplog.set_level(logging.DEBUG)
        storage.absorb_configured_join_token()
        assert "COMFYTV_EXECUTOR_JOIN_TOKEN" not in __import__("os").environ
        assert TOKEN not in caplog.text
        assert TOKEN.encode() not in _db_bytes(reset_db)

        joined = storage.join_executor(
            token=TOKEN, name="pod-a", base_url="http://100.64.0.8:8188",
            gpu_class="rtx5090-32g",
        )
        assert joined["kind"] == "ephemeral"
        with pytest.raises(storage.JoinRejected):
            storage.join_executor(
                token=TOKEN, name="pod-b", base_url="http://100.64.0.9:8188",
                gpu_class="rtx5090-32g",
            )

    def test_wrong_token_does_not_register_or_burn_the_real_one(self, reset_db, caplog):
        from ComfyTV import storage

        storage.configure_join_token(TOKEN)
        caplog.set_level(logging.DEBUG)
        with pytest.raises(storage.JoinRejected):
            storage.join_executor(
                token=OTHER, name="pod-a", base_url="http://100.64.0.8:8188",
            )
        assert OTHER not in caplog.text
        assert storage.list_executors() == []

        joined = storage.join_executor(
            token=TOKEN, name="pod-a", base_url="http://100.64.0.8:8188",
            gpu_class="rtx5090-32g", provider={"pod_id": "pod-a"},
        )
        assert joined["provider"] == {"pod_id": "pod-a"}
        assert joined["gpu_class"] == "rtx5090-32g"

    def test_issued_token_works_once_and_is_not_stored(self, reset_db):
        from ComfyTV import storage

        issued = storage.issue_join_token()
        assert issued != ""
        assert issued.encode() not in _db_bytes(reset_db)
        storage.join_executor(
            token=issued, name="pod-a", base_url="http://100.64.0.8:8188",
        )
        with pytest.raises(storage.JoinRejected):
            storage.join_executor(
                token=issued, name="pod-b", base_url="http://100.64.0.9:8188",
            )

    def test_rejoin_updates_the_ephemeral_executor(self, reset_db):
        from ComfyTV import storage

        first = storage.issue_join_token()
        created = storage.join_executor(
            token=first, name="pod-a", base_url="http://100.64.0.8:8188",
            gpu_class="rtx5090-32g",
        )
        second = storage.issue_join_token()
        again = storage.join_executor(
            token=second, name="pod-a", base_url="http://100.64.0.18:8188",
            gpu_class="rtxpro6000-96g", provider={"pod_id": "new"},
        )
        assert again["id"] == created["id"]
        assert again["base_url"] == "http://100.64.0.18:8188"
        assert again["gpu_class"] == "rtxpro6000-96g"
        assert again["kind"] == "ephemeral"
        assert len(storage.list_executors()) == 1

    def test_short_token_is_rejected_without_being_stored(self, reset_db):
        from ComfyTV import storage

        with pytest.raises(ValueError):
            storage.configure_join_token("short")
        with reset_db.get_session() as s:
            assert s.execute(select(ExecutorJoinToken)).scalars().all() == []


class TestRegistry:
    def test_empty_registry_lists_the_local_executor(self, reset_db, monkeypatch):
        from ComfyTV import storage
        import server

        monkeypatch.setattr(server.PromptServer.instance, "port", 8188, raising=False)
        listed = storage.list_executors(ensure_default=True)
        assert len(listed) == 1
        row = listed[0]
        assert row["name"] == "local"
        assert row["kind"] == "permanent"
        assert row["is_default"] is True
        assert row["base_url"] == "http://127.0.0.1:8188"
        assert row["max_concurrency"] == 1
        assert row["enabled"] is True
        assert row["draining"] is False
        assert row["health"] == "unknown"
        assert row["idle"] is False
        assert row["accepts_work"] is False
        assert row["admission_reason"] == "unknown"
        again = storage.list_executors(ensure_default=True)
        assert [r["id"] for r in again] == [row["id"]]

    def test_register_update_get_remove(self, reset_db):
        from ComfyTV import storage

        created = _register(storage)
        assert created["gpu_class"] == "gb10-128g"
        assert created["labels"] == {"arch": "arm64"}
        assert created["capabilities"] == {"video": True}
        assert created["provider"] == {"site": "home"}
        assert created["kind"] == "permanent"

        updated = storage.update_executor(created["id"], {
            "gpu_class": "gb10-128g",
            "max_concurrency": 2,
            "enabled": False,
            "labels": {"arch": "arm64", "pool": "a"},
        })
        assert updated["max_concurrency"] == 2
        assert updated["enabled"] is False
        assert updated["labels"]["pool"] == "a"
        assert storage.get_executor(created["id"])["max_concurrency"] == 2

        assert storage.remove_executor(created["id"]) is True
        assert storage.get_executor(created["id"]) is None
        assert storage.remove_executor(created["id"]) is False

    def test_duplicate_name_is_rejected(self, reset_db):
        from ComfyTV import storage

        _register(storage)
        with pytest.raises(storage.ExecutorConflict):
            _register(storage, base_url="http://100.64.0.3:8188")

    def test_base_url_rejects_credentials(self, reset_db):
        from ComfyTV import storage

        with pytest.raises(ValueError):
            _register(storage, base_url="http://user:secret@100.64.0.2:8188")
        assert storage.list_executors() == []

    def test_removing_the_last_executor_restores_the_local_default(self, reset_db):
        from ComfyTV import storage

        created = _register(storage)
        storage.remove_executor(created["id"])
        listed = storage.list_executors(ensure_default=True)
        assert [r["name"] for r in listed] == ["local"]
        assert listed[0]["is_default"] is True

    def test_existing_executors_are_not_replaced_by_the_default(self, reset_db):
        from ComfyTV import storage

        created = _register(storage)
        listed = storage.list_executors(ensure_default=True)
        assert [r["id"] for r in listed] == [created["id"]]


class TestHealthAndDrain:
    async def _checker(self, fake: FakeComfy, clock: Clock):
        from ComfyTV.api.executor_health import ExecutorHealthChecker
        return ExecutorHealthChecker(fetch=fake, clock=clock, interval_s=15)

    async def test_transitions_idle_and_foreign_prompts(self, reset_db):
        from ComfyTV import storage

        created = _register(storage, max_concurrency=1)
        fake = FakeComfy()
        clock = Clock()
        checker = await self._checker(fake, clock)

        first = (await checker.check_all())[0]
        assert first["health"] == "reachable"
        assert first["idle"] is True
        assert first["running_count"] == 0
        assert first["pending_count"] == 0
        assert first["idle_since"] is not None
        assert first["gpu"]["devices"][0]["vram_total"] == 34359738368
        assert first["gpu"]["devices"][0]["vram_free"] == 31100000000
        assert first["gpu"]["comfyui_version"] == "0.45.0"
        assert first["accepts_work"] is True
        assert first["admission_reason"] is None
        idle_since = first["idle_since"]

        clock.advance(minutes=5)
        again = (await checker.check_all())[0]
        assert again["idle"] is True
        assert again["idle_since"] == idle_since

        fake.queue = {"queue_running": [], "queue_pending": [FOREIGN]}
        clock.advance(minutes=1)
        busy = (await checker.check_all())[0]
        assert busy["idle"] is False
        assert busy["idle_since"] is None
        assert busy["pending_count"] == 1
        assert busy["running_count"] == 0
        assert busy["accepts_work"] is False
        assert busy["admission_reason"] == "at_capacity"

        fake.queue = {"queue_running": [FOREIGN], "queue_pending": []}
        running = (await checker.check_all())[0]
        assert running["idle"] is False
        assert running["running_count"] == 1
        assert running["accepts_work"] is False

        fake.fail = True
        clock.advance(minutes=1)
        down = (await checker.check_all())[0]
        assert down["health"] == "unreachable"
        assert down["idle"] is False
        assert down["idle_since"] is None
        assert down["accepts_work"] is False
        assert down["admission_reason"] == "unreachable"
        assert down["last_seen"] == busy["last_seen"]
        assert "secret.example" not in (down.get("health_error") or "")

        fake.fail = False
        fake.queue = {"queue_running": [], "queue_pending": []}
        clock.advance(seconds=30)
        back = (await checker.check_all())[0]
        assert back["health"] == "reachable"
        assert back["idle"] is True
        assert back["idle_since"] is not None
        assert back["idle_since"] != idle_since

        methods = {method for method, _url in fake.calls}
        assert methods == {"GET"}
        assert all(
            url.endswith("/system_stats") or url.endswith("/queue")
            for _method, url in fake.calls
        )

    async def test_queue_http_error_is_not_idle(self, reset_db):
        from ComfyTV import storage

        _register(storage)
        fake = FakeComfy()
        fake.queue_status = 503
        clock = Clock()
        checker = await self._checker(fake, clock)
        row = (await checker.check_all())[0]
        assert row["health"] == "unreachable"
        assert row["idle"] is False
        assert row["idle_since"] is None

    async def test_missing_queue_lists_are_not_idle(self, reset_db):
        from ComfyTV import storage

        _register(storage)
        fake = FakeComfy()
        fake.queue = {"queue_running": []}
        clock = Clock()
        checker = await self._checker(fake, clock)
        row = (await checker.check_all())[0]
        assert row["health"] == "reachable"
        assert row["idle"] is False
        assert row["idle_since"] is None
        assert row["accepts_work"] is False

    async def test_higher_concurrency_allows_one_foreign_prompt(self, reset_db):
        from ComfyTV import storage

        _register(storage, max_concurrency=2)
        fake = FakeComfy()
        fake.queue = {"queue_running": [FOREIGN], "queue_pending": []}
        checker = await self._checker(fake, Clock())
        row = (await checker.check_all())[0]
        assert row["idle"] is False
        assert row["accepts_work"] is True
        assert row["admission_reason"] is None

        fake.queue = {"queue_running": [FOREIGN], "queue_pending": [FOREIGN]}
        full = (await checker.check_all())[0]
        assert full["accepts_work"] is False
        assert full["admission_reason"] == "at_capacity"

    async def test_drain_stops_admission_and_keeps_idle_since(self, reset_db, caplog):
        from ComfyTV import storage

        created = _register(storage)
        fake = FakeComfy()
        clock = Clock()
        checker = await self._checker(fake, clock)
        idle = (await checker.check_all())[0]
        assert idle["accepts_work"] is True

        caplog.set_level(logging.DEBUG)
        drained = storage.drain_executor(created["id"])
        assert drained["draining"] is True
        assert drained["enabled"] is True
        assert drained["idle"] is True
        assert drained["idle_since"] == idle["idle_since"]
        assert drained["accepts_work"] is False
        assert drained["admission_reason"] == "draining"
        assert "/interrupt" not in caplog.text
        assert "/free" not in caplog.text

        clock.advance(minutes=12)
        still = (await checker.check_all())[0]
        assert still["draining"] is True
        assert still["idle"] is True
        assert still["idle_since"] == idle["idle_since"]
        assert still["accepts_work"] is False

        opened = storage.drain_executor(created["id"], draining=False)
        assert opened["draining"] is False
        assert opened["accepts_work"] is True

    async def test_disabled_executor_is_not_probed(self, reset_db):
        from ComfyTV import storage

        created = _register(storage)
        storage.update_executor(created["id"], {"enabled": False})
        fake = FakeComfy()
        checker = await self._checker(fake, Clock())
        rows = await checker.check_all()
        assert rows[0]["id"] == created["id"]
        assert rows[0]["accepts_work"] is False
        assert rows[0]["admission_reason"] == "disabled"
        assert fake.calls == []

    def test_sources_do_not_interrupt_or_free(self):
        root = Path(__file__).resolve().parents[1]
        for name in (
            "api/queue_state.py",
            "api/executor_health.py",
            "api/executors.py",
            "api/mcp_tools/executors.py",
            "storage/executors.py",
        ):
            text = (root / name).read_text(encoding="utf-8")
            assert "/interrupt" not in text
            assert "/free" not in text
            assert "interrupt_processing" not in text
            assert TOKEN not in text


class TestHttpAndMcp:
    @pytest.fixture()
    async def client(self, reset_db):
        from ComfyTV import storage
        storage.set_settings({"enable-mcp": True})
        from ComfyTV import api  # noqa: F401
        import server
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        app = web.Application()
        app.router.add_routes(server.PromptServer.instance.routes)
        test_server = TestServer(app)
        test_client = TestClient(test_server)
        await test_client.start_server()
        yield test_client
        await test_client.close()

    async def test_http_crud_drain_and_join(self, client):
        listed = await client.get("/comfytv/executors")
        assert listed.status == 200
        body = await listed.json()
        assert [row["name"] for row in body["executors"]] == ["local"]

        created = await client.post("/comfytv/executors", json={
            "name": "spark",
            "kind": "permanent",
            "base_url": "http://100.64.0.2:8188",
            "gpu_class": "gb10-128g",
            "labels": {"arch": "arm64"},
            "max_concurrency": 1,
        })
        assert created.status == 201
        executor = (await created.json())["executor"]
        assert executor["gpu_class"] == "gb10-128g"

        fetched = await client.get(f"/comfytv/executors/{executor['id']}")
        assert fetched.status == 200
        assert (await fetched.json())["executor"]["name"] == "spark"

        patched = await client.patch(
            f"/comfytv/executors/{executor['id']}",
            json={"gpu_class": "gb10-128g", "max_concurrency": 2},
        )
        assert patched.status == 200
        assert (await patched.json())["executor"]["max_concurrency"] == 2

        drained = await client.post(f"/comfytv/executors/{executor['id']}/drain", json={})
        assert drained.status == 200
        assert (await drained.json())["executor"]["draining"] is True

        missing = await client.get("/comfytv/executors/missing")
        assert missing.status == 404

        minted = await client.post("/comfytv/executors/join-tokens", json={})
        assert minted.status == 201
        token = (await minted.json())["token"]
        assert token
        joined = await client.post("/comfytv/executors/join", json={
            "join_token": token,
            "name": "pod-a",
            "base_url": "http://100.64.0.9:8188",
            "gpu_class": "rtx5090-32g",
            "provider": {"pod_id": "pod-a"},
        })
        assert joined.status == 201
        pod = (await joined.json())["executor"]
        assert pod["kind"] == "ephemeral"
        assert "join_token" not in pod
        assert token not in (await joined.text())

        reused = await client.post("/comfytv/executors/join", json={
            "join_token": token,
            "name": "pod-b",
            "base_url": "http://100.64.0.10:8188",
        })
        assert reused.status == 401

        removed = await client.delete(f"/comfytv/executors/{pod['id']}")
        assert removed.status == 200
        assert (await removed.json())["ok"] is True

    async def test_mcp_register_and_drain(self, client):
        import json

        resp = await client.post("/comfytv/mcp", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "executors",
                "arguments": {
                    "action": "register",
                    "name": "spark",
                    "kind": "permanent",
                    "base_url": "http://100.64.0.2:8188",
                    "gpu_class": "gb10-128g",
                },
            },
        })
        payload = await resp.json()
        registered = json.loads(payload["result"]["content"][0]["text"])
        executor_id = registered["executor"]["id"]
        assert registered["executor"]["kind"] == "permanent"

        resp = await client.post("/comfytv/mcp", json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "executors",
                "arguments": {"action": "drain", "id": executor_id},
            },
        })
        payload = await resp.json()
        drained = json.loads(payload["result"]["content"][0]["text"])
        assert drained["executor"]["draining"] is True
        assert drained["executor"]["accepts_work"] is False
