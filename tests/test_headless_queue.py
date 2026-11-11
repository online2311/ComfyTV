"""Headless MCP runs: no open page, one ComfyUI job at a time, resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ComfyTV.api.canvas_state import clear_canvas_state, store_canvas_state
from ComfyTV.api.headless_queue import (
    ComfyQueueClient,
    HeadlessQueue,
    QueueBusy,
    apply_completed,
    reset_for_tests,
    set_queue,
)


SUCCESS = {
    "outputs": {
        "9": {"images": [{"filename": "x.png", "subfolder": "", "type": "output"}]},
    },
    "status": {"status_str": "success", "completed": True, "messages": []},
}

ERROR = {
    "outputs": {},
    "status": {
        "status_str": "error",
        "completed": False,
        "messages": [[
            "execution_error",
            {"node_id": "3", "node_type": "KSampler", "exception_message": "boom"},
        ]],
    },
}

PROMPT = {
    "1": {"class_type": "ComfyTV.ImageStage", "inputs": {"main_prompt": "a cat"}},
}


class FakeComfy:
    def __init__(self):
        self.running: list[str] = []
        self.pending: list[str] = []
        self.history: dict[str, dict] = {}
        self.progress: dict[str, dict] = {}
        self.completed: dict[str, dict] = {}
        self.submitted: list[dict] = []
        self._n = 0

    def queue_ids(self):
        return list(self.running), list(self.pending)

    def busy(self):
        running, pending = self.queue_ids()
        return bool(running or pending)

    def history_for(self, prompt_id):
        return self.history.get(prompt_id)

    def progress_for(self, prompt_id):
        return self.progress.get(prompt_id)

    def completed_for(self, prompt_id):
        return dict(self.completed.get(prompt_id) or {})

    async def submit(self, prompt, partial):
        if self.busy():
            raise QueueBusy("queue was not empty")
        self._n += 1
        prompt_id = f"prompt-{self._n}"
        self.submitted.append({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "partial": partial,
        })
        self.running.append(prompt_id)
        return prompt_id

    def finish(self, prompt_id, entry):
        self.running = [p for p in self.running if p != prompt_id]
        self.pending = [p for p in self.pending if p != prompt_id]
        self.history[prompt_id] = entry


@pytest.fixture()
def headless(reset_db):
    clear_canvas_state()
    fake = FakeComfy()
    queue = HeadlessQueue(client=fake, autostart=False)
    set_queue(queue)
    yield {"queue": queue, "comfy": fake}
    reset_for_tests()
    clear_canvas_state()


class TestApplyCompleted:
    def test_rewires_downstream_and_drops_finished_node(self):
        prompt = {
            "1": {"class_type": "A", "inputs": {"text": "hi"}},
            "2": {"class_type": "B", "inputs": {"image": ["1", 0], "seed": 1}},
        }
        reduced, partial = apply_completed(
            prompt,
            {"1": {"value": "/view?filename=a.png&type=output", "slot": 0}},
            ["1", "2"],
        )
        assert "1" not in reduced
        assert reduced["2"]["inputs"]["image"] == "/view?filename=a.png&type=output"
        assert reduced["2"]["inputs"]["seed"] == 1
        assert partial == ["2"]

    def test_keeps_a_node_when_its_output_was_not_saved(self):
        prompt = {"1": {"class_type": "A", "inputs": {}}}
        reduced, partial = apply_completed(prompt, {"1": {"slot": 0}}, ["1"])
        assert "1" in reduced
        assert partial == ["1"]


class TestHeadlessRun:
    async def test_graph_run_returns_job_id_without_submitting(self, headless):
        from ComfyTV.api.mcp_tools import _graph_run

        out = await _graph_run({"prompt": PROMPT})
        assert out["status"] == "queued"
        assert out["job_id"]
        assert headless["comfy"].submitted == []

        await headless["queue"].tick()
        assert len(headless["comfy"].submitted) == 1
        sent = headless["comfy"].submitted[0]
        assert sent["prompt"]["1"]["inputs"]["main_prompt"] == "a cat"
        status = await _graph_run({"job_id": out["job_id"]})
        assert status["job"]["status"] == "running"
        assert status["job"]["prompt_id"] == sent["prompt_id"]

    async def test_status_reports_progress_and_outputs(self, headless):
        from ComfyTV.api.headless_queue import queue_status as _queue_status
        from ComfyTV.api.mcp_tools import _graph_run

        queued = await _graph_run({"prompt": PROMPT})
        await headless["queue"].tick()
        prompt_id = headless["comfy"].submitted[0]["prompt_id"]
        headless["comfy"].progress[prompt_id] = {"value": 3, "max": 20, "node": "1"}
        await headless["queue"].tick()
        running = await _queue_status({"job_id": queued["job_id"]})
        assert running["job"]["progress"] == {"value": 3, "max": 20, "node": "1"}

        headless["comfy"].finish(prompt_id, SUCCESS)
        done = await _queue_status({"job_id": queued["job_id"]})
        assert done["job"]["status"] == "done"
        assert done["job"]["outputs"] == ["/view?filename=x.png&subfolder=&type=output"]

    async def test_error_history_marks_the_job(self, headless):
        from ComfyTV.api.headless_queue import queue_status as _queue_status
        from ComfyTV.api.mcp_tools import _graph_run

        queued = await _graph_run({"prompt": PROMPT})
        await headless["queue"].tick()
        prompt_id = headless["comfy"].running[0]
        headless["comfy"].finish(prompt_id, ERROR)
        done = await _queue_status({"job_id": queued["job_id"]})
        assert done["job"]["status"] == "error"
        assert "boom" in done["job"]["error"]

    async def test_one_at_a_time_and_waits_for_external_work(self, headless):
        from ComfyTV.api.headless_queue import list_queue
        from ComfyTV.api.mcp_tools import _graph_run

        headless["comfy"].running.append("external")
        first = await _graph_run({"prompt": PROMPT})
        second = await _graph_run({"prompt": {
            "1": {"class_type": "ComfyTV.ImageStage", "inputs": {"main_prompt": "b"}},
        }})
        await headless["queue"].tick()
        assert headless["comfy"].submitted == []

        headless["comfy"].running.remove("external")
        await headless["queue"].tick()
        assert len(headless["comfy"].submitted) == 1
        assert headless["comfy"].submitted[0]["prompt"]["1"]["inputs"]["main_prompt"] == "a cat"
        await headless["queue"].tick()
        assert len(headless["comfy"].submitted) == 1

        listed = await list_queue({})
        ids = [job["job_id"] for job in listed["jobs"]]
        assert first["job_id"] in ids and second["job_id"] in ids
        assert sum(1 for job in listed["jobs"] if job["status"] == "running") == 1

        headless["comfy"].finish(headless["comfy"].running[0], SUCCESS)
        await headless["queue"].tick()
        assert len(headless["comfy"].submitted) == 2
        assert headless["comfy"].submitted[1]["prompt"]["1"]["inputs"]["main_prompt"] == "b"

    async def test_run_stage_queues_only_that_stage(self, headless):
        from ComfyTV.api.mcp_tools import _add_stage, _connect_stages, _run_stage

        a = await _add_stage({"node_class": "ImageStage", "prompt": "upstream"})
        b = await _add_stage({"node_class": "ImageStage", "prompt": "downstream"})
        await _connect_stages({
            "from_node": a["graph_node_id"],
            "to_node": b["graph_node_id"],
            "to_slot": "ref",
        })
        queued = await _run_stage({"node": b["uid"]})
        assert queued["status"] == "queued"
        assert queued["job_id"]
        await headless["queue"].tick()
        sent = headless["comfy"].submitted[0]
        assert set(sent["prompt"]) == {a["graph_node_id"], b["graph_node_id"]}
        assert sent["partial"] == [b["graph_node_id"]]
        assert sent["prompt"][b["graph_node_id"]]["inputs"]["ref"] == [a["graph_node_id"], 0]
        assert sent["prompt"][b["graph_node_id"]]["inputs"]["main_prompt"] == "downstream"

    async def test_editor_stage_is_not_queued(self, headless):
        from ComfyTV.api.mcp_tools import _add_stage, _run_stage

        added = await _add_stage({"node_class": "CropStage"})
        with pytest.raises(ValueError, match="editor stage"):
            await _run_stage({"node": added["graph_node_id"]})
        assert headless["comfy"].submitted == []

    async def test_remote_server_is_not_queued_locally(self, headless):
        from ComfyTV import storage
        from ComfyTV.api.mcp_tools import _add_stage, _run_stage, _set_stage

        added = await _add_stage({"node_class": "ImageStage", "prompt": "x"})
        server = storage.create_server(label="other", host="10.0.0.8", port=8188)
        await _set_stage({"node": added["graph_node_id"], "server": str(server["id"])})
        with pytest.raises(ValueError, match="local"):
            await _run_stage({"node": added["graph_node_id"]})

    async def test_asset_ref_is_in_the_prompt(self, headless):
        from ComfyTV import storage
        from ComfyTV.api.mcp_tools import _add_stage, _run_stage, _set_stage

        asset = storage.create_asset(
            name="ref",
            payload_url="/view?filename=ref.png&type=input",
            media_type="image",
            category_ids=[],
        )
        added = await _add_stage({"node_class": "ImageStage", "prompt": "use the ref"})
        await _set_stage({
            "node": added["graph_node_id"],
            "asset_refs": [{"asset_id": asset["id"], "type": "image"}],
        })
        await _run_stage({"node": added["graph_node_id"]})
        await headless["queue"].tick()
        inputs = headless["comfy"].submitted[0]["prompt"][added["graph_node_id"]]["inputs"]
        assert inputs["images"]["image_1"] == "/view?filename=ref.png&type=input"

    async def test_cancel_drops_a_queued_job_and_does_not_interrupt(self, headless):
        from ComfyTV.api.mcp_tools import _cancel_stage, _run_stage, _add_stage

        added = await _add_stage({"node_class": "ImageStage", "prompt": "x"})
        queued = await _run_stage({"node": added["graph_node_id"]})
        dropped = await _cancel_stage({"node": added["graph_node_id"]})
        assert dropped["status"] == "cancelled"
        assert dropped["job_id"] == queued["job_id"]
        await headless["queue"].tick()
        assert headless["comfy"].submitted == []

        again = await _run_stage({"node": added["graph_node_id"]})
        await headless["queue"].tick()
        with pytest.raises(ValueError, match="does not interrupt"):
            await _cancel_stage({"node": added["graph_node_id"]})
        assert headless["comfy"].running
        assert again["job_id"] != queued["job_id"]

    async def test_wait_stage_sees_the_headless_job(self, headless):
        from ComfyTV.api.mcp_tools import _run_stage, _add_stage, _wait_stage

        added = await _add_stage({"node_class": "TextStage", "prompt": "hello"})
        await _run_stage({"node": added["graph_node_id"]})
        await headless["queue"].tick()
        prompt_id = headless["comfy"].running[0]
        headless["comfy"].finish(prompt_id, SUCCESS)
        out = await _wait_stage({"node": added["graph_node_id"], "timeout_s": 2})
        assert out["status"] == "done"
        assert out["outputs"] == ["/view?filename=x.png&subfolder=&type=output"]

    async def test_open_page_still_relays(self, headless, monkeypatch):
        from ComfyTV.api import mcp_tools

        store_canvas_state("default", [], client_id="tab-1")
        calls = []

        async def submit(action, payload, timeout=15.0):
            calls.append((action, timeout))
            return {"queued": True, "prompt_id": "from-page"}

        monkeypatch.setattr(mcp_tools._shared, "submit_command", submit)
        out = await mcp_tools._graph_run({"wait_s": 0})
        assert calls == [("graph_run", 60.0)]
        assert out["prompt_id"] == "from-page"
        assert headless["comfy"].submitted == []


class TestResume:
    async def test_restart_skips_completed_stage(self, headless):
        from ComfyTV import storage
        from ComfyTV.api.headless_queue import queue_status as _queue_status

        prompt = {
            "1": {"class_type": "ComfyTV.ImageStage", "inputs": {"main_prompt": "done"}},
            "2": {
                "class_type": "ComfyTV.ImageStage",
                "inputs": {"main_prompt": "next", "image": ["1", 0]},
            },
        }
        row = storage.create_headless_job(
            project_id="default", kind="graph", prompt=prompt, partial=["1", "2"],
        )
        storage.update_headless_job(
            row["id"],
            status="running",
            prompt_id="lost-prompt",
            completed_json=json.dumps({
                "1": {"value": "/view?filename=kept.png&type=output", "slot": 0},
            }),
        )
        restarted = HeadlessQueue(client=FakeComfy(), autostart=False)
        set_queue(restarted)
        resumed = await restarted.recover()
        assert resumed == [row["id"]]
        fresh = storage.get_headless_job(row["id"])
        assert fresh["status"] == "queued"
        assert fresh["prompt_id"] is None
        assert fresh["attempt"] == 2

        await restarted.tick()
        sent = restarted.client.submitted[0]
        assert "1" not in sent["prompt"]
        assert sent["prompt"]["2"]["inputs"]["image"] == "/view?filename=kept.png&type=output"
        assert sent["partial"] == ["2"]
        status = await _queue_status({"job_id": row["id"]})
        assert status["job"]["status"] == "running"
        assert status["job"]["attempt"] == 2

    async def test_finished_stage_is_not_submitted_again(self, headless):
        from ComfyTV import storage

        row = storage.create_headless_job(
            project_id="default",
            kind="stage",
            prompt={"7": {"class_type": "ComfyTV.ImageStage", "inputs": {"main_prompt": "only"}}},
            partial=["7"],
            stage_ref="7",
        )
        storage.update_headless_job(
            row["id"],
            status="running",
            prompt_id="lost",
            completed_json=json.dumps({
                "7": {"value": "/view?filename=only.png&type=output", "slot": 0},
            }),
        )
        await headless["queue"].recover()
        await headless["queue"].tick()
        assert headless["comfy"].submitted == []
        fresh = storage.get_headless_job(row["id"])
        assert fresh["status"] == "done"
        assert "/view?filename=only.png&type=output" in json.loads(fresh["outputs_json"])

    async def test_history_success_is_not_rerun(self, headless):
        from ComfyTV import storage

        row = storage.create_headless_job(
            project_id="default", kind="graph", prompt=PROMPT,
        )
        storage.update_headless_job(row["id"], status="running", prompt_id="p-done")
        headless["comfy"].history["p-done"] = SUCCESS
        await headless["queue"].recover()
        await headless["queue"].tick()
        assert headless["comfy"].submitted == []
        assert storage.get_headless_job(row["id"])["status"] == "done"

    async def test_resume_queue_tool_requeues_an_error(self, headless):
        from ComfyTV.api.headless_queue import resume_queue as _resume_queue
        from ComfyTV.api.mcp_tools import _graph_run

        queued = await _graph_run({"prompt": PROMPT})
        await headless["queue"].tick()
        headless["comfy"].finish(headless["comfy"].running[0], ERROR)
        await headless["queue"].tick()
        out = await _resume_queue({"job_id": queued["job_id"]})
        assert out["job"]["status"] == "queued"
        await headless["queue"].tick()
        assert len(headless["comfy"].submitted) == 2


class TestHeadlessGraph:
    async def test_edit_roundtrip_and_subgraph_needs_a_page(self, headless):
        from ComfyTV.api.mcp_tools import _graph_edit, _graph_get, _graph_run

        edited = await _graph_edit({"ops": [
            {"op": "add_node", "type": "KSampler", "widgets": {"seed": 4, "steps": 8}},
            {"op": "set_widget", "node": "1", "name": "cfg", "value": 1},
            {"op": "set_mode", "node": "1", "mode": "bypass"},
        ]})
        assert edited["applied"][0]["node_id"] == "1"
        snap = await _graph_get({})
        assert snap["headless"] is True
        assert snap["nodes"][0]["widgets"]["cfg"] == 1
        assert snap["nodes"][0]["mode"] == 4
        with pytest.raises(ValueError, match="LiteGraph"):
            await _graph_edit({"ops": [{"op": "convert_to_subgraph", "nodes": ["1"]}]})
        queued = await _graph_run({})
        await headless["queue"].tick()
        sent = headless["comfy"].submitted[0]["prompt"]["1"]
        assert sent["class_type"] == "KSampler"
        assert sent["inputs"]["seed"] == 4
        assert sent["inputs"]["steps"] == 8
        assert queued["job_id"]

    async def test_projects_do_not_share_a_graph(self, headless):
        from ComfyTV.api.mcp_tools import _add_stage, _graph_get

        await _add_stage({
            "node_class": "ImageStage", "prompt": "only-a", "project_id": "proj-a",
        })
        empty = await _graph_get({"project_id": "proj-b"})
        assert empty["node_count"] == 0


class TestPageOnlyTools:
    async def test_editor_and_interrupt_fail_fast(self, headless):
        from ComfyTV.api.mcp_tools import _canvas_command, _director_get

        with pytest.raises(ValueError, match="open ComfyTV page"):
            await _director_get({"node": "1"})
        with pytest.raises(ValueError, match="not issued"):
            await _canvas_command({"command": "Comfy.Interrupt"})
        with pytest.raises(ValueError, match="not issued"):
            await _canvas_command({"command": "Comfy.ClearPendingTasks"})
        assert headless["comfy"].submitted == []


class TestComfyClient:
    async def test_submit_uses_the_prompt_queue_and_skips_a_busy_one(self, monkeypatch):
        import execution
        import server

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

        queue = _Queue()
        monkeypatch.setattr(server.PromptServer.instance, "prompt_queue", queue, raising=False)

        async def validate(prompt_id, prompt, partial):
            return (True, None, partial or list(prompt), {})

        monkeypatch.setattr(execution, "validate_prompt", validate, raising=False)
        client = ComfyQueueClient()
        prompt_id = await client.submit(PROMPT, ["1"])
        assert queue.items[0][1] == prompt_id
        assert len(queue.items[0]) == 6
        assert queue.items[0][5] == {}
        assert client.busy() is True

        queue.running.append((0, "external", {}, {}, [], {}))
        with pytest.raises(QueueBusy):
            await client.submit(PROMPT, ["1"])
        assert len(queue.items) == 1

    def test_new_modules_do_not_interrupt_or_free(self):
        root = Path(__file__).resolve().parents[1]
        for name in (
            "api/headless_queue.py",
            "api/headless_graph.py",
            "api/headless_dispatch.py",
        ):
            text = (root / name).read_text(encoding="utf-8")
            assert "/interrupt" not in text
            assert "/free" not in text
            assert "interrupt_processing" not in text
            assert "wipe_queue" not in text
            assert "127.0.0.1" not in text or "bind" not in text.lower()
