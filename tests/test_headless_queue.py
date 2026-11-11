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


class TestJobClockAndNodes:
    async def test_done_job_has_utc_timestamps_and_nodes(self, headless):
        from ComfyTV import storage
        from ComfyTV.api.headless_queue import list_queue
        from ComfyTV.api.headless_queue import queue_status as _queue_status
        from ComfyTV.api.mcp_tools import _graph_run

        queued = await _graph_run({"prompt": PROMPT})
        waiting = storage.get_headless_job(queued["job_id"])
        assert waiting["status"] == "queued"
        assert waiting["created_at"].endswith("+00:00")
        assert waiting["updated_at"].endswith("+00:00")
        assert waiting.get("started_at") is None
        assert waiting.get("finished_at") is None

        running = await _queue_status({"job_id": queued["job_id"]})
        assert running["job"]["status"] == "running"
        assert running["job"]["started_at"].endswith("+00:00")
        assert running["job"].get("finished_at") is None

        prompt_id = headless["comfy"].submitted[0]["prompt_id"]
        headless["comfy"].completed[prompt_id] = {
            "1": {"value": "/view?filename=live.png", "slot": 0, "slots": {"0": "/view?filename=live.png"}},
        }
        await headless["queue"].tick()
        progressed = await _queue_status({"job_id": queued["job_id"]})
        assert progressed["job"]["completed_nodes"] == ["1"]

        headless["comfy"].finish(prompt_id, SUCCESS)
        done = await _queue_status({"job_id": queued["job_id"]})
        assert done["job"]["status"] == "done"
        assert done["job"]["finished_at"].endswith("+00:00")
        assert done["job"]["started_at"] <= done["job"]["finished_at"]
        assert "1" in done["job"]["completed_nodes"]
        listed = await list_queue({})
        listed_job = next(job for job in listed["jobs"] if job["job_id"] == queued["job_id"])
        assert listed_job["finished_at"].endswith("+00:00")
        assert "1" in listed_job["completed_nodes"]

    async def test_cancel_and_error_set_finished_at(self, headless):
        from ComfyTV.api.headless_queue import queue_status as _queue_status
        from ComfyTV.api.mcp_tools import _add_stage, _cancel_stage, _graph_run, _run_stage

        added = await _add_stage({"node_class": "ImageStage", "prompt": "x"})
        queued = await _run_stage({"node": added["graph_node_id"]})
        await _cancel_stage({"node": added["graph_node_id"]})
        cancelled = await _queue_status({"job_id": queued["job_id"]})
        assert cancelled["job"]["status"] == "cancelled"
        assert cancelled["job"].get("started_at") is None
        assert cancelled["job"]["finished_at"].endswith("+00:00")

        failed = await _graph_run({"prompt": PROMPT})
        await headless["queue"].tick()
        prompt_id = headless["comfy"].running[0]
        headless["comfy"].finish(prompt_id, ERROR)
        errored = await _queue_status({"job_id": failed["job_id"]})
        assert errored["job"]["status"] == "error"
        assert errored["job"]["finished_at"].endswith("+00:00")
        assert errored["job"]["completed_nodes"] == []

    async def test_error_keeps_nodes_finished_before_the_failure(self, headless):
        from ComfyTV.api.headless_queue import queue_status as _queue_status
        from ComfyTV.api.mcp_tools import _graph_run

        failed = await _graph_run({"prompt": {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {}},
            "3": {"class_type": "KSampler", "inputs": {}},
        }})
        await headless["queue"].tick()
        prompt_id = headless["comfy"].running[0]
        headless["comfy"].finish(prompt_id, {
            "outputs": {},
            "status": {
                "status_str": "error",
                "messages": [[
                    "execution_error",
                    {
                        "node_id": "3",
                        "node_type": "KSampler",
                        "executed": ["1", "2", "3"],
                        "exception_message": "boom",
                    },
                ]],
            },
        })
        errored = await _queue_status({"job_id": failed["job_id"]})
        assert errored["job"]["status"] == "error"
        assert errored["job"]["completed_nodes"] == ["1", "2"]

        stopped = await _graph_run({"prompt": PROMPT})
        await headless["queue"].tick()
        prompt_id = headless["comfy"].running[0]
        headless["comfy"].finish(prompt_id, {
            "outputs": {},
            "status": {
                "status_str": "error",
                "messages": [[
                    "execution_interrupted",
                    {"node_id": "1", "node_type": "KSampler", "executed": ["9"]},
                ]],
            },
        })
        interrupted = await _queue_status({"job_id": stopped["job_id"]})
        assert interrupted["job"]["completed_nodes"] == ["9"]

    def test_naive_timestamps_parse_as_utc(self):
        from datetime import datetime, timezone

        from ComfyTV.storage.headless import format_utc_timestamp, parse_utc_timestamp

        naive = "2026-11-11T11:39:03.218265"
        parsed = parse_utc_timestamp(naive)
        assert parsed == datetime(2026, 11, 11, 11, 39, 3, 218265, tzinfo=timezone.utc)
        assert format_utc_timestamp(naive) == "2026-11-11T11:39:03.218265+00:00"
        assert format_utc_timestamp(naive + "Z") == "2026-11-11T11:39:03.218265+00:00"
        assert format_utc_timestamp(naive + "+00:00") == "2026-11-11T11:39:03.218265+00:00"
        shifted = parse_utc_timestamp("2026-11-11T19:39:03.218265+08:00")
        assert shifted == datetime(2026, 11, 11, 11, 39, 3, 218265, tzinfo=timezone.utc)


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
        assert inputs["images.image0"] == "/view?filename=ref.png&type=input"

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
        resumed = await restarted.recover(resubmit=True)
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


class TestRestartRecord:
    async def _lost(self, headless, prompt_id, prompt=None, partial=None):
        from ComfyTV import storage
        row = storage.create_headless_job(
            project_id="default",
            kind="graph",
            prompt=prompt or PROMPT,
            partial=partial,
        )
        storage.update_headless_job(row["id"], status="running", prompt_id=prompt_id)
        return row

    async def test_still_running_reattaches_and_does_not_submit(self, headless):
        from ComfyTV import storage
        headless["comfy"].running.append("still-running")
        headless["comfy"].history["still-running"] = SUCCESS
        row = await self._lost(headless, "still-running")
        resumed = await headless["queue"].recover()
        assert resumed == []
        fresh = storage.get_headless_job(row["id"])
        assert fresh["status"] == "running"
        assert fresh["prompt_id"] == "still-running"
        await headless["queue"].tick()
        assert headless["comfy"].submitted == []

    async def test_still_pending_reattaches_and_blocks_other_submits(self, headless):
        from ComfyTV import storage
        from ComfyTV.api.mcp_tools import _graph_run
        headless["comfy"].pending.append("still-pending")
        row = await self._lost(headless, "still-pending")
        await headless["queue"].recover()
        assert storage.get_headless_job(row["id"])["status"] == "running"
        assert storage.get_headless_job(row["id"])["prompt_id"] == "still-pending"
        await _graph_run({"prompt": PROMPT})
        await headless["queue"].tick()
        assert headless["comfy"].submitted == []

    async def test_history_is_collected_and_not_resubmitted(self, headless):
        from ComfyTV import storage
        headless["comfy"].history["p-done"] = SUCCESS
        row = await self._lost(headless, "p-done")
        await headless["queue"].recover()
        fresh = storage.get_headless_job(row["id"])
        assert fresh["status"] == "done"
        assert json.loads(fresh["outputs_json"]) == [
            "/view?filename=x.png&subfolder=&type=output",
        ]
        await headless["queue"].tick()
        assert headless["comfy"].submitted == []

    async def test_no_record_is_not_auto_resumed(self, headless):
        from ComfyTV import storage
        row = await self._lost(headless, "gone")
        resumed = await headless["queue"].recover()
        assert resumed == []
        fresh = storage.get_headless_job(row["id"])
        assert fresh["status"] == "interrupted"
        assert fresh["prompt_id"] == "gone"
        await headless["queue"].tick()
        assert headless["comfy"].submitted == []
        assert storage.get_setting("headless-auto-resume") is False

    async def test_resume_queue_submits_only_after_no_record(self, headless):
        from ComfyTV import storage
        from ComfyTV.api.headless_queue import resume_queue
        row = await self._lost(headless, "gone")
        await headless["queue"].recover()
        out = await resume_queue({"job_id": row["id"]})
        assert out["job"]["status"] == "queued"
        await headless["queue"].tick()
        assert len(headless["comfy"].submitted) == 1
        assert storage.get_headless_job(row["id"])["prompt_id"] == "prompt-1"


class TestStoredDataTools:
    async def test_storyboard_round_trip_without_a_page(self, headless):
        from ComfyTV.api.mcp_tools import _add_stage
        from ComfyTV.api.mcp_tools.storyboard_tool import _storyboard

        added = await _add_stage({"node_class": "StoryboardStage"})
        document = {
            "episode": "E01",
            "series": "gumu",
            "shots": [{
                "shot_id": "E01S01_01",
                "custom_note": "keep",
                "refs": [{"role": "first_frame", "src": "CHR_xiaolongnv@v1#front"}],
            }],
        }
        saved = await _storyboard({
            "action": "set", "node": added["graph_node_id"], "document": document,
        })
        assert saved["headless"] is True
        assert saved["shots"] == 1
        got = await _storyboard({"action": "get", "node": added["uid"]})
        assert got["headless"] is True
        assert got["document"]["shots"][0]["custom_note"] == "keep"
        assert got["episode"] == "E01"
        report = await _storyboard({"action": "validate", "document": document})
        assert "errors" in report and "warnings" in report
        assert report["ok"] is False

    async def test_series_library_reads_disk_without_a_page(self, headless, tmp_path):
        from ComfyTV.api.mcp_tools.series_library import _series_library

        root = tmp_path / "series"
        root.mkdir()
        (root / "library.yaml").write_text(
            "schema_version: 1\nseries:\n  id: demo\nassets: []\n",
            encoding="utf-8",
        )
        listed = await _series_library({"action": "list", "root": str(root)})
        assert listed["assets"] == []
        assert listed["series"]["id"] == "demo"


class TestReviewFixes:
    async def test_stage_prompt_includes_schema_defaults_and_autogrow_slot(self, headless):
        from ComfyTV.api.mcp_tools import _add_stage, _connect_stages, _run_stage

        upstream = await _add_stage({"node_class": "ImageStage", "prompt": "ref"})
        stage = await _add_stage({"node_class": "ImageStage", "prompt": "cat"})
        from ComfyTV.api import headless_graph as graph
        target = graph.find_node(graph.load_document("default"), stage["uid"])
        assert any(slot["name"] == "images.image0" for slot in target["inputs"])
        linked = await _connect_stages({
            "from_node": upstream["graph_node_id"],
            "to_node": stage["graph_node_id"],
            "to_slot": "images.image0",
        })
        assert linked["input"] == "images.image0"
        await _run_stage({"node": stage["uid"]})
        await headless["queue"].tick()
        inputs = headless["comfy"].submitted[0]["prompt"][stage["graph_node_id"]]["inputs"]
        assert inputs["main_prompt"] == "cat"
        assert inputs["resolution"] == "1K"
        assert inputs["batch_size"] == 1
        assert "workflow" in inputs
        assert inputs["images.image0"] == [upstream["graph_node_id"], 0]
        node = graph.find_node(graph.load_document("default"), stage["uid"])
        assert node["properties"]["comfytv_stage_uid"] == stage["uid"]

    async def test_imported_workflow_uid_and_widget_edit(self, headless):
        from ComfyTV.api.headless_graph import adopt_workflow
        from ComfyTV.api.mcp_tools import _graph_edit, _run_stage

        workflow = {
            "nodes": [{
                "id": 3,
                "type": "ComfyTV.ImageStage",
                "properties": {"comfytv_stage_uid": "uid-from-save"},
                "widgets_values": ["keep-token"],
                "inputs": [],
                "outputs": [{"name": "images", "type": "COMFYTV_IMAGES", "links": []}],
            }],
            "links": [],
        }
        adopt_workflow("default", workflow)
        await _graph_edit({"ops": [
            {"op": "set_widget", "node": "3", "name": "main_prompt", "value": "new prompt"},
        ]})
        await _run_stage({"node": "uid-from-save"})
        await headless["queue"].tick()
        inputs = headless["comfy"].submitted[0]["prompt"]["3"]["inputs"]
        assert inputs["main_prompt"] == "new prompt"
        assert inputs["resolution"] == "1K"
        assert inputs["force_run_token"] == "keep-token" or isinstance(inputs["force_run_token"], int)

    async def test_stage_history_payload_is_an_output(self, headless):
        from ComfyTV.api.headless_queue import queue_status
        from ComfyTV.api.mcp_tools import _graph_run

        queued = await _graph_run({"prompt": PROMPT})
        await headless["queue"].tick()
        prompt_id = headless["comfy"].running[0]
        headless["comfy"].finish(prompt_id, {
            "outputs": {"1": {"output": ["/view?filename=stage.png&type=output"],
                              "picked": ["/view?filename=picked.png&type=output"]}},
            "status": {"status_str": "success", "completed": True, "messages": []},
        })
        done = await queue_status({"job_id": queued["job_id"]})
        assert "/view?filename=stage.png&type=output" in done["job"]["outputs"]
        assert "/view?filename=picked.png&type=output" in done["job"]["outputs"]

    async def test_job_id_status_does_not_queue_when_a_page_opens(self, headless, monkeypatch):
        from ComfyTV.api import mcp_tools
        from ComfyTV.api.canvas_state import store_canvas_state
        from ComfyTV.api.mcp_tools import _graph_run

        queued = await _graph_run({"prompt": PROMPT})
        store_canvas_state("default", [], client_id="tab")
        calls = []

        async def submit(action, payload, timeout=15.0):
            calls.append(action)
            return {"queued": True, "prompt_id": "page"}

        monkeypatch.setattr(mcp_tools._shared, "submit_command", submit)
        status = await _graph_run({"job_id": queued["job_id"]})
        assert calls == []
        assert status["job"]["job_id"] == queued["job_id"]
        assert status["job"]["status"] in ("queued", "running")

    async def test_page_prompt_still_running_is_not_missing(self, headless):
        from ComfyTV.api.headless_queue import queue_status

        headless["comfy"].pending.append("page-prompt")
        status = await queue_status({"prompt_id": "page-prompt"})
        assert status["status"] == "running"
        assert headless["comfy"].submitted == []

    async def test_resume_one_job_leaves_the_other_interrupted(self, headless):
        from ComfyTV import storage
        from ComfyTV.api.headless_queue import resume_queue

        first = storage.create_headless_job(project_id="default", kind="graph", prompt=PROMPT)
        second = storage.create_headless_job(project_id="default", kind="graph", prompt={
            "9": {"class_type": "ComfyTV.ImageStage", "inputs": {"main_prompt": "other"}},
        })
        storage.update_headless_job(first["id"], status="running", prompt_id="gone-a")
        storage.update_headless_job(second["id"], status="running", prompt_id="gone-b")
        await headless["queue"].recover()
        await resume_queue({"job_id": first["id"]})
        assert storage.get_headless_job(first["id"])["status"] == "queued"
        assert storage.get_headless_job(second["id"])["status"] == "interrupted"

    async def test_cancel_loses_to_a_claimed_submit(self, headless):
        from ComfyTV import storage
        from ComfyTV.api.mcp_tools import _add_stage, _cancel_stage, _run_stage

        added = await _add_stage({"node_class": "ImageStage", "prompt": "x"})
        queued = await _run_stage({"node": added["graph_node_id"]})
        claimed = storage.claim_headless_job(
            queued["job_id"], expect="queued", become="submitting",
        )
        assert claimed["status"] == "submitting"
        with pytest.raises(ValueError, match="does not interrupt"):
            await _cancel_stage({"node": added["graph_node_id"]})
        storage.claim_headless_job(
            queued["job_id"], expect="submitting", become="running", prompt_id="kept",
        )
        assert storage.get_headless_job(queued["job_id"])["status"] == "running"
        assert storage.get_headless_job(queued["job_id"])["prompt_id"] == "kept"

    async def test_stage_lookup_is_not_hidden_by_newer_jobs(self, headless):
        from ComfyTV import storage
        from ComfyTV.api.mcp_tools import _add_stage, _run_stage

        first = await _add_stage({"node_class": "ImageStage", "prompt": "old"})
        queued = await _run_stage({"node": first["graph_node_id"]})
        for i in range(45):
            storage.create_headless_job(
                project_id="default", kind="stage", prompt={"1": {}},
                stage_ref=str(1000 + i), stage_uid=f"other-stage-{i:02d}",
            )
        found = storage.latest_headless_job_for_stage("default", first["uid"])
        assert found["id"] == queued["job_id"]


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
        assert queue.items[0][3]["client_id"] == "comfytv-headless"
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
