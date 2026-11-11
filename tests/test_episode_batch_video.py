"""Episode batch video: plan 22 shots without a model, then submit one at a time."""

from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "e01_shots.json"

_KIND = {
    "CHR": "character",
    "OUT": "outfit",
    "LOOK": "look",
    "LOC": "location",
    "PRP": "prop",
    "STY": "style",
}


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")


def _wav(path: Path, seconds: float, rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = max(1, int(round(seconds * rate)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * count)


def _kind_of(asset_id: str) -> str:
    prefix = asset_id.split("_", 1)[0]
    if prefix == "LOC" and "__" in asset_id:
        return "camera"
    return _KIND[prefix]


def _default_view(kind: str) -> str:
    if kind == "location":
        return "empty"
    if kind == "style":
        return "style"
    return "front"


def _e01_inputs(tmp_path: Path) -> dict:
    from ComfyTV.storage.series_library import import_series_manifest, iter_shot_refs
    import folder_paths

    doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    library = tmp_path / "library"
    library.mkdir()
    grouped: dict[tuple[str, int], set] = {}
    frames: set[str] = set()
    for shot in doc["shots"]:
        for parsed in iter_shot_refs(shot):
            if parsed["kind"] == "episode":
                frames.add(str(parsed["path"]))
            elif parsed["kind"] == "unresolved":
                grouped.setdefault((parsed["proposed_id"], 1), set()).add(parsed.get("view"))
            elif parsed["kind"] == "asset":
                grouped.setdefault(
                    (parsed["id"], int(parsed.get("version") or 1)), set(),
                ).add(parsed.get("view"))
    assets = []
    for (asset_id, version), views in sorted(grouped.items()):
        if asset_id.startswith(("VOX_", "BIB_")):
            continue
        kind = _kind_of(asset_id)
        names = {view for view in views if isinstance(view, str) and view}
        if not names or any(view in (None, "") for view in views):
            names.add(_default_view(kind))
        files = {}
        for view in sorted(names):
            rel = f"{asset_id}/v{version}/{view}.png"
            _png(library / rel)
            files[view] = {
                "path": rel,
                "media_type": "image",
                "usage": ["image_ref"],
            }
        assets.append({
            "id": asset_id,
            "version": version,
            "kind": kind,
            "status": "locked",
            "approved_by": "test",
            "files": files,
        })
    import_series_manifest(library, {
        "schema_version": 1,
        "series": {"id": "gumu-sx", "view_url_subfolder": "comfytv/series/e01"},
        "assets": assets,
    })
    frames_root = Path(folder_paths.get_input_directory()) / "e01-frames"
    for rel in frames:
        _png(frames_root / rel)
    speech = Path(folder_paths.get_output_directory()) / "e01-speech"
    for shot in doc["shots"]:
        for line in shot.get("dialogue") or []:
            if not isinstance(line, dict) or not str(line.get("text") or "").strip():
                continue
            seconds = 4.0 if shot["shot_id"] == "E01S01_02" else 0.4
            _wav(speech / shot["shot_id"] / f"{line['line_id']}.wav", seconds)
    return {
        "document": doc,
        "library_root": str(library),
        "frames_root": str(frames_root),
        "work_dir": str(Path(folder_paths.get_output_directory()) / "e01-dialogue"),
        "audio": {"dir": str(speech), "pattern": "{shot_id}/{line_id}.wav"},
        "base_seed": 20261111,
    }


def _plan(kwargs: dict):
    from ComfyTV.nodes.stages.common.episode_video import plan_episode_videos
    return plan_episode_videos(kwargs["document"], **{
        key: kwargs[key]
        for key in ("library_root", "frames_root", "work_dir", "audio", "base_seed")
    })


class TestE01DryRun:
    def test_twenty_two_shots_are_planned_without_submitting(self, tmp_path, monkeypatch):
        from ComfyTV.api import headless_queue

        def _refuse(*_args, **_kwargs):
            raise AssertionError("episode plan must not submit")

        monkeypatch.setattr(headless_queue, "enqueue_stage", _refuse)
        monkeypatch.setattr(headless_queue, "enqueue_graph", _refuse)
        kwargs = _e01_inputs(tmp_path)
        plan = _plan(kwargs)
        again = _plan(kwargs)

        assert plan["dry_run"] is True
        assert plan["ok"] is True, plan["errors"]
        assert plan["episode"] == "E01"
        assert plan["shot_count"] == 22
        assert len(plan["shots"]) == 22
        assert plan["duration_s_total"] == 126
        assert plan["errors"] == []

        dialogue = [shot for shot in plan["shots"] if shot["audio"]]
        silent = [shot for shot in plan["shots"] if shot["audio"] is None]
        assert len(dialogue) == 9
        assert len(silent) == 13
        assert all(shot["workflow"] == "Local MiniMax H3 R2V" for shot in plan["shots"])
        assert all(shot["mode"] == "Ref2VA" for shot in plan["shots"])
        assert all(shot["mode_source"] == "shot.mode" for shot in plan["shots"])
        assert all(shot["stage"]["node_class"] == "ComfyTV.VideoStage" for shot in plan["shots"])
        assert all(shot["stage"]["widgets"]["generate_audio"] is False for shot in plan["shots"])

        seeds = [shot["seed"] for shot in plan["shots"]]
        assert len(set(seeds)) == 22
        assert seeds == [shot["seed"] for shot in again["shots"]]
        assert all(shot["seed_source"] == "derived" for shot in plan["shots"])

        first = plan["shots"][0]
        assert first["shot_id"] == "E01S01_01"
        assert first["duration_s"] == 6
        assert first["cut_to"] == 5
        assert first["duration_source_s"] == 6
        assert first["images"][0]["role"] == "first_frame"
        assert first["images"][0]["url"].startswith("/view?")
        assert "In a candlelit stone chamber" in first["prompt"]
        assert any(image.get("asset_id") == "CHR_xiaolongnv" for image in first["images"])
        assert 1 <= len(first["images"]) <= 9

        spoken = plan["shots"][1]
        assert spoken["audio"]["line_ids"] == ["L001", "L002"]
        assert spoken["audio"]["duration_s"] == 8
        assert spoken["stage"]["widgets"]["audio.audio0"].startswith("/view?")
        assert "<Audio 1>" in spoken["prompt"]
        assert any("E01S01_02" in warning and "not trimmed" in warning for warning in plan["warnings"])
        assert all(shot["audio"] is None or shot["shot_id"] == "E01S01_02" or not shot["warnings"]
                   for shot in plan["shots"])

        bridged = next(shot for shot in plan["shots"] if shot["shot_id"] == "E01S03_05")
        assert bridged["frame_mode"] == "FL2VA"
        assert bridged["mode"] == "Ref2VA"
        assert bridged["mode_source"] == "shot.mode"
        assert bridged["reason"] is None
        assert any(image["role"] == "end_composition" for image in bridged["images"])
        assert all(shot["reason"] is None for shot in plan["shots"])

    def test_e01_frame_shot_with_dialogue_downgrades_to_ref2va(self, tmp_path):
        from ComfyTV.nodes.stages.common.episode_video import plan_episode_videos
        import folder_paths

        doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
        shot = next(item for item in doc["shots"] if item["shot_id"] == "E01S03_05")
        shot["mode"] = "FL2VA"
        shot.pop("video_mode", None)
        shot["characters"] = []
        shot["location"] = None
        shot["props"] = []
        shot["dialogue"] = [{
            "line_id": "L008",
            "text": "热是上来了。",
        }]
        frames = Path(folder_paths.get_input_directory()) / "e01-downgrade-frames"
        for ref in shot["refs"]:
            _png(frames / ref["src"].split("episode:", 1)[1])
        speech = Path(folder_paths.get_output_directory()) / "e01-downgrade" / "L008.wav"
        _wav(speech, 0.4)
        for other in doc["shots"]:
            if other["shot_id"] != "E01S03_05":
                other["dialogue"] = []
                other["refs"] = [{"role": "first_frame", "src": "/view?filename=a.png&type=input"}]
                other["characters"] = []
                other["location"] = None
                other["props"] = []
        planned = plan_episode_videos(
            doc,
            frames_root=str(frames),
            audio={"by_line": {"L008": str(speech)}},
            work_dir=str(tmp_path / "work"),
        )
        assert planned["ok"] is True, planned["errors"]
        affected = next(item for item in planned["shots"] if item["shot_id"] == "E01S03_05")
        assert affected["frame_mode"] == "FL2VA"
        assert affected["mode"] == "Ref2VA"
        assert affected["reason"] == "dialogue_downgrade"
        assert affected["mode_source"] == "dialogue_downgrade"
        assert affected["workflow"] == "Local MiniMax H3 R2V"
        assert affected["workflow_source"] == "dialogue_downgrade"
        roles = [image["role"] for image in affected["images"]]
        assert roles[0] == "first_frame"
        assert "end_composition" in roles
        assert len(affected["images"]) <= 9
        assert affected["audio"]["line_ids"] == ["L008"]
        assert affected["stage"]["widgets"]["workflow"] == "Local MiniMax H3 R2V"
        assert affected["stage"]["widgets"]["audio.audio0"].startswith("/view?")
        assert affected["stage"]["widgets"]["images.image0"] == affected["images"][0]["url"]
        assert "<Audio 1>" in affected["prompt"]
        assert not any("not sent to the model" in warning for warning in planned["warnings"])
        assert all(
            item["reason"] is None
            for item in planned["shots"]
            if item["shot_id"] != "E01S03_05"
        )

    async def test_plan_tool_returns_the_same_dry_run(self, tmp_path):
        from ComfyTV.api.mcp_tools.episode_batch import _episode_batch
        kwargs = _e01_inputs(tmp_path)
        result = await _episode_batch({"action": "plan", **kwargs})
        assert result["shot_count"] == 22
        assert result["dry_run"] is True
        assert result["ok"] is True


class TestShotRules:
    def _shot(self, **extra) -> dict:
        shot = {
            "shot_id": "E01S09_01",
            "order": 1,
            "scene_id": "E01S09",
            "duration_s": 5,
            "cut_to": 3,
            "dialogue": [],
            "refs": [
                {"role": "first_frame", "src": "/view?filename=a.png&type=input"},
                {"role": "end_composition", "src": "/view?filename=b.png&type=input"},
            ],
            "video_prompt": "a still figure turns",
        }
        shot.update(extra)
        return {"episode": "E01", "shots": [shot]}

    def test_frames_pick_fl2va_until_a_shot_field_overrides(self):
        from ComfyTV.nodes.stages.common.episode_video import plan_episode_videos

        planned = plan_episode_videos(self._shot())
        shot = planned["shots"][0]
        assert planned["ok"] is True
        assert shot["mode"] == "FL2VA"
        assert shot["mode_source"] == "frames"
        assert shot["reason"] is None
        assert shot["frame_mode"] == "FL2VA"
        assert shot["workflow"] == "Local MiniMax H3 FLF2V"
        assert [image["role"] for image in shot["images"]] == ["first_frame", "end_composition"]

        overridden = plan_episode_videos(self._shot(mode="Ref2VA", video_mode="Ref2VA"))
        shot = overridden["shots"][0]
        assert shot["mode"] == "Ref2VA"
        assert shot["mode_source"] == "shot.video_mode"
        assert shot["frame_mode"] == "FL2VA"
        assert shot["workflow"] == "Local MiniMax H3 R2V"
        assert shot["duration_s"] == 5
        assert shot["cut_to"] == 3

    def test_cut_longer_than_the_shot_is_an_error(self):
        from ComfyTV.nodes.stages.common.episode_video import plan_episode_videos

        planned = plan_episode_videos(self._shot(duration_s=5, cut_to=9))
        assert planned["ok"] is False
        assert any("cut_to 9.0 is longer than duration_s 5.0" in error for error in planned["errors"])

    def test_shot_seed_overrides_the_derived_seed(self):
        from ComfyTV.nodes.stages.common.episode_video import plan_episode_videos

        first = plan_episode_videos(self._shot(), base_seed=7)
        second = plan_episode_videos(self._shot(), base_seed=7)
        owned = plan_episode_videos(self._shot(seed=42), base_seed=7)
        assert first["shots"][0]["seed"] == second["shots"][0]["seed"]
        assert owned["shots"][0]["seed"] == 42
        assert owned["shots"][0]["seed_source"] == "shot"

    def test_dialogue_downgrades_fl2va_unless_video_mode_is_explicit(self, tmp_path):
        from ComfyTV.nodes.stages.common.episode_video import plan_episode_videos

        speech = tmp_path / "L001.wav"
        _wav(speech, 0.4)
        audio = {"by_line": {"L001": str(speech)}}
        doc = self._shot(
            mode="FL2VA",
            dialogue=[{"line_id": "L001", "text": "热上来了。"}],
        )
        downgraded = plan_episode_videos(doc, audio=audio, work_dir=str(tmp_path / "work"))
        shot = downgraded["shots"][0]
        assert downgraded["ok"] is True, downgraded["errors"]
        assert shot["mode"] == "Ref2VA"
        assert shot["reason"] == "dialogue_downgrade"
        assert shot["workflow"] == "Local MiniMax H3 R2V"
        assert [image["role"] for image in shot["images"]] == ["first_frame", "end_composition"]
        assert shot["stage"]["widgets"]["audio.audio0"].startswith("/view?")
        assert "<Audio 1>" in shot["prompt"]
        assert downgraded["warnings"] == []

        kept = plan_episode_videos(
            self._shot(
                mode="Ref2VA",
                video_mode="FL2VA",
                dialogue=[{"line_id": "L001", "text": "热上来了。"}],
            ),
            audio=audio,
            work_dir=str(tmp_path / "work"),
        )
        explicit = kept["shots"][0]
        assert explicit["mode"] == "FL2VA"
        assert explicit["reason"] is None
        assert explicit["mode_source"] == "shot.video_mode"
        assert explicit["workflow"] == "Local MiniMax H3 FLF2V"
        assert [image["role"] for image in explicit["images"]] == ["first_frame", "end_composition"]
        assert "audio.audio0" in explicit["stage"]["widgets"]
        assert "<Audio 1>" not in explicit["prompt"]
        assert any("does not bind upstream audio" in warning for warning in kept["warnings"])

        accepts = plan_episode_videos(
            doc,
            audio=audio,
            work_dir=str(tmp_path / "work"),
            workflows={"FL2VA": "Local MiniMax H3 R2V"},
        )
        stayed = accepts["shots"][0]
        assert stayed["mode"] == "FL2VA"
        assert stayed["reason"] is None
        assert stayed["workflow"] == "Local MiniMax H3 R2V"
        assert "<Audio 1>" in stayed["prompt"]
        assert accepts["warnings"] == []

    def test_downgrade_still_errors_when_references_exceed_nine(self, tmp_path):
        from ComfyTV.nodes.stages.common.episode_video import plan_episode_videos

        speech = tmp_path / "L001.wav"
        _wav(speech, 0.4)
        refs = [
            {"role": "first_frame", "src": "/view?filename=first.png&type=input"},
            {"role": "last_frame", "src": "/view?filename=last.png&type=input"},
        ]
        refs.extend(
            {"role": "ref", "src": f"/view?filename=p{i}.png&type=input"}
            for i in range(8)
        )
        planned = plan_episode_videos(
            self._shot(
                mode="FL2VA",
                refs=refs,
                dialogue=[{"line_id": "L001", "text": "一句。"}],
            ),
            audio={"by_line": {"L001": str(speech)}},
            work_dir=str(tmp_path / "work"),
        )
        shot = planned["shots"][0]
        assert shot["mode"] == "Ref2VA"
        assert shot["reason"] == "dialogue_downgrade"
        assert len(shot["images"]) == 10
        assert planned["ok"] is False
        assert any("none were dropped" in error for error in planned["errors"])
        assert shot["stage"] is None

    def test_extra_images_are_reported_and_kept(self):
        from ComfyTV.nodes.stages.common.episode_video import plan_episode_videos

        refs = [
            {"role": "ref", "src": f"/view?filename=p{i}.png&type=input"}
            for i in range(10)
        ]
        planned = plan_episode_videos(self._shot(mode="Ref2VA", refs=refs))
        assert planned["ok"] is False
        assert len(planned["shots"][0]["images"]) == 10
        assert any("none were dropped" in error for error in planned["errors"])

    def test_missing_audio_names_the_shot_and_line(self, tmp_path):
        from ComfyTV.nodes.stages.common.episode_video import (
            EpisodeBatchError,
            plan_episode_videos,
        )

        doc = self._shot(dialogue=[{
            "line_id": "L009",
            "text": "寒玉床上不能练。",
        }])
        missing = plan_episode_videos(doc)
        assert missing["ok"] is False
        assert any(
            "E01S09_01 line L009" in error and "no audio mapped" in error
            for error in missing["errors"]
        )
        absent = tmp_path / "no-such.wav"
        mapped = plan_episode_videos(doc, audio={"by_line": {"L009": str(absent)}})
        assert any(
            "E01S09_01 line L009" in error and "not found" in error and str(absent) in error
            for error in mapped["errors"]
        )
        with pytest.raises(EpisodeBatchError, match="unknown shot"):
            plan_episode_videos(doc, shots=["E01S99"])

    def test_silent_shot_is_not_given_audio_and_long_audio_is_not_trimmed(self, tmp_path):
        from ComfyTV.nodes.stages.common.episode_video import plan_episode_videos

        short = tmp_path / "short.wav"
        _wav(short, 0.5)
        silent = plan_episode_videos(
            self._shot(),
            audio={"by_shot": {"E01S09_01": str(short)}},
        )
        assert silent["ok"] is True
        assert silent["shots"][0]["audio"] is None
        assert "audio.audio0" not in silent["shots"][0]["stage"]["widgets"]

        first = tmp_path / "l1.wav"
        second = tmp_path / "l2.wav"
        _wav(first, 2)
        _wav(second, 2)
        doc = self._shot(
            mode="Ref2VA",
            duration_s=5,
            cut_to=3,
            dialogue=[
                {"line_id": "L001", "text": "第一句。"},
                {"line_id": "L002", "text": "第二句。"},
            ],
        )
        planned = plan_episode_videos(
            doc,
            audio={"by_shot": {"E01S09_01": [str(first), str(second)]}},
            work_dir=str(tmp_path / "work"),
        )
        assert planned["ok"] is True, planned["errors"]
        audio = planned["shots"][0]["audio"]
        assert audio["line_ids"] == ["L001", "L002"]
        assert audio["duration_s"] == 4
        assert audio["url"].startswith("/view?")
        assert audio["files"] == [str(first), str(second)]
        assert Path(audio["path"]).stat().st_size > first.stat().st_size
        warning = planned["warnings"][0]
        assert "longer than duration_s" not in warning
        assert "longer than cut_to 3.00s" in warning
        assert "not trimmed" in warning

    def test_subset_keeps_order(self):
        from ComfyTV.nodes.stages.common.episode_video import plan_episode_videos

        doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
        for shot in doc["shots"]:
            shot["dialogue"] = []
            shot["refs"] = [{"role": "first_frame", "src": "/view?filename=a.png&type=input"}]
            shot["characters"] = []
            shot["location"] = None
            shot["props"] = []
        planned = plan_episode_videos(doc, shots=[22, "E01S01_02"])
        assert [shot["shot_id"] for shot in planned["shots"]] == ["E01S04_05", "E01S01_02"]
        assert planned["ok"] is True


class TestSerialSubmit:
    @pytest.fixture()
    def headless(self, reset_db, monkeypatch):
        from test_headless_queue import FakeComfy
        from ComfyTV.api.canvas_state import clear_canvas_state
        from ComfyTV.api.headless_queue import HeadlessQueue, reset_for_tests, set_queue
        from ComfyTV.runners import workflow_db
        from ComfyTV.runners.workflow_db import seed as workflow_seed

        monkeypatch.setattr(
            workflow_seed,
            "_LEGACY_WORKFLOWS_DIR",
            Path(__file__).resolve().parent.parent / "workflows",
        )
        workflow_db.seed_workflows_from_disk(("video",))
        clear_canvas_state()
        fake = FakeComfy()
        runner = HeadlessQueue(client=fake, autostart=False, poll_s=0.01)
        set_queue(runner)
        yield {"queue": runner, "comfy": fake}
        reset_for_tests()
        clear_canvas_state()

    def _doc(self, tmp_path: Path) -> tuple[dict, dict]:
        speech = tmp_path / "L001.wav"
        _wav(speech, 0.4)
        doc = {
            "episode": "E01",
            "series": "gumu-sx",
            "shots": [
                {
                    "shot_id": f"E01S01_{n:02d}",
                    "order": n,
                    "duration_s": 5,
                    "cut_to": 4,
                    "mode": "Ref2VA",
                    "dialogue": (
                        [{"line_id": "L001", "text": "你好。"}] if n == 2 else []
                    ),
                    "refs": [{"role": "first_frame", "src": "/view?filename=ref.png&type=input"}],
                    "video_prompt": f"shot {n}",
                }
                for n in (1, 2, 3)
            ],
        }
        args = {
            "document": doc,
            "audio": {"by_line": {"L001": str(speech)}},
            "work_dir": str(tmp_path / "work"),
            "state_path": str(tmp_path / "batch.json"),
            "base_seed": 11,
        }
        return doc, args

    async def _finish(self, comfy, expected: int, *, fail_first: bool = False):
        from test_headless_queue import ERROR, SUCCESS

        seen: list[str] = []
        deadline = asyncio.get_running_loop().time() + 5
        while len(seen) < expected:
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError(
                    f"timed out with {len(seen)} of {expected}; running={comfy.running}"
                )
            assert len(comfy.running) <= 1
            for prompt_id in list(comfy.running):
                if prompt_id in seen:
                    continue
                entry = ERROR if fail_first and not seen else SUCCESS
                comfy.finish(prompt_id, entry)
                seen.append(prompt_id)
            await asyncio.sleep(0.005)
        return seen

    async def test_run_submits_one_at_a_time_and_skips_completed(self, headless, tmp_path, monkeypatch):
        from ComfyTV.api.episode_batch import run_episode_batch
        from ComfyTV.nodes.stages.generators import VideoStage

        async def _no_model(*_args, **_kwargs):
            raise AssertionError("model was called")

        monkeypatch.setattr(VideoStage, "execute", _no_model)
        _doc, args = self._doc(tmp_path)
        state = {
            "completed": {"E01S01_01": {"job_id": "already", "outputs": []}},
            "nodes": {},
        }
        Path(args["state_path"]).write_text(json.dumps(state), encoding="utf-8")
        comfy = headless["comfy"]
        task = asyncio.create_task(run_episode_batch({"action": "run", **args}))
        await self._finish(comfy, 2)
        result = await task

        assert result["dry_run"] is False
        assert result["submitted"] == 2
        statuses = {row["shot_id"]: row["status"] for row in result["results"]}
        assert statuses == {
            "E01S01_01": "skipped",
            "E01S01_02": "done",
            "E01S01_03": "done",
        }
        assert len(comfy.submitted) == 2
        for item in comfy.submitted:
            assert len(item["prompt"]) == 1
            node = next(iter(item["prompt"].values()))
            assert node["class_type"] == "ComfyTV.VideoStage"
            assert node["inputs"]["duration_s"] == 5
        spoken = next(
            node["inputs"]
            for item in comfy.submitted
            for node in item["prompt"].values()
            if node["inputs"].get("main_prompt", "").startswith("shot 2")
        )
        assert spoken["audio.audio0"]
        assert "audio.audio0" not in next(
            node["inputs"]
            for item in comfy.submitted
            for node in item["prompt"].values()
            if node["inputs"].get("main_prompt", "").startswith("shot 3")
        )

        before = len(comfy.submitted)
        again = await run_episode_batch({"action": "run", **args})
        assert again["submitted"] == 0
        assert len(comfy.submitted) == before
        assert all(row["status"] == "skipped" for row in again["results"])

    async def test_run_waits_until_the_queue_is_idle(self, headless, tmp_path):
        from ComfyTV.api.episode_batch import run_episode_batch

        _doc, args = self._doc(tmp_path)
        args["shots"] = [1]
        comfy = headless["comfy"]
        comfy.running.append("external")

        async def release():
            for _ in range(8):
                assert comfy.submitted == []
                await asyncio.sleep(0.01)
            comfy.running.remove("external")

        task = asyncio.create_task(run_episode_batch({"action": "run", **args}))
        await release()
        await self._finish(comfy, 1)
        result = await task
        assert result["submitted"] == 1
        assert comfy.submitted[0]["prompt_id"] != "external"

    async def test_a_failed_shot_is_retried_and_the_next_was_not_submitted(self, headless, tmp_path):
        from ComfyTV.api.episode_batch import run_episode_batch
        from ComfyTV.nodes.stages.common.episode_video import EpisodeBatchError

        _doc, args = self._doc(tmp_path)
        args["shots"] = [1, 2]
        comfy = headless["comfy"]
        task = asyncio.create_task(run_episode_batch({"action": "run", **args}))
        await self._finish(comfy, 1, fail_first=True)
        with pytest.raises(EpisodeBatchError, match="E01S01_01 failed"):
            await task
        assert len(comfy.submitted) == 1
        saved = json.loads(Path(args["state_path"]).read_text(encoding="utf-8"))
        assert "E01S01_01" not in saved["completed"]

        task = asyncio.create_task(run_episode_batch({"action": "run", **args}))
        await self._finish(comfy, 2)
        result = await task
        assert [row["status"] for row in result["results"]] == ["done", "done"]
        assert len({row["graph_node_id"] for row in result["results"] if row["shot_id"] == "E01S01_01"}) == 1

    async def test_plan_reads_a_storyboard_stage(self, headless, tmp_path):
        from ComfyTV.api.episode_batch import run_episode_batch
        from ComfyTV.api.headless_graph import storyboard_set
        from ComfyTV.api.mcp_tools.stages import _add_stage

        _doc, args = self._doc(tmp_path)
        added = await _add_stage({
            "node_class": "StoryboardStage",
            "project_id": "default",
        })
        storyboard_set({
            "node": added["graph_node_id"],
            "project_id": "default",
            "document": args["document"],
        })
        result = await run_episode_batch({
            "action": "plan",
            "node": added["graph_node_id"],
            "project_id": "default",
            "audio": args["audio"],
            "work_dir": args["work_dir"],
        })
        assert result["ok"] is True, result["errors"]
        assert result["shot_count"] == 3
        assert result["shots"][1]["audio"]["line_ids"] == ["L001"]

    async def test_run_submits_the_downgraded_ref2va_workflow(self, headless, tmp_path):
        from ComfyTV.api.episode_batch import run_episode_batch

        speech = tmp_path / "L008.wav"
        _wav(speech, 0.4)
        task = asyncio.create_task(run_episode_batch({
            "action": "run",
            "document": {
                "episode": "E01",
                "shots": [{
                    "shot_id": "E01S03_05",
                    "order": 15,
                    "duration_s": 5,
                    "cut_to": 3,
                    "mode": "FL2VA",
                    "dialogue": [{"line_id": "L008", "text": "热是上来了。"}],
                    "refs": [
                        {"role": "first_frame", "src": "/view?filename=first.png&type=input"},
                        {"role": "end_composition", "src": "/view?filename=last.png&type=input"},
                    ],
                    "video_prompt": "she reacts",
                }],
            },
            "audio": {"by_line": {"L008": str(speech)}},
            "work_dir": str(tmp_path / "work"),
            "state_path": str(tmp_path / "state.json"),
        }))
        await self._finish(headless["comfy"], 1)
        result = await task
        assert result["submitted"] == 1
        node = next(iter(headless["comfy"].submitted[0]["prompt"].values()))
        assert node["inputs"]["workflow"] == "Local MiniMax H3 R2V"
        assert node["inputs"]["images.image0"].endswith("first.png") or "first.png" in node["inputs"]["images.image0"]
        assert "last.png" in node["inputs"]["images.image1"]
        assert node["inputs"]["audio.audio0"]
        assert "<Audio 1>" in node["inputs"]["main_prompt"]

    async def test_http_plan_does_not_require_a_model(self, reset_db):
        from ComfyTV import api  # noqa: F401
        import server

        app = web.Application()
        app.router.add_routes(server.PromptServer.instance.routes)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post("/comfytv/episode_batch", json={
                "action": "plan",
                "document": {
                    "episode": "E01",
                    "shots": [{
                        "shot_id": "E01S01_01",
                        "order": 1,
                        "duration_s": 5,
                        "mode": "Ref2VA",
                        "dialogue": [],
                        "refs": [{"role": "first_frame", "src": "/view?filename=a.png&type=input"}],
                        "video_prompt": "hold",
                    }],
                },
            })
            assert response.status == 200
            body = await response.json()
            assert body["shot_count"] == 1
            assert body["dry_run"] is True
            assert body["shots"][0]["workflow"] == "Local MiniMax H3 R2V"
        finally:
            await client.close()
