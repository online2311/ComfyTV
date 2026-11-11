"""Consistency scoring and reshoot. The scorer in these tests is a fake."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
import yaml
from PIL import Image

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "series_library" / "library.yaml"


def materialize(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    text = FIXTURE.read_text(encoding="utf-8")
    (dest / "library.yaml").write_text(text, encoding="utf-8")
    doc = yaml.safe_load(text)
    for asset in doc["assets"]:
        for view, info in (asset.get("files") or {}).items():
            rel = (info or {}).get("path")
            if not rel:
                continue
            path = dest / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{asset['id']}:{view}".encode())
    return dest


class FakeScorer:
    name = "fake"

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []

    def score_pair(self, frame_path: str, reference_path: str) -> float:
        self.calls.append((frame_path, reference_path))
        if not self.scores:
            raise AssertionError("fake scorer ran out of take scores")
        return float(self.scores[0])

    def finish_take(self) -> None:
        if self.scores:
            self.scores.pop(0)


class ScriptedReader:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.reads = []

    def __call__(self):
        snap = self.snapshots.pop(0) if len(self.snapshots) > 1 else self.snapshots[0]
        self.reads.append(snap)
        return snap


class RecordingSubmitter:
    def __init__(self, videos, reader: ScriptedReader):
        self.videos = list(videos)
        self.reader = reader
        self.jobs = []
        self.reads_before = []

    def submit(self, job):
        snap = self.reader.reads[-1]
        self.reads_before.append(snap)
        assert not snap["running"] and not snap["pending"]
        self.jobs.append(job)
        return self.videos.pop(0)


def _shot():
    return {
        "shot_id": "E01S01_01",
        "characters": [
            {"id": "CHR_xiaolongnv@v1", "outfit": "OUT_xiaolongnv_baiyi@v1"},
        ],
        "motion_desc": "she turns her head",
    }


def _extract(video_url, count):
    frames = []
    for index in range(count):
        frames.append({
            "t": float(index),
            "path": f"{video_url}#frame{index}",
            "image_url": f"/view?filename={video_url}-{index}.png",
        })
    return frames


async def _run(**overrides):
    from ComfyTV.runners.consistency_rescore import run_consistency_rescore

    root = overrides.pop("root")
    reader = overrides.pop("reader")
    submitter = overrides.pop("submitter")
    scorer = overrides.pop("scorer")
    sheets = overrides.pop("sheets")
    kwargs = dict(
        video_url="take-1",
        shot=_shot(),
        series_root=root,
        scorer=scorer,
        reader=reader,
        submitter=submitter,
        threshold=0.70,
        seed=5,
        frame_count=2,
        extract_frames=_extract,
        write_contact_sheet=lambda takes: sheets.append(takes) or "/view?filename=sheet.png",
        sleep=lambda _seconds: None,
        clock=lambda: 0.0,
        queue_timeout_s=10.0,
    )
    kwargs.update(overrides)
    return await run_consistency_rescore(**kwargs)


async def test_reshoot_stops_at_the_configured_limit_and_keeps_every_take(tmp_path):
    root = materialize(tmp_path / "series")
    scorer = FakeScorer([0.10, 0.20, 0.30])
    reader = ScriptedReader([{"running": [], "pending": []}])
    submitter = RecordingSubmitter(["take-2", "take-3"], reader)
    sheets = []

    report = await _run(
        root=root, reader=reader, submitter=submitter, scorer=scorer, sheets=sheets,
        max_reshoots=2,
    )

    assert report["stop"] == "pick_take"
    assert report["max_reshoots"] == 2
    assert [take["video_url"] for take in report["takes_in_order"]] == [
        "take-1", "take-2", "take-3",
    ]
    assert [take["seed"] for take in report["takes_in_order"]] == [5, 6, 7]
    assert [take["score"] for take in report["takes"]] == [0.30, 0.20, 0.10]
    assert report["takes"][0]["take"] == 3
    assert report["selected_video"] == "take-3"
    assert report["contact_sheet"] == "/view?filename=sheet.png"
    assert report["images"][0]["label"] == "take 3  seed 7  score 0.300"
    assert report["images"][0]["video_url"] == "take-3"
    assert [job.seed for job in submitter.jobs] == [6, 7]
    assert scorer.calls
    assert all(path.endswith("headshot.png") for _frame, path in scorer.calls)
    assert sheets and sheets[0][0]["score"] == 0.30
    assert sheets[0][0]["frames"][0]["score"] == 0.30


async def test_a_passing_take_is_not_resubmitted(tmp_path):
    root = materialize(tmp_path / "series")
    scorer = FakeScorer([0.91])
    reader = ScriptedReader([{"running": ["busy"], "pending": []}])
    submitter = RecordingSubmitter(["should-not-run"], reader)
    sheets = []

    report = await _run(
        root=root, reader=reader, submitter=submitter, scorer=scorer, sheets=sheets,
    )

    assert submitter.jobs == []
    assert reader.reads == []
    assert report["reshoots"] == 0
    assert report["takes"][0]["passed"] is True
    assert report["backend"] == "fake"


async def test_default_reshoot_limit_is_two(tmp_path):
    root = materialize(tmp_path / "series")
    scorer = FakeScorer([0.1, 0.2, 0.95])
    reader = ScriptedReader([{"running": [], "pending": []}])
    submitter = RecordingSubmitter(["take-2", "take-3", "take-4"], reader)

    report = await _run(
        root=root, reader=reader, submitter=submitter, scorer=scorer, sheets=[],
    )

    assert report["max_reshoots"] == 2
    assert report["reshoots"] == 2
    assert len(report["takes_in_order"]) == 3


async def test_reshoot_waits_until_get_queue_is_empty_and_never_interrupts(tmp_path):
    root = materialize(tmp_path / "series")
    scorer = FakeScorer([0.1, 0.9])
    reader = ScriptedReader([
        {"running": ["other"], "pending": ["queued"]},
        {"running": [], "pending": []},
    ])
    submitter = RecordingSubmitter(["take-2"], reader)
    slept = []

    report = await _run(
        root=root, reader=reader, submitter=submitter, scorer=scorer, sheets=[],
        sleep=lambda seconds: slept.append(seconds),
    )

    assert slept
    assert submitter.reads_before == [{"running": [], "pending": []}]
    assert report["reshoots"] == 1
    source = Path(__file__).resolve().parents[1].joinpath(
        "runners", "consistency_rescore.py").read_text(encoding="utf-8")
    backend = Path(__file__).resolve().parents[1].joinpath(
        "runners", "consistency_backend.py").read_text(encoding="utf-8")
    stage = Path(__file__).resolve().parents[1].joinpath(
        "nodes", "stages", "consistency.py").read_text(encoding="utf-8")
    for text in (source, backend, stage):
        assert "/interrupt" not in text
        assert "/free" not in text
        assert "cancel_stage" not in text


async def test_a_busy_queue_does_not_submit(tmp_path):
    from ComfyTV.runners.consistency_rescore import ConsistencyQueueBusy

    root = materialize(tmp_path / "series")
    scorer = FakeScorer([0.1, 0.9])
    reader = ScriptedReader([{"running": ["other"], "pending": []}])
    submitter = RecordingSubmitter(["take-2"], reader)

    with pytest.raises(ConsistencyQueueBusy, match="not submitted") as raised:
        await _run(
            root=root, reader=reader, submitter=submitter, scorer=scorer, sheets=[],
            queue_timeout_s=0,
        )

    assert submitter.jobs == []
    assert raised.value.report["takes_in_order"][0]["score"] == 0.1


async def test_shot_score_is_the_weakest_character(tmp_path):
    from ComfyTV.runners.consistency_rescore import run_consistency_rescore

    root = tmp_path / "series"
    root.mkdir()
    (root / "library.yaml").write_text(
        """
schema_version: 1
series:
  id: demo
  view_url_subfolder: comfytv/series/demo
assets:
- id: CHR_one
  version: 1
  kind: character
  status: locked
  name_zh: 甲
  approved_by: 张景
  files:
    headshot:
      path: CHR_one/v1/headshot.png
      media_type: image
- id: CHR_two
  version: 1
  kind: character
  status: locked
  name_zh: 乙
  approved_by: 张景
  files:
    front:
      path: CHR_two/v1/front.png
      media_type: image
""".strip(),
        encoding="utf-8",
    )
    (root / "CHR_one/v1").mkdir(parents=True)
    (root / "CHR_two/v1").mkdir(parents=True)
    (root / "CHR_one/v1/headshot.png").write_bytes(b"one")
    (root / "CHR_two/v1/front.png").write_bytes(b"two")

    class ByRef:
        name = "fake"

        def score_pair(self, frame_path, reference_path):
            if reference_path.endswith("headshot.png"):
                return 0.95
            return 0.40

        def finish_take(self):
            return None

    report = await run_consistency_rescore(
        video_url="only",
        shot={
            "shot_id": "S",
            "characters": ["CHR_one@v1", {"id": "CHR_two@v1"}],
        },
        series_root=root,
        scorer=ByRef(),
        reader=lambda: {"running": [], "pending": []},
        submitter=RecordingSubmitter(["nope"], ScriptedReader([{"running": [], "pending": []}])),
        threshold=0.70,
        max_reshoots=0,
        frame_count=1,
        seed=1,
        extract_frames=_extract,
        write_contact_sheet=lambda takes: "/view?filename=sheet.png",
        sleep=lambda _seconds: None,
        clock=lambda: 0.0,
    )

    assert report["takes"][0]["score"] == pytest.approx(0.40)
    assert report["takes"][0]["passed"] is False
    views = {row["id"]: row["view"] for row in report["takes"][0]["characters"]}
    assert views == {"CHR_one": "headshot", "CHR_two": "front"}


def test_unlocked_character_is_rejected(tmp_path):
    from ComfyTV.runners.consistency_rescore import ConsistencyRescoreError

    root = tmp_path / "series"
    root.mkdir()
    (root / "library.yaml").write_text(
        """
schema_version: 1
series:
  id: demo
assets:
- id: CHR_hero
  version: 1
  kind: character
  status: draft
  name_zh: 甲
  files:
    headshot:
      path: CHR_hero/v1/headshot.png
      media_type: image
""".strip(),
        encoding="utf-8",
    )
    (root / "CHR_hero/v1").mkdir(parents=True)
    (root / "CHR_hero/v1/headshot.png").write_bytes(b"x")

    with pytest.raises(ConsistencyRescoreError, match="not locked"):
        from ComfyTV.runners.consistency_rescore import locked_character_refs
        locked_character_refs(
            {"characters": [{"id": "CHR_hero@v1"}]},
            root,
        )


def test_missing_backend_raises_and_does_not_offer_an_install(monkeypatch):
    from ComfyTV.runners.consistency_backend import (
        ConsistencyBackendUnavailable,
        build_backend,
    )
    import folder_paths
    import nodes

    monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", {}, raising=False)
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda _kind: [])

    with pytest.raises(ConsistencyBackendUnavailable, match="CLIPVision") as raised:
        build_backend("auto")

    message = str(raised.value)
    assert "face" in message.lower()
    assert "pip" in message.lower()


def test_clip_vision_backend_scores_with_the_registered_nodes(monkeypatch, tmp_path):
    from ComfyTV.runners import consistency_backend as backend
    import folder_paths
    import nodes

    class Loader:
        calls = []

        def load_clip(self, clip_name):
            Loader.calls.append(clip_name)
            return (f"vision:{clip_name}",)

    class Encoder:
        def encode(self, clip_vision, image, crop="center"):
            value = float(image)
            return (type("Out", (), {"image_embeds": [value, 1.0 - value]})(),)

    monkeypatch.setattr(nodes, "NODE_CLASS_MAPPINGS", {
        "CLIPVisionLoader": Loader,
        "CLIPVisionEncode": Encoder,
    }, raising=False)
    monkeypatch.setattr(
        folder_paths, "get_filename_list",
        lambda kind: ["vision.safetensors"] if kind == "clip_vision" else [],
    )
    monkeypatch.setattr(backend, "load_image_value", lambda path: float(Path(path).read_text()))

    left = tmp_path / "a.png"
    right = tmp_path / "b.png"
    same = tmp_path / "c.png"
    left.write_text("0.2")
    right.write_text("0.2")
    same.write_text("0.8")
    scorer = backend.build_backend("clip_vision")
    identical = scorer.score_pair(str(left), str(right))
    other = scorer.score_pair(str(left), str(same))

    assert identical == pytest.approx(1.0)
    assert other < 1.0
    assert Loader.calls == ["vision.safetensors"]


def test_registered_backend_replaces_the_builtin(monkeypatch):
    from ComfyTV.runners.consistency_backend import (
        build_backend,
        register_similarity_backend,
        unregister_similarity_backend,
    )

    class Custom:
        name = "custom"

        def score_pair(self, frame_path, reference_path):
            return 0.42

    register_similarity_backend("custom", lambda: Custom())
    try:
        got = build_backend("custom")
        assert got.score_pair("f", "r") == 0.42
        assert got.name == "custom"
    finally:
        unregister_similarity_backend("custom")


def test_ffmpeg_frame_times_and_command(tmp_path):
    from ComfyTV.runners.consistency_rescore import extract_shot_frames

    calls = []

    def run(cmd, **_kwargs):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"\x89PNG")
        return type("R", (), {"returncode": 0})()

    frames = extract_shot_frames(
        "/view?filename=shot.mp4",
        4,
        duration_s=8.0,
        source_path="/tmp/shot.mp4",
        run=run,
        output_path=lambda: tmp_path / f"f{len(calls)}.png",
        to_url=lambda path: f"/view?filename={Path(path).name}",
    )

    assert [round(frame["t"], 2) for frame in frames] == [0.0, 2.65, 5.3, 7.95]
    assert calls[0][0] == "ffmpeg"
    assert "-ss" in calls[0]
    assert "interrupt" not in " ".join(calls[0])


def test_contact_sheet_draws_one_row_per_take(tmp_path):
    from ComfyTV.runners.consistency_rescore import render_contact_sheet, take_label

    frames = []
    for name, color in (("a.png", (200, 10, 10)), ("b.png", (10, 10, 200))):
        path = tmp_path / name
        Image.new("RGB", (8, 8), color).save(path)
        frames.append({"t": 0.0, "path": str(path), "image_url": name, "score": 0.5})
    takes = [
        {"take": 2, "seed": 9, "score": 0.88, "frames": frames},
        {"take": 1, "seed": 4, "score": 0.12, "frames": frames[:1]},
    ]
    out = tmp_path / "sheet.png"
    render_contact_sheet(takes, out)
    image = Image.open(out)
    assert image.mode == "RGB"
    assert image.size[0] > 8 and image.size[1] > 8
    assert take_label(takes[0]) == "take 2  seed 9  score 0.880"


def test_queue_client_gets_queue_before_post_and_reads_history():
    from ComfyTV.runners.consistency_rescore import submit_reshoot_prompt

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(("GET", self.path))
            if self.path == "/queue":
                running = [] if seen.count(("GET", "/queue")) > 1 else [[0, "other", {}, {}, []]]
                body = json.dumps({"queue_running": running, "queue_pending": []}).encode()
            elif self.path.startswith("/history/"):
                body = json.dumps({
                    "pid-1": {
                        "status": {"status_str": "success", "completed": True},
                        "outputs": {
                            "9": {"videos": [{
                                "filename": "take.mp4",
                                "subfolder": "comfytv",
                                "type": "output",
                            }]},
                        },
                    },
                }).encode()
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            seen.append(("POST", self.path))
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            body = json.dumps({"prompt_id": "pid-1"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        url = submit_reshoot_prompt(
            base,
            {"1": {"class_type": "SaveVideo", "inputs": {}}},
            sleep=lambda _seconds: None,
            clock=lambda: 0.0,
            queue_timeout_s=5,
            history_timeout_s=5,
        )
    finally:
        server.shutdown()

    assert url == "/view?filename=take.mp4&subfolder=comfytv&type=output"
    methods = [(item[0], item[1].split("?")[0]) for item in seen]
    post_at = methods.index(("POST", "/prompt"))
    assert ("GET", "/queue") in methods[:post_at]
    assert methods[post_at - 1] == ("GET", "/queue")
    assert not any("interrupt" in path or path == "/free" for _method, path in methods)


async def test_empty_series_root_uses_the_bound_library(tmp_path, monkeypatch):
    root = materialize(tmp_path / "series")
    user = tmp_path / "user"
    user.mkdir()
    import folder_paths
    from ComfyTV.storage.series_library import bind_series_library

    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(user))
    bind_series_library(root)
    scorer = FakeScorer([0.88])
    reader = ScriptedReader([{"running": [], "pending": []}])
    report = await _run(
        root=None,
        reader=reader,
        submitter=RecordingSubmitter(["nope"], reader),
        scorer=scorer,
        sheets=[],
        max_reshoots=0,
    )
    assert report["takes"][0]["characters"][0]["view"] == "headshot"
    assert report["takes"][0]["passed"] is True


def test_reshoot_prompt_uses_the_new_seed(monkeypatch):
    from ComfyTV.runners.consistency_rescore import ReshootJob, build_reshoot_prompt

    captured = {}

    def prepare(runner_id, kinds, ctx):
        captured["id"] = runner_id
        captured["kinds"] = set(kinds)
        captured["seed"] = ctx.options["seed"]
        captured["prompt"] = ctx.main_prompt
        captured["images"] = list(ctx.upstream["images"])
        return {"1": {"class_type": "SaveVideo", "inputs": {}}}, {"node": "1"}

    monkeypatch.setattr(
        "ComfyTV.runners.consistency_rescore.prepare_workflow", prepare,
    )
    prompt = build_reshoot_prompt(ReshootJob(
        seed=6,
        shot={},
        prompt="she turns her head",
        images=["/view?filename=face.png"],
        workflow="GXS",
        options={"resolution": "720P"},
    ))
    assert prompt["1"]["class_type"] == "SaveVideo"
    assert captured["id"] == "video/GXS"
    assert captured["kinds"] == {"video"}
    assert captured["seed"] == 6
    assert captured["prompt"] == "she turns her head"
    assert captured["images"] == ["/view?filename=face.png"]


async def test_a_running_prompt_still_reshoots_once_pending_is_empty(tmp_path):
    root = materialize(tmp_path / "series")
    reader = ScriptedReader([
        {"running": ["this-stage"], "pending": ["other"]},
        {"running": ["this-stage"], "pending": []},
    ])

    class Submit:
        def __init__(self):
            self.jobs = []

        def submit(self, job):
            assert reader.reads[-1]["pending"] == []
            assert reader.reads[-1]["running"] == ["this-stage"]
            self.jobs.append(job)
            return "take-2"

    submitter = Submit()
    report = await _run(
        root=root,
        reader=reader,
        submitter=submitter,
        scorer=FakeScorer([0.1, 0.95]),
        sheets=[],
        allow_running=True,
    )
    assert [job.seed for job in submitter.jobs] == [6]
    assert report["reshoots"] == 1
    assert report["selected_video"] == "take-2"


def test_runner_payload_is_a_video_url():
    from ComfyTV.runners.consistency_rescore import coerce_video_url

    assert coerce_video_url("/view?filename=a.mp4") == "/view?filename=a.mp4"
    assert coerce_video_url('{"video_url": "/view?filename=b.mp4"}') == "/view?filename=b.mp4"


async def test_stage_schema_defaults_and_reuses_a_cached_report(tmp_path, reset_db, monkeypatch):
    from ComfyTV.nodes.stages import consistency as stage_mod
    from ComfyTV.nodes.stages.consistency import ConsistencyRescoreStage
    from ComfyTV.runners import consistency_rescore as rescore

    schema = ConsistencyRescoreStage.define_schema()
    assert schema.kw["node_id"] == "ComfyTV.ConsistencyRescoreStage"
    widgets = {item.args[0]: item.kw.get("default") for item in schema.kw["inputs"] if item.args}
    assert widgets["max_reshoots"] == 2
    assert widgets["threshold"] == pytest.approx(0.70)
    assert "video" in {output.args[0] for output in schema.kw["outputs"]}

    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return {
            "stop": "pick_take",
            "takes": [
                {"take": 1, "score": 0.2, "video_url": "low", "seed": 1, "passed": False, "frames": [], "characters": []},
                {"take": 2, "score": 0.9, "video_url": "high", "seed": 2, "passed": True, "frames": [], "characters": []},
            ],
            "takes_in_order": [],
            "contact_sheet": "/view?filename=sheet.png",
            "selected_index": kwargs.get("selected_index") or 1,
            "selected_video": "high" if (kwargs.get("selected_index") or 1) == 1 else "low",
            "threshold": 0.7,
            "max_reshoots": 2,
            "backend": "fake",
            "reshoots": 1,
            "shot_id": "E01S01_01",
            "shot_index": 1,
            "cache_key": "same",
        }

    monkeypatch.setattr(stage_mod, "run_consistency_rescore", fake_run)
    cache = tmp_path / "report.json"
    first = await ConsistencyRescoreStage.execute(
        project_id="default",
        video="/view?filename=shot.mp4",
        storyboard=json.dumps({"shots": [_shot()]}),
        shot_index=1,
        series_root=str(tmp_path),
        threshold=0.7,
        max_reshoots=2,
        frame_count=4,
        seed=5,
        workflow="GXS",
        selected_index=1,
        force_run_token=3,
        cache_path=str(cache),
    )
    assert calls and calls[0]["max_reshoots"] == 2
    assert first.values[1] == "high"
    assert first.values[2] == "/view?filename=sheet.png"

    cache.write_text(json.dumps({
        "cache_key": rescore.cache_key(
            video_url="/view?filename=shot.mp4",
            shot_index=1,
            threshold=0.7,
            max_reshoots=2,
            frame_count=4,
            seed=5,
            backend="auto",
            workflow="GXS",
            series_root=str(tmp_path),
            force_run_token=3,
        ),
        "stop": "pick_take",
        "takes": fake_run(selected_index=1)["takes"],
        "contact_sheet": "/view?filename=sheet.png",
        "threshold": 0.7,
        "max_reshoots": 2,
        "backend": "fake",
        "reshoots": 1,
        "shot_id": "E01S01_01",
        "shot_index": 1,
    }), encoding="utf-8")
    before = len(calls)
    second = await ConsistencyRescoreStage.execute(
        project_id="default",
        video="/view?filename=shot.mp4",
        storyboard=json.dumps({"shots": [_shot()]}),
        shot_index=1,
        series_root=str(tmp_path),
        threshold=0.7,
        max_reshoots=2,
        frame_count=4,
        seed=5,
        workflow="GXS",
        selected_index=2,
        force_run_token=3,
        cache_path=str(cache),
    )
    assert len(calls) == before
    assert second.values[1] == "low"
