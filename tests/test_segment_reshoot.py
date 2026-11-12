"""Audio-only segment reshoot: frame grid, versions, and a three-part splice."""

from __future__ import annotations

import hashlib
import uuid
import wave
from fractions import Fraction
from pathlib import Path

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")


@pytest.fixture(autouse=True)
def cpu_encoder(monkeypatch):
    monkeypatch.setenv("COMFYTV_VIDEO_ENCODER", "cpu")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_clip(path: Path, *, seconds: float, fps: int = 24, hz: float = 440.0,
                w: int = 64, h: int = 48):
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(round(seconds * fps))
    with av.open(str(path), "w") as out:
        video = out.add_stream("libx264", rate=fps)
        video.width = w
        video.height = h
        video.pix_fmt = "yuv420p"
        video.options = {"crf": "18", "bf": "0"}
        audio = out.add_stream("aac", rate=44100)
        audio.layout = "stereo"
        for index in range(n_frames):
            level = 32 + (index % 16) * 8
            frame = av.VideoFrame.from_ndarray(
                np.full((h, w, 3), level, dtype=np.uint8), format="rgb24",
            ).reformat(format="yuv420p")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
        total = int(round(seconds * 44100))
        samples = (0.2 * np.sin(2 * np.pi * hz * np.arange(total) / 44100)).astype(np.float32)
        stereo = np.stack([samples, samples])
        written = 0
        while written < total:
            chunk = stereo[:, written:written + 1024]
            aframe = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(chunk), format="fltp", layout="stereo",
            )
            aframe.sample_rate = 44100
            aframe.pts = written
            aframe.time_base = Fraction(1, 44100)
            written += chunk.shape[1]
            for packet in audio.encode(aframe):
                out.mux(packet)
        for packet in audio.encode():
            out.mux(packet)


def _write_tone(path: Path, seconds: float, hz: float, rate: int = 44100):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(round(seconds * rate))
    samples = (0.2 * np.sin(2 * np.pi * hz * np.arange(count) / rate) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def _output_file(name: str) -> Path:
    import folder_paths
    path = Path(folder_paths.get_output_directory()) / "comfytv" / "reshoot-src" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _view(path: Path) -> str:
    from ComfyTV.runners import media
    return media.path_to_view_url(path)


def _frames(path: Path):
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def _psnr(left, right) -> float:
    diff = left.astype(np.float64) - right.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    if mse == 0:
        return 99.0
    return 10 * np.log10((255.0 ** 2) / mse)


def _tone_hz(url: str, start_s: float, end_s: float) -> float:
    from ComfyTV.runners import media
    chunks = []
    with av.open(str(media.localize(url))) as container:
        stream = container.streams.audio[0]
        rate = stream.sample_rate or 44100
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray()[0])
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray()[0])
    samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    window = samples[int(start_s * rate):int(end_s * rate)]
    if window.size < 8:
        return 0.0
    signs = np.sign(window)
    signs[signs == 0] = 1
    crossings = np.where(np.diff(signs))[0]
    if crossings.size < 2:
        return 0.0
    return float(rate / (2 * np.mean(np.diff(crossings))))


def _decoded_audio_seconds(url: str) -> float:
    from ComfyTV.runners import media
    with av.open(str(media.localize(url))) as container:
        stream = container.streams.audio[0]
        rate = stream.sample_rate or 44100
        samples = sum(int(frame.samples or 0) for frame in container.decode(stream))
    return samples / rate


class TestWindowGrid:
    def test_snaps_to_whole_frames_and_records_handles(self):
        from ComfyTV.nodes.stages.common.shot_document import frames_for
        from ComfyTV.storage.segment_reshoot import align_window, grid_frames

        window = align_window(3.01, 6.99, Fraction(24, 1), 8 * 24, handles_s=0.5)
        assert window["frame_start"] == 72
        assert window["frame_end"] == 168
        assert window["t_start"] * 24 == 72
        assert window["t_end"] * 24 == 168
        assert window["handle_frames"] == 12
        assert window["gen_need_frames"] == 96 + 24
        assert window["gen_frames"] == 124
        assert grid_frames(window["gen_need_frames"]) == window["gen_frames"]
        for seconds in (5, 6, 8, 10):
            assert grid_frames(int(round(seconds * 24))) == frames_for(seconds)

    def test_rejects_an_empty_or_outside_window(self):
        from ComfyTV.storage.segment_reshoot import ReshootError, align_window

        with pytest.raises(ReshootError, match="length"):
            align_window(1, 1, Fraction(24, 1), 48)
        with pytest.raises(ReshootError, match="outside"):
            align_window(0, 9, Fraction(24, 1), 48)

    def test_prompt_id_is_stable_and_separate_from_the_parent_take(self):
        from ComfyTV.storage.segment_reshoot import prompt_id_for

        key = "E01S02_03/take_2/r1"
        assert prompt_id_for("gumu", key) == prompt_id_for("gumu", key)
        assert prompt_id_for("gumu", key) != prompt_id_for("gumu", "E01S02_03/take_2")
        assert prompt_id_for("gumu", key) != prompt_id_for("idol-mv-hit", key)


class TestPlanAndVersions:
    def test_plan_does_not_submit_and_reports_the_aligned_window(self, reset_db, monkeypatch):
        from ComfyTV.api import headless_queue
        from ComfyTV.storage.segment_reshoot import plan

        calls = []
        monkeypatch.setattr(
            headless_queue, "enqueue_graph",
            lambda *args, **kwargs: calls.append("graph"),
        )
        monkeypatch.setattr(
            headless_queue, "enqueue_stage",
            lambda *args, **kwargs: calls.append("stage"),
        )
        source = _output_file("align-8s.mp4")
        if not source.exists():
            _write_clip(source, seconds=8)
        planned = plan({
            "project_id": "grid-plan",
            "mode": "audio_only",
            "previews": False,
            "source": {"url": _view(source), "shot_id": "E01S02_03", "take": 2},
            "t_start": 3.01,
            "t_end": 6.99,
            "audio": str(_tone(reset_db)),
        })
        assert calls == []
        assert planned["status"] == "planned"
        assert planned["window"]["frame_start"] == 72
        assert planned["window"]["frame_end"] == 168
        assert planned["window"]["t_start"] * planned["source"]["fps"] == 72
        assert planned["window"]["t_end"] * planned["source"]["fps"] == 168
        assert planned["window"]["gen_frames"] == 124
        assert planned["jobs"][0]["kind"] == "splice"
        assert planned["output"] is None
        assert planned["source"]["sha256"] == _sha(source)

    def test_lipsync_and_regen_are_refused(self, reset_db):
        from ComfyTV.storage.segment_reshoot import ReshootError, plan

        source = _output_file("short.mp4")
        if not source.exists():
            _write_clip(source, seconds=2)
        audio = str(_tone(reset_db))
        with pytest.raises(ReshootError, match="lipsync has no backend"):
            plan(_call(source, audio, mode="lipsync", project="no-lips"))
        with pytest.raises(ReshootError, match="regen is not in this build"):
            plan(_call(source, audio, mode="regen", project="no-regen"))

    def test_fit_rejects_audio_that_is_longer_than_the_window(self, reset_db):
        from ComfyTV.storage.segment_reshoot import ReshootError, plan

        source = _output_file("fit-src.mp4")
        if not source.exists():
            _write_clip(source, seconds=4)
        long_audio = _output_file("long.wav")
        _write_tone(long_audio, 3.0, 880)
        with pytest.raises(ReshootError, match="length_policy 'fit'"):
            plan({
                **_call(source, str(long_audio), project="fit-long"),
                "t_start": 1.0,
                "t_end": 2.0,
            })

    def test_multiple_windows_are_off_until_asked(self, reset_db):
        from ComfyTV.storage.segment_reshoot import ReshootError, plan

        source = _output_file("multi-src.mp4")
        if not source.exists():
            _write_clip(source, seconds=4)
        short = _output_file("multi-tone.wav")
        _write_tone(short, 0.4, 880)
        audio = str(short)
        windows = [
            {"t_start": 0.5, "t_end": 1.0, "audio": audio},
            {"t_start": 2.0, "t_end": 2.5, "audio": audio},
        ]
        with pytest.raises(ReshootError, match="allow_multiple"):
            plan({**_call(source, audio, project="multi-off"), "windows": windows})
        planned = plan({
            **_call(source, audio, project="multi-on"),
            "windows": windows,
            "allow_multiple": True,
        })
        assert len(planned["windows"]) == 2
        assert planned["windows"][0]["frame_end"] <= planned["windows"][1]["frame_start"]

    def test_upscaled_profile_is_refused(self, reset_db):
        from ComfyTV.storage.segment_reshoot import ReshootError, plan

        source = _output_file("upscale-src.mp4")
        if not source.exists():
            _write_clip(source, seconds=2)
        with pytest.raises(ReshootError, match="pre-upscale"):
            plan({
                **_call(source, str(_tone(reset_db)), project="upscale"),
                "source": {
                    "url": _view(source),
                    "shot_id": "E01S02_03",
                    "take": 1,
                    "profile": "relay_vsr",
                },
            })


class TestAudioSplice:
    def test_replaces_the_window_and_keeps_versions(self, reset_db):
        from ComfyTV.storage.segment_reshoot import (
            ReshootError, adopt, plan, rollback, run,
        )

        source = _output_file("bars-6s.mp4")
        _write_clip(source, seconds=6, hz=440)
        source_sha = _sha(source)
        replacement = _output_file("tone-880.wav")
        _write_tone(replacement, 2.0, 880)
        project = f"audio-only-{uuid.uuid4().hex[:8]}"
        base = {
            "project_id": project,
            "mode": "audio_only",
            "previews": False,
            "source": {
                "url": _view(source),
                "shot_id": "E01S02_03",
                "take": 2,
                "prompt_id": "prompt-take-2",
            },
            "t_start": 2.0,
            "t_end": 4.0,
            "audio": str(replacement),
        }
        first = run(base)
        assert first["status"] == "done"
        assert first["source"]["sha256"] == source_sha
        assert _sha(source) == source_sha
        assert first["prompt_id"] != "prompt-take-2"
        url = first["output"]["url"]
        out_frames = _frames(Path(from_url(url)))
        src_frames = _frames(source)
        assert abs(len(out_frames) - len(src_frames)) <= 1
        picture_s = len(out_frames) / 24
        assert abs(_decoded_audio_seconds(url) - picture_s) <= 0.05
        assert abs(picture_s - 6.0) <= 1 / 24
        for index in list(range(0, 48)) + list(range(96, min(len(out_frames), len(src_frames)))):
            assert _psnr(out_frames[index], src_frames[index]) >= 40
        assert _tone_hz(url, 2.4, 3.6) == pytest.approx(880, abs=20)
        r1_sha = first["output"]["sha256"]
        again = run({"project_id": project, "reshoot_id": first["reshoot_id"]})
        assert again["output"]["sha256"] == r1_sha

        second = run(base)
        assert second["version_id"] == "take_2/r2"
        assert _sha(source) == source_sha
        assert _sha(from_url_path(first["output"]["url"])) == r1_sha

        with pytest.raises(ReshootError, match="approved_by"):
            adopt({"project_id": project, "reshoot_id": first["reshoot_id"]})
        with pytest.raises(ReshootError, match="张景"):
            adopt({
                "project_id": project,
                "reshoot_id": first["reshoot_id"],
                "approved_by": "助手",
            })
        adopted = adopt({
            "project_id": project,
            "reshoot_id": first["reshoot_id"],
            "approved_by": "张景",
            "cut_to": 5,
            "next_shot_id": "E01S02_04",
        })
        assert adopted["active"] == "take_2/r1"
        assert any(row["kind"] == "rough_cut" and row["rewrite"] is False for row in adopted["stale"])
        rolled = rollback({
            "project_id": project,
            "shot_id": "E01S02_03",
            "to": "take_2",
            "approved_by": "张景",
        })
        assert rolled["active"] == "take_2"
        assert any(row["kind"] == "subtitles" and row["rewrite"] is False for row in rolled["stale"])
        assert _sha(source) == source_sha

    def test_ripple_shifts_the_tail(self, reset_db):
        from ComfyTV.storage.segment_reshoot import run

        source = _output_file("ripple-4s.mp4")
        _write_clip(source, seconds=4, hz=440)
        before = len(_frames(source))
        audio = _output_file("ripple-3s.wav")
        _write_tone(audio, 3.0, 660)
        result = run({
            "project_id": f"ripple-{uuid.uuid4().hex[:8]}",
            "mode": "audio_only",
            "previews": False,
            "length_policy": "ripple",
            "source": {"url": _view(source), "shot_id": "E01S01_01", "take": 1},
            "t_start": 1.0,
            "t_end": 2.0,
            "audio": str(audio),
        })
        after = len(_frames(from_url_path(result["output"]["url"])))
        assert after > before
        assert abs(after - (before + 48)) <= 1

    def test_imported_takes_and_rough_cuts_stay_put(self, reset_db):
        from ComfyTV.storage.segment_reshoot import (
            ReshootError,
            _assert_destination_allowed,
            list_reshoots,
            prompt_id_for,
            register_imported_version,
            register_protected,
            run,
        )

        project = f"idol-mv-hit-{uuid.uuid4().hex[:8]}"
        shot = "E01S01_01"
        source = _output_file("idol-take-1.mp4")
        _write_clip(source, seconds=3, hz=440)
        historical = "colab-prompt-take-1"
        registered = register_imported_version(
            project, shot, take=1, sha256=_sha(source), prompt_id=historical,
            legacy_path="renders/E01S01_01/take_1.mp4", url=_view(source),
        )
        assert registered["active"] is None
        assert registered["shot_id"] == shot
        assert registered["versions"][0]["prompt_id"] == historical
        assert registered["versions"][0]["take"] == 1
        with pytest.raises(ReshootError, match="refusing to overwrite"):
            register_imported_version(
                project, shot, take=1, sha256="0" * 64, prompt_id=historical,
                legacy_path="renders/E01S01_01/take_1.mp4",
            )
        for legacy in (
            "cut/rough_cut.mp4",
            "cut/rough_cut_r5_backup.mp4",
            "cut/rough_cut_r6.mp4",
        ):
            register_protected(project, kind="rough_cut", legacy_path=legacy, sha256=_sha(source))
        with pytest.raises(ReshootError, match="rough_cut"):
            _assert_destination_allowed(
                project,
                _output_file("cut/rough_cut.mp4"),
            )
        before = _sha(source)
        audio = _output_file("idol-audio.wav")
        _write_tone(audio, 1.0, 880)
        result = run({
            "project_id": project,
            "mode": "audio_only",
            "previews": False,
            "source": {
                "url": _view(source),
                "shot_id": shot,
                "take": 1,
                "legacy_path": "renders/E01S01_01/take_1.mp4",
            },
            "t_start": 1.0,
            "t_end": 2.0,
            "audio": str(audio),
        })
        assert _sha(source) == before
        listed = list_reshoots({"project_id": project, "shot_id": shot})
        imported = next(row for row in listed["takes"]["versions"] if row["id"] == "take_1")
        assert imported["prompt_id"] == historical
        assert imported["sha256"] == before
        assert listed["takes"]["active"] is None
        assert result["prompt_id"] == prompt_id_for(project, f"{shot}/take_1/r1")
        assert "rough_cut" not in result["output"]["url"]
        assert result["storage"]["records_dir"] == f"comfytv/projects/{project}/reshoots/{shot}"


class TestHttp:
    @pytest.fixture()
    async def client(self, reset_db):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        from ComfyTV import api  # noqa: F401
        import server

        app = web.Application()
        app.router.add_routes(server.PromptServer.instance.routes)
        test_server = TestServer(app)
        test_client = TestClient(test_server)
        await test_client.start_server()
        yield test_client
        await test_client.close()

    async def test_plan_and_adopt_over_http(self, client):
        source = _output_file("http-src.mp4")
        if not source.exists():
            _write_clip(source, seconds=2)
        audio = _output_file("http-tone.wav")
        _write_tone(audio, 0.4, 550)
        response = await client.post("/comfytv/segment_reshoot/plan", json={
            "project_id": "http-plan",
            "mode": "audio_only",
            "previews": False,
            "source": {"url": _view(source), "shot_id": "E01S03_02", "take": 1},
            "t_start": 0.4,
            "t_end": 0.9,
            "audio": str(audio),
        })
        assert response.status == 200
        body = await response.json()
        assert body["shot_id"] == "E01S03_02"
        assert body["status"] == "planned"
        denied = await client.post("/comfytv/segment_reshoot/adopt", json={
            "project_id": "http-plan",
            "reshoot_id": body["reshoot_id"],
            "approved_by": "张景",
        })
        assert denied.status == 400
        listed = await client.get("/comfytv/segment_reshoot/list", params={
            "project_id": "http-plan",
            "shot_id": "E01S03_02",
        })
        assert listed.status == 200
        payload = await listed.json()
        assert payload["takes"]["versions"][0]["id"] == "take_1"
        assert payload["takes"]["versions"][0]["immutable"] is True


class TestReviewGuards:
    def test_project_id_is_required(self):
        from ComfyTV.storage.segment_reshoot import ReshootError, plan

        with pytest.raises(ReshootError, match="project_id"):
            plan({"mode": "audio_only", "shot_id": "E01S01_01"})

    def test_a_second_plan_reserves_the_next_revision(self, reset_db):
        from ComfyTV.storage.segment_reshoot import plan

        source = _output_file("reserve-src.mp4")
        if not source.exists():
            _write_clip(source, seconds=2)
        audio = _output_file("reserve-tone.wav")
        _write_tone(audio, 0.4, 660)
        project = f"reserve-{uuid.uuid4().hex[:8]}"
        first = plan({**_call(source, str(audio), project=project), "t_start": 0.2, "t_end": 0.7})
        second = plan({**_call(source, str(audio), project=project), "t_start": 1.0, "t_end": 1.4})
        assert first["version_id"] == "take_2/r1"
        assert second["version_id"] == "take_2/r2"
        assert first["windows"][0]["t_start"] != second["windows"][0]["t_start"]

    def test_windows_are_stored_in_time_order(self, reset_db):
        from ComfyTV.storage.segment_reshoot import plan

        source = _output_file("order-src.mp4")
        if not source.exists():
            _write_clip(source, seconds=4)
        audio = _output_file("order-tone.wav")
        _write_tone(audio, 0.4, 550)
        later = {"t_start": 2.0, "t_end": 2.4, "audio": str(audio)}
        earlier = {"t_start": 0.4, "t_end": 0.8, "audio": str(audio)}
        planned = plan({
            **_call(source, str(audio), project=f"order-{uuid.uuid4().hex[:8]}"),
            "windows": [later, earlier],
            "allow_multiple": True,
        })
        assert planned["windows"][0]["frame_start"] < planned["windows"][1]["frame_start"]
        assert planned["window"]["frame_start"] == planned["windows"][0]["frame_start"]

    def test_run_does_not_delete_an_existing_output(self, reset_db):
        from ComfyTV.storage.segment_reshoot import ReshootError, _media_path, plan, run

        source = _output_file("keep-src.mp4")
        if not source.exists():
            _write_clip(source, seconds=2)
        audio = _output_file("keep-tone.wav")
        _write_tone(audio, 0.4, 440)
        project = f"keep-file-{uuid.uuid4().hex[:8]}"
        planned = plan(_call(source, str(audio), project=project))
        dest = _media_path(project, "E01S02_03", planned["version_id"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"finished-version")
        with pytest.raises(ReshootError, match="overwrite"):
            run({"project_id": project, "reshoot_id": planned["reshoot_id"]})
        assert dest.read_bytes() == b"finished-version"

    def test_run_rejects_audio_that_no_longer_matches_the_plan(self, reset_db):
        from ComfyTV.storage.segment_reshoot import ReshootError, plan, run

        source = _output_file("hash-src.mp4")
        if not source.exists():
            _write_clip(source, seconds=2)
        audio = _output_file("hash-tone.wav")
        _write_tone(audio, 0.4, 440)
        project = f"hash-audio-{uuid.uuid4().hex[:8]}"
        planned = plan(_call(source, str(audio), project=project))
        _write_tone(audio, 0.4, 880)
        with pytest.raises(ReshootError, match="sha256"):
            run({"project_id": project, "reshoot_id": planned["reshoot_id"]})

    def test_bad_cut_to_does_not_move_the_pointer(self, reset_db):
        from ComfyTV.storage.segment_reshoot import ReshootError, adopt, list_reshoots, run

        source = _output_file("cut-src.mp4")
        if not source.exists():
            _write_clip(source, seconds=2)
        audio = _output_file("cut-tone.wav")
        _write_tone(audio, 0.4, 500)
        project = f"bad-cut-{uuid.uuid4().hex[:8]}"
        done = run(_call(source, str(audio), project=project))
        before = list_reshoots({"project_id": project, "shot_id": "E01S02_03"})
        with pytest.raises(ReshootError, match="cut_to"):
            adopt({
                "project_id": project,
                "reshoot_id": done["reshoot_id"],
                "approved_by": "张景",
                "cut_to": "later",
            })
        after = list_reshoots({"project_id": project, "shot_id": "E01S02_03"})
        assert after["takes"]["active"] == before["takes"]["active"]

    def test_a_path_outside_the_output_root_is_stored_absolutely(self, reset_db, tmp_path):
        from ComfyTV.storage.segment_reshoot import plan

        source = _output_file("abs-src.mp4")
        if not source.exists():
            _write_clip(source, seconds=2)
        audio = tmp_path / "outside.wav"
        _write_tone(audio, 0.4, 470)
        planned = plan(_call(source, str(audio), project=f"abs-path-{uuid.uuid4().hex[:8]}"))
        assert planned["inputs"]["audio"]["url"] == str(audio.resolve())
        assert not planned["inputs"]["audio"]["url"].startswith("/view")

    def test_previews_are_namespaced_by_source(self, reset_db):
        from ComfyTV.storage.segment_reshoot import plan

        first_src = _output_file("prev-a.mp4")
        second_src = _output_file("prev-b.mp4")
        _write_clip(first_src, seconds=2, hz=440)
        _write_clip(second_src, seconds=2, hz=880, w=32, h=32)
        audio = _output_file("prev-tone.wav")
        _write_tone(audio, 0.4, 500)
        project = f"previews-{uuid.uuid4().hex[:8]}"
        one = plan({
            **_call(first_src, str(audio), project=project),
            "previews": True,
            "source": {"url": _view(first_src), "shot_id": "E01S02_03", "take": 1},
        })
        two = plan({
            **_call(second_src, str(audio), project=project),
            "previews": True,
            "source": {"url": _view(second_src), "shot_id": "E01S02_03", "take": 2},
        })
        assert one["inputs"]["first_frame"]["sha256"] != two["inputs"]["first_frame"]["sha256"]
        assert one["inputs"]["first_frame"]["url"] != two["inputs"]["first_frame"]["url"]


def _tone(reset_db) -> Path:
    path = _output_file("default-tone.wav")
    if not path.exists():
        _write_tone(path, 2.0, 880)
    return path


def _call(source: Path, audio: str, *, mode: str = "audio_only", project: str) -> dict:
    return {
        "project_id": project,
        "mode": mode,
        "previews": False,
        "source": {"url": _view(source), "shot_id": "E01S02_03", "take": 2},
        "t_start": 0.5,
        "t_end": 1.0,
        "audio": audio,
    }


def from_url(url: str) -> str:
    from ComfyTV.runners import media
    return str(media.localize(url))


def from_url_path(url: str) -> Path:
    return Path(from_url(url))
