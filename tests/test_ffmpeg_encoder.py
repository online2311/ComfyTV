"""NVENC is preferred, then libx264. CI has no GPU, so the probe is mocked."""
import logging
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from ComfyTV.runners.ffmpeg_concat import TargetFormat, _can_stream_copy
from ComfyTV.runners.ffmpeg_encode import (
    clear_encoder_cache,
    resolve_video_encoder_mode,
    select_video_encoder,
)


class _Result:
    def __init__(self, code, out="", err=""):
        self.returncode = code
        self.stdout = out
        self.stderr = err


_LISTED = " V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)\n"
_DRIVER = (
    "Driver does not support the required nvenc API version. "
    "Required: 13.1 Found: 13.0"
)


@pytest.fixture(autouse=True)
def _reset_encoder_cache():
    clear_encoder_cache()
    yield
    clear_encoder_cache()


def _patch_run(monkeypatch, run):
    import ComfyTV.runners.ffmpeg_encode as ffmpeg_encode
    monkeypatch.setattr(ffmpeg_encode.subprocess, "run", run)


def _testsrc_size(cmd):
    for arg in cmd:
        if not str(arg).startswith("testsrc="):
            continue
        size = str(arg).split("size=", 1)[1].split(":", 1)[0]
        width, height = size.split("x", 1)
        return int(width), int(height)
    raise AssertionError(f"nvenc probe did not encode a testsrc frame: {cmd}")


def test_auto_uses_nvenc_after_a_real_encode(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(list(cmd))
        if "-encoders" in cmd:
            return _Result(0, _LISTED, "")
        return _Result(0, "", "")

    _patch_run(monkeypatch, run)
    assert select_video_encoder("/opt/ffmpeg-nv", "auto") == "h264_nvenc"
    # The probe result is cached for this ffmpeg binary. A second concat
    # must not start another encoder trial.
    assert select_video_encoder("/opt/ffmpeg-nv", "auto") == "h264_nvenc"
    assert len(calls) == 2
    assert "-encoders" in calls[0]
    assert "h264_nvenc" in calls[1]
    assert "lavfi" in calls[1]
    assert calls[1].count("-i") == 1
    width, height = _testsrc_size(calls[1])
    # 64x64 is below NVENC's minimum on current drivers. The trial frame
    # has to be even and at least 256 on each side.
    assert width % 2 == 0 and height % 2 == 0
    assert width >= 256 and height >= 256


_DIMENSION_ERROR = (
    "InitializeEncoder failed: invalid param (8): "
    "Frame Dimension less than the minimum supported value"
)


def test_nvenc_probe_failure_logs_frame_size_and_ffmpeg_reason(monkeypatch, caplog):
    calls = []

    def run(cmd, **kwargs):
        calls.append(list(cmd))
        if "-encoders" in cmd:
            return _Result(0, _LISTED, "")
        return _Result(1, "", _DIMENSION_ERROR)

    _patch_run(monkeypatch, run)
    with caplog.at_level(logging.WARNING):
        chosen = select_video_encoder("/opt/ffmpeg-dim", "auto")
    assert chosen == "libx264"
    assert "Frame Dimension less than the minimum supported value" in caplog.text
    assert "256x256" in caplog.text
    assert select_video_encoder("/opt/ffmpeg-dim", "auto") == "libx264"
    assert len(calls) == 2


def test_auto_falls_back_and_logs_the_driver_error(monkeypatch, caplog):
    def run(cmd, **kwargs):
        if "-encoders" in cmd:
            return _Result(0, _LISTED, "")
        return _Result(1, "", _DRIVER)

    _patch_run(monkeypatch, run)
    with caplog.at_level(logging.WARNING):
        chosen = select_video_encoder("/opt/ffmpeg-old", "auto")
    assert chosen == "libx264"
    assert "13.1" in caplog.text
    assert "13.0" in caplog.text
    assert select_video_encoder("/opt/ffmpeg-old", "auto") == "libx264"


def test_forced_nvenc_raises_the_ffmpeg_error(monkeypatch):
    def run(cmd, **kwargs):
        if "-encoders" in cmd:
            return _Result(0, _LISTED, "")
        return _Result(1, "", _DRIVER)

    _patch_run(monkeypatch, run)
    with pytest.raises(RuntimeError, match=r"Required: 13\.1 Found: 13\.0") as caught:
        select_video_encoder("/opt/ffmpeg-force", "nvenc")
    assert "libx264" not in str(caught.value)
    with pytest.raises(RuntimeError, match="13.0"):
        select_video_encoder("/opt/ffmpeg-force", "nvenc")


def test_missing_nvenc_encoder_does_not_try_a_frame(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(list(cmd))
        return _Result(0, " V....D libx264              libx264 H.264\n", "")

    _patch_run(monkeypatch, run)
    assert select_video_encoder("/opt/ffmpeg-cpu-only", "auto") == "libx264"
    assert len(calls) == 1
    assert "-encoders" in calls[0]


def test_cpu_mode_does_not_probe(monkeypatch):
    def run(cmd, **kwargs):
        raise AssertionError(f"cpu mode probed ffmpeg: {cmd}")

    _patch_run(monkeypatch, run)
    assert select_video_encoder("/opt/ffmpeg-any", "cpu") == "libx264"


def test_probe_cache_is_per_ffmpeg_path(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd[0])
        if cmd[0].endswith("good"):
            if "-encoders" in cmd:
                return _Result(0, _LISTED, "")
            return _Result(0, "", "")
        if "-encoders" in cmd:
            return _Result(0, _LISTED, "")
        return _Result(1, "", _DRIVER)

    _patch_run(monkeypatch, run)
    assert select_video_encoder("/opt/good", "auto") == "h264_nvenc"
    assert select_video_encoder("/opt/bad", "auto") == "libx264"
    assert select_video_encoder("/opt/good", "auto") == "h264_nvenc"
    assert calls.count("/opt/good") == 2
    assert calls.count("/opt/bad") == 2


def test_encoder_mode_env_and_node(monkeypatch):
    monkeypatch.delenv("COMFYTV_VIDEO_ENCODER", raising=False)
    assert resolve_video_encoder_mode(None) == "auto"
    assert resolve_video_encoder_mode("auto") == "auto"
    monkeypatch.setenv("COMFYTV_VIDEO_ENCODER", "cpu")
    assert resolve_video_encoder_mode("auto") == "cpu"
    assert resolve_video_encoder_mode("nvenc") == "nvenc"
    monkeypatch.setenv("COMFYTV_VIDEO_ENCODER", "qsv")
    with pytest.raises(RuntimeError, match="COMFYTV_VIDEO_ENCODER"):
        resolve_video_encoder_mode("auto")
    with pytest.raises(RuntimeError, match="video_encoder"):
        resolve_video_encoder_mode("vaapi")


def _stream(**overrides):
    stream = {
        "codec": "h264",
        "pix_fmt": "yuv420p",
        "width": 64,
        "height": 48,
        "avg_frame_rate": "12/1",
        "r_frame_rate": "12/1",
    }
    stream.update(overrides)
    return stream


def _clip(audio=False, **video_overrides):
    item = {
        "path": Path("clip.mp4"),
        "duration": 0.5,
        "video": _stream(**video_overrides),
        "audio": None,
    }
    if audio:
        item["audio"] = {
            "codec": "aac",
            "sample_rate": 44100,
            "channels": 2,
            "duration": 0.5,
        }
    return item


def test_matching_h264_clips_can_stream_copy():
    target = TargetFormat(64, 48, Fraction(12, 1))
    probed = [_clip(), _clip()]
    assert _can_stream_copy(probed, target, False) is True


def test_mixed_size_or_audio_cannot_stream_copy():
    target = TargetFormat(64, 48, Fraction(12, 1))
    assert _can_stream_copy(
        [_clip(), _clip(width=80)], target, False,
    ) is False
    assert _can_stream_copy(
        [_clip(audio=True), _clip()], target, True,
    ) is False
    long_audio = _clip(audio=True)
    long_audio["audio"]["duration"] = 1.5
    assert _can_stream_copy(
        [long_audio, _clip(audio=True)], target, True,
    ) is False


def test_normalize_uses_nvenc_quality_args(monkeypatch, tmp_path):
    from ComfyTV.runners.ffmpeg_concat import _normalize

    dest = tmp_path / "part.mov"
    cmds = []

    def capture(cmd, what):
        cmds.append(list(cmd))
        dest.write_bytes(b"")

    monkeypatch.setattr("ComfyTV.runners.ffmpeg_concat._run", capture)
    _normalize(
        "/opt/ffmpeg-nv",
        _clip(),
        TargetFormat(64, 48, Fraction(12, 1)),
        False,
        dest,
        label="clip 1/2",
        encoder="h264_nvenc",
    )
    cmd = cmds[0]
    assert "h264_nvenc" in cmd
    assert "p5" in cmd
    assert "19" in cmd
    assert "yuv420p" in cmd
    assert "libx264" not in cmd


def test_matching_files_are_copied_and_mixed_files_are_encoded(monkeypatch):
    pytest.importorskip("av")
    pytest.importorskip("numpy")
    import folder_paths
    from test_media_concat import _write_clip
    from ComfyTV.runners import media

    monkeypatch.setenv("COMFYTV_VIDEO_ENCODER", "cpu")
    monkeypatch.delenv("COMFYTV_FFMPEG", raising=False)
    src = Path(folder_paths.get_output_directory()) / "encoder-src"
    src.mkdir(parents=True, exist_ok=True)
    a = src / "a.mp4"
    b = src / "b.mp4"
    c = src / "c.mp4"
    _write_clip(a, w=64, h=48, fps=12, seconds=0.25, with_audio=False)
    _write_clip(b, w=64, h=48, fps=12, seconds=0.25, with_audio=False)
    _write_clip(c, w=80, h=48, fps=12, seconds=0.25, with_audio=False)

    commands = []
    real_run = subprocess.run

    def spy(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)):
            commands.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    report = {}
    media.concat_videos(
        [media.path_to_view_url(a), media.path_to_view_url(b)],
        report=report,
    )
    joined = [c for c in commands if "concat" in c and "-c" in c]
    assert joined
    assert "copy" in joined[0]
    assert not any("libx264" in c or "h264_nvenc" in c for c in commands)
    assert report["video_encoder"] == "copy"
    assert report["audio_encoder"] == "copy"

    commands.clear()
    report.clear()
    media.concat_videos(
        [media.path_to_view_url(a), media.path_to_view_url(c)],
        report=report,
    )
    assert any("libx264" in c for c in commands)
    assert report["video_encoder"] == "libx264"
    assert report["audio_encoder"] == "copy"


def test_sequence_report_keeps_audio_encoder(monkeypatch):
    from ComfyTV.runners import media_remap

    def fake_concat(urls, progress=None, *, video_encoder=None, report=None):
        if report is not None:
            report["video_encoder"] = "copy"
            report["audio_encoder"] = "reencode"
        return urls[0]

    monkeypatch.setattr(media_remap, "concat_videos", fake_concat)
    monkeypatch.setattr(
        media_remap, "get_video_info",
        lambda url: {"duration": 1.0, "fps": 24, "width": 64, "height": 48, "has_audio": True},
    )
    report = {}
    media_remap.render_sequence(
        [
            {"url": "/view?filename/a.mp4", "transition": "cut"},
            {"url": "/view?filename/b.mp4", "transition": "cut"},
        ],
        report=report,
    )
    assert report["video_encoder"] == "copy"
    assert report["audio_encoder"] == "reencode"


def test_concat_node_forwards_encoder_and_records_it(reset_db, monkeypatch):
    from ComfyTV.nodes.stages import video_edit

    captured = {}

    def fake_concat(urls, progress=None, *, video_encoder=None, report=None):
        captured["video_encoder"] = video_encoder
        if report is not None:
            report["video_encoder"] = "h264_nvenc"
            report["audio_encoder"] = "reencode"
        return "/view?filename=out.mp4"

    monkeypatch.setattr(video_edit, "concat_videos", fake_concat)
    out = video_edit.VideoConcatStage.execute(
        project_id="default",
        videos={
            "videos.video0": "/view?filename=a.mp4",
            "videos.video1": "/view?filename=b.mp4",
        },
        video_encoder="nvenc",
    )
    assert captured["video_encoder"] == "nvenc"
    assert out.ui["video_encoder"] == ["h264_nvenc"]
    assert out.ui["audio_encoder"] == ["reencode"]

    def fake_default(urls, progress=None, *, video_encoder=None, report=None):
        captured["default"] = video_encoder
        return "/view?filename=out.mp4"

    monkeypatch.setattr(video_edit, "concat_videos", fake_default)
    video_edit.VideoConcatStage.execute(
        project_id="default",
        videos={
            "videos.video0": "/view?filename=a.mp4",
            "videos.video1": "/view?filename=b.mp4",
        },
    )
    assert captured["default"] == "auto"
