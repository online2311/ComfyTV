"""ffmpeg discovery order and PyAV probe, including MKV/WebM packet duration."""
import os
import shutil
import stat
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")


def _exe(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _clear():
    from ComfyTV.runners.ffmpeg_bin import clear_ffmpeg_cache
    clear_ffmpeg_cache()


def test_env_beats_path(monkeypatch, tmp_path):
    from ComfyTV.runners.ffmpeg_bin import find_ffmpeg

    chosen = _exe(tmp_path / "chosen")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _exe(bindir / "ffmpeg")
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setenv("COMFYTV_FFMPEG", str(chosen))
    _clear()
    assert find_ffmpeg() == str(chosen.resolve())


def test_path_is_used_when_env_is_unset(monkeypatch, tmp_path):
    from ComfyTV.runners.ffmpeg_bin import find_ffmpeg

    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = _exe(bindir / "ffmpeg")
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.delenv("COMFYTV_FFMPEG", raising=False)
    _clear()
    assert find_ffmpeg() == str(exe.resolve())


def test_invalid_env_falls_through_to_path(monkeypatch, tmp_path):
    from ComfyTV.runners.ffmpeg_bin import find_ffmpeg

    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = _exe(bindir / "ffmpeg")
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setenv("COMFYTV_FFMPEG", str(tmp_path / "missing-ffmpeg"))
    _clear()
    assert find_ffmpeg() == str(exe.resolve())


def test_imageio_ffmpeg_when_path_and_env_miss(monkeypatch, tmp_path):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    from ComfyTV.runners.ffmpeg_bin import find_ffmpeg

    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("COMFYTV_FFMPEG", raising=False)
    monkeypatch.delenv("IMAGEIO_FFMPEG_EXE", raising=False)
    _clear()
    found = find_ffmpeg()
    assert os.path.samefile(found, imageio_ffmpeg.get_ffmpeg_exe())


def test_find_ffmpeg_caches_per_env_and_path(monkeypatch, tmp_path):
    from ComfyTV.runners.ffmpeg_bin import find_ffmpeg

    first = _exe(tmp_path / "first")
    second = _exe(tmp_path / "second")
    monkeypatch.setenv("COMFYTV_FFMPEG", str(first))
    _clear()
    assert find_ffmpeg() == str(first.resolve())
    first.unlink()
    assert find_ffmpeg() == str(first.resolve())
    monkeypatch.setenv("COMFYTV_FFMPEG", str(second))
    assert find_ffmpeg() == str(second.resolve())


def test_missing_ffmpeg_lists_every_checked_place(monkeypatch, tmp_path):
    import ComfyTV.runners.ffmpeg_bin as ffmpeg_bin

    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("COMFYTV_FFMPEG", str(tmp_path / "missing-ffmpeg"))
    monkeypatch.setattr(ffmpeg_bin, "_imageio_ffmpeg", None)
    _clear()
    with pytest.raises(RuntimeError) as caught:
        ffmpeg_bin.find_ffmpeg()
    message = str(caught.value)
    assert "COMFYTV_FFMPEG" in message
    assert "missing-ffmpeg" in message
    assert "shutil.which('ffmpeg')" in message
    assert "imageio_ffmpeg" in message
    assert "not installed" in message
    assert "absolute path" in message


def test_imageio_failure_is_reported(monkeypatch, tmp_path):
    import ComfyTV.runners.ffmpeg_bin as ffmpeg_bin

    class _Boom:
        @staticmethod
        def get_ffmpeg_exe():
            raise RuntimeError("wheel has no binary")

    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("COMFYTV_FFMPEG", raising=False)
    monkeypatch.setattr(ffmpeg_bin, "_imageio_ffmpeg", _Boom)
    _clear()
    with pytest.raises(RuntimeError) as caught:
        ffmpeg_bin.find_ffmpeg()
    message = str(caught.value)
    assert "COMFYTV_FFMPEG: not set" in message
    assert "shutil.which('ffmpeg') found nothing" in message
    assert "wheel has no binary" in message
    assert "absolute path" in message


def test_imageio_non_file_is_reported(monkeypatch, tmp_path):
    import ComfyTV.runners.ffmpeg_bin as ffmpeg_bin

    class _Bad:
        @staticmethod
        def get_ffmpeg_exe():
            return str(tmp_path / "not-a-binary")

    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("COMFYTV_FFMPEG", raising=False)
    monkeypatch.setattr(ffmpeg_bin, "_imageio_ffmpeg", _Bad)
    _clear()
    with pytest.raises(RuntimeError) as caught:
        ffmpeg_bin.find_ffmpeg()
    assert "not-a-binary" in str(caught.value)
    assert "absolute path" in str(caught.value)


def _mp4(path: Path) -> None:
    with av.open(str(path), "w") as out:
        video = out.add_stream("libx264", rate=12)
        video.width = 64
        video.height = 48
        video.pix_fmt = "yuv420p"
        audio = out.add_stream("aac", rate=44100)
        audio.layout = "stereo"
        for i in range(6):
            frame = av.VideoFrame(64, 48, "yuv420p")
            frame.pts = i
            frame.time_base = Fraction(1, 12)
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
        samples = np.zeros((2, 1024), dtype="float32")
        aframe = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
        aframe.sample_rate = 44100
        aframe.pts = 0
        aframe.time_base = Fraction(1, 44100)
        for packet in audio.encode(aframe):
            out.mux(packet)
        for packet in audio.encode():
            out.mux(packet)


def test_default_probe_uses_pyav_not_ffprobe(monkeypatch, tmp_path):
    from ComfyTV.runners.ffmpeg_concat import _probe

    path = tmp_path / "clip.mp4"
    _mp4(path)
    seen = []
    real_run = subprocess.run

    def spy(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)) and cmd:
            seen.append(Path(str(cmd[0])).name)
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    monkeypatch.delenv("COMFYTV_FFPROBE", raising=False)
    info = _probe(path, 0)
    assert info["video"]["width"] == 64
    assert info["video"]["height"] == 48
    assert info["audio"] is not None
    assert info["duration"] > 0
    assert not any(name.startswith("ffprobe") for name in seen)


def _long_av_container(dest: Path, ffmpeg: str, container: str) -> None:
    video_codec = "libx264" if container == "mkv" else "libvpx-vp9"
    audio_codec = "aac" if container == "mkv" else "libopus"
    pix = ["-pix_fmt", "yuv420p"] if container == "mkv" else []
    subprocess.check_call([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=red:s=64x48:r=24:d=0.5",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=1.5",
        "-c:v", video_codec, *pix, "-c:a", audio_codec,
        "-map", "0:v", "-map", "1:a", str(dest),
    ])


@pytest.mark.parametrize("container", ["mkv", "webm"])
def test_probe_mkv_and_webm_follow_video_packets(monkeypatch, tmp_path, container):
    """Container duration is the longer audio track. Picture length is the
    video packet timeline, for Matroska and WebM."""
    from ComfyTV.runners.ffmpeg_bin import find_ffmpeg
    from ComfyTV.runners.ffmpeg_concat import _probe

    monkeypatch.delenv("COMFYTV_FFPROBE", raising=False)
    dest = tmp_path / f"long.{container}"
    _long_av_container(dest, find_ffmpeg(), container)
    info = _probe(dest, 0)
    assert (info["video"]["width"], info["video"]["height"]) == (64, 48)
    assert info["audio"] is not None
    assert 0.4 <= info["duration"] <= 0.7


def test_explicit_ffprobe_is_optional(monkeypatch, tmp_path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        pytest.skip("ffprobe is not installed")
    from ComfyTV.runners.ffmpeg_concat import _probe

    path = tmp_path / "clip.mp4"
    _mp4(path)
    opened = {"n": 0}
    real_open = av.open

    def counting_open(*args, **kwargs):
        opened["n"] += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(av, "open", counting_open)
    monkeypatch.setenv("COMFYTV_FFPROBE", ffprobe)
    info = _probe(path, 0)
    assert info["video"]["width"] == 64
    assert info["audio"] is not None
    assert info["duration"] > 0
    assert opened["n"] == 0


def test_bad_ffprobe_override_explains_the_variable(monkeypatch, tmp_path):
    from ComfyTV.runners.ffmpeg_concat import _probe

    monkeypatch.setenv("COMFYTV_FFPROBE", str(tmp_path / "missing-ffprobe"))
    with pytest.raises(RuntimeError, match="COMFYTV_FFPROBE") as caught:
        _probe(tmp_path / "clip.mp4", 0)
    assert "PyAV" in str(caught.value)


def test_missing_av_falls_back_to_path_ffprobe(monkeypatch, tmp_path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        pytest.skip("ffprobe is not installed")
    import ComfyTV.runners.ffmpeg_concat as ffmpeg_concat

    path = tmp_path / "clip.mp4"
    _mp4(path)
    monkeypatch.setattr(ffmpeg_concat, "_av", None)
    monkeypatch.delenv("COMFYTV_FFPROBE", raising=False)
    info = ffmpeg_concat._probe(path, 0)
    assert info["video"]["width"] == 64
    assert info["duration"] > 0


def test_missing_av_and_ffprobe_names_both(monkeypatch, tmp_path):
    import ComfyTV.runners.ffmpeg_concat as ffmpeg_concat

    monkeypatch.setattr(ffmpeg_concat, "_av", None)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("COMFYTV_FFPROBE", raising=False)
    with pytest.raises(RuntimeError) as caught:
        ffmpeg_concat._probe(tmp_path / "clip.mp4", 3)
    message = str(caught.value)
    assert "av" in message
    assert "ffprobe" in message
    assert "COMFYTV_FFPROBE" in message
    assert "shutil.which('ffprobe')" in message


def test_probe_without_video_stream(monkeypatch, tmp_path):
    from ComfyTV.runners.ffmpeg_concat import _probe

    path = tmp_path / "tone.m4a"
    with av.open(str(path), "w") as out:
        audio = out.add_stream("aac", rate=44100)
        audio.layout = "mono"
        samples = np.zeros((1, 1024), dtype="float32")
        frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
        frame.sample_rate = 44100
        frame.pts = 0
        frame.time_base = Fraction(1, 44100)
        for packet in audio.encode(frame):
            out.mux(packet)
        for packet in audio.encode():
            out.mux(packet)
    monkeypatch.delenv("COMFYTV_FFPROBE", raising=False)
    with pytest.raises(RuntimeError, match="no video stream"):
        _probe(path, 1)
