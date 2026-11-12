"""Unlimited ffmpeg concat: many segments, silence padding, no in-process decode.

The autouse fixture removes ffmpeg and ffprobe from PATH. That is the Spark
production case: ComfyUI's venv Python does not see a system ffmpeg, and the
imageio-ffmpeg wheel ships ffmpeg without ffprobe.
"""
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

from test_media_concat import _write_clip  # noqa: E402


@pytest.fixture(autouse=True)
def hide_system_ffmpeg(monkeypatch, tmp_path):
    """Do not let a system ffmpeg/ffprobe satisfy concat."""
    empty = tmp_path / "empty-path"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.delenv("COMFYTV_FFMPEG", raising=False)
    monkeypatch.delenv("COMFYTV_FFPROBE", raising=False)
    monkeypatch.delenv("COMFYTV_VIDEO_ENCODER", raising=False)
    monkeypatch.delenv("IMAGEIO_FFMPEG_EXE", raising=False)
    from ComfyTV.runners.ffmpeg_bin import clear_ffmpeg_cache
    clear_ffmpeg_cache()


def _ffmpeg() -> str:
    from ComfyTV.runners.ffmpeg_bin import find_ffmpeg
    return find_ffmpeg()


def _urls(paths):
    from ComfyTV.runners import media
    return [media.path_to_view_url(p) for p in paths]


def _src_dir(name: str) -> Path:
    import folder_paths
    d = Path(folder_paths.get_output_directory()) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _videos_input(stage_cls):
    schema = stage_cls.define_schema()
    for inp in schema.kw["inputs"]:
        if inp.args and inp.args[0] == "videos" and "template" in getattr(inp, "kw", {}):
            return inp
    raise AssertionError(f"{stage_cls.__name__} has no videos autogrow input")


def _stream_durations(url):
    import av as avlib
    from ComfyTV.runners import media
    with avlib.open(str(media.localize(url))) as container:
        video = container.streams.video[0]
        video_s = float(video.duration * video.time_base)
        if not container.streams.audio:
            return video_s, None
        audio = container.streams.audio[0]
        audio_s = float(audio.duration * audio.time_base)
    return video_s, audio_s


def _decoded_audio_seconds(url):
    """Samples the decoder actually emits, divided by the sample rate.

    Stream duration and the MP4 edit list can hide AAC priming. ``frame.samples``
    does not.
    """
    import av as avlib
    from ComfyTV.runners import media
    with avlib.open(str(media.localize(url))) as container:
        stream = container.streams.audio[0]
        rate = stream.sample_rate or 44100
        samples = 0
        for frame in container.decode(stream):
            samples += frame.samples
    return samples / rate


def _mono_pcm(url):
    import av as avlib
    from ComfyTV.runners import media
    chunks = []
    with avlib.open(str(media.localize(url))) as container:
        stream = container.streams.audio[0]
        rate = stream.sample_rate or 44100
        resampler = avlib.AudioResampler(format="fltp", layout="mono", rate=rate)
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray()[0])
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray()[0])
    if not chunks:
        return np.zeros(0, dtype=np.float32), rate
    return np.concatenate(chunks), rate


def _tone_hz(samples, rate, start_s, end_s):
    window = samples[int(start_s * rate):int(end_s * rate)]
    if window.size < 8:
        return 0.0
    signs = np.sign(window)
    signs[signs == 0] = 1
    crossings = np.where(np.diff(signs))[0]
    if crossings.size < 2:
        return 0.0
    return float(rate / (2 * np.mean(np.diff(crossings))))


def _mono_rms(url, start_s, end_s):
    import av as avlib
    from ComfyTV.runners import media
    chunks = []
    with avlib.open(str(media.localize(url))) as container:
        stream = container.streams.audio[0]
        rate = stream.sample_rate or 44100
        resampler = avlib.AudioResampler(format="fltp", layout="mono", rate=rate)
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray()[0])
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray()[0])
    samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    window = samples[int(start_s * rate):int(end_s * rate)]
    if window.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))


class TestAssemblySlots:
    """Old canvases store videos.video0 … videos.video11. The group stays an
    autogrow prefix so those names still resolve, and the cap is high enough
    for a 22-shot episode plus the 64-segment check."""

    @pytest.mark.parametrize("cls_name", ["VideoConcatStage", "SequenceStage"])
    def test_autogrow_keeps_legacy_names_and_fits_64(self, cls_name):
        from ComfyTV.nodes.stages.common.inputs import VIDEO_ASSEMBLY_SLOT_MAX
        from ComfyTV.nodes import stages

        inp = _videos_input(getattr(stages, cls_name))
        template = inp.kw["template"]
        assert template.kw["prefix"] == "video"
        assert template.kw["min"] == 0
        assert template.kw["max"] == VIDEO_ASSEMBLY_SLOT_MAX
        assert template.kw["max"] >= 64
        assert template.kw["max"] >= 12


def test_concat_stage_forwards_every_slot(reset_db, monkeypatch):
    from ComfyTV.nodes.stages import video_edit

    captured = {}

    def fake_concat(urls, progress=None, **_ignored):
        captured["urls"] = list(urls)
        return "/view?filename=out.mp4"

    monkeypatch.setattr(video_edit, "concat_videos", fake_concat)
    videos = {f"videos.video{i}": f"/view?filename=c{i}.mp4" for i in range(64)}
    videos["videos.video3"] = ""
    video_edit.VideoConcatStage.execute(project_id="default", videos=videos)
    assert len(captured["urls"]) == 63
    assert captured["urls"][0].endswith("c0.mp4")
    # video3 is empty, so the legacy video11 slot is still present, one place earlier.
    assert captured["urls"][10].endswith("c11.mp4")
    assert captured["urls"][-1].endswith("c63.mp4")


def test_sequence_forwards_64_cuts(monkeypatch):
    from ComfyTV.runners import media_remap

    captured = {}

    def fake_concat(urls, progress=None, **_ignored):
        captured["n"] = len(list(urls))
        return list(urls)[0]

    monkeypatch.setattr(media_remap, "concat_videos", fake_concat)
    monkeypatch.setattr(
        media_remap, "get_video_info",
        lambda url: {"duration": 1.0, "fps": 24, "width": 64, "height": 48, "has_audio": True},
    )
    segs = [{"url": f"/view?filename/{i}.mp4", "transition": "cut"} for i in range(64)]
    media_remap.render_sequence(segs)
    assert captured["n"] == 64


def test_concat_uses_bundled_ffmpeg_and_does_not_decode_pictures(monkeypatch):
    """PATH has no ffmpeg/ffprobe. The bundled imageio-ffmpeg binary runs the
    concat, PyAV opens each clip to read timestamps, and no video frame is
    decoded in this process. No ffmpeg process is handed every source."""
    import imageio_ffmpeg
    from ComfyTV.runners import media

    bundled = imageio_ffmpeg.get_ffmpeg_exe()
    src = _src_dir("concat-ff")
    a = src / "guard-a.mp4"
    b = src / "guard-b.mp4"
    if not a.exists():
        _write_clip(a, w=64, h=48, fps=12, seconds=0.25, with_audio=True)
    if not b.exists():
        _write_clip(b, w=80, h=48, fps=15, seconds=0.2, with_audio=False)

    commands = []
    real_run = subprocess.run

    def spy(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)):
            commands.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)

    opened = {"n": 0}
    real_open = av.open
    real_decode = av.container.InputContainer.decode

    def guarding_open(*args, **kwargs):
        opened["n"] += 1
        return real_open(*args, **kwargs)

    def guarding_decode(self, *dargs, **dkwargs):
        if not dargs:
            raise AssertionError("concat decoded video frames with PyAV")
        for item in dargs:
            if getattr(item, "type", None) == "video":
                raise AssertionError("concat decoded video frames with PyAV")
        return real_decode(self, *dargs, **dkwargs)

    monkeypatch.setattr(av, "open", guarding_open)
    monkeypatch.setattr(av.container.InputContainer, "decode", guarding_decode)
    media.concat_videos(_urls([a, b]))

    assert opened["n"] >= 2
    assert commands
    assert not any(Path(c[0]).name.startswith("ffprobe") for c in commands)
    ffmpeg_cmds = [c for c in commands if c and Path(c[0]).name.startswith("ffmpeg")]
    assert ffmpeg_cmds
    for cmd in ffmpeg_cmds:
        assert os.path.samefile(cmd[0], bundled)
        assert cmd.count("-i") <= 2
    concat_cmds = [c for c in ffmpeg_cmds if "concat" in c]
    assert len(concat_cmds) == 1
    assert concat_cmds[0].count("-i") == 1


def test_silent_segment_does_not_shift_later_audio():
    """Middle clip has no audio. Silence is inserted for its video length so
    the third clip's tone still starts after clip1+clip2, not after clip1."""
    from ComfyTV.runners import media

    src = _src_dir("concat-sync")
    a = src / "tone-a.mp4"
    b = src / "silent.mp4"
    c = src / "tone-c.mp4"
    _write_clip(a, w=96, h=64, fps=24, seconds=1.0, with_audio=True)
    _write_clip(b, w=96, h=64, fps=24, seconds=0.5, with_audio=False)
    _write_clip(c, w=96, h=64, fps=24, seconds=1.0, with_audio=True)
    out = media.concat_videos(_urls([a, b, c]))

    video_s, audio_s = _stream_durations(out)
    assert audio_s is not None
    assert abs(video_s - audio_s) < 0.04
    assert abs(video_s - 2.5) < 0.04
    assert _mono_rms(out, 0.15, 0.85) > 0.02
    assert _mono_rms(out, 1.12, 1.38) < 0.01
    assert _mono_rms(out, 1.65, 2.35) > 0.02


def test_concat_22_segments_mixed_audio_and_size():
    """A 22-shot episode: silent and sounding clips, two frame sizes."""
    from ComfyTV.runners import media

    src = _src_dir("concat-22")
    paths = []
    for i in range(22):
        path = src / f"s{i:02d}.mp4"
        odd = i % 2 == 1
        if not path.exists():
            _write_clip(
                path,
                w=96 if odd else 64,
                h=64 if odd else 48,
                fps=24 if odd else 12,
                seconds=0.25,
                with_audio=i % 3 != 2,
            )
        paths.append(path)

    out = media.concat_videos(_urls(paths))
    info = media.get_video_info(out)
    assert (info["width"], info["height"]) == (64, 48)
    assert info["has_audio"] is True
    video_s, audio_s = _stream_durations(out)
    assert audio_s is not None
    assert abs(video_s - audio_s) < 0.08
    assert 5.0 <= video_s <= 6.2


def test_concat_64_segments_mixed_audio_and_geometry():
    from ComfyTV.runners import media

    src = _src_dir("concat-64")
    paths = []
    for i in range(64):
        path = src / f"s{i:02d}.mp4"
        odd = i % 4 == 3
        if not path.exists():
            _write_clip(
                path,
                w=80 if odd else 64,
                h=48,
                fps=15 if odd else 12,
                seconds=0.2 if odd else 0.25,
                with_audio=not odd,
            )
        paths.append(path)

    out = media.concat_videos(_urls(paths))
    info = media.get_video_info(out)
    assert (info["width"], info["height"]) == (64, 48)
    assert info["has_audio"] is True
    video_s, audio_s = _stream_durations(out)
    assert audio_s is not None
    assert abs(video_s - audio_s) < 0.05
    # 48 clips * 0.25s + 16 clips * 0.20s, quantized onto 12 fps.
    assert 14.0 <= video_s <= 16.5


def test_stream_copy_aac_duration_stays_within_one_frame():
    """Matching AAC clips are joined with a video stream copy. Encoder
    priming must not leave the audio more than one frame (~0.05s) away
    from the picture."""
    from ComfyTV.runners import media

    src = _src_dir("concat-aac-align")
    fps = 24
    paths = []
    for i in range(8):
        path = src / f"s{i:02d}.mp4"
        subprocess.check_call([
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-t", "0.5", "-i", f"color=c=red:s=64x48:r={fps}",
            "-f", "lavfi", "-t", "0.56", "-i", "sine=frequency=440:sample_rate=44100",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0",
            "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2",
            "-map", "0:v:0", "-map", "1:a:0",
            str(path),
        ])
        paths.append(path)

    report = {}
    out = media.concat_videos(_urls(paths), report=report)
    assert report["video_encoder"] == "copy"
    assert report["audio_encoder"] == "reencode"
    video_s, audio_s = _stream_durations(out)
    assert audio_s is not None
    assert video_s >= 8 * 0.5 - 0.05
    assert abs(video_s - audio_s) <= 0.05
    assert abs(_decoded_audio_seconds(out) - video_s) <= 0.05


def test_matching_aac_reencodes_audio_and_copies_video(monkeypatch):
    """Matching AAC clips no longer stay a byte-identical stream copy.

    PR #22 left the audio untouched when the container durations were
    already within one frame. That comparison misses encoder priming: the
    edit list keeps the container aligned while the decoded samples run
    long once several clips are joined. ``video_encoder`` stays ``copy``
    because the picture is not re-encoded. The audio is rebuilt, so the
    output is not the pure ``ffmpeg -c copy`` file.
    """
    from ComfyTV.runners import media

    src = _src_dir("concat-aac-pure-copy")
    paths = []
    for i in range(4):
        path = src / f"s{i:02d}.mp4"
        _write_clip(path, w=64, h=48, fps=24, seconds=0.5, with_audio=True)
        paths.append(path)

    commands = []
    real_run = subprocess.run

    def spy(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)):
            commands.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    report = {}
    out = media.concat_videos(_urls(paths), report=report)
    assert report["video_encoder"] == "copy"
    assert report["audio_encoder"] == "reencode"
    video_s, audio_s = _stream_durations(out)
    assert audio_s is not None
    assert abs(video_s - audio_s) <= 0.05
    assert abs(_decoded_audio_seconds(out) - video_s) <= 0.05

    ffmpeg_cmds = [c for c in commands if c and Path(c[0]).name.startswith("ffmpeg")]
    assert any("concat" in cmd and "copy" in cmd for cmd in ffmpeg_cmds)
    assert any("apad" in arg or "atrim" in arg for cmd in ffmpeg_cmds for arg in cmd)
    assert any(arg == "aac" for cmd in ffmpeg_cmds for arg in cmd)
    assert not any(
        arg in ("libx264", "h264_nvenc") for cmd in ffmpeg_cmds for arg in cmd
    )

    copied = media.localize(out)
    reference = src / "reference-copy.mp4"
    listing = src / "reference-concat.txt"
    listing.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in paths), encoding="utf-8")
    real_run([
        _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c", "copy", "-movflags", "+faststart", str(reference),
    ], check=True)
    assert hashlib.sha256(copied.read_bytes()).digest() != hashlib.sha256(reference.read_bytes()).digest()


def test_many_aac_segments_match_picture_by_decoded_samples():
    """22 short AAC clips: the joined container looks aligned, the decoder does not.

    Each source edit list hides about one AAC frame of priming. Stream-copy
    concat drops those lists, so ffprobe still reports audio within one frame
    of the picture while the samples PyAV emits run long. The copy path has
    to rebuild that audio. A pure ``-c copy`` of the same clips is the
    before-fix witness and must stay long.
    """
    from ComfyTV.runners import media

    n = 22
    segment_s = 0.5
    src = _src_dir("concat-aac-decoded")
    paths = []
    freqs = []
    for i in range(n):
        freq = 440 + (i % 5) * 220
        freqs.append(freq)
        path = src / f"s{i:02d}.mp4"
        subprocess.check_call([
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-t", str(segment_s), "-i", "color=c=red:s=64x48:r=24",
            "-f", "lavfi", "-t", str(segment_s), "-i",
            f"sine=frequency={freq}:sample_rate=44100",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0",
            "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2",
            "-map", "0:v:0", "-map", "1:a:0",
            str(path),
        ])
        paths.append(path)

    raw = src / "raw-copy.mp4"
    listing = src / "raw-concat.txt"
    listing.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in paths),
        encoding="utf-8",
    )
    subprocess.check_call([
        _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c", "copy", "-movflags", "+faststart", str(raw),
    ])
    raw_url = _urls([raw])[0]
    raw_video, raw_audio = _stream_durations(raw_url)
    raw_decoded = _decoded_audio_seconds(raw_url)
    assert abs(raw_video - raw_audio) <= 0.05
    assert raw_decoded - raw_video > 0.2

    report = {}
    out = media.concat_videos(_urls(paths), report=report)
    assert report["video_encoder"] == "copy"
    video_s, _audio_s = _stream_durations(out)
    decoded = _decoded_audio_seconds(out)
    expected = n * segment_s
    assert abs(video_s - expected) <= 0.05
    assert abs(decoded - video_s) <= 0.05
    assert abs(decoded - expected) <= 0.05
    assert report["audio_encoder"] == "reencode"

    pcm, rate = _mono_pcm(out)
    for i, freq in enumerate(freqs):
        start = i * segment_s + 0.12
        end = i * segment_s + 0.38
        heard = _tone_hz(pcm, rate, start, end)
        assert abs(heard - freq) < 40, f"segment {i}: heard {heard:.1f} Hz, expected {freq}"


def test_audio_refit_stops_when_progress_cancels():
    """Matching AAC clips are fitted one at a time. The progress callback is
    the cancellation check, so a cancel on the second clip must leave the
    later clips unprocessed instead of falling through to a re-encode."""
    from ComfyTV.runners import media

    class Cancelled(Exception):
        pass

    src = _src_dir("concat-aac-cancel")
    paths = []
    for i in range(4):
        path = src / f"s{i:02d}.mp4"
        _write_clip(path, w=64, h=48, fps=24, seconds=0.5, with_audio=True)
        paths.append(path)

    calls = []

    def progress(index, total, text=""):
        calls.append((index, total, text))
        if index == 1:
            raise Cancelled()

    with pytest.raises(Cancelled):
        media.concat_videos(_urls(paths), progress=progress)
    assert calls == [(0, 4, "clip 1/4"), (1, 4, "clip 2/4")]


def test_refit_keeps_a_leading_audio_gap():
    """A 1s picture whose AAC starts about 60ms late is still copy-compatible.

    Sample padding has to keep that opening silence. Dropping it and padding
    the end plays the tone early.
    """
    from ComfyTV.runners import media

    src = _src_dir("concat-aac-gap")
    paths = []
    for i in range(2):
        path = src / f"s{i:02d}.mp4"
        subprocess.check_call([
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-t", "1.0", "-i", "color=c=blue:s=64x48:r=24",
            "-itsoffset", "0.06", "-f", "lavfi", "-t", "0.94", "-i",
            "sine=frequency=880:sample_rate=44100",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0",
            "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2",
            "-map", "0:v:0", "-map", "1:a:0",
            str(path),
        ])
        paths.append(path)

    report = {}
    out = media.concat_videos(_urls(paths), report=report)
    assert report["video_encoder"] == "copy"
    assert report["audio_encoder"] == "reencode"
    assert _mono_rms(out, 0.0, 0.03) < 0.01
    assert _mono_rms(out, 0.12, 0.40) > 0.02
    video_s, _audio_s = _stream_durations(out)
    assert abs(_decoded_audio_seconds(out) - video_s) <= 0.05


def test_refit_decodes_the_probed_audio_stream():
    """The probe reads the first audio stream. A later default track must
    not replace it when the clip is decoded for the refit."""
    from ComfyTV.runners import media

    src = _src_dir("concat-aac-first-stream")
    built = src / "dual.mp4"
    subprocess.check_call([
        _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-t", "0.5", "-i", "color=c=red:s=64x48:r=24",
        "-f", "lavfi", "-t", "0.5", "-i", "sine=frequency=440:sample_rate=44100",
        "-f", "lavfi", "-t", "0.5", "-i", "sine=frequency=1760:sample_rate=44100",
        "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0",
        "-c:a:0", "aac", "-ac:a:0", "1", "-b:a:0", "96k",
        "-c:a:1", "aac", "-ac:a:1", "2", "-b:a:1", "160k",
        "-disposition:a:0", "0", "-disposition:a:1", "default",
        str(built),
    ])
    paths = [src / "a.mp4", src / "b.mp4"]
    for path in paths:
        path.write_bytes(built.read_bytes())

    report = {}
    out = media.concat_videos(_urls(paths), report=report)
    assert report["video_encoder"] == "copy"
    pcm, rate = _mono_pcm(out)
    heard = _tone_hz(pcm, rate, 0.12, 0.38)
    assert abs(heard - 440) < 40, heard


def test_all_silent_sources_stay_silent():
    from ComfyTV.runners import media

    src = _src_dir("concat-silent")
    a = src / "sa.mp4"
    b = src / "sb.mp4"
    if not a.exists():
        _write_clip(a, w=64, h=48, fps=12, seconds=0.25, with_audio=False)
    if not b.exists():
        _write_clip(b, w=64, h=48, fps=12, seconds=0.25, with_audio=False)
    info = media.get_video_info(media.concat_videos(_urls([a, b])))
    assert info["has_audio"] is False


def test_long_audio_does_not_freeze_the_picture():
    """A Matroska file can omit the video duration and report the longer
    audio track as the container duration. The picture length stays the
    video timeline, and the extra audio is cut to match."""
    from ComfyTV.runners import media

    src = _src_dir("concat-long-audio")
    long = src / "long-audio.mkv"
    other = src / "other.mp4"
    if not long.exists():
        subprocess.check_call([
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=red:s=64x48:r=24:d=0.5",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=1.5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-map", "0:v", "-map", "1:a", str(long),
        ])
    if not other.exists():
        _write_clip(other, w=64, h=48, fps=24, seconds=0.5, with_audio=True)
    out = media.concat_videos(_urls([long, other]))
    video_s, audio_s = _stream_durations(out)
    assert 0.9 <= video_s <= 1.2
    assert audio_s is not None
    assert abs(video_s - audio_s) < 0.04


def test_clip_shorter_than_half_a_frame_is_kept():
    """A 0.25s clip after a 1 fps leader is under half a target frame.
    It is held for one output frame instead of disappearing."""
    from ComfyTV.runners import media

    src = _src_dir("concat-short")
    slow = src / "onefps.mp4"
    brief = src / "quarter.mp4"
    if not slow.exists():
        subprocess.check_call([
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=red:s=64x48:r=1:d=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(slow),
        ])
    if not brief.exists():
        subprocess.check_call([
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=blue:s=80x48:r=24:d=0.25",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(brief),
        ])
    out = media.concat_videos(_urls([slow, brief]))
    video_s, audio_s = _stream_durations(out)
    assert audio_s is None
    assert 1.8 <= video_s <= 2.2


def test_missing_video_stream_is_an_error():
    from ComfyTV.runners import media

    src = _src_dir("concat-novid")
    video = src / "real.mp4"
    audio_only = src / "only-audio.m4a"
    if not video.exists():
        _write_clip(video, w=64, h=48, fps=12, seconds=0.25, with_audio=True)
    if not audio_only.exists():
        subprocess.check_call([
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.25",
            "-c:a", "aac", str(audio_only),
        ])
    with pytest.raises(RuntimeError, match="no video stream"):
        media.concat_videos(_urls([video, audio_only]))


def test_normalize_policy_is_explicit():
    """Mismatch is not a hard error: stretch to the first clip's even size
    and frame rate. Unusable first-clip geometry still errors."""
    from ComfyTV.runners.ffmpeg_concat import target_format

    target = target_format({
        "width": 321,
        "height": 241,
        "avg_frame_rate": "24000/1001",
        "r_frame_rate": "24000/1001",
        "duration": "1.0",
    })
    assert (target.width, target.height) == (320, 240)
    assert target.fps == pytest.approx(23.976, abs=0.001)

    with pytest.raises(RuntimeError, match="invalid dimensions"):
        target_format({
            "width": 1, "height": 1,
            "avg_frame_rate": "24/1", "duration": "1",
        })
    with pytest.raises(RuntimeError, match="frame rate"):
        target_format({
            "width": 320, "height": 240,
            "avg_frame_rate": "0/0", "r_frame_rate": "0/0",
            "duration": "1",
        })
