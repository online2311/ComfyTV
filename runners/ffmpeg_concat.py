"""Join clips with ffmpeg, one source at a time.

Policy when clips disagree (this is the rule, not a guess):

* Picture size and frame rate follow the first clip. Width and height are
  rounded down to even numbers. Later clips are stretched to that size and
  retimed with the ``fps`` filter. A first clip with no usable size or frame
  rate is an error. Frame rates above 480 fps are an error.
* If any clip has audio, every clip is given a stereo 44.1 kHz track that is
  exactly as long as its video (silence when the source has none, trimmed or
  padded when it does). The tracks are concatenated as PCM, then encoded to
  AAC once, so encoder delay does not accumulate per clip.
* If no clip has audio, the output has no audio track.

Pictures are never decoded in this process. Duration, size, frame rate, and
whether a clip has audio are read with PyAV. When the container omits a video
duration (Matroska and WebM often do, and then report the longer audio track
as the container duration), the picture length is the end of the video packet
timestamps. Clips that already share codec, size, frame rate, pixel format,
and audio layout are joined with ``-c copy``. Copied AAC keeps each clip's
priming. When that leaves the audio more than one frame off the picture, the
audio is padded or trimmed to the picture and encoded once; the picture stays
a stream copy. Anything else is normalized by its own ffmpeg subprocess
(h264_nvenc when the probe succeeds, otherwise libx264), then the concat
demuxer stream-copies those parts. The binary comes from
``ffmpeg_bin.find_ffmpeg``.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from ._audio_io import _AUDIO_RATE
from ._media_paths import fresh_output_path, localize, path_to_view_url
from .ffmpeg_bin import find_ffmpeg, usable_executable
from .ffmpeg_encode import (
    resolve_video_encoder_mode,
    select_video_encoder,
    video_encoder_args,
)

_log = logging.getLogger(__name__)

try:
    import av as _av
except ImportError:  # optional; probe falls back to ffprobe
    _av = None

_MAX_FPS = 480
# Stream-copied AAC may sit this far from the picture (about one frame).
_AUDIO_SLACK_S = 0.05


@dataclass(frozen=True)
class TargetFormat:
    width: int
    height: int
    fps_fraction: Fraction

    @property
    def fps(self) -> float:
        return float(self.fps_fraction)


def target_format(stream: dict) -> TargetFormat:
    """Even size and frame rate taken from the first clip's video stream."""
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    even_w = width - (width % 2)
    even_h = height - (height % 2)
    if even_w <= 0 or even_h <= 0:
        raise RuntimeError(
            f"concat: first clip has invalid dimensions ({width}x{height})"
        )
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(
        stream.get("r_frame_rate")
    )
    if fps is None:
        raise RuntimeError("concat: first clip has no usable frame rate")
    if fps > _MAX_FPS:
        raise RuntimeError(
            f"concat: first clip frame rate {float(fps):.3f} fps is above {_MAX_FPS}"
        )
    return TargetFormat(even_w, even_h, fps)


def concat_videos(view_urls: list, progress=None, *,
                  video_encoder: str | None = None, report: dict | None = None) -> str:
    urls = [u for u in (view_urls or []) if (u or "").strip()]
    if len(urls) < 2:
        raise RuntimeError(f"concat: need at least 2 videos, got {len(urls)}")

    ffmpeg = find_ffmpeg()
    sources = [localize(u) for u in urls]
    probed = [_probe(path, i) for i, path in enumerate(sources)]
    first_video = probed[0]["video"]
    if first_video is None:
        raise RuntimeError("concat: clip 1 has no video stream")
    target = target_format(first_video)
    want_audio = any(item["audio"] for item in probed)

    out = fresh_output_path(".mp4")
    try:
        used = _concat_sources(
            ffmpeg, probed, target, want_audio, out, progress, video_encoder,
        )
    except Exception:
        out.unlink(missing_ok=True)
        raise
    _log.info("concat: video encoder %s", used)
    if report is not None:
        report["video_encoder"] = used
    return path_to_view_url(out)


def _concat_sources(ffmpeg, probed, target, want_audio, out, progress,
                    video_encoder) -> str:
    if _can_stream_copy(probed, target, want_audio):
        try:
            _concat_copy(
                ffmpeg, [item["path"] for item in probed], out, faststart=True,
            )
            if want_audio:
                _fit_copied_audio(ffmpeg, out)
            return "copy"
        except RuntimeError as exc:
            _log.warning("concat: stream copy failed, re-encoding: %s", exc)
            out.unlink(missing_ok=True)
    encoder = select_video_encoder(ffmpeg, resolve_video_encoder_mode(video_encoder))
    with tempfile.TemporaryDirectory(prefix="comfytv-concat-") as tmp:
        tmp_dir = Path(tmp)
        parts: list[Path] = []
        total = len(probed)
        for i, item in enumerate(probed):
            if progress is not None:
                progress(i, total, f"clip {i + 1}/{total}")
            part = tmp_dir / f"part-{i:04d}.mov"
            _normalize(
                ffmpeg, item, target, want_audio, part,
                label=f"clip {i + 1}/{total}", encoder=encoder,
            )
            parts.append(part)
        joined = tmp_dir / "joined.mov"
        _concat_copy(ffmpeg, parts, joined)
        _remux(ffmpeg, joined, out, want_audio)
    return encoder


def _parse_rate(raw) -> Fraction | None:
    if raw in (None, "", "0/0", "N/A"):
        return None
    try:
        rate = Fraction(str(raw))
    except (ValueError, ZeroDivisionError):
        return None
    if rate <= 0:
        return None
    return rate


def _positive(raw) -> float | None:
    value = _nonnegative(raw)
    if value is None or value <= 0:
        return None
    return value


def _nonnegative(raw) -> float | None:
    if raw in (None, "", "N/A"):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _probe(path: Path, index: int) -> dict:
    """Default probe is PyAV. ffprobe is used only when asked, or when av is missing."""
    override = os.environ.get("COMFYTV_FFPROBE", "").strip()
    if override:
        exe = usable_executable(override)
        if exe is None:
            raise RuntimeError(
                f"concat: COMFYTV_FFPROBE={override} is not an executable file. "
                "Unset COMFYTV_FFPROBE to probe with PyAV, or set it to the "
                "absolute path of an ffprobe binary."
            )
        return _probe_ffprobe(exe, path, index)
    if _av is not None:
        return _probe_pyav(path, index)
    found = usable_executable(shutil.which("ffprobe"))
    if found:
        return _probe_ffprobe(found, path, index)
    raise RuntimeError(
        "concat: PyAV (package av) is not installed, and ffprobe was not found. "
        "Checked COMFYTV_FFPROBE (not set) and PATH "
        "(shutil.which('ffprobe') found nothing). "
        "Install av, or set COMFYTV_FFPROBE to the absolute path of an ffprobe binary."
    )


def _stream_rate(stream) -> Fraction | None:
    for attr in ("average_rate", "base_rate", "guessed_rate"):
        raw = getattr(stream, attr, None)
        if not raw:
            continue
        try:
            rate = Fraction(raw.numerator, raw.denominator)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            try:
                rate = Fraction(str(raw))
            except (ValueError, ZeroDivisionError):
                continue
        if rate > 0:
            return rate
    return None


def _rate_text(rate: Fraction | None) -> str:
    if rate is None:
        return "0/0"
    return f"{int(rate.numerator)}/{int(rate.denominator)}"


def _stream_seconds(stream) -> float | None:
    if stream.duration is None or not stream.time_base:
        return None
    try:
        seconds = float(stream.duration * stream.time_base)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if seconds <= 0:
        return None
    return seconds


def _packet_end(container, video, fps: Fraction | None) -> float | None:
    frame_dur = (1.0 / float(fps)) if fps else 0.0
    end = 0.0
    saw = False
    for packet in container.demux(video):
        if packet.pts is None or packet.pts < 0:
            continue
        time_base = packet.time_base or video.time_base
        if not time_base:
            continue
        pts = float(packet.pts * time_base)
        if pts < 0:
            continue
        pkt_dur = None
        if packet.duration is not None and packet.duration > 0:
            pkt_dur = float(packet.duration * time_base)
        if pkt_dur is None or pkt_dur <= 0:
            pkt_dur = frame_dur
        saw = True
        end = max(end, pts + pkt_dur)
    return end if saw and end > 0 else None


def _probe_pyav(path: Path, index: int) -> dict:
    if _av is None:
        raise ImportError("av is not installed")
    try:
        with _av.open(str(path)) as container:
            if not container.streams.video:
                raise RuntimeError(f"concat: clip {index + 1} has no video stream")
            video = container.streams.video[0]
            audio = container.streams.audio[0] if container.streams.audio else None
            fps = _stream_rate(video)
            duration = _stream_seconds(video)
            frames = int(video.frames or 0)
            if duration is None and fps is not None and frames > 0:
                duration = frames / float(fps)
            if not duration:
                duration = _packet_end(container, video, fps)
            if not duration:
                raise RuntimeError(f"concat: clip {index + 1} has no duration")
            width = int(video.width or 0) or int(getattr(video.codec_context, "width", 0) or 0)
            height = int(video.height or 0) or int(getattr(video.codec_context, "height", 0) or 0)
            rate = _rate_text(fps)
            stream = {
                "codec_type": "video",
                "codec": getattr(video.codec_context, "name", None),
                "pix_fmt": _pix_name(video),
                "width": width,
                "height": height,
                "avg_frame_rate": rate,
                "r_frame_rate": rate,
                "duration": f"{duration}",
                "nb_frames": str(frames) if frames > 0 else None,
            }
            audio_info = _audio_info(audio) if audio is not None else None
            return {
                "path": path,
                "video": stream,
                "audio": audio_info,
                "duration": duration,
            }
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"concat: failed to probe clip {index + 1}: {exc}") from exc


def _probe_ffprobe(ffprobe: str, path: Path, index: int) -> dict:
    proc = subprocess.run(
        [
            ffprobe, "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", str(path),
        ],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-800:]
        raise RuntimeError(f"concat: ffprobe failed on clip {index + 1}: {tail}")
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"concat: ffprobe returned invalid JSON for clip {index + 1}") from exc
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise RuntimeError(f"concat: clip {index + 1} has no video stream")
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    duration = _video_duration(video, fps)
    if duration is None:
        # Container duration can be the longer audio track (common on Matroska).
        # Read the video packets instead of extending the picture to that length.
        duration = _video_packet_end(ffprobe, path, fps)
    if not duration:
        raise RuntimeError(f"concat: clip {index + 1} has no duration")
    if video.get("codec_name") and not video.get("codec"):
        video["codec"] = video.get("codec_name")
    if audio is not None and audio.get("codec_name") and not audio.get("codec"):
        audio["codec"] = audio.get("codec_name")
    return {"path": path, "video": video, "audio": audio, "duration": duration}


def _video_duration(stream: dict, fps: Fraction | None) -> float | None:
    duration = _positive(stream.get("duration"))
    if duration is not None:
        return duration
    if fps is None:
        return None
    nb = stream.get("nb_frames")
    if nb in (None, "", "N/A"):
        return None
    try:
        frames = int(nb)
    except ValueError:
        return None
    if frames <= 0:
        return None
    return frames / float(fps)


def _video_packet_end(ffprobe: str, path: Path, fps: Fraction | None) -> float | None:
    proc = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts_time,duration_time",
            "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        return None
    frame_dur = (1.0 / float(fps)) if fps else 0.0
    end = 0.0
    saw = False
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        pts = _nonnegative(parts[0] if parts else None)
        if pts is None:
            continue
        pkt_dur = _positive(parts[1]) if len(parts) > 1 else None
        saw = True
        end = max(end, pts + (pkt_dur if pkt_dur is not None else frame_dur))
    return end if saw and end > 0 else None


def _fps_arg(fps: Fraction) -> str:
    if fps.denominator == 1:
        return str(int(fps.numerator))
    return f"{int(fps.numerator)}/{int(fps.denominator)}"


def _timescale(fps: Fraction) -> int:
    numerator = int(fps.numerator)
    if numerator < 1000:
        return numerator * 1000
    return numerator


def _run(cmd: list[str], what: str) -> None:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-1200:]
        raise RuntimeError(f"concat: {what} failed: {tail}")


_COPY_VIDEO_CODECS = {"h264", "hevc", "av1"}


def _video_codec(stream: dict) -> str | None:
    name = stream.get("codec") or stream.get("codec_name")
    return str(name).lower() if name else None


def _pix_fmt_name(stream: dict) -> str | None:
    pix = stream.get("pix_fmt")
    return str(pix).lower() if pix else None


def _as_int(raw) -> int | None:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _audio_duration(audio: dict) -> float | None:
    raw = audio.get("duration")
    if raw in (None, "", "N/A"):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _pix_name(video) -> str | None:
    fmt = getattr(video, "format", None)
    name = getattr(fmt, "name", None) if fmt is not None else None
    if name:
        return str(name)
    raw = getattr(getattr(video, "codec_context", None), "pix_fmt", None)
    named = getattr(raw, "name", None)
    if named:
        return str(named)
    return str(raw) if raw else None


def _audio_info(audio) -> dict:
    return {
        "codec_type": "audio",
        "codec": getattr(audio.codec_context, "name", None),
        "sample_rate": int(audio.sample_rate or 0) or None,
        "channels": int(getattr(audio, "channels", 0) or 0) or None,
        "duration": _stream_seconds(audio),
    }


def _can_stream_copy(probed: list, target: TargetFormat, want_audio: bool) -> bool:
    """True when the sources can be joined with -c copy, without a re-encode.

    Mixed sizes, frame rates, codecs, or a silent clip beside a clip with
    audio still go through the normalize path. A video track shorter than
    its audio (typical Matroska) is re-encoded so the picture is not held.
    """
    if len(probed) < 2:
        return False
    first = probed[0]
    codec = _video_codec(first["video"])
    pix = _pix_fmt_name(first["video"])
    if codec not in _COPY_VIDEO_CODECS or pix != "yuv420p":
        return False
    first_audio = first["audio"]
    for item in probed:
        video = item["video"]
        if _video_codec(video) != codec or _pix_fmt_name(video) != pix:
            return False
        if int(video.get("width") or 0) != target.width:
            return False
        if int(video.get("height") or 0) != target.height:
            return False
        rate = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(
            video.get("r_frame_rate")
        )
        if rate != target.fps_fraction:
            return False
        audio = item["audio"]
        if not want_audio:
            if audio is not None:
                return False
            continue
        if audio is None or first_audio is None:
            return False
        if (_video_codec(audio) or "") != "aac":
            return False
        if _video_codec(audio) != _video_codec(first_audio):
            return False
        if _as_int(audio.get("sample_rate")) != _as_int(first_audio.get("sample_rate")):
            return False
        if _as_int(audio.get("channels")) != _as_int(first_audio.get("channels")):
            return False
        heard = _audio_duration(audio)
        if heard is None or abs(heard - float(item["duration"])) > 0.08:
            return False
    return True


def _normalize(ffmpeg: str, item: dict, target: TargetFormat, want_audio: bool,
               dest: Path, *, label: str, encoder: str = "libx264") -> None:
    frames = max(1, int(round(item["duration"] * float(target.fps_fraction))))
    samples = int(round(frames / float(target.fps_fraction) * _AUDIO_RATE))
    fps_arg = _fps_arg(target.fps_fraction)
    video = (
        f"[0:v]scale={target.width}:{target.height}:flags=bilinear,"
        f"fps={fps_arg}:eof_action=pass,"
        f"setsar=1,format=yuv420p,setpts=PTS-STARTPTS,"
        f"tpad=stop_mode=clone:stop_duration=1,trim=end_frame={frames},"
        f"setpts=PTS-STARTPTS[v]"
    )
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(item["path"])]
    if want_audio and item["audio"] is not None:
        audio = (
            f"[0:a]aformat=sample_fmts=s16:channel_layouts=stereo,"
            f"aresample={_AUDIO_RATE}:async=1:first_pts=0,"
            f"apad=pad_len={samples},atrim=end_sample={samples},asetpts=PTS-STARTPTS[a]"
        )
        filters = video + ";" + audio
    elif want_audio:
        pad_s = f"{(frames / float(target.fps_fraction)) + 0.5:.3f}"
        audio = f"[1:a]atrim=end_sample={samples},asetpts=PTS-STARTPTS[a]"
        filters = video + ";" + audio
        cmd += [
            "-f", "lavfi", "-t", pad_s, "-i",
            f"anullsrc=channel_layout=stereo:sample_rate={_AUDIO_RATE}",
        ]
    else:
        filters = video
    cmd += ["-filter_complex", filters, "-map", "[v]"]
    if want_audio:
        cmd += ["-map", "[a]", "-c:a", "pcm_s16le"]
    else:
        cmd += ["-an"]
    cmd += [
        *video_encoder_args(encoder),
        "-video_track_timescale", str(_timescale(target.fps_fraction)),
        "-f", "mov", str(dest),
    ]
    _run(cmd, label)


def _concat_line(path: Path) -> str:
    escaped = path.as_posix().replace("'", r"'\''")
    return f"file '{escaped}'\n"


def _concat_copy(ffmpeg: str, parts: list[Path], dest: Path, *,
                 faststart: bool = False) -> None:
    listing = dest.with_name(f"{dest.stem}-concat.txt")
    listing.write_text("".join(_concat_line(p) for p in parts), encoding="utf-8")
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c", "copy",
    ]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(dest))
    try:
        _run(cmd, "concat")
    finally:
        listing.unlink(missing_ok=True)


def _fit_copied_audio(ffmpeg: str, path: Path) -> None:
    """Make a stream-copied audio track the same length as the picture.

    ``-c copy`` cannot cut inside an AAC frame, and each clip's priming is
    kept, so the joined audio runs long. Video packets stay copied. The
    audio is padded or trimmed to the picture duration and encoded once.
    """
    info = _probe(path, 0)
    if info["audio"] is None:
        return
    video_s = float(info["duration"])
    if video_s <= 0:
        raise RuntimeError("concat: copied picture has no duration")
    heard = _audio_duration(info["audio"])
    if heard is not None and abs(heard - video_s) <= _AUDIO_SLACK_S:
        return
    duration = f"{video_s:.6f}"
    dest = path.with_name(f"{path.stem}-audiofit{path.suffix}")
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-map", "0:v:0", "-c:v", "copy",
        "-map", "0:a:0",
        "-af", (
            f"apad=whole_dur={duration},atrim=end={duration},"
            "asetpts=PTS-STARTPTS"
        ),
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(dest),
    ]
    try:
        _run(cmd, "fit audio")
        dest.replace(path)
    finally:
        dest.unlink(missing_ok=True)
    if heard is None:
        _log.info("concat: fitted audio to picture %.3fs", video_s)
    else:
        _log.info(
            "concat: fitted audio to picture %.3fs (audio was %.3fs)",
            video_s, heard,
        )


def _remux(ffmpeg: str, src: Path, dest: Path, want_audio: bool) -> None:
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    if want_audio:
        cmd += [
            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
            "-ar", str(_AUDIO_RATE), "-ac", "2",
        ]
    else:
        cmd += ["-c", "copy", "-an"]
    cmd += ["-movflags", "+faststart", str(dest)]
    _run(cmd, "remux")
