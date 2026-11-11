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

Pictures are never decoded in this process. Each clip is normalized by its
own ffmpeg subprocess, then the concat demuxer stream-copies them.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from ._audio_io import _AUDIO_RATE
from ._media_paths import fresh_output_path, localize, path_to_view_url

_MAX_FPS = 480


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


def concat_videos(view_urls: list, progress=None) -> str:
    urls = [u for u in (view_urls or []) if (u or "").strip()]
    if len(urls) < 2:
        raise RuntimeError(f"concat: need at least 2 videos, got {len(urls)}")

    ffmpeg = _tool("ffmpeg")
    ffprobe = _tool("ffprobe")
    sources = [localize(u) for u in urls]
    probed = [_probe(ffprobe, path, i) for i, path in enumerate(sources)]
    first_video = probed[0]["video"]
    if first_video is None:
        raise RuntimeError("concat: clip 1 has no video stream")
    target = target_format(first_video)
    want_audio = any(item["audio"] for item in probed)

    out = fresh_output_path(".mp4")
    try:
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
                    label=f"clip {i + 1}/{total}",
                )
                parts.append(part)
            joined = tmp_dir / "joined.mov"
            _concat_copy(ffmpeg, parts, joined)
            _remux(ffmpeg, joined, out, want_audio)
    except Exception:
        out.unlink(missing_ok=True)
        raise
    return path_to_view_url(out)


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(
            "concat: ffmpeg/ffprobe not found on PATH. "
            "Concat runs ffmpeg as a subprocess and does not install Python packages."
        )
    return path


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
    if raw in (None, "", "N/A"):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _probe(ffprobe: str, path: Path, index: int) -> dict:
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
    duration = _positive(video.get("duration"))
    if duration is None and fps is not None:
        nb = video.get("nb_frames")
        if nb not in (None, "", "N/A"):
            try:
                frames = int(nb)
            except ValueError:
                frames = 0
            if frames > 0:
                duration = frames / float(fps)
    if duration is None:
        duration = _positive((data.get("format") or {}).get("duration"))
    if not duration:
        raise RuntimeError(f"concat: clip {index + 1} has no duration")
    return {"path": path, "video": video, "audio": audio, "duration": duration}


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


def _normalize(ffmpeg: str, item: dict, target: TargetFormat, want_audio: bool,
               dest: Path, *, label: str) -> None:
    frames = max(1, int(round(item["duration"] * float(target.fps_fraction))))
    samples = int(round(frames / float(target.fps_fraction) * _AUDIO_RATE))
    fps_arg = _fps_arg(target.fps_fraction)
    video = (
        f"[0:v]scale={target.width}:{target.height}:flags=bilinear,fps={fps_arg},"
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
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-bf", "0",
        "-video_track_timescale", str(_timescale(target.fps_fraction)),
        "-f", "mov", str(dest),
    ]
    _run(cmd, label)


def _concat_line(path: Path) -> str:
    escaped = path.as_posix().replace("'", r"'\''")
    return f"file '{escaped}'\n"


def _concat_copy(ffmpeg: str, parts: list[Path], dest: Path) -> None:
    listing = dest.with_name("list.txt")
    listing.write_text("".join(_concat_line(p) for p in parts), encoding="utf-8")
    _run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c", "copy", str(dest),
        ],
        "concat",
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
