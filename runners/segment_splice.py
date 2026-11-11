"""Cut one video by frame number and join the pieces with ffmpeg_concat.

The join is ``concat_videos``. This module only builds the pieces: a source
span, an optional replacement audio track, and an exact output frame count.
A longer count clones the last picture frame (ripple). A shorter count is a
shorter source span, chosen by the caller. Picture cuts go through ffmpeg so
the frame numbers come from the decoder, not from a keyframe seek.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from fractions import Fraction
from pathlib import Path

import av

from ._media_paths import localize, path_to_view_url
from .ffmpeg_bin import find_ffmpeg
from .ffmpeg_concat import concat_videos
from .ffmpeg_encode import video_encoder_args

_AUDIO_RATE = 44100


def probe_video(path: Path) -> dict:
    """Frame rate, frame count, size, and whether the file has audio.

    Frame count prefers the container. When that field is missing the frames
    are decoded, because a packet count is not a frame count.
    """
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError(f"splice: no video stream in {path}")
        video = container.streams.video[0]
        fps = _rate(video)
        if fps is None:
            raise RuntimeError(f"splice: no frame rate in {path}")
        frames = int(video.frames or 0)
        if frames <= 0:
            frames = sum(1 for _ in container.decode(video))
        audio = container.streams.audio[0] if container.streams.audio else None
        sample_rate = int(audio.sample_rate) if audio is not None and audio.sample_rate else None
        width = int(video.width or 0)
        height = int(video.height or 0)
        has_audio = audio is not None
    if frames <= 0:
        raise RuntimeError(f"splice: no frames in {path}")
    return {
        "fps": fps,
        "frame_count": frames,
        "width": width,
        "height": height,
        "duration_s": frames / float(fps),
        "has_audio": has_audio,
        "audio_rate": sample_rate,
    }


def probe_audio_duration(path: Path) -> float:
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise RuntimeError(f"splice: no audio stream in {path}")
        audio = container.streams.audio[0]
        if audio.duration is not None and audio.time_base:
            seconds = float(audio.duration * audio.time_base)
            if seconds > 0:
                return seconds
        if audio.frames and audio.sample_rate:
            return int(audio.frames) / int(audio.sample_rate)
        if container.duration:
            return container.duration / 1_000_000
    raise RuntimeError(f"splice: no audio duration in {path}")


def splice_pieces(source: Path, dest: Path, pieces: list[dict], *,
                  fade_s: float = 0.0, video_encoder: str | None = None) -> dict:
    """Write ``dest`` from ``pieces`` cut out of ``source``.

    Each piece is ``frame_start``, ``frame_end`` (source, end exclusive),
    ``out_frames``, ``audio`` (a Path, or None to keep the source audio),
    and ``role``. ``dest``'s parent must already exist. The file is created
    only after the join succeeds.
    """
    if not pieces:
        raise RuntimeError("splice: no pieces")
    if dest.exists():
        raise RuntimeError(f"splice: refusing to overwrite {dest}")
    probe = probe_video(source)
    ffmpeg = find_ffmpeg()
    work = dest.parent / f".work-{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True)
    report: dict = {
        "segments": [],
        "video_encoder": None,
        "audio_encoder": None,
    }
    produced: list[Path] = []
    try:
        for index, piece in enumerate(pieces):
            part = work / f"part-{index:02d}.mp4"
            _cut_piece(ffmpeg, source, part, piece, probe, fade_s)
            got = _decoded_frame_count(part)
            if got != int(piece["out_frames"]):
                raise RuntimeError(
                    f"splice: piece {index} has {got} frames, expected {piece['out_frames']}"
                )
            produced.append(part)
            report["segments"].append({
                "role": piece.get("role") or "keep",
                "frame_start": int(piece["frame_start"]),
                "frame_end": int(piece["frame_end"]),
                "out_frames": int(piece["out_frames"]),
                "duration_s": int(piece["out_frames"]) / float(probe["fps"]),
            })
        if len(produced) == 1:
            shutil.copyfile(produced[0], dest)
            report["video_encoder"] = "libx264"
            report["audio_encoder"] = "aac"
        else:
            urls = [path_to_view_url(path) for path in produced]
            concat_report: dict = {}
            joined = concat_videos(
                urls, video_encoder=video_encoder, report=concat_report,
            )
            shutil.copyfile(localize(joined), dest)
            report["video_encoder"] = concat_report.get("video_encoder")
            report["audio_encoder"] = concat_report.get("audio_encoder")
            Path(localize(joined)).unlink(missing_ok=True)
        measured = _measure(dest, probe["fps"])
        report.update(measured)
        return report
    finally:
        shutil.rmtree(work, ignore_errors=True)


def extract_frame_png(source: Path, frame_index: int, dest: Path) -> None:
    """Write one PNG. An existing file is left untouched."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source),
        "-vf", f"select=eq(n\\,{int(frame_index)})",
        "-frames:v", "1",
        str(dest),
    ]
    _run(cmd, f"frame {frame_index}")
    if not dest.is_file():
        raise RuntimeError(f"splice: frame {frame_index} was not written")


def _cut_piece(ffmpeg: str, source: Path, dest: Path, piece: dict,
               probe: dict, fade_s: float) -> None:
    start = int(piece["frame_start"])
    end = int(piece["frame_end"])
    out_frames = int(piece["out_frames"])
    span = end - start
    if span <= 0 or out_frames <= 0:
        raise RuntimeError("splice: empty piece")
    if out_frames < span:
        raise RuntimeError("splice: out_frames is shorter than the source span")
    fps: Fraction = probe["fps"]
    extra_s = (out_frames - span) / float(fps)
    # One extra second of cloned frames is trimmed back to out_frames so a
    # filter that drops or repeats a single frame still lands on the count.
    stop = extra_s + 1.0
    fps_arg = f"{fps.numerator}/{fps.denominator}"
    video = (
        f"[0:v]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,"
        f"fps={fps_arg}:eof_action=pass,setsar=1,format=yuv420p,"
        f"tpad=stop_mode=clone:stop_duration={stop:.6f},trim=end_frame={out_frames},"
        f"setpts=PTS-STARTPTS[v]"
    )
    duration = out_frames / float(fps)
    audio_path = piece.get("audio")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source)]
    if audio_path:
        cmd.append("-i")
        cmd.append(str(audio_path))
        fade = min(float(fade_s or 0), duration / 4) if duration > 0 else 0.0
        fades = ""
        if fade > 0:
            out_at = max(0.0, duration - fade)
            fades = (
                f",afade=t=in:st=0:d={fade:.6f},"
                f"afade=t=out:st={out_at:.6f}:d={fade:.6f}"
            )
        audio = (
            f"[1:a]aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"aresample={_AUDIO_RATE},apad,atrim=end={duration:.6f}"
            f"{fades},asetpts=PTS-STARTPTS[a]"
        )
    elif probe["has_audio"]:
        t0 = start / float(fps)
        t1 = end / float(fps)
        audio = (
            f"[0:a]atrim=start={t0:.6f}:end={t1:.6f},asetpts=PTS-STARTPTS,"
            f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"aresample={_AUDIO_RATE},apad,atrim=end={duration:.6f},"
            f"asetpts=PTS-STARTPTS[a]"
        )
    else:
        pad = f"{duration + 0.5:.3f}"
        cmd += [
            "-f", "lavfi", "-t", pad, "-i",
            f"anullsrc=channel_layout=stereo:sample_rate={_AUDIO_RATE}",
        ]
        audio = (
            f"[1:a]atrim=end={duration:.6f},asetpts=PTS-STARTPTS[a]"
        )
    cmd += [
        "-filter_complex", video + ";" + audio,
        "-map", "[v]", "-map", "[a]",
        *video_encoder_args("libx264"),
        "-c:a", "aac", "-b:a", "192k", "-ar", str(_AUDIO_RATE), "-ac", "2",
        str(dest),
    ]
    _run(cmd, f"cut {start}:{end}")


def _measure(path: Path, fps: Fraction) -> dict:
    frames = _decoded_frame_count(path)
    picture_s = frames / float(fps)
    audio_s = _decoded_audio_seconds(path)
    return {
        "output_frames": frames,
        "output_video_s": picture_s,
        "output_audio_s": audio_s,
        "av_delta_s": None if audio_s is None else audio_s - picture_s,
    }


def _decoded_frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        return sum(1 for _ in container.decode(video))


def _decoded_audio_seconds(path: Path) -> float | None:
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        rate = int(stream.sample_rate or _AUDIO_RATE)
        samples = 0
        for frame in container.decode(stream):
            samples += int(frame.samples or 0)
    if rate <= 0:
        return None
    return samples / rate


def _rate(video) -> Fraction | None:
    for attr in ("average_rate", "base_rate", "guessed_rate"):
        raw = getattr(video, attr, None)
        if not raw:
            continue
        try:
            rate = Fraction(raw.numerator, raw.denominator)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return rate
    return None


def _run(cmd: list[str], what: str) -> None:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-1200:]
        raise RuntimeError(f"splice: {what} failed: {tail}")

