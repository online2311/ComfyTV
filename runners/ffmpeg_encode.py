"""Choose h264_nvenc or libx264 for a concat re-encode.

``auto`` lists encoders, then encodes a few lavfi frames. That result is
cached for the ffmpeg binary for the life of the process. ``cpu`` never
probes. ``nvenc`` raises with ffmpeg's own stderr instead of falling back.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading

_log = logging.getLogger(__name__)

_MODES = ("auto", "nvenc", "cpu")
_LOCK = threading.Lock()
_CACHE: dict[str, tuple[bool, str]] = {}
_PROBE_TIMEOUT_S = 20


def clear_encoder_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def resolve_video_encoder_mode(node_value: str | None = None) -> str:
    """Node ``nvenc``/``cpu`` wins. ``auto`` (the default) reads the env var."""
    node = (node_value or "auto").strip().lower()
    if node not in _MODES:
        raise RuntimeError(
            f"video_encoder={node_value!r} is not one of auto, nvenc, cpu"
        )
    if node != "auto":
        return node
    env = os.environ.get("COMFYTV_VIDEO_ENCODER", "auto").strip().lower() or "auto"
    if env not in _MODES:
        raise RuntimeError(
            f"COMFYTV_VIDEO_ENCODER={env!r} is not one of auto, nvenc, cpu"
        )
    return env


def video_encoder_args(encoder: str) -> list[str]:
    if encoder == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-rc", "vbr",
            "-cq", "19",
            "-b:v", "0",
            "-bf", "0",
            "-pix_fmt", "yuv420p",
        ]
    if encoder == "libx264":
        return [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-bf", "0",
            "-pix_fmt", "yuv420p",
        ]
    raise RuntimeError(f"concat: unknown video encoder {encoder!r}")


def select_video_encoder(ffmpeg: str, mode: str) -> str:
    chosen = resolve_video_encoder_mode(mode)
    if chosen == "cpu":
        return "libx264"
    ok, reason = _probe_nvenc(ffmpeg)
    if ok:
        return "h264_nvenc"
    if chosen == "nvenc":
        raise RuntimeError(
            "concat: video encoder nvenc was requested but h264_nvenc failed:\n"
            + reason
        )
    _log.warning("concat: h264_nvenc unavailable, using libx264: %s", reason)
    return "libx264"


def _probe_nvenc(ffmpeg: str) -> tuple[bool, str]:
    with _LOCK:
        cached = _CACHE.get(ffmpeg)
    if cached is not None:
        return cached
    result = _probe_nvenc_uncached(ffmpeg)
    with _LOCK:
        _CACHE[ffmpeg] = result
    return result


def _encoder_listed(text: str, name: str) -> bool:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == name:
            return True
    return False


def _probe_nvenc_uncached(ffmpeg: str) -> tuple[bool, str]:
    try:
        listed = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, "ffmpeg -encoders timed out"
    listing = f"{listed.stdout or ''}\n{listed.stderr or ''}"
    if not _encoder_listed(listing, "h264_nvenc"):
        return False, "h264_nvenc is not listed by ffmpeg -encoders"
    try:
        trial = subprocess.run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24:duration=0.04",
                *video_encoder_args("h264_nvenc"),
                "-an", "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, "h264_nvenc probe timed out"
    if trial.returncode != 0:
        reason = (trial.stderr or trial.stdout or "h264_nvenc encode failed").strip()
        return False, (reason or "h264_nvenc encode failed")[-2000:]
    return True, ""
