"""Locate the ffmpeg binary ComfyUI can actually execute.

Search order, first hit wins:

1. ``COMFYTV_FFMPEG`` (absolute path, or a name on ``PATH``)
2. ``shutil.which("ffmpeg")``
3. ``imageio_ffmpeg.get_ffmpeg_exe()`` (the wheel ships ffmpeg and not ffprobe)

A hit is cached for that environment and ``PATH``. ``imageio_ffmpeg`` is
optional: a failed import is one of the checked places, not a crash.
"""
from __future__ import annotations

import os
import shutil
import threading

try:
    import imageio_ffmpeg as _imageio_ffmpeg
except ImportError:  # optional; discovery falls through to the error below
    _imageio_ffmpeg = None

_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple[str, str], str] = {}


def clear_ffmpeg_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def usable_executable(raw: str | None) -> str | None:
    """Absolute path of an executable file, or None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    expanded = os.path.expanduser(text)
    if os.path.isabs(expanded) or os.path.isfile(expanded):
        candidate = expanded
    else:
        found = shutil.which(expanded)
        if not found:
            return None
        candidate = found
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return os.path.abspath(candidate)
    return None


def find_ffmpeg() -> str:
    env_raw = os.environ.get("COMFYTV_FFMPEG", "")
    path_raw = os.environ.get("PATH", "")
    key = (env_raw, path_raw)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached
    found, attempts = _search_ffmpeg()
    if found is None:
        raise RuntimeError(_missing_ffmpeg_message(attempts))
    with _CACHE_LOCK:
        _CACHE[key] = found
    return found


def _search_ffmpeg() -> tuple[str | None, list[str]]:
    attempts: list[str] = []
    env_raw = os.environ.get("COMFYTV_FFMPEG", "").strip()
    if not env_raw:
        attempts.append("COMFYTV_FFMPEG: not set")
    else:
        env_exe = usable_executable(env_raw)
        if env_exe:
            return env_exe, attempts
        attempts.append(f"COMFYTV_FFMPEG={env_raw}: not an executable file")

    which = shutil.which("ffmpeg")
    which_exe = usable_executable(which) if which else None
    if which_exe:
        return which_exe, attempts
    if which:
        attempts.append(
            f"PATH: shutil.which('ffmpeg') returned {which}, which is not executable"
        )
    else:
        attempts.append("PATH: shutil.which('ffmpeg') found nothing")

    bundled, note = _bundled_ffmpeg()
    attempts.append(note)
    if bundled:
        return bundled, attempts
    return None, attempts


def _bundled_ffmpeg() -> tuple[str | None, str]:
    if _imageio_ffmpeg is None:
        return None, "imageio_ffmpeg.get_ffmpeg_exe(): imageio_ffmpeg is not installed"
    try:
        raw = _imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        return None, f"imageio_ffmpeg.get_ffmpeg_exe(): failed ({exc})"
    exe = usable_executable(None if raw is None else str(raw))
    if not exe:
        return None, (
            f"imageio_ffmpeg.get_ffmpeg_exe(): returned {raw!r}, "
            "which is not an executable file"
        )
    return exe, f"imageio_ffmpeg.get_ffmpeg_exe(): {exe}"


def _missing_ffmpeg_message(attempts: list[str]) -> str:
    checked = "\n".join(f"- {item}" for item in attempts)
    return (
        "ffmpeg not found. Checked:\n"
        f"{checked}\n"
        "Set COMFYTV_FFMPEG to the absolute path of an ffmpeg binary, "
        "then restart ComfyUI."
    )
