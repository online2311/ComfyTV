"""Segment reshoot records for one shot.

Files live under the ComfyUI input directory at
``<input>/comfytv/projects/<project_id>/`` (see ``storage.asset_root``).
Finished media is a new file under the ComfyUI output directory so ``/view``
can serve it. The database backup directory is not used for these records.

Nothing here writes a legacy project tree. An imported take keeps its
shot id, take number, prompt id, legacy path, and sha256. A reshoot appends
``take_<n>/r<k>`` and a new mp4. Paths registered as protected (delivered
rough cuts included) are never opened for writing, and a file whose name
matches ``rough_cut*.mp4`` is never a reshoot output.

``regen`` and ``lipsync`` are refused. Those modes belong to a later change
that can reuse the episode-batch workflows.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import folder_paths

from ..runners._media_paths import localize, path_to_view_url
from ..runners.segment_splice import (
    extract_frame_png,
    probe_audio_duration,
    probe_video,
    splice_pieces,
)
from .asset_root import project_dir

APPROVER = "张景"
SCHEMA = "comfytv/reshoot.v1"
TAKES_SCHEMA = "comfytv/takes.v1"
PROTECTED_SCHEMA = "comfytv/protected.v1"

DEFAULT_HANDLES_S = 0.5
DEFAULT_LENGTH_POLICY = "fit"
DEFAULT_ALLOW_MULTIPLE = False
DEFAULT_FADE_MS = 15
MODES = ("audio_only", "lipsync", "regen")
POLICIES = ("fit", "ripple")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
_RESHOOT_RE = re.compile(
    r"^(?P<shot>[A-Za-z0-9][A-Za-z0-9_.-]*)/take_(?P<take>[0-9]+)/r(?P<k>[0-9]+)$"
)
_ROUGH_CUT_RE = re.compile(r"^rough_cut.*\.mp4$", re.IGNORECASE)
_UPSCALE_RE = re.compile(r"relay|vsr|upscale|superres|superscale", re.IGNORECASE)
_IDENTITY_KEYS = ("sha256", "prompt_id", "legacy_path", "url", "take", "shot_id")
_LOCK = threading.RLock()
_HISTORY_SEQ = 0


class ReshootError(ValueError):
    def __init__(self, message: str, *, status: int = 400):
        super().__init__(message)
        self.http_status = status


def project_namespace(project_id: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"comfytv:project:{project_id}")


def prompt_id_for(project_id: str, key: str) -> str:
    """uuid5 of ``key`` in the project namespace.

    New reshoots use ``<shot_id>/take_<n>/r<k>``. An imported take keeps the
    prompt id it was registered with; this function is not applied to it.
    """
    return str(uuid.uuid5(project_namespace(project_id), key))


def grid_frames(need: int) -> int:
    """Smallest 17n+5 count that can hold ``need`` frames.

    Same rounding as ``frames_for`` in the shot document: ``n`` is
    ``max(0, -(-(need - 5) // 17))``.
    """
    n = max(0, -(-(int(need) - 5) // 17))
    return 17 * n + 5


def align_window(t_start: float, t_end: float, fps: Fraction, n_frames: int, *,
                 handles_s: float = DEFAULT_HANDLES_S) -> dict:
    """Snap ``[t_start, t_end)`` to source frames and record the 17n+5 plan.

    The splice uses ``frame_start`` / ``frame_end``. ``gen_frames`` is the
    length a later picture regen would ask for: the snapped window, plus
    ``handles_s`` on each side, then rounded up onto the 17n+5 grid. Audio
    replacement does not generate those extra frames.
    """
    if n_frames <= 0 or fps <= 0:
        raise ReshootError("source has no frames")
    try:
        start_s = float(t_start)
        end_s = float(t_end)
        handles = float(handles_s)
    except (TypeError, ValueError) as exc:
        raise ReshootError("t_start, t_end, and handles_s must be numbers") from exc
    if handles < 0:
        raise ReshootError("handles_s must be >= 0")
    if end_s <= start_s:
        raise ReshootError("window length must be > 0")
    duration = n_frames / float(fps)
    slack = 0.5 / float(fps)
    if start_s < -slack or end_s > duration + slack:
        raise ReshootError(
            f"window [{start_s}, {end_s}) is outside the source "
            f"({duration:.6f}s, {n_frames} frames)"
        )
    frame_start = int(round(start_s * float(fps)))
    frame_end = int(round(end_s * float(fps)))
    frame_start = min(max(frame_start, 0), n_frames)
    frame_end = min(max(frame_end, 0), n_frames)
    if frame_end <= frame_start:
        raise ReshootError("aligned window is empty")
    handle_frames = int(round(handles * float(fps)))
    gen_need = (frame_end - frame_start) + (2 * handle_frames)
    gen = grid_frames(gen_need)
    aligned_start = frame_start / float(fps)
    aligned_end = frame_end / float(fps)
    warnings: list[str] = []
    if gen != gen_need:
        warnings.append(
            f"generation length {gen_need} frames rounds up to {gen} (17n+5)"
        )
    if frame_start < handle_frames:
        warnings.append(
            "the leading handle runs past the start of the file; "
            "a later regen would not have a source frame there"
        )
    if frame_end + handle_frames > n_frames:
        warnings.append(
            "the trailing handle runs past the end of the file; "
            "a later regen would not have a source frame there"
        )
    return {
        "t_start": aligned_start,
        "t_end": aligned_end,
        "input_t_start": start_s,
        "input_t_end": end_s,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "handles_s": handles,
        "handle_frames": handle_frames,
        "gen_need_frames": gen_need,
        "gen_frames": gen,
        "delta_s": {
            "t_start": aligned_start - start_s,
            "t_end": aligned_end - end_s,
        },
        "warnings": warnings,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_imported_version(project_id: str, shot_id: str, *, take: int,
                              sha256: str, prompt_id: str | None = None,
                              legacy_path: str | None = None,
                              url: str | None = None) -> dict:
    """Append one existing take. Does not copy or open the media.

    A second call with the same identity is a no-op. A different sha256,
    prompt id, legacy path, or url is refused.
    """
    project = _require_id(project_id, "project_id")
    shot = _require_id(shot_id, "shot_id")
    number = _require_take(take)
    digest = str(sha256 or "").strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ReshootError("sha256 must be 64 hex characters")
    version = {
        "id": f"take_{number}",
        "take": number,
        "parent": None,
        "role": "imported",
        "origin": "imported",
        "shot_id": shot,
        "prompt_id": _optional_text(prompt_id),
        "sha256": digest,
        "legacy_path": _optional_text(legacy_path),
        "url": _optional_text(url),
        "immutable": True,
        "reshoot_id": None,
    }
    with _LOCK:
        index = _load_takes(project, shot)
        if index is None:
            index = _empty_takes(project, shot, active=None)
        _merge_version(index, version)
        _save_takes(project, shot, index)
        return _public_takes(index)


def register_protected(project_id: str, *, kind: str, legacy_path: str,
                       sha256: str | None = None, url: str | None = None) -> dict:
    """Remember a file reshoot must not write. Does not open the file."""
    project = _require_id(project_id, "project_id")
    label = str(kind or "").strip()
    legacy = str(legacy_path or "").strip()
    if not label or not legacy:
        raise ReshootError("kind and legacy_path are required")
    if legacy.startswith("/") or ".." in Path(legacy).parts:
        raise ReshootError("legacy_path must be a relative path without ..")
    digest = _optional_text(sha256)
    if digest is not None:
        digest = digest.lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ReshootError("sha256 must be 64 hex characters")
    entry = {
        "kind": label,
        "legacy_path": legacy.replace("\\", "/"),
        "sha256": digest,
        "url": _optional_text(url),
        "immutable": True,
        "rewrite": False,
    }
    with _LOCK:
        doc = _load_protected(project)
        found = next((row for row in doc["entries"] if row["legacy_path"] == entry["legacy_path"]), None)
        if found is None:
            doc["entries"].append(entry)
        else:
            for key in ("kind", "sha256", "url"):
                old = found.get(key)
                new = entry.get(key)
                if _blank(new) or _blank(old) or old == new:
                    continue
                raise ReshootError(
                    f"refusing to overwrite protected {entry['legacy_path']} field {key}"
                )
        _save_protected(project, doc)
        return json.loads(json.dumps(doc))


def _project_id(args: dict) -> str:
    if "project_id" not in args or not str(args.get("project_id") or "").strip():
        raise ReshootError("project_id is required")
    return _require_id(str(args["project_id"]), "project_id")


def plan(args: dict) -> dict:
    with _LOCK:
        return _plan(args)


def run(args: dict) -> dict:
    with _LOCK:
        if args.get("reshoot_id"):
            record = _require_record(args)
        else:
            record = _plan(args)
        return _execute(record)


def status(args: dict) -> dict:
    with _LOCK:
        return _public_record(_require_record(args))


def list_reshoots(args: dict) -> dict:
    project = _project_id(args)
    shot = args.get("shot_id")
    with _LOCK:
        protected = _load_protected(project)
        if shot:
            shot_id = _require_id(str(shot), "shot_id")
            return {
                "project_id": project,
                "shot_id": shot_id,
                "takes": _public_takes(_load_takes(project, shot_id) or _empty_takes(project, shot_id, None)),
                "reshoots": [_public_record(row) for row in _load_reshoots(project, shot_id)],
                "protected": protected["entries"],
            }
        shots = []
        root = _project_dir(project) / "reshoots"
        if root.is_dir():
            for name in sorted(path.name for path in root.iterdir() if path.is_dir()):
                index = _load_takes(project, name)
                shots.append({
                    "shot_id": name,
                    "active": None if index is None else index.get("active"),
                    "version_ids": [] if index is None else [row["id"] for row in index["versions"]],
                })
        return {
            "project_id": project,
            "shots": shots,
            "protected": protected["entries"],
        }


def adopt(args: dict) -> dict:
    with _LOCK:
        _require_approver(args.get("approved_by"))
        record = _require_record(args)
        if record["status"] not in ("done", "adopted"):
            raise ReshootError(f"reshoot {record['reshoot_id']} is {record['status']}, not done")
        return _point_at(
            record["project_id"], record["shot_id"], record["version_id"],
            approved_by=APPROVER, adopted_id=record["reshoot_id"],
            cut_to=args.get("cut_to"), next_shot_id=args.get("next_shot_id"),
        )


def rollback(args: dict) -> dict:
    with _LOCK:
        _require_approver(args.get("approved_by"))
        project = _project_id(args)
        shot = _require_id(str(args.get("shot_id") or ""), "shot_id")
        target = str(args.get("to") or "").strip()
        if not target:
            raise ReshootError("to is required")
        index = _load_takes(project, shot)
        if index is None or not any(row["id"] == target for row in index["versions"]):
            raise ReshootError(f"version {target} is not on {shot}", status=404)
        version = next(row for row in index["versions"] if row["id"] == target)
        if version.get("reshoot_id"):
            record = _read_record(project, shot, version["reshoot_id"])
            if record["status"] not in ("done", "adopted", "rolled_back"):
                raise ReshootError(f"{target} has no finished file")
        return _point_at(
            project, shot, target, approved_by=APPROVER, adopted_id=version.get("reshoot_id"),
        )


def _plan(args: dict) -> dict:
    mode = str(args.get("mode") or "").strip()
    _reject_unavailable_mode(mode)
    if mode != "audio_only":
        raise ReshootError("mode must be audio_only, lipsync, or regen")
    project = _project_id(args)
    source = _source_args(args)
    shot = _require_id(str(source.get("shot_id") or args.get("shot_id") or ""), "shot_id")
    take = _require_take(source.get("take", 1))
    policy = str(args.get("length_policy") or DEFAULT_LENGTH_POLICY).strip()
    if policy not in POLICIES:
        raise ReshootError("length_policy must be fit or ripple")
    handles = args.get("handles_s", DEFAULT_HANDLES_S)
    allow_multiple = args.get("allow_multiple", DEFAULT_ALLOW_MULTIPLE)
    if not isinstance(allow_multiple, bool):
        allow_multiple = DEFAULT_ALLOW_MULTIPLE
    profile = str(source.get("profile") or "")
    if _UPSCALE_RE.search(profile):
        raise ReshootError(
            "this source is an upscaled take; reshoot the pre-upscale take, then upscale again"
        )
    src_path = _resolve_media(str(source.get("url") or ""), "source")
    probed = probe_video(src_path)
    fps: Fraction = probed["fps"]
    raw_windows = _raw_windows(args, allow_multiple)
    changes = args.get("changes") if isinstance(args.get("changes"), dict) else {}
    if changes.get("dialogue") and not args.get("audio") and not any(
        isinstance(item, dict) and item.get("audio") for item in raw_windows
    ):
        raise ReshootError(
            "dialogue was changed but no audio file was passed. "
            "Voice cloning is not in this build; pass the replacement audio."
        )
    if len(raw_windows) > 1 and any(not item.get("audio") for item in raw_windows):
        raise ReshootError("each window needs its own audio when more than one window is set")
    aligned = []
    warnings: list[str] = []
    for raw in raw_windows:
        window = align_window(
            raw["t_start"], raw["t_end"], fps, probed["frame_count"], handles_s=handles,
        )
        warnings.extend(window["warnings"])
        audio_url = raw.get("audio") or args.get("audio")
        audio_path = _resolve_media(str(audio_url or ""), "audio")
        audio_s = probe_audio_duration(audio_path)
        window_s = (window["frame_end"] - window["frame_start"]) / float(fps)
        if policy == "fit" and audio_s > window_s + (1 / float(fps)):
            raise ReshootError(
                f"audio is {audio_s:.3f}s and the aligned window is {window_s:.3f}s; "
                "length_policy 'fit' refuses to spill past the window. "
                "Pass length_policy 'ripple' to let this segment change length "
                "and shift everything after it."
            )
        if policy == "fit" and audio_s + (1 / float(fps)) < window_s:
            warnings.append(
                f"audio is {audio_s:.3f}s inside a {window_s:.3f}s window; "
                "the rest of the window is silence"
            )
        audio_sha = file_sha256(audio_path)
        window["audio"] = {"url": _media_ref(audio_path, str(audio_url)), "sha256": audio_sha}
        window["audio_duration_s"] = audio_s
        aligned.append(window)
    aligned.sort(key=lambda row: (int(row["frame_start"]), int(row["frame_end"])))
    _reject_overlaps(aligned)
    warnings.append(
        "audio_only replaces sound inside the aligned window and keeps the picture. "
        "gen_frames includes handles and the 17n+5 grid for a later regen; "
        "this run does not generate those frames."
    )
    recommendation = "audio_only"
    dialogue = changes.get("dialogue")
    if isinstance(dialogue, list) and any(
        isinstance(line, dict) and line.get("lips_visible") is False for line in dialogue
    ):
        recommendation = "audio_only"
        warnings.append("lips_visible is false; audio_only is the recommended mode")
    source_sha = file_sha256(src_path)
    source_url = _media_ref(src_path, str(source.get("url")))
    supplied_prompt = _optional_text(source.get("prompt_id"))
    index = _load_takes(project, shot)
    if index is None:
        index = _empty_takes(project, shot, active=f"take_{take}")
    _merge_version(index, {
        "id": f"take_{take}",
        "take": take,
        "parent": None,
        "role": "source",
        "origin": "source",
        "shot_id": shot,
        "prompt_id": supplied_prompt,
        "sha256": source_sha,
        "legacy_path": _optional_text(source.get("legacy_path")),
        "url": source_url,
        "immutable": True,
        "reshoot_id": None,
    })
    k = _next_revision(project, shot, index, take)
    reshoot_id = f"{shot}/take_{take}/r{k}"
    version_id = f"take_{take}/r{k}"
    prompt_id = prompt_id_for(project, reshoot_id)
    previews = _previews(
        args, src_path, project, shot, aligned[0], probed["frame_count"],
        source_sha, warnings,
    )
    now = _now()
    record = {
        "schema": SCHEMA,
        "reshoot_id": reshoot_id,
        "version_id": version_id,
        "project_id": project,
        "shot_id": shot,
        "source": {
            "url": source_url,
            "sha256": source_sha,
            "take": take,
            "prompt_id": supplied_prompt,
            "legacy_path": _optional_text(source.get("legacy_path")),
            "fps": float(fps),
            "fps_rational": f"{fps.numerator}/{fps.denominator}",
            "frame_count": probed["frame_count"],
            "width": probed["width"],
            "height": probed["height"],
            "duration_s": probed["duration_s"],
            "audio_rate": probed["audio_rate"],
            "profile": profile,
        },
        "window": _window_public(aligned[0]),
        "windows": [_window_public(row) for row in aligned],
        "mode": "audio_only",
        "workflow": _optional_text(args.get("workflow")),
        "length_policy": policy,
        "allow_multiple": allow_multiple,
        "audio_fade_ms": _fade_ms(args),
        "changes": changes,
        "inputs": {
            "audio": aligned[0]["audio"],
            "first_frame": previews.get("first_frame"),
            "last_frame": previews.get("last_frame"),
        },
        "jobs": [{
            "kind": "splice",
            "workflow": "runners/ffmpeg_concat.py",
            "prompt_id": prompt_id,
            "seed": None,
            "status": "planned",
            "started_at": None,
            "finished_at": None,
        }],
        "output": None,
        "splice_report": None,
        "prompt_id": prompt_id,
        "recommendation": recommendation,
        "warnings": warnings,
        "stale": [],
        "status": "planned",
        "adopted_by": None,
        "adopted_at": None,
        "created_by": _optional_text(args.get("created_by")) or "",
        "created_at": now,
        "storage": _storage_note(project, shot, version_id),
    }
    _assert_destination_allowed(project, _media_path(project, shot, version_id))
    _save_takes(project, shot, index)
    _write_json(_record_path(project, shot, reshoot_id), record)
    return _public_record(record)


def _execute(record: dict) -> dict:
    if record["status"] in ("done", "adopted", "rolled_back") and record.get("output"):
        return _public_record(record)
    if record["mode"] != "audio_only":
        _reject_unavailable_mode(record["mode"])
    project = record["project_id"]
    shot = record["shot_id"]
    dest = _media_path(project, shot, record["version_id"])
    _assert_destination_allowed(project, dest)
    if dest.exists():
        raise ReshootError(f"refusing to overwrite {dest.name}")
    record["status"] = "running"
    record["jobs"][0]["status"] = "running"
    record["jobs"][0]["started_at"] = _now()
    _write_json(_record_path(project, shot, record["reshoot_id"]), record)
    try:
        src = _resolve_media(record["source"]["url"], "source")
        if file_sha256(src) != record["source"]["sha256"]:
            raise ReshootError("source file sha256 no longer matches the plan")
        probed = probe_video(src)
        fps = Fraction(record["source"]["fps_rational"])
        pieces = _pieces(record, probed["frame_count"], fps)
        fade_s = float(record.get("audio_fade_ms") or 0) / 1000
        dest.parent.mkdir(parents=True, exist_ok=True)
        report = splice_pieces(src, dest, pieces, fade_s=fade_s, video_encoder=None)
    except Exception as exc:
        record["status"] = "failed"
        record["jobs"][0]["status"] = "failed"
        record["jobs"][0]["finished_at"] = _now()
        record["error"] = str(exc)
        _write_json(_record_path(project, shot, record["reshoot_id"]), record)
        if isinstance(exc, ReshootError):
            raise
        raise ReshootError(str(exc)) from exc
    output_sha = file_sha256(dest)
    record["output"] = {"url": path_to_view_url(dest), "sha256": output_sha}
    record["splice_report"] = report
    record["status"] = "done"
    record["jobs"][0]["status"] = "done"
    record["jobs"][0]["finished_at"] = _now()
    record.pop("error", None)
    index = _load_takes(project, shot)
    if index is None:
        index = _empty_takes(project, shot, active=f"take_{record['source']['take']}")
    parent = f"take_{record['source']['take']}"
    _merge_version(index, {
        "id": record["version_id"],
        "take": record["source"]["take"],
        "parent": parent,
        "role": "reshoot",
        "origin": "reshoot",
        "shot_id": shot,
        "prompt_id": record["prompt_id"],
        "sha256": output_sha,
        "legacy_path": None,
        "url": record["output"]["url"],
        "immutable": True,
        "reshoot_id": record["reshoot_id"],
    })
    _save_takes(project, shot, index)
    _write_json(_record_path(project, shot, record["reshoot_id"]), record)
    return _public_record(record)


def _pieces(record: dict, n_frames: int, fps: Fraction) -> list[dict]:
    pieces: list[dict] = []
    cursor = 0
    windows = sorted(record["windows"], key=lambda row: int(row["frame_start"]))
    for window in windows:
        start = int(window["frame_start"])
        end = int(window["frame_end"])
        if start < cursor:
            raise ReshootError("windows are out of order")
        if start > cursor:
            pieces.append(_keep(cursor, start))
        span = end - start
        audio = _checked_audio(record, window, fps)
        if record["length_policy"] == "ripple":
            audio_s = probe_audio_duration(audio)
            out_frames = max(1, int(round(audio_s * float(fps))))
        else:
            out_frames = span
        src_end = end
        if out_frames < span:
            src_end = start + out_frames
        pieces.append({
            "frame_start": start,
            "frame_end": src_end,
            "out_frames": out_frames,
            "audio": audio,
            "role": "replace",
        })
        cursor = end
    if cursor < n_frames:
        pieces.append(_keep(cursor, n_frames))
    return [piece for piece in pieces if piece["out_frames"] > 0]


def _checked_audio(record: dict, window: dict, fps: Fraction) -> Path:
    audio = _resolve_media(window["audio"]["url"], "audio")
    expected = str((window.get("audio") or {}).get("sha256") or "")
    actual = file_sha256(audio)
    if expected and actual != expected:
        raise ReshootError("replacement audio sha256 no longer matches the plan")
    start = int(window["frame_start"])
    end = int(window["frame_end"])
    window_s = (end - start) / float(fps)
    if record["length_policy"] == "fit":
        audio_s = probe_audio_duration(audio)
        if audio_s > window_s + (1 / float(fps)):
            raise ReshootError(
                f"audio is {audio_s:.3f}s and the planned window is {window_s:.3f}s; "
                "length_policy 'fit' refuses to spill past the window"
            )
    return audio


def _keep(start: int, end: int) -> dict:
    return {
        "frame_start": start,
        "frame_end": end,
        "out_frames": end - start,
        "audio": None,
        "role": "keep",
    }


def _point_at(project: str, shot: str, version_id: str, *, approved_by: str,
              adopted_id: str | None, cut_to=None, next_shot_id=None) -> dict:
    index = _load_takes(project, shot)
    if index is None:
        raise ReshootError(f"no takes for {shot}", status=404)
    previous = index.get("active")
    stale = _selection_stale(
        project, shot, index, version_id, previous, cut_to=cut_to, next_shot_id=next_shot_id,
    )
    index["active"] = version_id
    index["active_approved_by"] = approved_by
    index["active_at"] = _now()
    index["stale"] = stale
    _save_takes(project, shot, index)
    if adopted_id:
        record = _read_record(project, shot, adopted_id)
        record["status"] = "adopted"
        record["adopted_by"] = approved_by
        record["adopted_at"] = index["active_at"]
        record["stale"] = stale
        _write_json(_record_path(project, shot, adopted_id), record)
    if previous and previous != version_id:
        previous_row = next((row for row in index["versions"] if row["id"] == previous), None)
        if previous_row and previous_row.get("reshoot_id") and previous_row["reshoot_id"] != adopted_id:
            other = _read_record(project, shot, previous_row["reshoot_id"])
            if other["status"] == "adopted":
                other["status"] = "rolled_back"
                _write_json(_record_path(project, shot, other["reshoot_id"]), other)
    return {
        "project_id": project,
        "shot_id": shot,
        "active": version_id,
        "approved_by": approved_by,
        "stale": stale,
        "takes": _public_takes(_load_takes(project, shot)),
    }


def _selection_stale(project: str, shot: str, index: dict, version_id: str,
                     previous: str | None, *, cut_to, next_shot_id) -> list[dict]:
    """Invalidations for a pointer change. Raises before any file is written."""
    if cut_to is not None and (isinstance(cut_to, bool) or not isinstance(cut_to, (int, float))):
        raise ReshootError("cut_to must be a number of seconds")
    rows = [
        {"kind": kind, "shot_id": shot, "stale": True, "rewrite": False}
        for kind in ("subtitles", "rough_cut", "upscale")
    ]
    target = next((row for row in index["versions"] if row["id"] == version_id), None)
    records = []
    if target and target.get("reshoot_id"):
        records.append(_read_record(project, shot, target["reshoot_id"]))
    if previous and previous != version_id:
        previous_row = next((row for row in index["versions"] if row["id"] == previous), None)
        if previous_row and previous_row.get("reshoot_id"):
            records.append(_read_record(project, shot, previous_row["reshoot_id"]))
    covered = False
    frame = None
    for record in records:
        hit, frame = _window_covers(record, cut_to)
        covered = covered or hit
    if covered:
        rows.append({
            "kind": "prev_last_frame",
            "shot_id": _optional_text(next_shot_id),
            "frame": frame,
            "stale": True,
            "rewrite": False,
        })
    return rows


def _window_covers(record: dict, cut_to) -> tuple[bool, int]:
    fps = Fraction(record["source"]["fps_rational"])
    n_frames = int(record["source"]["frame_count"])
    windows = record.get("windows") or [record["window"]]
    if cut_to is None:
        frame = n_frames - 1
        covers = any(int(window["frame_end"]) >= n_frames for window in windows)
        return covers, frame
    if isinstance(cut_to, bool):
        raise ReshootError("cut_to must be a number of seconds")
    try:
        cut_s = float(cut_to)
    except (TypeError, ValueError) as exc:
        raise ReshootError("cut_to must be a number of seconds") from exc
    frame = max(0, int(cut_s * float(fps)) - 1)
    covers = any(
        int(window["frame_start"]) <= frame < int(window["frame_end"])
        for window in windows
    )
    return covers, frame


def _previews(args, src: Path, project: str, shot: str, window: dict,
              n_frames: int, source_sha: str, warnings: list[str]) -> dict:
    if args.get("previews") is False:
        return {}
    start = int(window["frame_start"])
    end = int(window["frame_end"])
    before = start - 1 if start > 0 else None
    after = end if end < n_frames else None
    out: dict = {}
    folder = _media_dir(project, shot) / "previews" / source_sha[:12]
    try:
        if before is not None:
            dest = folder / f"f{before:06d}.png"
            extract_frame_png(src, before, dest)
            out["first_frame"] = {"url": path_to_view_url(dest), "sha256": file_sha256(dest), "frame": before}
        else:
            warnings.append("window starts at frame 0; there is no leading frame")
        if after is not None:
            dest = folder / f"f{after:06d}.png"
            extract_frame_png(src, after, dest)
            out["last_frame"] = {"url": path_to_view_url(dest), "sha256": file_sha256(dest), "frame": after}
        else:
            warnings.append("window ends on the last frame; there is no trailing frame")
    except (RuntimeError, OSError) as exc:
        warnings.append(f"frame preview was not written: {exc}")
    return out


def _window_public(window: dict) -> dict:
    keys = (
        "t_start", "t_end", "input_t_start", "input_t_end",
        "frame_start", "frame_end", "handles_s", "handle_frames",
        "gen_need_frames", "gen_frames", "delta_s", "audio", "audio_duration_s",
    )
    return {key: window[key] for key in keys if key in window}


def _raw_windows(args: dict, allow_multiple: bool) -> list[dict]:
    raw = args.get("windows")
    if raw is None:
        if "t_start" not in args or "t_end" not in args:
            raise ReshootError("t_start and t_end are required")
        return [{"t_start": args["t_start"], "t_end": args["t_end"], "audio": args.get("audio")}]
    if not isinstance(raw, list) or not raw:
        raise ReshootError("windows must be a non-empty list")
    if len(raw) > 1 and not allow_multiple:
        raise ReshootError(
            "multiple windows are off by default (D1.4); pass allow_multiple=true"
        )
    rows = []
    for item in raw:
        if not isinstance(item, dict) or "t_start" not in item or "t_end" not in item:
            raise ReshootError("each window needs t_start and t_end")
        rows.append(item)
    return rows


def _reject_overlaps(windows: list[dict]) -> None:
    ordered = sorted(windows, key=lambda row: int(row["frame_start"]))
    for left, right in zip(ordered, ordered[1:]):
        if int(left["frame_end"]) > int(right["frame_start"]):
            raise ReshootError("windows overlap")


def _reject_unavailable_mode(mode: str) -> None:
    if mode == "lipsync":
        raise ReshootError(
            "lipsync has no backend in this build (F1-b, waiting on the lip-sync "
            "adapter). Refusing to switch to another mode."
        )
    if mode == "regen":
        raise ReshootError(
            "regen is not in this build (F1-b, waiting on #18). Use audio_only."
        )


def _source_args(args: dict) -> dict:
    source = args.get("source")
    if isinstance(source, str):
        return {"url": source}
    if isinstance(source, dict):
        return source
    raise ReshootError("source is required")


def _resolve_media(url: str, label: str) -> Path:
    text = str(url or "").strip()
    if not text:
        raise ReshootError(f"{label} is required")
    try:
        return localize(text)
    except (RuntimeError, ValueError):
        path = Path(text)
        if path.is_file():
            return path.resolve()
        raise ReshootError(f"{label} is not a readable file: {text}") from None


def _media_ref(path: Path, original: str) -> str:
    """A /view URL only when the file is inside a ComfyUI media root.

    ``path_to_view_url`` otherwise keeps just the file name, which can point
    at a different file in the output directory.
    """
    resolved = path.resolve()
    for type_ in ("output", "input", "temp"):
        base = Path(folder_paths.get_directory_by_type(type_)).resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            continue
        return path_to_view_url(resolved, type_)
    if original and not str(original).startswith("/view"):
        candidate = Path(original)
        if candidate.is_file() and candidate.resolve() == resolved:
            return str(resolved)
    return str(resolved)


def _require_approver(value) -> None:
    if str(value or "").strip() != APPROVER:
        raise ReshootError("approved_by must be 张景", status=403)


def _require_id(value: str, label: str) -> str:
    text = str(value or "").strip()
    if not _ID_RE.match(text):
        raise ReshootError(f"{label} must be a short token without slashes")
    return text


def _require_take(value) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ReshootError("take must be an integer") from exc
    if number < 1:
        raise ReshootError("take must be >= 1")
    return number


def _fade_ms(args: dict) -> int:
    raw = args.get("audio_fade_ms", DEFAULT_FADE_MS)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ReshootError("audio_fade_ms must be an integer") from exc
    if value < 0 or value > 200:
        raise ReshootError("audio_fade_ms must be between 0 and 200")
    return value


def _optional_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _blank(value) -> bool:
    return value is None or value == ""


def _next_revision(project: str, shot: str, index: dict, take: int) -> int:
    """Next r<k>, counting planned records as well as finished versions."""
    highest = 0
    prefix = f"take_{take}/r"
    for row in index["versions"]:
        ident = str(row.get("id") or "")
        if ident.startswith(prefix):
            try:
                highest = max(highest, int(ident[len(prefix):]))
            except ValueError:
                continue
    folder = _shot_dir(project, shot)
    marker = f"take_{take}_r"
    if folder.is_dir():
        for path in folder.glob(f"{marker}*.json"):
            if not path.stem.startswith(marker):
                continue
            try:
                highest = max(highest, int(path.stem[len(marker):]))
            except ValueError:
                continue
    return highest + 1


def _merge_version(index: dict, version: dict) -> None:
    found = next((row for row in index["versions"] if row["id"] == version["id"]), None)
    if found is None:
        index["versions"].append(version)
        return
    for key in _IDENTITY_KEYS:
        old = found.get(key)
        new = version.get(key)
        if _blank(old) or _blank(new) or old == new:
            continue
        raise ReshootError(f"refusing to overwrite {version['id']} field {key}")


def _empty_takes(project: str, shot: str, active: str | None) -> dict:
    return {
        "schema": TAKES_SCHEMA,
        "project_id": project,
        "shot_id": shot,
        "active": active,
        "active_approved_by": None,
        "active_at": None,
        "versions": [],
    }


def _storage_note(project: str, shot: str, version_id: str) -> dict:
    file_name = version_id.replace("/", "_")
    return {
        "records_dir": f"comfytv/projects/{project}/reshoots/{shot}",
        "record": f"{file_name}.json",
        "takes": "takes.json",
        "media": f"comfytv/reshoot/{project}/{shot}/{file_name}.mp4",
        "protected": f"comfytv/projects/{project}/protected.json",
    }


def _project_dir(project: str) -> Path:
    return project_dir(project)


def _shot_dir(project: str, shot: str) -> Path:
    return _project_dir(project) / "reshoots" / shot


def _record_path(project: str, shot: str, reshoot_id: str) -> Path:
    match = _RESHOOT_RE.match(reshoot_id)
    if match is None or match.group("shot") != shot:
        raise ReshootError(f"reshoot_id {reshoot_id!r} does not match {shot}")
    name = f"take_{match.group('take')}_r{match.group('k')}.json"
    return _shot_dir(project, shot) / name


def _media_dir(project: str, shot: str) -> Path:
    return Path(folder_paths.get_output_directory()) / "comfytv" / "reshoot" / project / shot


def _media_path(project: str, shot: str, version_id: str) -> Path:
    return _media_dir(project, shot) / f"{version_id.replace('/', '_')}.mp4"


def _assert_destination_allowed(project: str, dest: Path) -> None:
    if _ROUGH_CUT_RE.match(dest.name):
        raise ReshootError(f"refusing to write {dest.name}")
    posix = dest.as_posix()
    if "/comfytv/reshoot/" not in posix:
        raise ReshootError("reshoot output must stay under comfytv/reshoot")
    for entry in _load_protected(project)["entries"]:
        legacy = str(entry.get("legacy_path") or "")
        if legacy and posix.endswith(legacy.lstrip("./")):
            raise ReshootError(f"refusing to write protected path {legacy}")
        url = entry.get("url")
        if url:
            try:
                protected = localize(str(url)).resolve()
            except (RuntimeError, ValueError, OSError):
                protected = None
            if protected is not None and protected == dest.resolve():
                raise ReshootError(f"refusing to write protected file {legacy or url}")


def _load_takes(project: str, shot: str) -> dict | None:
    path = _shot_dir(project, shot) / "takes.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("shot_id") != shot:
        raise ReshootError(f"takes.json for {shot} is unreadable")
    data.setdefault("versions", [])
    return data


def _save_takes(project: str, shot: str, index: dict) -> None:
    path = _shot_dir(project, shot) / "takes.json"
    _snapshot(path, "takes")
    _write_json(path, index)


def _load_protected(project: str) -> dict:
    path = _project_dir(project) / "protected.json"
    if not path.is_file():
        return {"schema": PROTECTED_SCHEMA, "project_id": project, "entries": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ReshootError("protected.json is unreadable")
    data.setdefault("entries", [])
    return data


def _save_protected(project: str, doc: dict) -> None:
    path = _project_dir(project) / "protected.json"
    _snapshot(path, "protected")
    _write_json(path, doc)


def _load_reshoots(project: str, shot: str) -> list[dict]:
    folder = _shot_dir(project, shot)
    if not folder.is_dir():
        return []
    rows = []
    for path in sorted(folder.glob("take_*_r*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("reshoot_id"):
            rows.append(data)
    return rows


def _require_record(args: dict) -> dict:
    project = _project_id(args)
    reshoot_id = str(args.get("reshoot_id") or "").strip()
    match = _RESHOOT_RE.match(reshoot_id)
    if match is None:
        raise ReshootError("reshoot_id must look like <shot_id>/take_<n>/r<k>")
    return _read_record(project, match.group("shot"), reshoot_id)


def _read_record(project: str, shot: str, reshoot_id: str) -> dict:
    path = _record_path(project, shot, reshoot_id)
    if not path.is_file():
        raise ReshootError(f"reshoot {reshoot_id} was not found", status=404)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ReshootError(f"reshoot {reshoot_id} is unreadable")
    return data


def _snapshot(path: Path, kind: str) -> None:
    global _HISTORY_SEQ
    if not path.is_file():
        return
    _HISTORY_SEQ += 1
    history = path.parent / f"{kind}.history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    shutil.copy2(path, history / f"{kind}.{stamp}-{_HISTORY_SEQ}.json")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _public_record(record: dict) -> dict:
    return json.loads(json.dumps(record))


def _public_takes(index: dict | None) -> dict | None:
    if index is None:
        return None
    return json.loads(json.dumps(index))


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
