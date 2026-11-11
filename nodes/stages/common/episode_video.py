"""Plan one VideoStage per shot from a shots.json / Storyboard document.

The plan is pure data: prompts, seeds, reference images, duration, and
dialogue audio. It does not call a model and does not submit to ComfyUI.
``api.episode_batch`` turns a ready plan into headless VideoStage nodes and
submits them one at a time.

Workflow choice is field-driven. A shot with both a first frame
(``first_frame`` or ``prev_last_frame``) and a last frame (``last_frame`` or
``end_composition``) would use FL2VA. Anything else would use Ref2VA.
``video_mode`` overrides that, then the production ``mode`` field
(``Ref2VA`` / ``FL2VA``). ``auto`` keeps the frame rule.

Duration sent to the stage is ``duration_s`` (the shot length). ``cut_to``
must not exceed it and is kept as the edit point; it does not replace
``duration_s``.

Seeds are per shot. A shot ``seed`` is used as stored. Otherwise the seed is
``sha256(f"{base_seed}:{shot_id}")`` truncated to 31 bits, so the same base
and shot id always reproduce the same seed.

Dialogue audio is attached only when the mapping says where that line's
file is. Shots with no dialogue lines are left without an audio slot.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import folder_paths

from ....runners._media_paths import path_to_view_url, view_url_to_path
from ....storage.series_library import (
    SeriesLibraryError,
    iter_shot_refs,
    parse_series_ref,
    resolve_root_arg,
    resolve_series_ref,
)
from .constants import (
    ASPECT_RATIOS,
    RESOLUTIONS,
    VIDEO_DURATION_MAX_S,
    VIDEO_DURATION_MIN_S,
)
from .ref_slots import MAX_REFERENCE_IMAGES
from .shot_document import load_shots_document
from .shot_inputs import image_ref_url, shot_seed

DEFAULT_BASE_SEED = 20261111
DEFAULT_WORKFLOW_LABELS = {
    "Ref2VA": "Local MiniMax H3 R2V",
    "FL2VA": "Local MiniMax H3 FLF2V",
}
_MODES = ("Ref2VA", "FL2VA")
_FIRST_ROLES = ("first_frame", "prev_last_frame")
_LAST_ROLES = ("last_frame", "end_composition")
_SKIP_PREFIXES = ("VOX_", "BIB_")
_AUDIO_SLACK_S = 0.05
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


class EpisodeBatchError(ValueError):
    """The batch request or a shot's inputs cannot be planned."""


def plan_episode_videos(
    document: dict | str,
    *,
    library_root: str | None = None,
    frames_root: str | None = None,
    base_seed: int | None = None,
    shots: list | None = None,
    workflows: dict | None = None,
    audio: dict | None = None,
    resolution: str = "720P",
    aspect_ratio: str = "16:9",
    work_dir: str | None = None,
    require_locked: bool = True,
) -> dict:
    """Build the per-shot VideoStage plan. Does not submit or call a model."""
    doc = load_shots_document(document)
    _validate_request(resolution, aspect_ratio, audio)
    seed = _base_seed(base_seed)
    labels = _workflow_labels(workflows)
    selected = _select_shots(doc, shots)
    library = _library_root(library_root, doc.get("series"))
    frames = Path(frames_root) if frames_root else None
    work = Path(work_dir) if work_dir else None

    planned: list[dict] = []
    errors: list[str] = []
    warnings: list[str] = []
    for index, shot in enumerate(selected):
        item = _plan_shot(
            shot,
            index=index,
            episode=str(doc.get("episode") or ""),
            library=library,
            frames=frames,
            base_seed=seed,
            labels=labels,
            audio=audio if isinstance(audio, dict) else None,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            work=work,
            require_locked=require_locked,
        )
        planned.append(item)
        errors.extend(item["errors"])
        warnings.extend(item["warnings"])

    total = 0.0
    for item in planned:
        source = item.get("duration_source_s")
        if isinstance(source, (int, float)) and not isinstance(source, bool):
            total += float(source)
    return {
        "ok": not errors,
        "dry_run": True,
        "episode": doc.get("episode"),
        "series": doc.get("series"),
        "shot_count": len(planned),
        "selected": [item["shot_id"] for item in planned],
        "duration_s_total": total,
        "base_seed": seed,
        "errors": errors,
        "warnings": warnings,
        "shots": planned,
    }


def _validate_request(resolution: str, aspect_ratio: str, audio: Any) -> None:
    if resolution not in RESOLUTIONS:
        raise EpisodeBatchError(
            f"resolution {resolution!r} must be one of: {', '.join(RESOLUTIONS)}"
        )
    if aspect_ratio not in ASPECT_RATIOS:
        raise EpisodeBatchError(
            f"aspect_ratio {aspect_ratio!r} must be one of: {', '.join(ASPECT_RATIOS)}"
        )
    if audio is None:
        return
    if not isinstance(audio, dict):
        raise EpisodeBatchError("audio must be an object")
    for key in ("by_line", "by_shot"):
        value = audio.get(key)
        if value is not None and not isinstance(value, dict):
            raise EpisodeBatchError(f"audio.{key} must be an object")
    pattern = audio.get("pattern")
    directory = audio.get("dir")
    if pattern is not None and not isinstance(pattern, str):
        raise EpisodeBatchError("audio.pattern must be a string")
    if directory is not None and not isinstance(directory, str):
        raise EpisodeBatchError("audio.dir must be a string")
    if pattern and not directory:
        raise EpisodeBatchError("audio.pattern requires audio.dir")
    if directory and not pattern:
        raise EpisodeBatchError("audio.dir requires audio.pattern")
    if pattern:
        _check_pattern(pattern)


def _check_pattern(pattern: str) -> None:
    sample = {
        "episode": "E01",
        "shot_id": "E01S01_01",
        "order": 1,
        "line_id": "L001",
        "scene_id": "E01S01",
    }
    try:
        rel = pattern.format(**sample)
    except KeyError as exc:
        raise EpisodeBatchError(
            f"audio.pattern has unknown field {exc}. "
            "Use episode, shot_id, order, line_id, scene_id."
        ) from exc
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts:
        raise EpisodeBatchError(f"audio.pattern escapes audio.dir: {pattern}")


def _base_seed(value: int | None) -> int:
    if value is None:
        return DEFAULT_BASE_SEED
    if isinstance(value, bool) or not isinstance(value, int):
        raise EpisodeBatchError("base_seed must be an integer")
    if value < 0 or value > 2_147_483_647:
        raise EpisodeBatchError("base_seed must be between 0 and 2147483647")
    return value


def _workflow_labels(overrides: dict | None) -> dict[str, str]:
    labels = dict(DEFAULT_WORKFLOW_LABELS)
    if overrides is None:
        return labels
    if not isinstance(overrides, dict):
        raise EpisodeBatchError("workflows must be an object of Ref2VA and FL2VA labels")
    for key, value in overrides.items():
        if key not in labels:
            raise EpisodeBatchError(
                f"workflows key {key!r} must be Ref2VA or FL2VA"
            )
        if not isinstance(value, str) or not value.strip():
            raise EpisodeBatchError(f"workflows.{key} must be a workflow label")
        labels[key] = value.strip()
    return labels


def _library_root(explicit: str | None, series: Any) -> Path | None:
    if explicit:
        return Path(explicit)
    if not isinstance(series, str) or not series.strip():
        return None
    try:
        return resolve_root_arg(None, series.strip())
    except SeriesLibraryError:
        return None


def _select_shots(doc: dict, shots: list | None) -> list[dict]:
    all_shots = [shot for shot in doc.get("shots") or [] if isinstance(shot, dict)]
    if shots is None:
        return all_shots
    if not isinstance(shots, list):
        raise EpisodeBatchError("shots must be a list of shot_id or 1-based order")
    if not shots:
        raise EpisodeBatchError("shots is empty; omit it to plan every shot")
    by_id = {}
    by_order = {}
    for shot in all_shots:
        shot_id = shot.get("shot_id")
        if isinstance(shot_id, str) and shot_id:
            by_id[shot_id] = shot
        order = shot.get("order")
        if isinstance(order, int) and not isinstance(order, bool):
            by_order[order] = shot
    chosen: list[dict] = []
    seen: set[str] = set()
    missing: list[str] = []
    for raw in shots:
        shot = None
        label = str(raw)
        if isinstance(raw, bool):
            missing.append(label)
            continue
        elif isinstance(raw, int):
            shot = by_order.get(raw)
        elif isinstance(raw, str):
            text = raw.strip()
            label = text
            if text.isdigit():
                shot = by_order.get(int(text))
            if shot is None:
                shot = by_id.get(text)
        else:
            missing.append(label)
            continue
        if shot is None:
            missing.append(label)
            continue
        shot_id = str(shot.get("shot_id") or shot.get("order"))
        if shot_id in seen:
            continue
        seen.add(shot_id)
        chosen.append(shot)
    if missing:
        known = ", ".join(str(shot.get("shot_id")) for shot in all_shots[:3])
        raise EpisodeBatchError(
            "unknown shot(s): "
            + ", ".join(missing)
            + f". Orders are 1..{len(all_shots)}. Known shot_id values start with {known}."
        )
    return chosen


def _plan_shot(
    shot: dict,
    *,
    index: int,
    episode: str,
    library: Path | None,
    frames: Path | None,
    base_seed: int,
    labels: dict[str, str],
    audio: dict | None,
    resolution: str,
    aspect_ratio: str,
    work: Path | None,
    require_locked: bool,
) -> dict:
    tag = _tag(shot, index)
    errors: list[str] = []
    warnings: list[str] = []
    duration_s, duration_source, cut_to = _duration(shot, tag, errors)
    mode, mode_source, frame_mode = _mode(shot, tag, errors)
    seed, seed_source = _seed_for(shot, tag, base_seed, errors)
    prompt = _video_prompt(shot, tag, errors)
    workflow, workflow_source = _workflow_label(shot, mode, labels, tag, errors)
    images = _images(
        shot, tag, mode, library, frames, require_locked, errors,
    )
    dialogue = _attach_audio(
        shot, tag, episode, duration_source, cut_to, audio, work, errors, warnings,
    )
    if dialogue is not None and mode == "FL2VA":
        warnings.append(
            f"{tag}: dialogue is on the VideoStage audio slot. "
            "The shipped FL2VA workflow does not bind that slot, "
            "so this file is stored on the stage and is not sent to the model."
        )
    elif dialogue is not None and mode == "Ref2VA" and "<Audio " not in prompt and "@audio_" not in prompt:
        prompt = prompt + "\n\nDialogue reference: <Audio 1>"

    ready = not errors and duration_s is not None and bool(prompt) and bool(workflow)
    stage = None
    if ready:
        widgets: dict[str, Any] = {
            "workflow": workflow,
            "main_prompt": prompt,
            "duration_s": duration_s,
            "seed": seed,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "generate_audio": False,
        }
        for offset, image in enumerate(images):
            widgets[f"images.image{offset}"] = image["url"]
        if dialogue is not None:
            widgets["audio.audio0"] = dialogue["url"]
        stage = {
            "node_class": "ComfyTV.VideoStage",
            "title": tag,
            "workflow": workflow,
            "prompt": prompt,
            "widgets": widgets,
        }
    return {
        "shot_id": tag,
        "order": shot.get("order") if isinstance(shot.get("order"), int) else index + 1,
        "ready": ready,
        "mode": mode,
        "mode_source": mode_source,
        "frame_mode": frame_mode,
        "workflow": workflow,
        "workflow_source": workflow_source,
        "duration_s": duration_s,
        "duration_source_s": duration_source,
        "cut_to": cut_to,
        "seed": seed,
        "seed_source": seed_source,
        "prompt": prompt,
        "images": images,
        "audio": dialogue,
        "errors": errors,
        "warnings": warnings,
        "stage": stage,
    }


def _tag(shot: dict, index: int) -> str:
    shot_id = shot.get("shot_id")
    if isinstance(shot_id, str) and shot_id.strip():
        return shot_id.strip()
    order = shot.get("order")
    if isinstance(order, int) and not isinstance(order, bool):
        return str(order)
    return f"shots[{index}]"


def _duration(shot: dict, tag: str, errors: list[str]) -> tuple[int | None, float | None, float | None]:
    raw = shot.get("duration_s")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        errors.append(f"{tag}: duration_s must be a number")
        return None, None, None
    duration = float(raw)
    cut_raw = shot.get("cut_to")
    cut: float | None = None
    if cut_raw is not None:
        if isinstance(cut_raw, bool) or not isinstance(cut_raw, (int, float)):
            errors.append(f"{tag}: cut_to must be a number or null")
        else:
            cut = float(cut_raw)
            if cut > duration + 1e-6:
                errors.append(
                    f"{tag}: cut_to {cut} is longer than duration_s {duration}"
                )
    seconds = int(round(duration))
    if seconds < VIDEO_DURATION_MIN_S or seconds > VIDEO_DURATION_MAX_S:
        errors.append(
            f"{tag}: duration_s {duration} is outside "
            f"{VIDEO_DURATION_MIN_S}..{VIDEO_DURATION_MAX_S}s"
        )
        return None, duration, cut
    return seconds, duration, cut


def _mode(shot: dict, tag: str, errors: list[str]) -> tuple[str, str, str]:
    frame_mode = "FL2VA" if _has_role(shot, _FIRST_ROLES) and _has_role(shot, _LAST_ROLES) else "Ref2VA"
    for key, source in (("video_mode", "shot.video_mode"), ("mode", "shot.mode")):
        raw = shot.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        token = raw.strip()
        if token == "auto":
            return frame_mode, "frames", frame_mode
        if token not in _MODES:
            errors.append(
                f"{tag}: {key} {token!r} must be Ref2VA, FL2VA, or auto"
            )
            return frame_mode, "frames", frame_mode
        return token, source, frame_mode
    return frame_mode, "frames", frame_mode


def _has_role(shot: dict, roles: tuple[str, ...]) -> bool:
    return _pick_role(_ref_dicts(shot), roles) is not None


def _ref_dicts(shot: dict) -> list[dict]:
    raw = shot.get("refs")
    if not isinstance(raw, list):
        return []
    found = []
    for item in raw:
        if isinstance(item, dict) and str(item.get("src") or "").strip():
            found.append(item)
    return found


def _pick_role(refs: list[dict], roles: tuple[str, ...]) -> dict | None:
    for role in roles:
        for ref in refs:
            if ref.get("role") == role:
                return ref
    return None


def _seed_for(shot: dict, tag: str, base_seed: int, errors: list[str]) -> tuple[int | None, str]:
    own = shot_seed(shot)
    if own is None:
        return _derived_seed(base_seed, tag), "derived"
    if isinstance(own, bool) or not isinstance(own, int) or own < 0 or own > 2_147_483_647:
        errors.append(f"{tag}: seed must be an integer from 0 to 2147483647")
        return None, "shot"
    return own, "shot"


def _derived_seed(base_seed: int, shot_id: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}:{shot_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    return value or 1


def _video_prompt(shot: dict, tag: str, errors: list[str]) -> str:
    explicit = shot.get("video_prompt")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    prompts = shot.get("prompts") if isinstance(shot.get("prompts"), dict) else {}
    for key in ("video", "h3_video", "ref2va", "fl2va"):
        value = prompts.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    motion = shot.get("motion_prompt")
    if isinstance(motion, str) and motion.strip():
        return motion.strip()
    plan = shot.get("video_plan") if isinstance(shot.get("video_plan"), dict) else {}
    parts: list[str] = []
    summary = plan.get("summary_en")
    if isinstance(summary, str) and summary.strip():
        parts.append(summary.strip())
    segments = plan.get("segments") if isinstance(plan.get("segments"), list) else []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        bits: list[str] = []
        for key in ("camera", "subject", "action", "environment", "lighting", "ending"):
            value = segment.get(key)
            if isinstance(value, str) and value.strip():
                bits.append(value.strip())
        avoid = segment.get("avoid")
        if isinstance(avoid, str) and avoid.strip():
            bits.append("Avoid: " + avoid.strip())
        if bits:
            parts.append(" ".join(bits))
    sound = plan.get("soundscape_en")
    if isinstance(sound, str) and sound.strip():
        parts.append(sound.strip())
    if parts:
        return "\n".join(parts)
    motion_desc = shot.get("motion_desc")
    if isinstance(motion_desc, str) and motion_desc.strip():
        return motion_desc.strip()
    errors.append(
        f"{tag}: no video prompt (video_prompt, prompts.video, video_plan, or motion_desc)"
    )
    return ""


def _workflow_label(
    shot: dict, mode: str, labels: dict[str, str], tag: str, errors: list[str],
) -> tuple[str, str]:
    override = shot.get("video_workflow")
    if isinstance(override, str) and override.strip():
        return override.strip(), "shot.video_workflow"
    if mode not in labels:
        errors.append(f"{tag}: no workflow label for mode {mode}")
        return "", "default"
    return labels[mode], "default" if labels[mode] == DEFAULT_WORKFLOW_LABELS.get(mode) else "workflows"


def _images(
    shot: dict,
    tag: str,
    mode: str,
    library: Path | None,
    frames: Path | None,
    require_locked: bool,
    errors: list[str],
) -> list[dict]:
    slots = _image_slots(shot, mode, tag, errors)
    images: list[dict] = []
    for slot in slots:
        if slot.get("missing"):
            continue
        try:
            resolved = _resolve_image(
                slot, library=library, frames=frames, require_locked=require_locked,
            )
        except EpisodeBatchError as exc:
            errors.append(f"{tag}: {exc}")
            continue
        resolved["role"] = slot.get("role")
        resolved["picture"] = len(images) + 1
        images.append(resolved)
    if len(images) > MAX_REFERENCE_IMAGES:
        errors.append(
            f"{tag}: {len(images)} reference images exceeds "
            f"{MAX_REFERENCE_IMAGES}; none were dropped"
        )
    return images


def _image_slots(shot: dict, mode: str, tag: str, errors: list[str]) -> list[dict]:
    refs = _ref_dicts(shot)
    if mode == "FL2VA":
        first = _pick_role(refs, _FIRST_ROLES)
        last = _pick_role(refs, _LAST_ROLES)
        slots: list[dict] = []
        if first is None:
            errors.append(
                f"{tag}: FL2VA needs a first frame (first_frame or prev_last_frame)"
            )
            slots.append({"role": "first_frame", "src": "", "missing": "first"})
        else:
            slots.append(first)
        if last is None:
            errors.append(
                f"{tag}: FL2VA needs a last frame (last_frame or end_composition)"
            )
            slots.append({"role": "last_frame", "src": "", "missing": "last"})
        else:
            slots.append(last)
        return slots

    ordered: list[dict] = []
    seen: set[str] = set()
    first = _pick_role(refs, ("first_frame",))
    sequence = []
    if first is not None:
        sequence.append(first)
    for ref in refs:
        if first is not None and ref is first:
            continue
        sequence.append(ref)
    for ref in sequence:
        src = str(ref.get("src") or "").strip()
        if src in seen:
            continue
        seen.add(src)
        ordered.append(ref)
    for parsed in iter_shot_refs(shot):
        src = str(parsed.get("ref") or "").strip()
        if not src or src in seen or parsed.get("kind") in ("episode", "invalid"):
            continue
        identity = parsed.get("id") or parsed.get("proposed_id") or ""
        if any(str(identity).startswith(prefix) or src.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue
        seen.add(src)
        ordered.append({
            "role": parsed.get("kind") or "ref",
            "src": src,
            "view": parsed.get("view"),
        })
    if not ordered:
        errors.append(f"{tag}: no reference image slots or asset codes")
    return ordered


def _resolve_image(
    slot: dict, *, library: Path | None, frames: Path | None, require_locked: bool,
) -> dict:
    src = str(slot.get("src") or "").strip()
    view = slot.get("view")
    view_name = str(view).strip() if isinstance(view, str) and view.strip() else None
    direct = image_ref_url(src)
    if direct:
        return {"ref": src, "url": direct, "resolved": True, "kind": "url"}
    parsed = parse_series_ref(src)
    if parsed["kind"] == "episode":
        if frames is None:
            raise EpisodeBatchError(
                f"frame {src} needs frames_root (episode files are not in the series library)"
            )
        path = _safe_child(frames, str(parsed.get("path") or ""))
        if not path.is_file():
            raise EpisodeBatchError(f"frame file not found for {src}: {path}")
        return {
            "ref": src,
            "url": _stage_url(path),
            "path": str(path),
            "resolved": True,
            "kind": "episode",
        }
    if parsed["kind"] == "unresolved":
        code = str(parsed.get("proposed_id") or "")
        if view_name:
            code = f"{code}#{view_name}"
        resolved = _resolve_asset(code, library, require_locked, src)
        return resolved
    if parsed["kind"] != "asset":
        raise EpisodeBatchError(f"cannot resolve reference {src!r}")
    code = str(parsed.get("ref") or src)
    if view_name and not parsed.get("view"):
        code = f"{code}#{view_name}"
    return _resolve_asset(code, library, require_locked, src)


def _resolve_asset(code: str, library: Path | None, require_locked: bool, src: str) -> dict:
    if library is None:
        raise EpisodeBatchError(
            f"{src} needs library_root (series library is not bound)"
        )
    try:
        resolved = resolve_series_ref(library, code, require_locked=require_locked)
    except SeriesLibraryError as exc:
        raise EpisodeBatchError(f"could not resolve {code}: {exc}") from exc
    if resolved.get("media_type") not in (None, "image"):
        raise EpisodeBatchError(
            f"{code} is {resolved.get('media_type')}, not an image"
        )
    url = resolved.get("payload_url") or resolved.get("absolute_path")
    if not url:
        raise EpisodeBatchError(f"{code} resolved without a file")
    return {
        "ref": src,
        "url": str(url),
        "path": resolved.get("absolute_path"),
        "resolved": True,
        "kind": resolved.get("kind") or "asset",
        "asset_id": resolved.get("id"),
    }


def _safe_child(root: Path, rel: str) -> Path:
    rel_path = Path(rel)
    if not rel or rel_path.is_absolute() or ".." in rel_path.parts:
        raise EpisodeBatchError(f"frame path escapes frames_root: {rel}")
    base = root.resolve()
    path = (base / rel_path).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise EpisodeBatchError(f"frame path escapes frames_root: {rel}") from exc
    return path


def _stage_url(path: Path) -> str:
    """A ``/view`` URL ComfyUI can load.

    Files already under an input, output, or temp directory are addressed in
    place. Anything else is copied into the output directory, because the
    video workflow only loads ``/view`` URLs.
    """
    resolved = path.resolve()
    for kind in ("input", "output", "temp"):
        try:
            base = Path(folder_paths.get_directory_by_type(kind)).resolve()
        except Exception:
            continue
        try:
            resolved.relative_to(base)
        except ValueError:
            continue
        return path_to_view_url(resolved, kind)
    dest_dir = Path(folder_paths.get_output_directory()) / "comfytv" / "episode-batch"
    dest_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    suffix = resolved.suffix.lower() if resolved.suffix else ".bin"
    dest = dest_dir / f"{digest}{suffix}"
    if not dest.exists() or dest.stat().st_mtime < resolved.stat().st_mtime:
        shutil.copy2(resolved, dest)
    return path_to_view_url(dest, "output")


def _dialogue_lines(shot: dict) -> list[dict]:
    raw = shot.get("dialogue")
    if not isinstance(raw, list):
        return []
    lines = []
    for item in raw:
        if isinstance(item, dict) and str(item.get("text") or "").strip():
            lines.append(item)
    return lines


def _attach_audio(
    shot: dict,
    tag: str,
    episode: str,
    duration_s: float | None,
    cut_to: float | None,
    audio: dict | None,
    work: Path | None,
    errors: list[str],
    warnings: list[str],
) -> dict | None:
    lines = _dialogue_lines(shot)
    if not lines:
        return None
    if not _audio_configured(audio):
        for line in lines:
            line_id = line.get("line_id") or "?"
            errors.append(
                f"{tag} line {line_id}: no audio mapped. "
                "Pass audio.by_line, audio.by_shot, or audio.dir with audio.pattern."
            )
        return None
    paths: list[Path] = []
    urls: list[str] = []
    line_ids: list[str] = []
    for index, line in enumerate(lines):
        line_id = str(line.get("line_id") or f"line{index + 1}")
        line_ids.append(line_id)
        try:
            found = _line_audio(audio, shot, episode, line, index, line_id, tag)
            path, url = _existing_audio(found, tag, line_id)
        except EpisodeBatchError as exc:
            errors.append(str(exc))
            continue
        paths.append(path)
        urls.append(url)
    if len(paths) != len(lines):
        return None
    try:
        url, path, heard = _combine_audio(paths, urls, tag, work)
    except EpisodeBatchError as exc:
        errors.append(str(exc))
        return None
    warning = _audio_length_warning(tag, heard, duration_s, cut_to)
    if warning:
        warnings.append(warning)
    return {
        "url": url,
        "path": str(path),
        "duration_s": round(heard, 3),
        "line_ids": line_ids,
        "files": [str(item) for item in paths],
    }


def _audio_configured(audio: dict | None) -> bool:
    if not isinstance(audio, dict):
        return False
    if audio.get("by_line") or audio.get("by_shot"):
        return True
    return bool(audio.get("dir") and audio.get("pattern"))


def _line_audio(
    audio: dict, shot: dict, episode: str, line: dict, index: int, line_id: str, tag: str,
) -> Any:
    by_line = audio.get("by_line") if isinstance(audio.get("by_line"), dict) else {}
    if line_id in by_line:
        return by_line[line_id]
    by_shot = audio.get("by_shot") if isinstance(audio.get("by_shot"), dict) else {}
    for key in (str(shot.get("shot_id") or ""), str(shot.get("order") or "")):
        if key and key in by_shot:
            return _shot_entry(by_shot[key], index, len(_dialogue_lines(shot)), tag, line_id)
    if audio.get("dir") and audio.get("pattern"):
        return str(_pattern_path(audio, shot, episode, line_id))
    raise EpisodeBatchError(
        f"{tag} line {line_id}: no audio mapped. "
        "Set audio.by_line, audio.by_shot for this shot, or audio.dir with audio.pattern."
    )


def _shot_entry(entry: Any, index: int, count: int, tag: str, line_id: str) -> Any:
    if isinstance(entry, list):
        if len(entry) != count:
            raise EpisodeBatchError(
                f"{tag}: audio.by_shot has {len(entry)} file(s) for {count} "
                f"dialogue line(s); list them in dialogue order"
            )
        return entry[index]
    if count == 1:
        return entry
    raise EpisodeBatchError(
        f"{tag} line {line_id}: audio.by_shot must be a list of {count} "
        "files in dialogue order, not one path"
    )


def _pattern_path(audio: dict, shot: dict, episode: str, line_id: str) -> Path:
    rel = str(audio["pattern"]).format(
        episode=episode,
        shot_id=str(shot.get("shot_id") or ""),
        order=int(shot.get("order") or 0),
        line_id=line_id,
        scene_id=str(shot.get("scene_id") or ""),
    )
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts:
        raise EpisodeBatchError(f"audio.pattern escapes audio.dir: {rel}")
    return Path(str(audio["dir"])) / path


def _existing_audio(value: Any, tag: str, line_id: str) -> tuple[Path, str]:
    text = _coerce_audio_ref(value)
    if not text:
        raise EpisodeBatchError(f"{tag} line {line_id}: audio mapping is empty")
    if text.startswith(("http://", "https://")):
        raise EpisodeBatchError(
            f"{tag} line {line_id}: audio must be a local file or /view URL, got {text}"
        )
    if text.startswith("/view?"):
        path = view_url_to_path(text)
        if path is None:
            raise EpisodeBatchError(f"{tag} line {line_id}: audio file not found: {text}")
        return path, text
    path = Path(text)
    if not path.is_file():
        raise EpisodeBatchError(f"{tag} line {line_id}: audio file not found: {path}")
    return path, _stage_url(path)


def _coerce_audio_ref(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "{[":
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                return text
            return _coerce_audio_ref(parsed)
        return text
    if isinstance(value, dict):
        for key in ("audio_url", "url", "path", "payload_url", "file"):
            found = value.get(key)
            if isinstance(found, str) and found.strip():
                return found.strip()
        nested = value.get("audio")
        if isinstance(nested, (str, dict)):
            return _coerce_audio_ref(nested)
    return ""


def _combine_audio(
    paths: list[Path], urls: list[str], tag: str, work: Path | None,
) -> tuple[str, Path, float]:
    if len(paths) == 1:
        return urls[0], paths[0], _probe_duration(paths[0])
    if work is None:
        raise EpisodeBatchError(
            f"{tag}: work_dir is required to concatenate {len(paths)} dialogue files"
        )
    safe = _SAFE_NAME.sub("_", tag) or "shot"
    dest = work / f"{safe}.dialogue.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _wav_params(paths) is not None:
        _concat_wavs(paths, dest)
    else:
        _ffmpeg_concat(paths, dest)
    return _stage_url(dest), dest, _probe_duration(dest)


def _audio_length_warning(
    tag: str, heard: float, duration_s: float | None, cut_to: float | None,
) -> str | None:
    over_shot = duration_s is not None and heard > float(duration_s) + _AUDIO_SLACK_S
    over_cut = cut_to is not None and heard > float(cut_to) + _AUDIO_SLACK_S
    if not over_shot and not over_cut:
        return None
    parts = [f"{tag}: dialogue audio is {heard:.2f}s"]
    if over_shot:
        parts.append(f"longer than duration_s {float(duration_s):.2f}s")
    if over_cut:
        parts.append(f"longer than cut_to {float(cut_to):.2f}s")
    sentence = parts[0] + ", " + " and ".join(parts[1:]) + "."
    return sentence + " The file was kept whole and was not trimmed."


def _probe_duration(path: Path) -> float:
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as src:
                rate = src.getframerate()
                if rate <= 0:
                    raise EpisodeBatchError(f"audio has no sample rate: {path}")
                return src.getnframes() / float(rate)
        except wave.Error:
            pass
    return _ffprobe_duration(path)


def _wav_params(paths: list[Path]) -> tuple | None:
    params = None
    for path in paths:
        if path.suffix.lower() != ".wav":
            return None
        try:
            with wave.open(str(path), "rb") as src:
                current = (src.getnchannels(), src.getsampwidth(), src.getframerate())
        except wave.Error:
            return None
        if params is None:
            params = current
        elif current != params:
            return None
    return params


def _concat_wavs(paths: list[Path], dest: Path) -> None:
    frames: list[bytes] = []
    params = None
    with wave.open(str(paths[0]), "rb") as first:
        params = first.getparams()
    for path in paths:
        with wave.open(str(path), "rb") as src:
            frames.append(src.readframes(src.getnframes()))
    with wave.open(str(dest), "wb") as out:
        out.setparams(params)
        for chunk in frames:
            out.writeframes(chunk)


def _ffprobe_duration(path: Path) -> float:
    if shutil.which("ffprobe") is None:
        raise EpisodeBatchError(
            f"cannot read duration of {path.name}: not a wav, and ffprobe is not on PATH"
        )
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "").strip()[:300]
        raise EpisodeBatchError(f"ffprobe failed for {path.name}: {tail}") from exc
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise EpisodeBatchError(f"ffprobe returned no duration for {path.name}") from exc


def _ffmpeg_concat(paths: list[Path], dest: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise EpisodeBatchError(
            "ffmpeg is required to concatenate these dialogue files "
            "(they are not wavs with the same format)"
        )
    listing = dest.with_suffix(".concat.txt")
    lines = []
    for path in paths:
        escaped = str(path.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(listing),
                "-c:a", "pcm_s16le",
                str(dest),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "").strip()[-300:]
        raise EpisodeBatchError(f"ffmpeg could not concatenate dialogue: {tail}") from exc
