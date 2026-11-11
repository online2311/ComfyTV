"""Optional shot-prompt fields on a production shots.json document.

The fields are the inputs for a later renderer (F2-b). This module does not
write H3 or Seedance prompt text. Empty fields are not inserted on load or
save, so an older document round-trips unchanged.

Asset codes (``CHR_…@vN#view``, ``episode:…``) name files ComfyTV stores.
A view URL or another machine's path is not a second copy of the asset.

H3 base modes (T2VA, I2VA, FL2VA, L2VA) need an alignment, then
``integrated_multimodal_description``, ``overall_soundscape``, and
``non_diegetic_music``. Ref2VA needs ``subject_definitions``, ``summary``,
``retention_analysis``, ``detailed_description``, ``overall_soundscape``, and
``non_diegetic_music``, in that order. ``h3_inputs`` gathers those inputs
from the shot. Picture slots stay dense from 1 because an empty picture slot
would renumber later pictures. Video and audio references use their own
1-based sequences and do not occupy picture slots.

``parse_asset_ref`` lives in ``shot_document``, which calls back into this
module. Import it inside functions so that cycle stays lazy.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, replace
from typing import Any

MAX_REFERENCE_IMAGES = 9

ASPECT_RATIOS = ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9")
REF_ROLES = (
    "first_frame",
    "prev_last_frame",
    "last_frame",
    "end_composition",
    "character",
    "location",
    "prop",
    "style",
    "action",
    "camera",
    "voice",
)
MEDIA_KINDS = ("picture", "video", "audio")
CUT_TYPES = ("opening", "cut", "oner")
END_HOLD_TYPES = ("action", "line")
SLOT_ORDERS = ("pipeline", "libtv")
SUBJECT_KINDS = (
    "identity",
    "object",
    "environment",
    "costume",
    "style",
    "action",
    "pose",
    "effect",
)
VISIBLE_RETENTION = (
    "fully_preserved",
    "partially_preserved",
    "attribute_transfer",
    "weak_reference",
)
AUDIO_RETENTION = (
    "fully_copy",
    "partially_copy",
    "reference",
    "weak_reference",
)
SUMMARY_TASKS = (
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
)
H3_REF2VA_SECTIONS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
H3_BASE_SECTIONS = (
    "alignment",
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)
DEFAULT_CONSTRAINTS = (
    "人物与道具保持一致",
    "不重复已完成动作",
    "无额外字幕、水印、配乐",
)
STORYBOARD_YAML_FIELDS = (
    "chain_from",
    "chain_method",
    "motion_source",
    "eyeline",
)
OPENING_KEYS = (
    "positions",
    "poses",
    "facing",
    "eyeline",
    "holding",
    "environment",
)
SPATIAL_KEYS = ("anchor", "distances", "occlusion", "camera")
_START_ROLES = ("first_frame", "prev_last_frame")
_END_ROLES = ("last_frame", "end_composition")
_CANONICAL_SRC = re.compile(
    r"^(?:(?P<series>[A-Z]+_[A-Za-z0-9_]+)@v(?P<version>[0-9]+)"
    r"(?:#(?P<view>[A-Za-z0-9_]+))?|(?P<episode>episode:.+)|(?P<new>NEW:.+))$"
)
_SHEET_VIEWS = frozenset({"sheet"})


@dataclass(frozen=True)
class ShotPromptConfig:
    """Defaults keep today's documents valid. Callers opt into stricter checks.

    ``slot_order`` ``pipeline`` numbers the start frame first (张景, D2.1).
    ``libtv`` is the tutorial order: characters, then the scene, then the
    tail frame. ``merge_storyboard_yaml`` stays off (D2.3). A ``sheet`` view
    stays out of video reference slots (D3.1).
    """

    strict: bool = False
    slot_order: str = "pipeline"
    merge_storyboard_yaml: bool = False
    allow_sheet_in_video_refs: bool = False
    require_chain_tail_frame: bool = False
    frames_mismatch: str = "warning"
    unresolved_new_refs: str = "warning"
    default_constraints: tuple[str, ...] = DEFAULT_CONSTRAINTS
    max_refs: int = MAX_REFERENCE_IMAGES

    def __post_init__(self) -> None:
        if self.slot_order not in SLOT_ORDERS:
            raise ValueError(
                f"slot_order must be one of {', '.join(SLOT_ORDERS)}"
            )
        if self.frames_mismatch not in ("warning", "error"):
            raise ValueError("frames_mismatch must be warning or error")
        if self.unresolved_new_refs not in ("warning", "error"):
            raise ValueError("unresolved_new_refs must be warning or error")
        constraints = self.default_constraints
        if isinstance(constraints, list):
            constraints = tuple(constraints)
            object.__setattr__(self, "default_constraints", constraints)
        if not isinstance(constraints, tuple) or not all(
            isinstance(item, str) for item in constraints
        ):
            raise ValueError("default_constraints must be strings")
        if isinstance(self.max_refs, bool) or not isinstance(self.max_refs, int):
            raise ValueError("max_refs must be an integer")
        if self.max_refs < 0 or self.max_refs > MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"max_refs must be from 0 to {MAX_REFERENCE_IMAGES}"
            )


@dataclass(frozen=True)
class AliasIndex:
    """Name or alias to one canonical ``ID@vN#view`` code."""

    by_name: dict[str, str]
    conflicts: dict[str, tuple[str, ...]]

    def lookup(self, name: str) -> tuple[str | None, tuple[str, ...] | None]:
        if name in self.conflicts:
            return None, self.conflicts[name]
        return self.by_name.get(name), None


def normalize_config(
    config: ShotPromptConfig | None = None,
    *,
    strict: bool | None = None,
    allow_sheet_in_video_refs: bool | None = None,
    require_chain_tail_frame: bool | None = None,
    frames_mismatch: str | None = None,
    unresolved_new_refs: str | None = None,
    max_refs: int | None = None,
    slot_order: str | None = None,
    merge_storyboard_yaml: bool | None = None,
) -> ShotPromptConfig:
    """Apply explicit overrides. ``strict=True`` tightens the checks it implies
    unless that check is passed explicitly.
    """
    cfg = config or ShotPromptConfig()
    updates: dict[str, Any] = {}
    if strict is True:
        updates["strict"] = True
        if require_chain_tail_frame is None:
            updates["require_chain_tail_frame"] = True
        if frames_mismatch is None:
            updates["frames_mismatch"] = "error"
        if unresolved_new_refs is None:
            updates["unresolved_new_refs"] = "error"
    elif strict is False:
        updates["strict"] = False
    if allow_sheet_in_video_refs is not None:
        updates["allow_sheet_in_video_refs"] = allow_sheet_in_video_refs
    if require_chain_tail_frame is not None:
        updates["require_chain_tail_frame"] = require_chain_tail_frame
    if frames_mismatch is not None:
        updates["frames_mismatch"] = frames_mismatch
    if unresolved_new_refs is not None:
        updates["unresolved_new_refs"] = unresolved_new_refs
    if isinstance(max_refs, int) and not isinstance(max_refs, bool):
        updates["max_refs"] = max(0, min(MAX_REFERENCE_IMAGES, max_refs))
    if slot_order is not None:
        updates["slot_order"] = slot_order
    if merge_storyboard_yaml is not None:
        updates["merge_storyboard_yaml"] = merge_storyboard_yaml
    if not updates:
        return cfg
    return replace(cfg, **updates)


def _parse_ref(raw: str) -> dict:
    from .shot_document import parse_asset_ref

    return parse_asset_ref(raw)


def _asset_id(raw: Any) -> str:
    return str(_parse_ref(str(raw or "")).get("id") or "")


def _canonical_match(text: str) -> re.Match[str] | None:
    return _CANONICAL_SRC.match(text.strip())


def _view_of(code: str) -> str | None:
    match = _canonical_match(code)
    if match is None:
        return None
    return match.group("view")


def _is_external_locator(text: str) -> bool:
    lower = text.lower()
    return (
        lower.startswith(("http://", "https://", "data:", "/view", "/"))
        or "filename=" in lower
    )


def _media(ref: dict) -> str:
    media = ref.get("media")
    if media is None or media == "":
        return "picture"
    return str(media)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-6


def _bucket(errors: list[str], warns: list[str], severity: str) -> list[str]:
    if severity == "error":
        return errors
    return warns


def shot_aspect_ratio(doc: dict | None, shot: dict | None) -> str | None:
    """Shot value wins. Otherwise the document value. Neither means unset."""
    if isinstance(shot, dict):
        shot_value = shot.get("aspect_ratio")
        if isinstance(shot_value, str) and shot_value.strip():
            return shot_value
    if isinstance(doc, dict):
        envelope = doc.get("aspect_ratio")
        if isinstance(envelope, str) and envelope.strip():
            return envelope
    return None


def default_align_s(ref: dict, duration_s: Any) -> float | None:
    """Seconds at which a picture matches the target video.

    An explicit ``align_s`` wins. A start frame is 0. A last frame is the
    shot length. Other pictures stay unset until a caller fills them in.
    """
    if not isinstance(ref, dict):
        return None
    explicit = _number(ref.get("align_s"))
    if explicit is not None and "align_s" in ref:
        return explicit
    role = ref.get("role")
    if role in _START_ROLES:
        return 0.0
    duration = _number(duration_s)
    if role in _END_ROLES and duration is not None:
        return duration
    return None


def inferred_summary_tasks(shot: dict) -> list[str]:
    """Task prefixes a renderer can use when ``summary_tasks`` is absent."""
    if not isinstance(shot, dict):
        return []
    mode = str(shot.get("mode") or "")
    tasks: list[str] = []
    if mode in ("FL2VA", "I2VA", "L2VA"):
        tasks.append("keyframe completion")
    elif mode in ("Ref2VA", "T2VA"):
        tasks.append("reference generation")
    has_audio = False
    has_video = False
    has_tail = False
    refs = shot.get("refs")
    if isinstance(refs, list):
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            if _media(ref) == "audio":
                has_audio = True
            elif _media(ref) == "video":
                has_video = True
            if ref.get("role") == "prev_last_frame":
                has_tail = True
    if (has_tail or has_video) and "video continuation" not in tasks:
        tasks.append("video continuation")
    if has_audio and "audio reference" not in tasks:
        tasks.append("audio reference")
    return tasks


def soundscape_text(shot: dict) -> str | None:
    plan = shot.get("video_plan") if isinstance(shot, dict) else None
    if isinstance(plan, dict):
        text = plan.get("soundscape_en")
        if isinstance(text, str) and text.strip():
            return text
    audio = shot.get("audio") if isinstance(shot, dict) else None
    if isinstance(audio, dict):
        ambience = audio.get("ambience")
        if isinstance(ambience, str) and ambience.strip():
            return ambience
    return None


def non_diegetic_music_text(shot: dict) -> str | None:
    """Authored score text. ``audio.h3_music: none`` means there is no score."""
    plan = shot.get("video_plan") if isinstance(shot, dict) else None
    if isinstance(plan, dict):
        text = plan.get("non_diegetic_music")
        if isinstance(text, str):
            return text
    audio = shot.get("audio") if isinstance(shot, dict) else None
    if isinstance(audio, dict):
        music = audio.get("h3_music")
        if isinstance(music, str) and music not in ("", "none"):
            return music
    return None


def _constraint_defaults(shot_doc: dict | None, config: ShotPromptConfig) -> tuple[str, ...]:
    if isinstance(shot_doc, dict) and isinstance(shot_doc.get("default_constraints"), list):
        return tuple(
            item for item in shot_doc["default_constraints"] if isinstance(item, str)
        )
    return config.default_constraints


def effective_constraints(
    shot: dict,
    doc: dict | None = None,
    *,
    config: ShotPromptConfig | None = None,
) -> list[str]:
    """Project defaults, minus shot items prefixed with ``-``, plus the rest.

    The shot is not modified. Absent ``constraints`` means the defaults only.
    """
    cfg = config or ShotPromptConfig()
    defaults = list(_constraint_defaults(doc, cfg))
    disabled: set[str] = set()
    extra: list[str] = []
    raw = shot.get("constraints") if isinstance(shot, dict) else None
    if isinstance(raw, list):
        known = set(defaults)
        for item in raw:
            if not isinstance(item, str):
                continue
            if item.startswith("-") and item[1:] in known:
                disabled.add(item[1:])
            elif not item.startswith("-"):
                extra.append(item)
    return [item for item in defaults if item not in disabled] + extra


def _video_view(record: dict, *, allow_sheet: bool) -> str | None:
    files = record.get("files") if isinstance(record.get("files"), dict) else {}
    kind = str(record.get("kind") or "")

    def has(name: str) -> bool:
        return isinstance(files.get(name), dict)

    if kind in ("character", "look", "outfit", "camera") and has("front"):
        return "front"
    if kind == "style" and has("style"):
        return "style"
    if kind == "location":
        cameras = [
            name for name in files
            if isinstance(name, str) and len(name) >= 3 and name[0] in "ABW"
            and name[1] == "_" and has(name)
        ]
        if len(cameras) == 1:
            return cameras[0]
        if has("empty"):
            return "empty"
    if kind == "voice" and has("voice_ref"):
        return "voice_ref"
    images = []
    for name, info in files.items():
        if not isinstance(info, dict):
            continue
        if name == "sheet" and not allow_sheet:
            continue
        media = info.get("media_type")
        if media in (None, "", "image"):
            images.append(str(name))
    if len(images) == 1:
        return images[0]
    return None


def _preferred_records(assets: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        asset_id = str(asset.get("id") or "").strip()
        if asset_id:
            grouped.setdefault(asset_id, []).append(asset)
    chosen: list[dict] = []
    for rows in grouped.values():
        locked = [row for row in rows if row.get("status") == "locked"]
        pool = locked or rows

        def version_of(row: dict) -> int:
            try:
                return int(row.get("version") or 0)
            except (TypeError, ValueError):
                return 0

        chosen.append(max(pool, key=version_of))
    return chosen


def _record_code(record: dict, *, allow_sheet: bool) -> str | None:
    asset_id = str(record.get("id") or "").strip()
    try:
        version = int(record.get("version"))
    except (TypeError, ValueError):
        return None
    if not asset_id or version < 1:
        return None
    view = _video_view(record, allow_sheet=allow_sheet)
    if view == "sheet" and not allow_sheet:
        return None
    code = f"{asset_id}@v{version}"
    if view:
        return f"{code}#{view}"
    return code


def _names_of(record: dict) -> list[str]:
    found: list[str] = []
    name = record.get("name_zh")
    if isinstance(name, str) and name.strip():
        found.append(name.strip())
    aliases = record.get("aliases")
    if isinstance(aliases, list):
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                found.append(alias.strip())
    return found


def build_alias_index(
    aliases: dict[str, str] | None = None,
    assets: list[dict] | None = None,
    *,
    allow_sheet: bool = False,
) -> AliasIndex:
    """Map Chinese names and aliases to canonical asset codes.

    ``assets`` are series-library records (``name_zh``, optional ``aliases``).
    A name that points at two different codes is a conflict, not a guess.
    """
    by_name: dict[str, str] = {}
    conflict_sets: dict[str, set[str]] = {}

    def add(name: str, code: str) -> None:
        if not name or not code:
            return
        existing = by_name.get(name)
        if existing is None:
            by_name[name] = code
            return
        if existing == code:
            return
        conflict_sets.setdefault(name, {existing}).add(code)

    for record in _preferred_records(list(assets or [])):
        code = _record_code(record, allow_sheet=allow_sheet)
        if code is None:
            continue
        for name in _names_of(record):
            add(name, code)
    for name, code in (aliases or {}).items():
        if isinstance(name, str) and isinstance(code, str):
            add(name.strip(), code.strip())
    conflicts = {
        name: tuple(sorted(codes))
        for name, codes in conflict_sets.items()
    }
    for name in conflicts:
        by_name.pop(name, None)
    return AliasIndex(by_name=by_name, conflicts=conflicts)


def _rewrite_src(text: str, index: AliasIndex | None) -> tuple[str, str | None]:
    """Return ``(src, issue)``. A rewritten alias has no issue."""
    stripped = text.strip()
    match = _canonical_match(stripped)
    if match is not None:
        if match.group("new") is not None:
            return stripped, f"unresolved: {stripped}"
        return stripped, None
    if index is not None:
        code, conflicts = index.lookup(stripped)
        if conflicts:
            joined = ", ".join(conflicts)
            return stripped, f"conflict: {stripped} -> {joined}"
        if code:
            return code, None
    if _is_external_locator(stripped):
        return stripped, f"external: {stripped}"
    return stripped, f"unresolved: {stripped}"


def canonicalize_shot(
    shot: dict,
    index: AliasIndex | None = None,
    *,
    apply: bool = False,
) -> dict:
    """Rewrite alias ``src`` values to canonical ids.

    ``apply=False`` leaves ``shot`` unchanged. ``NEW:`` refs stay unresolved.
    """
    issues: list[str] = []

    def rewrite_list(raw: Any) -> list[Any] | None:
        if not isinstance(raw, list):
            return None
        rewritten: list[Any] = []
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("src"), str):
                rewritten.append(copy.deepcopy(item) if isinstance(item, dict) else item)
                continue
            copy_item = copy.deepcopy(item)
            src, issue = _rewrite_src(copy_item["src"], index)
            copy_item["src"] = src
            if issue:
                issues.append(issue)
            rewritten.append(copy_item)
        return rewritten

    refs = rewrite_list(shot.get("refs")) or []
    subjects = rewrite_list(shot.get("subjects"))
    if apply:
        if isinstance(shot.get("refs"), list):
            shot["refs"] = refs
        if subjects is not None and isinstance(shot.get("subjects"), list):
            shot["subjects"] = subjects
    result: dict[str, Any] = {"refs": refs, "issues": issues}
    if subjects is not None:
        result["subjects"] = subjects
    return result


def _speaker_ids(shot: dict) -> list[str]:
    found: list[str] = []
    dialogue = shot.get("dialogue")
    if not isinstance(dialogue, list):
        return found
    for line in dialogue:
        if not isinstance(line, dict):
            continue
        ident = _asset_id(line.get("speaker"))
        if ident and ident not in found:
            found.append(ident)
    return found


def _protagonist_id(shot: dict) -> str | None:
    characters = shot.get("characters")
    if not isinstance(characters, list):
        return None
    for character in characters:
        if isinstance(character, dict):
            ident = _asset_id(character.get("id"))
        elif isinstance(character, str):
            ident = _asset_id(character)
        else:
            ident = ""
        if ident:
            return ident
    return None


def _picture_rank(ref: dict, *, order: str, speakers: list[str], protagonist: str | None) -> int:
    role = str(ref.get("role") or "")
    ident = _asset_id(ref.get("src"))
    speaker_rank = speakers.index(ident) if ident in speakers else None
    if order == "pipeline":
        table = {
            "first_frame": 0,
            "prev_last_frame": 10,
            "location": 50,
            "prop": 60,
            "last_frame": 70,
            "end_composition": 80,
            "style": 90,
            "action": 100,
            "camera": 110,
            "voice": 120,
        }
        if role == "character":
            if speaker_rank is not None:
                return 20 + speaker_rank
            if protagonist and ident == protagonist:
                return 30
            return 40
        return table.get(role, 500)
    if order == "libtv":
        table = {
            "location": 30,
            "prop": 40,
            "action": 50,
            "camera": 60,
            "style": 70,
            "first_frame": 80,
            "prev_last_frame": 90,
            "last_frame": 100,
            "end_composition": 110,
            "voice": 120,
        }
        if role == "character":
            if speaker_rank is not None:
                return speaker_rank
            if protagonist and ident == protagonist:
                return 10
            return 20
        return table.get(role, 500)
    raise ValueError(f"slot_order must be one of {', '.join(SLOT_ORDERS)}")


def assign_ref_slots(
    shot: dict,
    *,
    config: ShotPromptConfig | None = None,
    apply: bool = False,
) -> list[dict]:
    """Number references densely from 1.

    Pictures follow ``slot_order`` (default ``pipeline``: start frame first).
    Video and audio keep their own 1-based sequences and are listed after the
    pictures, so a picture list passed to the model matches ``<Picture N>``.
    Nothing is dropped and nothing is invented. ``apply=False`` does not
    write the shot.
    """
    cfg = normalize_config(config)
    raw = shot.get("refs") if isinstance(shot, dict) and isinstance(shot.get("refs"), list) else []
    pictures: list[tuple[int, dict]] = []
    videos: list[dict] = []
    audios: list[dict] = []
    others: list[Any] = []
    for index, ref in enumerate(raw):
        if not isinstance(ref, dict):
            others.append(ref)
            continue
        media = ref.get("media") or "picture"
        if media == "video":
            videos.append(copy.deepcopy(ref))
        elif media == "audio":
            audios.append(copy.deepcopy(ref))
        else:
            pictures.append((index, copy.deepcopy(ref)))
    speakers = _speaker_ids(shot if isinstance(shot, dict) else {})
    protagonist = _protagonist_id(shot if isinstance(shot, dict) else {})
    pictures.sort(
        key=lambda item: (
            _picture_rank(
                item[1],
                order=cfg.slot_order,
                speakers=speakers,
                protagonist=protagonist,
            ),
            item[0],
        )
    )
    ordered: list[dict] = []
    for slot, (_index, ref) in enumerate(pictures, 1):
        ref["slot"] = slot
        ordered.append(ref)
    for slot, ref in enumerate(videos, 1):
        ref["slot"] = slot
        ordered.append(ref)
    for slot, ref in enumerate(audios, 1):
        ref["slot"] = slot
        ordered.append(ref)
    ordered.extend(others)
    if apply and isinstance(shot, dict):
        shot["refs"] = ordered
    return ordered


def merge_storyboard_fields(
    shot: dict,
    source: dict | None = None,
    *,
    config: ShotPromptConfig | None = None,
) -> dict:
    """Copy storyboard.yaml fields into the shot when the switch is on.

    The default leaves those keys untouched and does not create ``opening``
    or fill ``continuity.prev``. Source keys are not deleted.
    """
    cfg = config or ShotPromptConfig()
    merged = copy.deepcopy(shot)
    if not cfg.merge_storyboard_yaml:
        return merged
    incoming = source if isinstance(source, dict) else shot
    chain_from = incoming.get("chain_from")
    continuity = merged.get("continuity")
    if not isinstance(continuity, dict):
        continuity = {}
        merged["continuity"] = continuity
    if chain_from and not continuity.get("prev"):
        continuity["prev"] = chain_from
    eyeline = incoming.get("eyeline")
    if isinstance(eyeline, str) and eyeline.strip():
        opening = merged.get("opening")
        if not isinstance(opening, dict):
            opening = {}
            merged["opening"] = opening
        if not opening.get("eyeline"):
            opening["eyeline"] = eyeline
    for key in STORYBOARD_YAML_FIELDS:
        if key in incoming and key not in merged:
            merged[key] = copy.deepcopy(incoming[key])
    return merged


def _string_block(
    value: Any,
    tag: str,
    label: str,
    keys: tuple[str, ...],
    errors: list[str],
) -> None:
    if not isinstance(value, dict):
        errors.append(f"M {tag}: {label} must be an object")
        return
    for key in keys:
        if key in value and not isinstance(value.get(key), str):
            errors.append(f"M {tag}: {label}.{key} must be a string")


def _check_aspect_value(value: Any, where: str, errors: list[str]) -> None:
    if not isinstance(value, str) or value not in ASPECT_RATIOS:
        choices = ", ".join(ASPECT_RATIOS)
        errors.append(f"M {where}: aspect_ratio {value!r} is not one of {choices}")


def check_document_spec(
    doc: dict,
    errors: list[str],
    _warns: list[str],
    config: ShotPromptConfig,
) -> None:
    if "aspect_ratio" in doc:
        _check_aspect_value(doc.get("aspect_ratio"), "document", errors)
    raw = doc.get("default_constraints")
    if "default_constraints" in doc and (
        not isinstance(raw, list) or not all(isinstance(item, str) for item in raw)
    ):
        errors.append("M document.default_constraints must be an array of strings")
    if config.strict and "aspect_ratio" not in doc:
        shots = doc.get("shots")
        if isinstance(shots, list) and shots and not any(
            isinstance(shot, dict) and "aspect_ratio" in shot for shot in shots
        ):
            errors.append("M document.aspect_ratio is required")


def _check_named_strings(shot: dict, tag: str, errors: list[str]) -> None:
    for key in ("performance", "optics", "physics"):
        if key in shot and not isinstance(shot.get(key), str):
            errors.append(f"M {tag}: {key} must be a string")
    if "opening" in shot:
        _string_block(shot.get("opening"), tag, "opening", OPENING_KEYS, errors)
    if "spatial" in shot:
        _string_block(shot.get("spatial"), tag, "spatial", SPATIAL_KEYS, errors)
    if "visual_style" in shot:
        style = shot.get("visual_style")
        if not isinstance(style, dict):
            errors.append(f"M {tag}: visual_style must be an object")
        else:
            note = style.get("note")
            if "note" in style and not isinstance(note, str):
                errors.append(f"M {tag}: visual_style.note must be a string")
            if "style_ref" in style:
                ref = style.get("style_ref")
                match = _canonical_match(ref) if isinstance(ref, str) else None
                if match is None or match.group("series") is None:
                    errors.append(
                        f"M {tag}: visual_style.style_ref {ref!r} is not an asset code"
                    )
                elif not str(match.group("series")).startswith("STY_"):
                    errors.append(
                        f"M {tag}: visual_style.style_ref {ref!r} is not a STY_ code"
                    )
    if "positive_locks" in shot:
        locks = shot.get("positive_locks")
        if not isinstance(locks, list) or not all(isinstance(item, str) for item in locks):
            errors.append(f"M {tag}: positive_locks must be an array of strings")


def _check_constraints(
    shot: dict,
    doc: dict,
    tag: str,
    errors: list[str],
    config: ShotPromptConfig,
) -> None:
    if "constraints" not in shot:
        return
    raw = shot.get("constraints")
    if not isinstance(raw, list):
        errors.append(f"M {tag}: constraints must be an array of strings")
        return
    defaults = set(_constraint_defaults(doc, config))
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            errors.append(f"M {tag}: constraints[{index}] must be a string")
            continue
        if item.startswith("-") and item[1:] not in defaults:
            errors.append(
                f"M {tag}: constraints[{index}] does not match a default "
                "and cannot be turned off"
            )


def _check_end_hold(shot: dict, tag: str, errors: list[str], *, required: bool) -> None:
    if "end_hold" not in shot:
        if required:
            errors.append(f"M {tag}: end_hold is required")
        return
    hold = shot.get("end_hold")
    if not isinstance(hold, dict):
        errors.append(f"M {tag}: end_hold must be an object")
        return
    kind = hold.get("type")
    ref = hold.get("ref")
    if kind not in END_HOLD_TYPES:
        errors.append(f"M {tag}: end_hold.type must be action or line")
    if not isinstance(ref, str) or not ref.strip():
        errors.append(f"M {tag}: end_hold.ref must be a string")
        return
    if kind == "line":
        ids = {
            line.get("line_id")
            for line in shot.get("dialogue") or []
            if isinstance(line, dict)
        }
        if ref not in ids:
            errors.append(f"M {tag}: end_hold.ref {ref} is not a dialogue line_id")


def _has_opening(shot: dict) -> bool:
    desc = shot.get("first_frame_desc")
    if isinstance(desc, str) and desc.strip():
        return True
    opening = shot.get("opening")
    if isinstance(opening, dict):
        return any(isinstance(value, str) and value.strip() for value in opening.values())
    return False


def _has_ending_picture(shot: dict) -> bool:
    for key in ("last_frame_desc", "end_state"):
        value = shot.get(key)
        if isinstance(value, str) and value.strip():
            return True
    plan = shot.get("video_plan")
    segments = plan.get("segments") if isinstance(plan, dict) else None
    if isinstance(segments, list) and segments and isinstance(segments[-1], dict):
        ending = segments[-1].get("ending")
        if isinstance(ending, str) and ending.strip():
            return True
    return False


def _check_segments(
    shot: dict,
    tag: str,
    duration: Any,
    errors: list[str],
    *,
    required: bool,
) -> None:
    plan = shot.get("video_plan")
    if plan is None:
        if required:
            errors.append(f"M {tag}: video_plan.segments must contain at least one segment")
        return
    if not isinstance(plan, dict):
        errors.append(f"M {tag}: video_plan must be an object")
        return
    if "segments" not in plan:
        if required:
            errors.append(f"M {tag}: video_plan.segments must contain at least one segment")
        return
    segments = plan.get("segments")
    if not isinstance(segments, list):
        errors.append(f"M {tag}: video_plan.segments must be an array")
        return
    if not segments:
        if required:
            errors.append(f"M {tag}: video_plan.segments must contain at least one segment")
        return
    line_ids = {
        line.get("line_id")
        for line in shot.get("dialogue") or []
        if isinstance(line, dict)
    }
    previous: float | None = None
    duration_n = _number(duration)
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            errors.append(f"M {tag}: segments[{index}] must be an object")
            continue
        if "cut_type" in segment and segment.get("cut_type") not in CUT_TYPES:
            errors.append(
                f"M {tag}: segments[{index}].cut_type {segment.get('cut_type')!r} "
                "is not one of opening, cut, oner"
            )
        if "shot_size" in segment and not isinstance(segment.get("shot_size"), str):
            errors.append(f"M {tag}: segments[{index}].shot_size must be a string")
        if "focus" in segment and not isinstance(segment.get("focus"), str):
            errors.append(f"M {tag}: segments[{index}].focus must be a string")
        cited = segment.get("line_ids")
        if "line_ids" in segment:
            if not isinstance(cited, list):
                errors.append(f"M {tag}: segments[{index}].line_ids must be an array")
            else:
                for line_id in cited:
                    if line_id not in line_ids:
                        errors.append(
                            f"M {tag}: video_plan segment cites {line_id} "
                            "which is not in this shot's dialogue"
                        )
        start = _number(segment.get("t0"))
        end = _number(segment.get("t1"))
        if start is None or end is None:
            errors.append(f"M {tag}: segments[{index}] t0 and t1 must be numbers")
            previous = None
            continue
        if start >= end:
            errors.append(
                f"M {tag}: segments[{index}] t0 {start:g} is not before t1 {end:g}"
            )
        if index == 0 and not _close(start, 0.0):
            errors.append(f"M {tag}: segments must start at 0")
        if previous is not None and not _close(start, previous):
            errors.append(
                f"M {tag}: segments[{index}] t0 {start:g} does not meet "
                f"the previous t1 {previous:g}"
            )
        previous = end
    if previous is not None and duration_n is not None and not _close(previous, duration_n):
        errors.append(
            f"M {tag}: segments end at {previous:g}, not duration_s {duration_n:g}"
        )


def _check_video_plan_h3(shot: dict, tag: str, errors: list[str]) -> None:
    plan = shot.get("video_plan")
    if not isinstance(plan, dict):
        return
    for key in (
        "retention_analysis",
        "integrated_description",
        "detailed_description",
        "non_diegetic_music",
    ):
        if key in plan and not isinstance(plan.get(key), str):
            errors.append(f"M {tag}: video_plan.{key} must be a string")
    if "summary_tasks" not in plan:
        return
    tasks = plan.get("summary_tasks")
    if not isinstance(tasks, list):
        errors.append(f"M {tag}: video_plan.summary_tasks must be an array")
        return
    allowed = set(SUMMARY_TASKS)
    for index, task in enumerate(tasks):
        if task not in allowed:
            errors.append(
                f"M {tag}: video_plan.summary_tasks[{index}] {task!r} "
                "is not an H3 summary task"
            )


def _retention_allowed(ref: dict) -> tuple[str, ...]:
    if _media(ref) == "audio":
        return AUDIO_RETENTION
    return VISIBLE_RETENTION


def _check_one_src(
    text: str,
    where: str,
    errors: list[str],
    warns: list[str],
    config: ShotPromptConfig,
    alias_index: AliasIndex | None,
) -> None:
    match = _canonical_match(text)
    if match is not None:
        if match.group("new") is not None:
            _bucket(errors, warns, config.unresolved_new_refs).append(
                f"M {where}: src {text} is unresolved"
            )
            return
        view = match.group("view")
        if view in _SHEET_VIEWS and not config.allow_sheet_in_video_refs:
            errors.append(
                f"M {where}: view 'sheet' is not a video reference "
                "(标准角色卡不进视频参考槽)"
            )
        return
    if alias_index is not None:
        code, conflicts = alias_index.lookup(text)
        if conflicts:
            errors.append(
                f"M {where}: src {text} matches more than one asset "
                f"({', '.join(conflicts)})"
            )
            return
        if code:
            if _view_of(code) in _SHEET_VIEWS and not config.allow_sheet_in_video_refs:
                errors.append(
                    f"M {where}: view 'sheet' is not a video reference "
                    "(标准角色卡不进视频参考槽)"
                )
                return
            warns.append(
                f"M {where}: src {text} is an alias of {code}; "
                "save should write the canonical id"
            )
            return
    if _is_external_locator(text):
        _bucket(errors, warns, "error" if config.strict else "warning").append(
            f"M {where}: src {text} is not a ComfyTV asset code or episode path"
        )
        return
    errors.append(f"M {where}: src {text} is unresolved")


def _check_ref_objects(
    shot: dict,
    tag: str,
    duration: Any,
    errors: list[str],
    warns: list[str],
    config: ShotPromptConfig,
    alias_index: AliasIndex | None,
    ref_limit_reason: str | None,
) -> None:
    refs = shot.get("refs")
    if not isinstance(refs, list):
        return
    grouped: dict[str, list[tuple[int, dict]]] = {kind: [] for kind in MEDIA_KINDS}
    for index, ref in enumerate(refs):
        if not isinstance(ref, dict):
            if config.strict or any(
                isinstance(item, dict) and "slot" in item for item in refs
            ):
                errors.append(f"M {tag}: refs[{index}] must be an object")
            continue
        media = ref.get("media")
        if media is not None and media not in MEDIA_KINDS:
            errors.append(
                f"M {tag}: refs[{index}].media {media!r} is not picture, video, or audio"
            )
            continue
        grouped[_media(ref)].append((index, ref))
        if "role" in ref and ref.get("role") not in REF_ROLES:
            warns.append(
                f"M {tag}: refs[{index}].role {ref.get('role')!r} is not a known reference role"
            )
        if "label" in ref and not isinstance(ref.get("label"), str):
            errors.append(f"M {tag}: refs[{index}].label must be a string")
        if "stale" in ref and not isinstance(ref.get("stale"), bool):
            errors.append(f"M {tag}: refs[{index}].stale must be a boolean")
        if "sha256" in ref:
            digest = ref.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                errors.append(f"M {tag}: refs[{index}].sha256 must be 64 hex characters")
        if "retention_note" in ref and not isinstance(ref.get("retention_note"), str):
            errors.append(f"M {tag}: refs[{index}].retention_note must be a string")
        if "retention" in ref and ref.get("retention") not in _retention_allowed(ref):
            errors.append(
                f"M {tag}: refs[{index}].retention {ref.get('retention')!r} "
                "does not match its media"
            )
        if "align_s" in ref:
            align = _number(ref.get("align_s"))
            length = _number(duration)
            if align is None:
                errors.append(f"M {tag}: refs[{index}].align_s must be a number")
            elif align < 0 or (length is not None and align - length > 1e-6):
                errors.append(
                    f"M {tag}: refs[{index}].align_s {align:g} is outside the shot"
                )
        src = ref.get("src")
        if isinstance(src, str) and src.strip():
            _check_one_src(
                src.strip(),
                f"{tag}: refs[{index}]",
                errors,
                warns,
                config,
                alias_index,
            )
        elif "slot" in ref or config.strict:
            errors.append(f"M {tag}: refs[{index}].src is required")
    for media, items in grouped.items():
        _check_slot_group(items, media, tag, errors)
    picture_count = len(grouped["picture"])
    limit = min(MAX_REFERENCE_IMAGES, config.max_refs)
    if picture_count > limit:
        detail = f" ({ref_limit_reason})" if ref_limit_reason else ""
        errors.append(
            f"M {tag}: refs has {picture_count} picture items, above the limit {limit}{detail}"
        )
    for media in ("video", "audio"):
        count = len(grouped[media])
        if count > MAX_REFERENCE_IMAGES:
            errors.append(
                f"M {tag}: {media} refs has {count} items, above the limit "
                f"{MAX_REFERENCE_IMAGES}"
            )
    locations = [
        ref for _index, ref in grouped["picture"] if ref.get("role") == "location"
    ]
    if len(locations) > 1:
        errors.append(
            f"M {tag}: a shot takes one location reference, found {len(locations)}"
        )


def _check_slot_group(
    items: list[tuple[int, dict]],
    media: str,
    tag: str,
    errors: list[str],
) -> None:
    if not items or not any("slot" in ref for _index, ref in items):
        return
    slots: list[int] = []
    valid = True
    for index, ref in items:
        if "slot" not in ref:
            errors.append(f"M {tag}: refs[{index}].slot is required once any {media} slot is set")
            valid = False
            continue
        slot = ref.get("slot")
        if isinstance(slot, bool) or not isinstance(slot, int):
            errors.append(f"M {tag}: refs[{index}].slot must be an integer")
            valid = False
            continue
        if slot < 1 or slot > MAX_REFERENCE_IMAGES:
            errors.append(f"M {tag}: refs[{index}].slot {slot} is outside 1–9")
            valid = False
            continue
        slots.append(slot)
    if valid and set(slots) != set(range(1, len(items) + 1)):
        shown = ", ".join(str(slot) for slot in slots)
        errors.append(
            f"M {tag}: {media} slots must be 1..{len(items)} with no gaps, got [{shown}]"
        )


def _cited_slots(refs: list, media: str) -> set[int] | None:
    """Slots of one media, or None when that media is not numbered yet."""
    found: list[int] = []
    saw_slot = False
    for ref in refs:
        if not isinstance(ref, dict) or _media(ref) != media:
            continue
        if "slot" not in ref:
            return None
        saw_slot = True
        slot = ref.get("slot")
        if isinstance(slot, int) and not isinstance(slot, bool):
            found.append(slot)
    if not saw_slot:
        return None
    return set(found)


def _check_subjects(
    shot: dict,
    tag: str,
    errors: list[str],
    warns: list[str],
    config: ShotPromptConfig,
    alias_index: AliasIndex | None,
) -> None:
    if "subjects" not in shot:
        return
    subjects = shot.get("subjects")
    if not isinstance(subjects, list):
        errors.append(f"M {tag}: subjects must be an array")
        return
    refs = shot.get("refs") if isinstance(shot.get("refs"), list) else []
    picture_slots = _cited_slots(refs, "picture")
    video_slots = _cited_slots(refs, "video")
    audio_slots = _cited_slots(refs, "audio")
    numbers: list[int] = []
    saw_number = False
    for index, subject in enumerate(subjects):
        if not isinstance(subject, dict):
            errors.append(f"M {tag}: subjects[{index}] must be an object")
            continue
        if "n" in subject:
            saw_number = True
            number = subject.get("n")
            if isinstance(number, bool) or not isinstance(number, int):
                errors.append(f"M {tag}: subjects[{index}].n must be an integer")
            else:
                numbers.append(number)
        if "kind" in subject and subject.get("kind") not in SUBJECT_KINDS:
            errors.append(
                f"M {tag}: subjects[{index}].kind {subject.get('kind')!r} "
                "is not an H3 subject kind"
            )
        for key in ("label", "text", "retention_note"):
            if key in subject and not isinstance(subject.get(key), str):
                errors.append(f"M {tag}: subjects[{index}].{key} must be a string")
        cites_audio = False
        cites_visible = False
        for key, pool, marker in (
            ("pictures", picture_slots, "picture"),
            ("videos", video_slots, "video"),
            ("audios", audio_slots, "audio"),
        ):
            if key not in subject:
                continue
            cited = subject.get(key)
            if not isinstance(cited, list) or not all(
                isinstance(item, int) and not isinstance(item, bool) for item in cited
            ):
                errors.append(f"M {tag}: subjects[{index}].{key} must be an array of integers")
                continue
            if key == "audios" and cited:
                cites_audio = True
            if key in ("pictures", "videos") and cited:
                cites_visible = True
            if pool is None:
                continue
            for slot in cited:
                if slot not in pool:
                    errors.append(
                        f"M {tag}: subjects[{index}].{key} cites {marker} slot {slot} "
                        "which is not on this shot"
                    )
        if "retention" in subject:
            retention = subject.get("retention")
            allowed = VISIBLE_RETENTION
            if cites_audio and not cites_visible:
                allowed = AUDIO_RETENTION
            elif cites_audio and cites_visible:
                allowed = VISIBLE_RETENTION + AUDIO_RETENTION
            if retention not in allowed:
                errors.append(
                    f"M {tag}: subjects[{index}].retention {retention!r} "
                    "does not match the cited media"
                )
        src = subject.get("src")
        if isinstance(src, str) and src.strip():
            _check_one_src(
                src.strip(),
                f"{tag}: subjects[{index}]",
                errors,
                warns,
                config,
                alias_index,
            )
    if saw_number and set(numbers) != set(range(1, len(subjects) + 1)):
        errors.append(
            f"M {tag}: subject numbers must be 1..{len(subjects)} with no gaps, "
            f"got {numbers}"
        )


def _check_dialogue_h3(shot: dict, tag: str, errors: list[str]) -> None:
    dialogue = shot.get("dialogue")
    if not isinstance(dialogue, list):
        return
    for index, line in enumerate(dialogue):
        if not isinstance(line, dict):
            continue
        if "language" in line and not isinstance(line.get("language"), str):
            errors.append(f"M {tag}: dialogue[{index}].language must be a string")
        for key in ("cutoff", "cross_cut"):
            if key in line and not isinstance(line.get(key), bool):
                errors.append(f"M {tag}: dialogue[{index}].{key} must be a boolean")


def _check_chain_tail(shot: dict, tag: str, errors: list[str]) -> None:
    continuity = shot.get("continuity") if isinstance(shot.get("continuity"), dict) else {}
    if not continuity.get("prev") or continuity.get("chain_break_reason"):
        return
    refs = shot.get("refs") if isinstance(shot.get("refs"), list) else []
    if any(isinstance(ref, dict) and ref.get("role") == "prev_last_frame" for ref in refs):
        return
    errors.append(f"M {tag}: continuity.prev is set without a prev_last_frame ref")


def check_shot_spec(
    doc: dict,
    shot: dict,
    tag: str,
    errors: list[str],
    warns: list[str],
    config: ShotPromptConfig,
    *,
    alias_index: AliasIndex | None = None,
    ref_limit_reason: str | None = None,
) -> None:
    """Validate optional prompt-spec and H3 fields. Does not change the shot."""
    if "aspect_ratio" in shot:
        _check_aspect_value(shot.get("aspect_ratio"), tag, errors)
    elif config.strict and not (
        isinstance(doc.get("aspect_ratio"), str) and doc.get("aspect_ratio") in ASPECT_RATIOS
    ):
        errors.append(f"M {tag}: aspect_ratio is required")
    _check_named_strings(shot, tag, errors)
    _check_constraints(shot, doc, tag, errors, config)
    _check_end_hold(shot, tag, errors, required=config.strict)
    if config.strict and not _has_opening(shot):
        errors.append(
            f"M {tag}: opening picture is required (opening or first_frame_desc)"
        )
    if config.strict and not _has_ending_picture(shot):
        errors.append(
            f"M {tag}: ending picture is required "
            "(last_frame_desc, end_state, or the last segment ending)"
        )
    _check_segments(
        shot,
        tag,
        shot.get("duration_s"),
        errors,
        required=config.strict,
    )
    _check_video_plan_h3(shot, tag, errors)
    _check_ref_objects(
        shot,
        tag,
        shot.get("duration_s"),
        errors,
        warns,
        config,
        alias_index,
        ref_limit_reason,
    )
    _check_subjects(shot, tag, errors, warns, config, alias_index)
    _check_dialogue_h3(shot, tag, errors)
    if config.require_chain_tail_frame:
        _check_chain_tail(shot, tag, errors)


def h3_inputs(
    shot: dict,
    doc: dict | None = None,
    *,
    config: ShotPromptConfig | None = None,
) -> dict:
    """Structured H3 inputs. This is not a prompt.

    Authored paragraphs win when they are present. Otherwise the renderer
    (F2-b) assembles them from segments, subjects, and the shot fields.
    """
    cfg = config or ShotPromptConfig()
    if not isinstance(shot, dict):
        return {}
    plan = shot.get("video_plan") if isinstance(shot.get("video_plan"), dict) else {}
    duration = shot.get("duration_s")
    refs = shot.get("refs") if isinstance(shot.get("refs"), list) else []
    grouped: dict[str, list[dict]] = {"picture": [], "video": [], "audio": []}
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        media = _media(ref)
        if media not in grouped:
            continue
        grouped[media].append({
            "slot": ref.get("slot"),
            "role": ref.get("role"),
            "src": ref.get("src"),
            "label": ref.get("label"),
            "sha256": ref.get("sha256"),
            "align_s": default_align_s(ref, duration) if media == "picture" else ref.get("align_s"),
            "retention": ref.get("retention"),
            "retention_note": ref.get("retention_note"),
        })
    tasks = plan.get("summary_tasks")
    if not isinstance(tasks, list):
        tasks = inferred_summary_tasks(shot)
    return {
        "mode": shot.get("mode"),
        "aspect_ratio": shot_aspect_ratio(doc, shot),
        "duration_s": duration,
        "sections": {
            "ref2va": H3_REF2VA_SECTIONS,
            "base": H3_BASE_SECTIONS,
        },
        "pictures": grouped["picture"],
        "videos": grouped["video"],
        "audios": grouped["audio"],
        "subjects": shot.get("subjects") if isinstance(shot.get("subjects"), list) else [],
        "summary": plan.get("summary_en") if isinstance(plan.get("summary_en"), str) else None,
        "summary_tasks": list(tasks),
        "retention_analysis": plan.get("retention_analysis")
        if isinstance(plan.get("retention_analysis"), str) else None,
        "detailed_description": plan.get("detailed_description")
        if isinstance(plan.get("detailed_description"), str) else None,
        "integrated_description": plan.get("integrated_description")
        if isinstance(plan.get("integrated_description"), str) else None,
        "soundscape": soundscape_text(shot),
        "non_diegetic_music": non_diegetic_music_text(shot),
        "opening": shot.get("opening") if isinstance(shot.get("opening"), dict) else None,
        "first_frame_desc": shot.get("first_frame_desc")
        if isinstance(shot.get("first_frame_desc"), str) else None,
        "spatial": shot.get("spatial") if isinstance(shot.get("spatial"), dict) else None,
        "performance": shot.get("performance") if isinstance(shot.get("performance"), str) else None,
        "optics": shot.get("optics") if isinstance(shot.get("optics"), str) else None,
        "physics": shot.get("physics") if isinstance(shot.get("physics"), str) else None,
        "lighting": shot.get("lighting") if isinstance(shot.get("lighting"), str) else None,
        "positive_locks": list(shot.get("positive_locks") or [])
        if isinstance(shot.get("positive_locks"), list) else [],
        "constraints": effective_constraints(shot, doc, config=cfg),
        "segments": plan.get("segments") if isinstance(plan.get("segments"), list) else [],
        "end_hold": shot.get("end_hold") if isinstance(shot.get("end_hold"), dict) else None,
        "ending_picture": shot.get("last_frame_desc")
        if isinstance(shot.get("last_frame_desc"), str) else None,
        "visual_style": shot.get("visual_style")
        if isinstance(shot.get("visual_style"), dict) else None,
        "dialogue": shot.get("dialogue") if isinstance(shot.get("dialogue"), list) else [],
    }


def reference_cap_for_workflow(workflow: str | None) -> tuple[int | None, str | None]:
    """Picture cap for a workflow label. Unknown labels are not guessed past 9."""
    if not isinstance(workflow, str) or not workflow.strip():
        return None, None
    label = workflow.strip()
    folded = re.sub(r"[\s_\-]+", "", label).lower()
    if "fl2va" in folded or "flf2v" in folded or "imagetovideo" in folded:
        return 2, (
            f"{label} accepts at most 2 reference images (MiniMaxH3ImageToVideo)"
        )
    if (
        "ref2va" in folded
        or "referencetovideo" in folded
        or folded.endswith("r2v")
        or "h3r2v" in folded
    ):
        return 9, (
            f"{label} accepts at most 9 reference images (MiniMaxH3ReferenceToVideo)"
        )
    try:
        from .ref_slots import load_image_cap
        cap = load_image_cap("video", label)
    except Exception:
        cap = None
    if cap is not None:
        return cap.limit, cap.reason
    return None, f"workflow {label} has no known reference-image cap; using 9"
