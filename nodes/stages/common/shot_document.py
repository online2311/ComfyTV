"""Extensible storyboard documents.

The drama pipeline's episode file (``shots.json``) is stored as-is. A
document is a JSON object with a ``shots`` array plus any envelope keys
(``series``, ``episode``, ``title_zh``, ``fps``, ``target_duration_s``,
``new_assets``, and anything else the production file carries).

Each shot uses the field names from the production validator. ``extras`` is
the integer crowd count from that format, not a bag of custom data. Keys
this module does not know are kept on the shot, in place, and round-trip
through save, load, MCP, and the Storyboard / Shot Images stages.

Asset references stay codes (``CHR_…@vN#view``, ``NEW:…``, ``episode:…``).
Those codes name files ComfyTV stores. Resolving them to library files is
a separate asset index. Optional prompt-spec fields (``aspect_ratio``,
``refs[].slot``, ``subjects``, H3 inputs) are defined in ``shot_prompt_spec``.
They are not inserted when absent.

The 16 Chinese columns (``shot_no``, ``scene_purpose``, ``image_prompt``, …)
remain the legacy editor view. Import does not rewrite a production document
into that view.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .shot_prompt_spec import (
    AliasIndex,
    ShotPromptConfig,
    check_document_spec,
    check_shot_spec,
    normalize_config,
)

# Envelope keys the production file uses. Other envelope keys are preserved.
DOCUMENT_FIELDS = (
    "series",
    "episode",
    "title_zh",
    "fps",
    "target_duration_s",
    "new_assets",
    "aspect_ratio",
    "default_constraints",
    "shots",
)

# Shot fields the production validator and the E01 list actually use.
# ``extras`` is a crowd-count integer.
SHOT_FIELDS = (
    "shot_id",
    "scene_id",
    "order",
    "beats",
    "duration_s",
    "cut_to",
    "frames",
    "mode",
    "shot_size",
    "angle",
    "move",
    "lens",
    "cam_idx",
    "purpose",
    "reveal",
    "characters",
    "extras",
    "location",
    "props",
    "start_state",
    "unique_action",
    "end_state",
    "variation_type",
    "dialogue",
    "continuity",
    "first_frame_desc",
    "last_frame_desc",
    "motion_desc",
    "lighting",
    "audio",
    "refs",
    "subjects",
    "prompts",
    "video_plan",
    "aspect_ratio",
    "visual_style",
    "opening",
    "spatial",
    "performance",
    "optics",
    "physics",
    "end_hold",
    "constraints",
    "positive_locks",
)

# Legacy StoryboardStage columns. Still first-class, still preserved.
LEGACY_FIELDS = (
    "shot_no",
    "duration",
    "scene_purpose",
    "character",
    "character_desc",
    "character_img",
    "reference_img",
    "action",
    "emotion",
    "scene_tags",
    "sfx",
    "prompt",
    "image_prompt",
    "motion_prompt",
    "image_url",
    "reference_images",
    "seed",
)

_DOC_MARKERS = ("episode", "series", "target_duration_s", "new_assets")
_REQUIRED_SHOT = (
    "shot_id", "order", "scene_id", "duration_s", "characters",
    "dialogue", "mode", "refs", "variation_type",
)

_REF_RE = re.compile(
    r"^(?P<id>[A-Z]+_[A-Za-z0-9_]+)"
    r"(?:@v(?P<version>\d+))?"
    r"(?:#(?P<view>[A-Za-z0-9_]+))?$"
)

# Mechanical checks ported from the production validate_shots.py.
# JSON Schema (schema/shots.schema.json) is not bundled here.
CPS = 4.0
RESERVE_S = 1.0
RED_LINES = [
    r"\bnude\b", r"\bnaked\b", r"\bnudity\b", r"\btopless\b", r"\bnipples?\b",
    r"\bgenital", r"\bbreasts?\b", r"\bbare (chest|back|torso|breast)",
    r"\bundress(ed)? completely\b", r"\bsex\b", r"\bintercourse\b",
    r"\berotic\b", r"\bunderwear\b", r"\blingerie\b", r"\bcleavage\b",
    "裸", "赤身", "一丝不挂", "全身衣服脱", "衣衫尽褪", "褪尽",
    "交合", "性爱", "做爱", "胸部", "乳", "下体", "生殖", "肚兜", "内衣",
]
RED_RE = [re.compile(p, re.I) for p in RED_LINES]
TEXT_FIELDS = [
    "purpose", "reveal", "start_state", "unique_action", "end_state",
    "first_frame_desc", "last_frame_desc", "motion_desc", "lighting",
]
_PROMPT_LOCK_RE = re.compile(
    r"\b(ponytail|wears|dressed in|hair (?:falls|hangs))\b", re.I,
)
_DIALOGUE_PUNCT_RE = re.compile(r"[\s，。！？、…—“”‘’,.!?\"'：:；;（）()\[\]]")
_HIDDEN_VISIBLE = ("back", "offscreen", "hands_only", "silhouette")
_FRAME_ROLES = ("first_frame", "prev_last_frame", "end_composition", "last_frame")


def parse_asset_ref(raw: str) -> dict:
    """Split ``CHR_yangguo@v1#front`` / ``NEW:LOC_x`` / ``episode:…``.

    ``id`` is the bare asset code (no version, no view). Callers compare
    dialogue speakers to that id. This does not look up a library.
    """
    text = str(raw or "")
    if text.startswith("NEW:"):
        return {"type": "new", "id": text[4:], "version": None, "view": None}
    if text.startswith("episode:"):
        return {"type": "episode", "id": text, "version": None, "view": None}
    match = _REF_RE.match(text)
    if not match:
        return {"type": "unknown", "id": text, "version": None, "view": None}
    version = match.group("version")
    return {
        "type": "series",
        "id": match.group("id"),
        "version": int(version) if version else None,
        "view": match.group("view"),
    }


def is_production_shot(shot: Any) -> bool:
    if not isinstance(shot, dict):
        return False
    if isinstance(shot.get("shot_id"), str):
        return True
    if "duration_s" in shot:
        return True
    return isinstance(shot.get("dialogue"), list)


def is_production_document(doc: Any) -> bool:
    if not isinstance(doc, dict) or not isinstance(doc.get("shots"), list):
        return False
    if any(key in doc for key in _DOC_MARKERS):
        return True
    return any(is_production_shot(shot) for shot in doc["shots"])


def copy_unknown_fields(dst: dict, src: dict) -> dict:
    """Copy keys from ``src`` that ``dst`` does not already define."""
    for key, value in src.items():
        if key not in dst:
            dst[key] = value
    return dst


def shot_image_prompt(shot: Any) -> str:
    """Prompt Shot Images should render.

    Production shots keep the model prompts under ``prompts``. Legacy shots
    keep ``image_prompt`` / ``prompt``. Asset-code ``refs`` are not prompts
    and are not turned into URLs here.
    """
    if not isinstance(shot, dict):
        return ""
    prompts = shot.get("prompts")
    if isinstance(prompts, dict):
        for key in ("h3_image", "krea2"):
            value = prompts.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("image_prompt", "prompt"):
        value = shot.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_shots_document(raw: str | dict) -> dict:
    if isinstance(raw, str):
        try:
            doc = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError("shots document is not valid JSON") from exc
    elif isinstance(raw, dict):
        doc = raw
    else:
        raise ValueError("shots document must be a JSON object")
    if not isinstance(doc, dict) or not isinstance(doc.get("shots"), list):
        raise ValueError("shots document must be an object with a shots array")
    if not all(isinstance(shot, dict) for shot in doc["shots"]):
        raise ValueError("every shot must be an object")
    return doc


def dump_shots_document(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False)


def frames_for(duration_s: float, fps: int = 24) -> int:
    need = int(round(float(duration_s) * fps))
    n = max(0, -(-(need - 5) // 17))
    return 17 * n + 5


def _iter_text(shot: dict):
    for field in TEXT_FIELDS:
        value = shot.get(field)
        if isinstance(value, str):
            yield field, value
    dialogue = shot.get("dialogue") or []
    if isinstance(dialogue, list):
        for index, line in enumerate(dialogue):
            if isinstance(line, dict):
                yield f"dialogue[{index}].text", str(line.get("text", ""))
    prompts = shot.get("prompts") or {}
    if isinstance(prompts, dict):
        for key, value in prompts.items():
            if isinstance(value, str):
                yield f"prompts.{key}", value
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    yield f"prompts.{key}[{index}]", json.dumps(item, ensure_ascii=False)
    for field in ("performance", "optics", "physics"):
        value = shot.get(field)
        if isinstance(value, str):
            yield field, value
    for field in ("opening", "spatial"):
        block = shot.get(field)
        if isinstance(block, dict):
            for key, value in block.items():
                if isinstance(value, str):
                    yield f"{field}.{key}", value
    style = shot.get("visual_style")
    if isinstance(style, dict) and isinstance(style.get("note"), str):
        yield "visual_style.note", style["note"]
    hold = shot.get("end_hold")
    if isinstance(hold, dict) and isinstance(hold.get("ref"), str):
        yield "end_hold.ref", hold["ref"]
    for field in ("constraints", "positive_locks"):
        items = shot.get(field)
        if isinstance(items, list):
            for index, item in enumerate(items):
                if isinstance(item, str) and not item.startswith("-"):
                    yield f"{field}[{index}]", item
    subjects = shot.get("subjects")
    if isinstance(subjects, list):
        for index, subject in enumerate(subjects):
            if not isinstance(subject, dict):
                continue
            for key in ("text", "retention_note", "label"):
                value = subject.get(key)
                if isinstance(value, str):
                    yield f"subjects[{index}].{key}", value
    plan = shot.get("video_plan")
    if isinstance(plan, dict):
        for key in (
            "retention_analysis",
            "integrated_description",
            "detailed_description",
            "non_diegetic_music",
        ):
            value = plan.get(key)
            if isinstance(value, str):
                yield f"video_plan.{key}", value


def _character_ids(shot: dict) -> set[str]:
    found: set[str] = set()
    characters = shot.get("characters")
    if not isinstance(characters, list):
        return found
    for character in characters:
        if isinstance(character, dict):
            found.add(str(parse_asset_ref(str(character.get("id") or "")).get("id") or ""))
    return found


# Per-shot length bounds. The episode-target check uses the max: a board
# that cannot reach target_duration_s even if every shot were this long is a
# single shot or a partial extract, not a failed episode.
_SHOT_S_MIN = 5
_SHOT_S_MAX = 15


def validate_shots_document(
    doc: dict,
    *,
    config: ShotPromptConfig | None = None,
    shot_ids: list[str] | None = None,
    max_refs: int | None = None,
    ref_limit_reason: str | None = None,
    alias_index: AliasIndex | None = None,
) -> tuple[list[str], list[str]]:
    """Mechanical checks from the production ``validate_shots.py``.

    Returns ``(errors, warnings)``. Schema-file checks are not included.
    A duration outside 5–15s is an error, matching that script's stated rule.
    Optional prompt-spec fields are checked when present. Missing ones are
    not errors unless ``config.strict`` is set. The default config leaves an
    older document on the same error list as before.

    The episode total sums ``duration_s`` (the shot length). ``cut_to`` is
    only compared with that length; it is not the shot's contribution to the
    episode. The comparison with ``target_duration_s`` is skipped when the
    board cannot reach the target even at 15s per shot.
    """
    cfg = normalize_config(config)
    if isinstance(max_refs, int) and not isinstance(max_refs, bool):
        cfg = normalize_config(cfg, max_refs=min(cfg.max_refs, max(0, max_refs)))
    selected = None if shot_ids is None else {str(item) for item in shot_ids}
    errors: list[str] = []
    warns: list[str] = []
    if not isinstance(doc, dict):
        return ["M document must be an object"], warns
    episode = doc.get("episode")
    shots = doc.get("shots")
    if not isinstance(episode, str) or not episode:
        errors.append("M document.episode is required")
    if not isinstance(shots, list):
        errors.append("M document.shots must be an array")
        return errors, warns
    check_document_spec(doc, errors, warns, cfg)

    seen: set[str] = set()
    prev_by_scene: dict[str, dict] = {}
    total = 0.0
    counted = 0
    for index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            errors.append(f"M shots[{index}] must be an object")
            continue
        missing = [key for key in _REQUIRED_SHOT if key not in shot]
        shot_id = shot.get("shot_id")
        tag = str(shot_id) if isinstance(shot_id, str) and shot_id else f"shots[{index}]"
        if selected is not None and str(shot_id or "") not in selected:
            duration = shot.get("duration_s")
            if isinstance(duration, (int, float)) and not isinstance(duration, bool):
                total += float(duration)
                counted += 1
            continue
        for key in missing:
            errors.append(f"M {tag}: missing {key}")
        if missing:
            continue
        if shot_id in seen:
            errors.append(f"M {tag}: duplicate shot_id")
        seen.add(str(shot_id))
        if shot.get("order") != index + 1:
            errors.append(f"M {tag}: order {shot.get('order')} != position {index + 1}")
        scene_id = str(shot.get("scene_id") or "")
        if not str(shot_id).startswith(scene_id + "_") or not str(shot_id).startswith(str(episode or "")):
            errors.append(f"M {tag}: shot_id must start with scene_id and episode")
        duration = shot.get("duration_s")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            errors.append(f"M {tag}: duration_s must be a number")
            continue
        if isinstance(duration, float):
            errors.append(f"M {tag}: duration_s must be an integer number of seconds")
        if duration < _SHOT_S_MIN or duration > _SHOT_S_MAX:
            errors.append(f"M {tag}: duration_s {duration} outside 5–15s")
        cut = shot.get("cut_to")
        if cut is not None and isinstance(cut, (int, float)) and not isinstance(cut, bool) and cut > duration:
            errors.append(f"M {tag}: cut_to {cut} > duration {duration}")
        total += float(duration)
        counted += 1
        expected_frames = frames_for(duration)
        if shot.get("frames") not in (None, expected_frames):
            message = (
                f"M {tag}: frames {shot.get('frames')} -> compile will use "
                f"{expected_frames} (24fps, 17n+5)"
            )
            if cfg.frames_mismatch == "error":
                errors.append(message)
            else:
                warns.append(message)
        if not isinstance(shot.get("characters"), list) or not isinstance(shot.get("dialogue"), list):
            errors.append(f"M {tag}: characters and dialogue must be arrays")
            continue
        if not isinstance(shot.get("refs"), list):
            errors.append(f"M {tag}: refs must be an array")
            continue
        _check_dialogue(shot, tag, duration, errors, warns)
        _check_mode(shot, tag, errors, warns)
        _check_chain(shot, tag, prev_by_scene, errors, warns)
        prev_by_scene[scene_id] = shot
        _check_plan_and_prompts(shot, tag, errors, warns)
        _check_refs(shot, tag, errors, warns)
        check_shot_spec(
            doc, shot, tag, errors, warns, cfg,
            alias_index=alias_index,
            ref_limit_reason=ref_limit_reason,
        )
        extras = shot.get("extras")
        if isinstance(extras, (int, float)) and not isinstance(extras, bool) and extras > 0:
            warns.append(f"S {tag}: extras={extras} — make sure crowd is planned")
        for field, text in _iter_text(shot):
            for pattern in RED_RE:
                found = pattern.search(text or "")
                if found:
                    errors.append(f"M {tag}: content red line '{found.group(0)}' in {field}")
                    break
    target = doc.get("target_duration_s")
    if (
        isinstance(target, (int, float))
        and not isinstance(target, bool)
        and target
        and counted * _SHOT_S_MAX >= float(target)
    ):
        if abs(total - target) > 0.15 * target:
            errors.append(f"M episode: total {total:.1f}s vs target {target}s (±15%)")
        elif abs(total - target) > 0.05 * target:
            warns.append(f"M episode: total {total:.1f}s vs target {target}s")
    return errors, warns


def _check_dialogue(shot: dict, tag: str, duration: float, errors: list[str], warns: list[str]) -> None:
    ids = _character_ids(shot)
    spoken = 0
    for line in shot["dialogue"]:
        if not isinstance(line, dict):
            errors.append(f"M {tag}: dialogue entries must be objects")
            continue
        text = str(line.get("text") or "")
        spoken += len(_DIALOGUE_PUNCT_RE.sub("", text))
        line_id = line.get("line_id")
        if line.get("kind") == "onscreen":
            speaker = line.get("speaker")
            if speaker not in ids:
                errors.append(
                    f"M {tag}: onscreen line {line_id} speaker {speaker} not in characters"
                )
            else:
                visible = next(
                    (
                        character.get("visible")
                        for character in shot["characters"]
                        if isinstance(character, dict)
                        and parse_asset_ref(str(character.get("id") or "")).get("id") == speaker
                    ),
                    None,
                )
                if visible in _HIDDEN_VISIBLE:
                    errors.append(
                        f"M {tag}: {line_id} spoken onscreen but speaker visible={visible} "
                        "(use inner/voiceover or turn to camera)"
                    )
            if line.get("lips_visible") is False:
                warns.append(f"S {tag}: {line_id} onscreen but lips_visible=false")
        start = line.get("start_s")
        if isinstance(start, (int, float)) and not isinstance(start, bool) and start >= duration:
            errors.append(f"M {tag}: {line_id} start_s beyond duration")
    if spoken and spoken / CPS + RESERVE_S > duration + 1e-6:
        errors.append(
            f"M {tag}: {spoken} 字 needs {spoken / CPS + RESERVE_S:.1f}s > duration {duration}s"
        )


def _check_mode(shot: dict, tag: str, errors: list[str], warns: list[str]) -> None:
    visible_people = [
        character for character in shot["characters"]
        if isinstance(character, dict) and character.get("visible") not in ("offscreen",)
    ]
    refs = [ref for ref in shot["refs"] if isinstance(ref, dict)]
    if shot.get("mode") == "FL2VA":
        if len(visible_people) > 1:
            errors.append(
                f"M {tag}: FL2VA with {len(visible_people)} people in frame — "
                "use Ref2VA (FL2VA has no identity refs)"
            )
        if any(ref.get("role") not in ("first_frame", "prev_last_frame", "last_frame") for ref in refs):
            warns.append(f"M {tag}: FL2VA ignores non-frame refs (they will be dropped for video)")
    if shot.get("variation_type") == "large" and shot.get("mode") == "Ref2VA":
        warns.append(f"S {tag}: variation_type large on Ref2VA — consider FL2VA or splitting")
    if shot.get("mode") == "T2VA":
        warns.append(f"S {tag}: T2VA has no refs — identity not locked; avoid for characters")


def _check_chain(shot: dict, tag: str, prev_by_scene: dict[str, dict], errors: list[str], warns: list[str]) -> None:
    continuity = shot.get("continuity") or {}
    if not isinstance(continuity, dict):
        errors.append(f"M {tag}: continuity must be an object")
        continuity = {}
    previous = prev_by_scene.get(str(shot.get("scene_id")))
    if previous is None:
        if continuity.get("prev") and not continuity.get("chain_break_reason"):
            warns.append(
                f"M {tag}: first shot of scene points prev={continuity.get('prev')} "
                "without chain_break_reason"
            )
    elif continuity.get("prev") != previous.get("shot_id") and not continuity.get("chain_break_reason"):
        errors.append(
            f"M {tag}: same-scene shot must chain prev={previous.get('shot_id')} "
            "or give chain_break_reason"
        )
    if not str(continuity.get("cut_motive") or "").strip():
        errors.append(f"M {tag}: continuity.cut_motive empty")


def _check_plan_and_prompts(shot: dict, tag: str, errors: list[str], warns: list[str]) -> None:
    line_ids = {
        line.get("line_id")
        for line in shot["dialogue"]
        if isinstance(line, dict)
    }
    plan = shot.get("video_plan") or {}
    segments = plan.get("segments") if isinstance(plan, dict) else None
    segment_lines: list = []
    if isinstance(segments, list):
        for segment in segments:
            if isinstance(segment, dict):
                segment_lines.extend(segment.get("lines") or [])
    for line_id in segment_lines:
        if line_id not in line_ids:
            errors.append(f"M {tag}: video_plan segment cites {line_id} which is not in this shot's dialogue")
    for line in shot["dialogue"]:
        if isinstance(line, dict) and line.get("kind") == "onscreen" and line.get("line_id") not in segment_lines:
            warns.append(f"M {tag}: onscreen line {line.get('line_id')} not placed in any video_plan segment")
    prompts = shot.get("prompts") if isinstance(shot.get("prompts"), dict) else {}
    for key in ("krea2", "h3_image"):
        if _PROMPT_LOCK_RE.search(str(prompts.get(key) or "")):
            warns.append(f"S {tag}: prompts.{key} describes hair/outfit; locks are pasted by the compiler, remove it")


def _check_refs(shot: dict, tag: str, errors: list[str], warns: list[str]) -> None:
    refs = [ref for ref in shot["refs"] if isinstance(ref, dict)]
    for ref in refs:
        if ref.get("role") in _FRAME_ROLES and not str(ref.get("src") or "").startswith("episode:"):
            warns.append(f"M {tag}: {ref.get('role')} ref should be an episode: frame, got {ref.get('src')}")
    if shot.get("mode") == "FL2VA" and not any(ref.get("role") in ("first_frame", "prev_last_frame") for ref in refs):
        errors.append(f"M {tag}: FL2VA needs a first_frame or prev_last_frame ref")
    continuity = shot.get("continuity") if isinstance(shot.get("continuity"), dict) else {}
    if continuity.get("prev") and shot.get("mode") == "FL2VA" and not any(ref.get("role") == "prev_last_frame" for ref in refs):
        warns.append(f"M {tag}: chained FL2VA usually starts from prev_last_frame")
