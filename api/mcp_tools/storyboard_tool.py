"""MCP read/write for a Storyboard stage's shot document.

``storyboard_data`` can be larger than get_stage's widget preview, so agents
read and write the full shots.json through this tool. The page stores the
document on the hidden widget and the open card reloads it without dropping
unknown keys.
"""

from __future__ import annotations

import json

from ...nodes.stages.common.shot_document import (
    load_shots_document,
    validate_shots_document,
)
from ...nodes.stages.common.shot_prompt_spec import (
    build_alias_index,
    normalize_config,
    reference_cap_for_workflow,
)
from . import _shared
from ._shared import _command_payload


def _document_arg(args: dict) -> dict:
    raw = args.get("document")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError("document is not valid JSON") from exc
    try:
        return load_shots_document(raw)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def _optional_bool(args: dict, key: str) -> bool | None:
    if key not in args:
        return None
    value = args.get(key)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be a boolean")


def _validate_report(args: dict) -> dict:
    document = _document_arg(args)
    strict = _optional_bool(args, "strict")
    allow_sheet = _optional_bool(args, "allow_sheet_in_video_refs")
    require_tail = _optional_bool(args, "require_chain_tail_frame")
    frames_mismatch = args.get("frames_mismatch")
    if frames_mismatch is not None and frames_mismatch not in ("warning", "error"):
        raise ValueError("frames_mismatch must be warning or error")
    unresolved = args.get("unresolved_new_refs")
    if unresolved is not None and unresolved not in ("warning", "error"):
        raise ValueError("unresolved_new_refs must be warning or error")
    max_refs = args.get("max_refs")
    if max_refs is not None and (isinstance(max_refs, bool) or not isinstance(max_refs, int)):
        raise ValueError("max_refs must be an integer")
    reason = None
    workflow = args.get("workflow")
    workflow_warning = None
    if isinstance(workflow, str) and workflow.strip():
        cap, note = reference_cap_for_workflow(workflow)
        if cap is None:
            workflow_warning = note
        elif not isinstance(max_refs, int) or cap < max_refs:
            max_refs = cap
            reason = note
    shot_ids = args.get("shot_ids")
    if shot_ids is not None and (
        not isinstance(shot_ids, list) or not all(isinstance(item, str) for item in shot_ids)
    ):
        raise ValueError("shot_ids must be an array of strings")
    aliases = args.get("aliases")
    alias_index = None
    if aliases is not None:
        if not isinstance(aliases, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in aliases.items()
        ):
            raise ValueError("aliases must be an object of name to asset code")
        alias_index = build_alias_index(aliases)
    config = normalize_config(
        strict=strict,
        allow_sheet_in_video_refs=allow_sheet,
        require_chain_tail_frame=require_tail,
        frames_mismatch=frames_mismatch if isinstance(frames_mismatch, str) else None,
        unresolved_new_refs=unresolved if isinstance(unresolved, str) else None,
    )
    errors, warnings = validate_shots_document(
        document,
        config=config,
        shot_ids=shot_ids,
        max_refs=max_refs if isinstance(max_refs, int) else None,
        ref_limit_reason=reason,
        alias_index=alias_index,
    )
    if workflow_warning:
        warnings = list(warnings)
        warnings.append(f"M document: {workflow_warning}")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


async def _storyboard(args: dict) -> dict:
    action = str(args.get("action") or "get")
    if action == "validate":
        return _validate_report(args)
    if action not in ("get", "set"):
        raise ValueError("action must be get, set, or validate")
    node = args.get("node")
    if not node:
        raise ValueError("node is required (StoryboardStage uid or graph node id)")
    payload = _command_payload(args, ("project_id",))
    payload["node"] = str(node)
    if action == "get":
        return await _shared.submit_or_headless("storyboard_get", payload)
    payload["document"] = _document_arg(args)
    return await _shared.submit_or_headless("storyboard_set", payload)


TOOLS: dict[str, dict] = {
    "storyboard": {
        "description": (
            "Read or write a Storyboard stage's full shot document "
            "(the storyboard_data widget). action 'get' returns the document "
            "exactly as stored, including drama-pipeline fields (shot_id, "
            "duration_s, characters, dialogue lines with speaker, refs as "
            "asset codes like CHR_name@vN#view or episode:shot/file, "
            "prompts.krea2 / prompts.h3_image, first_frame_desc / "
            "last_frame_desc, camera fields angle/move/lens/cam_idx) and any "
            "custom keys. action 'set' replaces that document; pass document "
            "as an object with a shots array (a shots.json file). Unknown "
            "keys are kept. extras on a shot is the integer crowd count, not "
            "a custom-field bag. Asset codes are not resolved. action "
            "'validate' checks document with the production mechanical rules "
            "and does not need an open page. Optional prompt-spec fields "
            "(aspect_ratio, refs.slot, subjects, H3 summary_tasks, retention, "
            "align_s, picture/video/audio refs) are checked when present. "
            "strict=true also requires aspect ratio, an opening picture, an "
            "ending picture, end_hold, and at least one segment. Asset src "
            "values are ComfyTV series codes (ID@vN#view) or episode: paths. "
            "aliases maps a name to a canonical code. workflow sets the "
            "picture cap (FL2VA/FLF2V is 2, Ref2VA is 9). The episode total sums "
            "duration_s (cut_to is only checked against that length). The "
            "target_duration_s comparison is skipped when the board cannot "
            "reach the target even at 15s per shot (a single shot or a short "
            "partial extract). 'get' and 'set' use the open "
            "page when one is mirroring; with no page they read and write "
            "storyboard_data on the persisted headless graph. get_stage truncates a large "
            "storyboard_data preview — use this tool for the full document. "
            "node is the StoryboardStage uid or graph node id."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "set", "validate"],
                },
                "node": {"type": "string"},
                "document": {"type": "object"},
                "project_id": {"type": "string"},
                "shot_ids": {"type": "array", "items": {"type": "string"}},
                "strict": {"type": "boolean"},
                "max_refs": {"type": "integer"},
                "workflow": {"type": "string"},
                "allow_sheet_in_video_refs": {"type": "boolean"},
                "require_chain_tail_frame": {"type": "boolean"},
                "frames_mismatch": {"type": "string", "enum": ["warning", "error"]},
                "unresolved_new_refs": {"type": "string", "enum": ["warning", "error"]},
                "aliases": {"type": "object", "additionalProperties": {"type": "string"}},
            },
            "required": [],
            "additionalProperties": False,
        },
        "handler": _storyboard,
    },
}
