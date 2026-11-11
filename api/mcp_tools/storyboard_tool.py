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


async def _storyboard(args: dict) -> dict:
    action = str(args.get("action") or "get")
    if action == "validate":
        document = _document_arg(args)
        errors, warnings = validate_shots_document(document)
        return {"ok": not errors, "errors": errors, "warnings": warnings}
    if action not in ("get", "set"):
        raise ValueError("action must be get, set, or validate")
    node = args.get("node")
    if not node:
        raise ValueError("node is required (StoryboardStage uid or graph node id)")
    payload = _command_payload(args, ("project_id",))
    payload["node"] = str(node)
    if action == "get":
        return await _shared.submit_command("storyboard_get", payload)
    payload["document"] = _document_arg(args)
    return await _shared.submit_command("storyboard_set", payload)


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
            "and does not need an open page. get_stage truncates a large "
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
            },
            "required": [],
            "additionalProperties": False,
        },
        "handler": _storyboard,
    },
}
