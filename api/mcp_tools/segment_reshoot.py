"""MCP tool for planning and adopting an audio-only segment reshoot."""

from ...storage.segment_reshoot import (
    ReshootError,
    adopt,
    list_reshoots,
    plan,
    rollback,
    run,
    status,
)


async def _segment_reshoot(args: dict) -> dict:
    action = str(args.get("action") or "")
    try:
        if action == "plan":
            return plan(args)
        if action == "run":
            return run(args)
        if action == "status":
            return status(args)
        if action == "list":
            return list_reshoots(args)
        if action == "adopt":
            return adopt(args)
        if action == "rollback":
            return rollback(args)
    except ReshootError as exc:
        raise ValueError(str(exc)) from exc
    raise ValueError(
        f"unknown action {action!r} — valid: plan, run, status, list, adopt, rollback"
    )


TOOLS: dict[str, dict] = {
    "segment_reshoot": {
        "description": (
            "Reshoot one time range inside a single take, audio only. "
            "action 'plan' aligns [t_start, t_end) to the source frame rate, "
            "records handles and the 17n+5 generation length, and writes a "
            "reshoot record. It does not submit a ComfyUI prompt. "
            "action 'run' splices head, replacement, and tail with the ffmpeg "
            "concat runner and returns when that local splice finishes. "
            "'status' and 'list' read the records. 'adopt' and 'rollback' "
            "move the active version pointer and require approved_by '张景'. "
            "Records are JSON under the ComfyTV user directory at "
            "comfytv/projects/<project_id>/reshoots/<shot_id>/ "
            "(takes.json plus take_<n>_r<k>.json). New media is "
            "comfytv/reshoot/<project_id>/<shot_id>/take_<n>_r<k>.mp4 in the "
            "output directory. Imported takes and protected rough_cut files "
            "are not overwritten. lipsync and regen are refused. "
            "length_policy defaults to fit; ripple and allow_multiple are opt-in."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["plan", "run", "status", "list", "adopt", "rollback"],
                },
                "project_id": {"type": "string"},
                "shot_id": {"type": "string"},
                "reshoot_id": {"type": "string"},
                "source": {"type": "object"},
                "t_start": {"type": "number"},
                "t_end": {"type": "number"},
                "windows": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "mode": {"type": "string", "enum": ["audio_only", "lipsync", "regen"]},
                "changes": {"type": "object"},
                "audio": {"type": "string"},
                "handles_s": {"type": "number"},
                "length_policy": {"type": "string", "enum": ["fit", "ripple"]},
                "allow_multiple": {"type": "boolean"},
                "audio_fade_ms": {"type": "integer"},
                "workflow": {"type": "string"},
                "approved_by": {"type": "string"},
                "to": {"type": "string"},
                "cut_to": {"type": "number"},
                "next_shot_id": {"type": "string"},
                "created_by": {"type": "string"},
                "previews": {"type": "boolean"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "handler": _segment_reshoot,
    },
}
