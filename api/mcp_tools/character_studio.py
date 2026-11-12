"""MCP tool for the character studio (records, stills, voice auditions)."""

from ...storage.character_studio import (
    attach_job,
    bind_voice,
    compose_outfit,
    derive_views,
    diff_characters,
    get_character,
    import_face,
    make_sheet,
    pick_candidate,
    refine,
    resolve_character_name,
    voice_audition,
)
from ...storage.series_library import SeriesLibraryError, resolve_root_arg
from ..headless_queue import enqueue_graph


def _folder(args: dict):
    return resolve_root_arg(args.get("root"), args.get("series"))


def _optional_int(args: dict, key: str):
    value = args.get(key)
    if value is None or value == "":
        return None
    return int(value)


async def _queue(planned: dict, folder) -> dict:
    queued = await enqueue_graph({"prompt": planned["prompt"]})
    attach_job(folder, planned["code"], planned["op_id"], queued["job_id"])
    planned["job_id"] = queued["job_id"]
    planned["status"] = queued.get("status") or "queued"
    return planned


async def _character_studio(args: dict) -> dict:
    action = str(args.get("action") or "")
    folder = _folder(args)
    try:
        if action == "get":
            return get_character(folder, str(args.get("code") or ""))
        if action == "import_face":
            return import_face(
                folder,
                str(args.get("code") or ""),
                args.get("files") or [],
                str(args.get("provenance") or ""),
                args.get("note"),
            )
        if action == "compose_outfit":
            planned = compose_outfit(
                folder,
                str(args.get("code") or ""),
                str(args.get("face") or ""),
                str(args.get("outfit_body") or ""),
                batch=_optional_int(args, "batch"),
                seed=_optional_int(args, "seed"),
                image_1=args.get("image_1"),
            )
            return await _queue(planned, folder)
        if action == "make_sheet":
            planned = make_sheet(
                folder,
                str(args.get("code") or ""),
                source=args.get("from") or None,
                seed=_optional_int(args, "seed"),
            )
            return await _queue(planned, folder)
        if action == "derive_views":
            views = args.get("views") if isinstance(args.get("views"), list) else None
            planned = derive_views(
                folder,
                str(args.get("code") or ""),
                views,
                seed=_optional_int(args, "seed"),
            )
            return await _queue(planned, folder)
        if action == "refine":
            planned = refine(
                folder,
                str(args.get("code") or ""),
                str(args.get("part") or ""),
                str(args.get("instruction") or ""),
                seed=_optional_int(args, "seed"),
            )
            return await _queue(planned, folder)
        if action == "pick":
            return pick_candidate(
                folder,
                str(args.get("code") or ""),
                str(args.get("op_id") or ""),
                int(args.get("index") or 0),
                str(args.get("as_view") or ""),
                source=args.get("source") or None,
            )
        if action == "voice_audition":
            seeds = args.get("seeds") if isinstance(args.get("seeds"), list) else []
            planned = voice_audition(
                folder,
                str(args.get("voice_code") or ""),
                str(args.get("text") or ""),
                seeds,
                desc_voice_en=args.get("desc_voice_en"),
            )
            return await _queue(planned, folder)
        if action == "bind_voice":
            return bind_voice(
                folder,
                str(args.get("code") or ""),
                str(args.get("voice_code") or ""),
            )
        if action == "diff":
            return diff_characters(folder, str(args.get("a") or ""), str(args.get("b") or ""))
        if action == "resolve_alias":
            try:
                return {"resolved": resolve_character_name(folder, str(args.get("name") or ""))}
            except SeriesLibraryError as exc:
                conflicts = exc.details.get("conflicts")
                if conflicts:
                    return {
                        "resolved": False,
                        "conflicts": conflicts,
                        "name": exc.details.get("name"),
                    }
                raise
    except SeriesLibraryError:
        raise
    raise ValueError(
        f"unknown action {action!r} — valid: get, import_face, compose_outfit, "
        "make_sheet, derive_views, refine, pick, voice_audition, bind_voice, "
        "diff, resolve_alias"
    )


TOOLS: dict[str, dict] = {
    "character_studio": {
        "description": (
            "Character studio on a series library.yaml. Records stay in that "
            "file; images and audio stay under <ID>/v<n>/. Still-image actions "
            "assemble a MiniMax H3 Image graph (image VAE, <Picture N> matching "
            "image_1..N, image_1 is the picture being edited) and queue one "
            "headless job. voice_audition assembles the existing comfytv_s2pro "
            "node (ComfyTV.S2Pro by default) and also returns a job_id. "
            "Neither path calls interrupt, free, or cancel_stage. "
            "get accepts CHR_x, CHR_x@vN, a Chinese name, an alias, a "
            "legacy_ids entry, or a P01_/P02_ name when that CHR_ exists. "
            "resolve_alias returns the canonical CHR_@vN#front (highest locked "
            "version) or a conflict list. import_face requires provenance "
            "self, licensed, or ai_generated. compose_outfit defaults image_1 "
            "to the outfit/body picture. make_sheet writes a review sheet that "
            "is not a video reference unless comfytv.character_studio."
            "sheet_in_video_ref is true. derive_views always starts from front. "
            "refine starts from the draft's root image, not the previous refine, "
            "and a locked version becomes a new draft first. pick is 1-based "
            "and only writes a draft or candidate. Lock and new_version stay on "
            "series_library; lock requires approved_by 张景 and locked OUT_/LOOK_/VOX_ refs. "
            "status candidate is picked-but-unlocked and is never treated as locked."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "get", "import_face", "compose_outfit", "make_sheet",
                        "derive_views", "refine", "pick", "voice_audition",
                        "bind_voice", "diff", "resolve_alias",
                    ],
                },
                "root": {"type": "string"},
                "series": {"type": "string"},
                "code": {"type": "string"},
                "files": {"type": "array", "items": {"type": "string"}},
                "provenance": {"type": "string"},
                "note": {"type": "string"},
                "face": {"type": "string"},
                "outfit_body": {"type": "string"},
                "batch": {"type": "integer"},
                "seed": {"type": "integer"},
                "image_1": {"type": "string"},
                "from": {"type": "string"},
                "views": {"type": "array", "items": {"type": "string"}},
                "part": {"type": "string"},
                "instruction": {"type": "string"},
                "op_id": {"type": "string"},
                "index": {"type": "integer"},
                "as_view": {"type": "string"},
                "source": {"type": "string"},
                "voice_code": {"type": "string"},
                "text": {"type": "string"},
                "seeds": {"type": "array", "items": {"type": "integer"}},
                "desc_voice_en": {"type": "string"},
                "a": {"type": "string"},
                "b": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "handler": _character_studio,
    },
}
