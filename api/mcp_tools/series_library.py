"""MCP tool for the series library.yaml asset index."""

from ...storage.series_library import (
    bind_series_library,
    get_series_asset,
    import_series_manifest,
    list_series_assets,
    lock_series_asset,
    new_series_asset_version,
    resolve_root_arg,
    resolve_series_ref,
    resolve_series_shot,
)


def _folder(args: dict):
    return resolve_root_arg(args.get("root"), args.get("series"))


def _flag(args: dict, key: str, default: bool) -> bool:
    value = args.get(key, default)
    return value if isinstance(value, bool) else default


async def _series_library(args: dict) -> dict:
    action = str(args.get("action") or "")
    if action == "bind":
        root = args.get("root")
        if not root:
            raise ValueError("root is required")
        return bind_series_library(root)
    if action == "list":
        return list_series_assets(
            _folder(args),
            kind=args.get("kind") or None,
            status=args.get("status") or None,
            episode=args.get("episode") or None,
        )
    if action == "get":
        code = str(args.get("code") or "").strip()
        if not code:
            raise ValueError("code is required")
        return {"asset": get_series_asset(_folder(args), code)}
    if action == "register":
        asset = args.get("asset")
        if not isinstance(asset, dict):
            raise ValueError("asset object is required")
        return {"asset": register_series_asset(_folder(args), asset)}
    if action == "lock":
        code = str(args.get("code") or "").strip()
        if not code:
            raise ValueError("code is required")
        return {"asset": lock_series_asset(
            _folder(args),
            code,
            approved_by=str(args.get("approved_by") or ""),
            approved_at=args.get("approved_at"),
            approval_ref=args.get("approval_ref"),
        )}
    if action == "new_version":
        code = str(args.get("code") or "").strip()
        if not code:
            raise ValueError("code is required")
        files = args.get("files") if isinstance(args.get("files"), dict) else None
        return {"asset": new_series_asset_version(_folder(args), code, files=files)}
    if action == "resolve":
        folder = _folder(args)
        require_locked = _flag(args, "require_locked", True)
        if isinstance(args.get("shot"), dict):
            return resolve_series_shot(folder, args["shot"], require_locked=require_locked)
        refs = args.get("refs")
        if isinstance(refs, list):
            return {"results": [
                resolve_series_ref(
                    folder, str(ref), require_locked=require_locked,
                    verify=_flag(args, "verify", False),
                )
                for ref in refs
            ]}
        ref = args.get("ref")
        if not ref:
            raise ValueError("ref, refs, or shot is required")
        return {"resolved": resolve_series_ref(
            folder, str(ref), require_locked=require_locked,
            verify=_flag(args, "verify", False),
        )}
    if action == "import_manifest":
        manifest = args.get("manifest") if isinstance(args.get("manifest"), dict) else None
        manifest_path = args.get("manifest_path")
        if manifest is None and not manifest_path:
            raise ValueError("manifest or manifest_path is required")
        return import_series_manifest(
            _folder(args),
            manifest,
            manifest_path=str(manifest_path) if manifest_path else None,
        )
    raise ValueError(
        f"unknown action {action!r} — valid: list, get, register, lock, "
        "new_version, resolve, import_manifest, bind"
    )


TOOLS: dict[str, dict] = {
    "series_library": {
        "description": (
            "Series asset index stored as library.yaml inside a series folder. "
            "Does not need an open ComfyTV page; it only reads and writes that "
            "file on disk. "
            "(schema_version 1, the production kit format). Codes look like "
            "CHR_xiaolongnv@v1#front, LOC_gumu_shishi@v1#empty, "
            "PRP_zhutai@v1#lit, STY_01@v1#style, VOX_xiaolongnv@v1#voice_ref. "
            "Kinds and prefixes: CHR_ character, OUT_ outfit, LOOK_ look, "
            "LOC_ location (LOC_<place>__<card> camera), PRP_ prop, VOX_ voice, "
            "STY_ style, BIB_ bible. Locked versions are immutable; changing an "
            "asset creates a new version and the previous library.yaml is copied "
            "to library.history/library.<timestamp>.yaml. action 'bind' points "
            "at a series folder (saved under the ComfyUI user dir) so later "
            "calls can omit root. 'list'/'get' read the file as-is. 'register' "
            "appends one draft or locked record whose files already exist under "
            "<ID>/v<n>/. 'lock' sets a draft to locked (approved_by required) "
            "and makes that version directory read-only. 'new_version' appends "
            "v(n+1) as a draft copy and leaves the old version untouched. "
            "'resolve' maps a code, a list of codes, or a shot "
            "(characters/location/props/refs, including {CHR_...} and "
            "episode:/NEW: refs) to files. 'import_manifest' bulk-registers a "
            "JSON or YAML manifest in the same record shape; it will not "
            "modify a locked or draft version that already exists. Omit "
            "#view to use the kind default (character/look/outfit front, "
            "location empty or the only camera card, voice voice_ref, style "
            "style). A prop with more than one state needs #view."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list", "get", "register", "lock", "new_version",
                        "resolve", "import_manifest", "bind",
                    ],
                },
                "root": {"type": "string"},
                "series": {"type": "string"},
                "kind": {"type": "string"},
                "status": {"type": "string"},
                "episode": {"type": "string"},
                "code": {"type": "string"},
                "asset": {"type": "object"},
                "approved_by": {"type": "string"},
                "approved_at": {"type": "string"},
                "approval_ref": {"type": "string"},
                "files": {"type": "object"},
                "ref": {"type": "string"},
                "refs": {"type": "array", "items": {"type": "string"}},
                "shot": {"type": "object"},
                "manifest": {"type": "object"},
                "manifest_path": {"type": "string"},
                "require_locked": {"type": "boolean"},
                "verify": {"type": "boolean"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "handler": _series_library,
    },
}
