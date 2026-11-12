"""Series asset index backed by a folder's library.yaml (schema_version 1).

The file in the series folder is the index. Locked versions are not edited;
a change is a new version. Writes copy the previous file to library.history/
first. Asset codes follow LIBRARY_FORMAT: ``CHR_xiaolongnv@v1#front``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import stat
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import yaml

from ..backup import data_dir
from .asset_root import comfytv_asset_root, series_dir

_WRITE_LOCK = threading.Lock()
_SNAPSHOT_SEQ = 0

_DEFAULT_PREAMBLE = (
    "# 系列资产库索引（唯一真相）。只追加；已锁定版本永不修改。"
    "格式见 docs/LIBRARY_FORMAT.md\n"
)

_KIND_PREFIX = {
    "character": "CHR_",
    "outfit": "OUT_",
    "look": "LOOK_",
    "location": "LOC_",
    "camera": "LOC_",
    "prop": "PRP_",
    "voice": "VOX_",
    "style": "STY_",
    "bible": "BIB_",
}

_ASSET_RE = re.compile(
    r"^(?P<id>(?:CHR|OUT|LOOK|LOC|PRP|VOX|STY|BIB)_[A-Za-z0-9_]+)"
    r"(?:@v(?P<version>[0-9]+))?"
    r"(?:#(?P<view>[A-Za-z0-9_]+))?$"
)
_VIEW_RE = re.compile(r"^[A-Za-z0-9_]+$")
_CAMERA_VIEW_RE = re.compile(r"^[ABW]_")

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus"}
_TEXT_EXT = {".md", ".txt", ".yaml", ".yml", ".json", ".csv"}

_DROP_ON_NEW_VERSION = (
    "approved_by",
    "approved_at",
    "approval_ref",
    "locked_at",
    "approval_history",
)
_COMPUTED_FILE_KEYS = ("sha256", "bytes", "payload_url")


class SeriesLibraryError(ValueError):
    def __init__(self, message: str, *, status: int = 400, details: dict | None = None):
        super().__init__(message)
        self.http_status = status
        self.details = details or {}


_ASSET_STATUSES = ("draft", "candidate", "locked")
_MUTABLE_STATUSES = ("draft", "candidate")
_LOCK_APPROVER = "张景"
_PROVENANCE = {"self", "licensed", "ai_generated"}
_POLICY_CACHE: dict | None = None


def parse_series_ref(text: str) -> dict:
    """Parse ``ID@vN#view``, ``{ID@vN#view}``, ``episode:...``, or ``NEW:...``."""
    original = str(text or "").strip()
    raw = original
    if len(raw) >= 2 and raw.startswith("{") and raw.endswith("}"):
        raw = raw[1:-1].strip()
    if raw.startswith("episode:"):
        return {
            "kind": "episode",
            "ref": original,
            "path": raw[len("episode:"):],
            "resolved": False,
        }
    if raw.startswith("NEW:"):
        return {
            "kind": "unresolved",
            "ref": original,
            "proposed_id": raw[len("NEW:"):],
            "resolved": False,
        }
    match = _ASSET_RE.match(raw)
    if match is None:
        return {"kind": "invalid", "ref": original, "resolved": False}
    version = match.group("version")
    return {
        "kind": "asset",
        "ref": raw,
        "id": match.group("id"),
        "version": int(version) if version is not None else None,
        "view": match.group("view"),
        "resolved": False,
    }


def bind_series_library(root) -> dict:
    folder = _as_root(root)
    doc, _text = _load(folder)
    series = doc.get("series") if isinstance(doc.get("series"), dict) else {}
    series_id = str(series.get("id") or folder.name)
    data = _read_bindings()
    roots = data.setdefault("roots", {})
    if not isinstance(roots, dict):
        roots = {}
        data["roots"] = roots
    roots[series_id] = str(folder)
    data["active"] = series_id
    _write_bindings(data)
    return {"series_id": series_id, "root": str(folder), "active": series_id}


def list_series_assets(root, *, kind=None, status=None, episode=None) -> dict:
    folder = _as_root(root)
    doc, _text = _load(folder)
    rows = []
    for asset in doc.get("assets") or []:
        if not isinstance(asset, dict) or not asset.get("id"):
            continue
        if kind and asset.get("kind") != kind:
            continue
        if status and asset.get("status") != status:
            continue
        if episode and not _matches_episode(asset, str(episode)):
            continue
        version = int(asset.get("version") or 0)
        rows.append({
            "id": asset.get("id"),
            "version": version,
            "kind": asset.get("kind"),
            "status": asset.get("status"),
            "name_zh": asset.get("name_zh") or "",
            "code": f"{asset.get('id')}@v{version}",
            "views": sorted((asset.get("files") or {}).keys()),
            "approved_by": asset.get("approved_by"),
        })
    series = doc.get("series") if isinstance(doc.get("series"), dict) else {}
    return {
        "schema_version": doc.get("schema_version") or 1,
        "series": series,
        "root": str(folder),
        "assets": rows,
    }


def get_series_asset(root, code: str) -> dict:
    folder = _as_root(root)
    doc, _text = _load(folder)
    parsed = _require_asset_code(code)
    record, _defaulted = _select_record(doc, parsed, require_locked=False)
    return copy.deepcopy(record)


def register_series_asset(root, asset: dict) -> dict:
    folder = _as_root(root)
    with _WRITE_LOCK:
        doc, preamble = _open_for_write(folder)
        prepared = _prepare_record(folder, asset)
        if _by_version(doc, prepared["id"], prepared["version"]) is not None:
            raise SeriesLibraryError(
                f"{prepared['id']}@v{prepared['version']} already exists",
                status=409,
            )
        doc.setdefault("assets", []).append(prepared)
        if prepared.get("kind") == "character":
            _assert_alias_unique(doc, prepared["id"])
        _require_locked_dependencies(doc, [prepared])
        _save(folder, doc, preamble)
        if prepared["status"] == "locked":
            _chmod_version(folder, prepared["id"], prepared["version"])
        return copy.deepcopy(prepared)


def asset_store_policy() -> dict:
    """Migration and directory policy. Box paths in the file are notes only."""
    global _POLICY_CACHE
    if _POLICY_CACHE is None:
        path = Path(__file__).with_name("asset_store.yaml")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise SeriesLibraryError("asset_store.yaml must be a mapping")
        _POLICY_CACHE = loaded
    return _POLICY_CACHE


def read_series_document(root) -> dict:
    folder = _as_root(root)
    doc, _text = _load(folder)
    return copy.deepcopy(doc)


def character_dependencies(root, code: str) -> list[dict]:
    """Lock state of the OUT_, LOOK_, and VOX_ refs on a character version."""
    folder = _as_root(root)
    doc, _text = _load(folder)
    parsed = _require_asset_code(code)
    record, _defaulted = _select_record(doc, parsed, require_locked=False)
    if record.get("kind") != "character":
        return []
    return character_dependency_rows(doc, record)


def character_dependency_rows(doc: dict, record: dict) -> list[dict]:
    return [_resolve_dependency(doc, ref) for ref in _character_dep_refs(record)]


def _require_locked_dependencies(doc: dict, records: list[dict]) -> None:
    """Locked characters in this write must not point at missing or unlocked deps."""
    for record in records:
        if record.get("kind") != "character" or record.get("status") != "locked":
            continue
        gaps = [row for row in character_dependency_rows(doc, record) if not row["locked"]]
        if not gaps:
            continue
        label = f"{record['id']}@v{int(record['version'])}"
        listed = ", ".join(f"{row['code']} ({row['status']})" for row in gaps)
        raise SeriesLibraryError(
            f"cannot lock {label}; dependencies are not locked: {listed}")


def lock_series_asset(root, code: str, *, approved_by: str,
                      approved_at: str | None = None,
                      approval_ref: str | None = None) -> dict:
    approver = str(approved_by or "").strip()
    if not approver:
        raise SeriesLibraryError("approved_by is required to lock a version")
    if approver != _LOCK_APPROVER:
        raise SeriesLibraryError("approved_by must be 张景")
    folder = _as_root(root)
    with _WRITE_LOCK:
        doc, preamble = _open_for_write(folder)
        parsed = _require_asset_code(code)
        record = _tip_record(doc, parsed["id"], parsed["version"])
        label = f"{record['id']}@v{int(record['version'])}"
        if record.get("status") == "locked":
            raise SeriesLibraryError(f"{label} is already locked", status=409)
        if record.get("kind") == "character":
            gaps = [
                row for row in character_dependency_rows(doc, record)
                if not row["locked"]
            ]
            if gaps:
                listed = ", ".join(
                    f"{row['code']} ({row['status']})" for row in gaps)
                raise SeriesLibraryError(
                    f"cannot lock {label}; dependencies are not locked: {listed}")
        now = _now()
        record["status"] = "locked"
        record["approved_by"] = approver
        record["approved_at"] = approved_at or now
        record["locked_at"] = now
        if approval_ref:
            record["approval_ref"] = approval_ref
        _save(folder, doc, preamble)
        _chmod_version(folder, record["id"], int(record["version"]))
        return copy.deepcopy(record)


def new_series_asset_version(root, code: str, *, files: dict | None = None) -> dict:
    folder = _as_root(root)
    with _WRITE_LOCK:
        doc, preamble = _open_for_write(folder)
        parsed = _require_asset_code(code)
        source = _tip_record(doc, parsed["id"], parsed["version"])
        asset_id = source["id"]
        existing = [
            int(row.get("version") or 0)
            for row in _versions(doc, asset_id)
        ]
        new_version = max(existing) + 1
        if _by_version(doc, asset_id, new_version) is not None:
            raise SeriesLibraryError(
                f"{asset_id}@v{new_version} already exists", status=409)
        fresh = copy.deepcopy(source)
        for key in _DROP_ON_NEW_VERSION:
            fresh.pop(key, None)
        fresh["version"] = new_version
        fresh["status"] = "draft"
        fresh["supersedes"] = f"{asset_id}@v{int(source['version'])}"
        if files is None:
            fresh["files"] = _copy_files(folder, source, new_version)
        else:
            fresh["files"] = _prepare_files(folder, asset_id, new_version, files)
        doc.setdefault("assets", []).append(fresh)
        _save(folder, doc, preamble)
        return copy.deepcopy(fresh)


def update_draft_series_asset(root, code: str, changes: dict) -> dict:
    """Replace mutable fields on a draft or candidate. Locked versions are refused."""
    if not isinstance(changes, dict):
        raise SeriesLibraryError("changes must be a mapping")
    folder = _as_root(root)
    with _WRITE_LOCK:
        doc, preamble = _open_for_write(folder)
        parsed = _require_asset_code(code)
        record = _tip_record(doc, parsed["id"], parsed["version"])
        label = f"{record['id']}@v{int(record['version'])}"
        if record.get("status") not in _MUTABLE_STATUSES:
            raise SeriesLibraryError(
                f"{label} is {record.get('status')}; create a new version",
                status=409,
            )
        for key in ("id", "kind", "version"):
            if key in changes and changes[key] != record.get(key):
                raise SeriesLibraryError(f"cannot change {key} on {label}")
        if "status" in changes:
            raise SeriesLibraryError("status changes go through lock")
        for key in _DROP_ON_NEW_VERSION:
            if key in changes:
                raise SeriesLibraryError(f"{key} is only set by lock")
        files = changes.get("files")
        for key, value in changes.items():
            if key in ("files", "id", "kind", "version"):
                continue
            record[key] = copy.deepcopy(value)
        if record.get("status") != "locked":
            for key in _DROP_ON_NEW_VERSION:
                record.pop(key, None)
        if files is not None:
            merged = dict(record.get("files") or {})
            if not isinstance(files, dict):
                raise SeriesLibraryError("files must be a mapping")
            merged.update(files)
            record["files"] = _prepare_files(
                folder, record["id"], int(record["version"]), merged)
        _validate_record_extras(record)
        _normalize_name_lists(record)
        if record.get("kind") == "character":
            _assert_alias_unique(doc, record["id"])
        _save(folder, doc, preamble)
        return copy.deepcopy(record)


def resolve_series_ref(root, ref: str, *, require_locked: bool = True,
                       verify: bool = False) -> dict:
    folder = _as_root(root)
    parsed = parse_series_ref(ref)
    if parsed["kind"] in ("episode", "unresolved"):
        return parsed
    if parsed["kind"] != "asset":
        raise SeriesLibraryError(f"not an asset code: {ref!r}")
    doc, _text = _load(folder)
    return _resolve_parsed(folder, doc, parsed, require_locked=require_locked,
                           verify=verify)


def resolve_series_shot(root, shot: dict, *, require_locked: bool = True) -> dict:
    folder = _as_root(root)
    doc, _text = _load(folder)
    results = []
    for parsed in iter_shot_refs(shot):
        if parsed["kind"] != "asset":
            results.append(parsed)
            continue
        try:
            results.append(_resolve_parsed(
                folder, doc, parsed, require_locked=require_locked, verify=False))
        except SeriesLibraryError as exc:
            results.append({
                "kind": "missing",
                "ref": parsed.get("ref"),
                "id": parsed.get("id"),
                "resolved": False,
                "error": str(exc),
            })
    return {"results": results}


def import_series_manifest(root, manifest: dict | None = None, *,
                           manifest_path: str | None = None,
                           allow_excluded: bool = False) -> dict:
    if manifest_path:
        if manifest is not None:
            raise SeriesLibraryError("pass manifest or manifest_path, not both")
        manifest = _read_manifest_file(manifest_path)
    if not isinstance(manifest, dict):
        raise SeriesLibraryError("manifest must be an object")
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        raise SeriesLibraryError("manifest assets must be a list")
    _reject_excluded_manifest(manifest, allow_excluded=allow_excluded)
    declared = _declared_manifest_files(manifest)
    _check_manifest_inventory(manifest, declared)
    folder = _as_root_allow_empty(root)
    with _WRITE_LOCK:
        path = folder / "library.yaml"
        if path.is_file():
            doc, text = _load(folder)
            preamble = _preamble_from(text)
        else:
            series = manifest.get("series") if isinstance(manifest.get("series"), dict) else {}
            series = dict(series)
            series.setdefault("id", folder.name)
            doc = {"schema_version": 1, "series": series, "assets": []}
            if isinstance(manifest.get("comfytv"), dict):
                doc["comfytv"] = copy.deepcopy(manifest["comfytv"])
            preamble = _DEFAULT_PREAMBLE
        added: list[str] = []
        skipped: list[str] = []
        staged: list[dict] = []
        for raw in assets:
            if not isinstance(raw, dict):
                raise SeriesLibraryError("each manifest asset must be a mapping")
            asset_id = str(raw.get("id") or "").strip()
            try:
                version = int(raw.get("version"))
            except (TypeError, ValueError):
                raise SeriesLibraryError(f"asset {asset_id or '?'} needs a version")
            label = f"{asset_id}@v{version}"
            existing = _by_version(doc, asset_id, version)
            prepared = _prepare_record(folder, raw)
            if existing is None:
                staged.append(prepared)
                added.append(label)
                continue
            if _same_record(existing, prepared):
                skipped.append(label)
                continue
            if existing.get("status") == "locked":
                raise SeriesLibraryError(
                    f"locked version {label} cannot be changed", status=409)
            raise SeriesLibraryError(
                f"{existing.get('status')} version {label} cannot be changed; "
                "lock it or create a new version",
                status=409,
            )
        if staged or not path.is_file():
            folder.mkdir(parents=True, exist_ok=True)
            doc.setdefault("assets", []).extend(staged)
            for prepared in staged:
                if prepared.get("kind") == "character":
                    _assert_alias_unique(doc, prepared["id"])
            _require_locked_dependencies(doc, staged)
            _save(folder, doc, preamble)
            for prepared in staged:
                if prepared.get("status") == "locked":
                    _chmod_version(folder, prepared["id"], int(prepared["version"]))
        checked = sum(1 for info in declared if str(info.get("sha256") or "").strip())
        return {
            "added": added,
            "skipped": skipped,
            "root": str(folder.resolve()),
            "file_count": len(declared),
            "sha256_checked": checked,
        }


def image_urls_for_shot(shot: dict, root=None) -> list[str]:
    """Image payload URLs for asset codes on a shot. Empty when unbound."""
    if not isinstance(shot, dict):
        return []
    try:
        folder = _as_root(root) if root else _peek_bound_root()
    except SeriesLibraryError:
        return []
    if folder is None:
        return []
    try:
        resolved = resolve_series_shot(folder, shot)
    except SeriesLibraryError:
        return []
    urls: list[str] = []
    for row in resolved["results"]:
        if not row.get("resolved") or row.get("media_type") != "image":
            continue
        url = row.get("payload_url") or row.get("absolute_path")
        if url and url not in urls:
            urls.append(url)
    return urls


def iter_shot_refs(shot: dict) -> list[dict]:
    """Asset codes already stored on a shot. Does not change the shot."""
    if not isinstance(shot, dict):
        return []
    found: list[dict] = []

    def add(ref, view=None) -> None:
        if ref is None:
            return
        text = str(ref).strip()
        if not text:
            return
        parsed = parse_series_ref(text)
        if parsed["kind"] == "invalid":
            return
        if view and parsed["kind"] == "asset" and not parsed.get("view"):
            view_name = str(view).strip()
            if view_name:
                parsed = dict(parsed)
                parsed["view"] = view_name
        found.append(parsed)

    for key in ("refs", "reference_images"):
        raw = shot.get(key)
        if isinstance(raw, str):
            add(raw)
        elif isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, str):
                    add(entry)
                elif isinstance(entry, dict):
                    add(entry.get("src") or entry.get("id"), entry.get("view"))
    characters = shot.get("characters")
    if isinstance(characters, list):
        for character in characters:
            if isinstance(character, str):
                add(character)
            elif isinstance(character, dict):
                add(character.get("id"))
                add(character.get("outfit"))
                if character.get("look"):
                    add(character.get("look"))
    location = shot.get("location")
    if isinstance(location, str):
        add(location)
    elif isinstance(location, dict):
        add(location.get("id"), location.get("camera_card"))
    props = shot.get("props")
    if isinstance(props, list):
        for prop in props:
            if isinstance(prop, str):
                add(prop)
            elif isinstance(prop, dict):
                add(prop.get("id"), prop.get("state") or prop.get("view"))
    add(shot.get("cam_idx"))
    return found


def resolve_root_arg(root=None, series: str | None = None) -> Path:
    """Folder that holds ``library.yaml``.

    ``series`` must be a single path segment. The folder, including a
    ``spark_root`` redirect, must stay inside ``comfytv_asset_root()``.
    """
    if series is not None:
        series = _require_series_name(series)
    if root:
        folder = _honor_spark_root(_as_root(root))
    else:
        try:
            folder = _bound_root(series)
        except SeriesLibraryError:
            if series is None:
                raise
            folder = None
        if folder is None:
            folder = _as_root(series_dir(series))
        folder = _honor_spark_root(folder)
    _require_inside_asset_root(folder)
    return folder


def _require_series_name(series: str) -> str:
    """Reject empty names, absolute paths, and any ``/``, ``\\``, or ``..``."""
    text = str(series or "").strip()
    if (
        not text
        or Path(text).is_absolute()
        or text.startswith("\\")
        or "/" in text
        or "\\" in text
        or ".." in text
    ):
        raise SeriesLibraryError(f"series name is not safe: {series!r}")
    return text


def _honor_spark_root(folder: Path) -> Path:
    library = folder / "library.yaml"
    if not library.is_file():
        return folder
    try:
        loaded = yaml.safe_load(library.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return folder
    if not isinstance(loaded, dict):
        return folder
    series = loaded.get("series") if isinstance(loaded.get("series"), dict) else {}
    raw = series.get("spark_root")
    if not isinstance(raw, str) or not raw.strip():
        raw = loaded.get("spark_root")
    if not isinstance(raw, str) or not raw.strip():
        return folder
    target = Path(raw).expanduser()
    if not target.is_dir():
        raise SeriesLibraryError(f"spark_root is not a directory: {raw}", status=404)
    return target.resolve()


def _require_inside_asset_root(folder: Path) -> None:
    root = comfytv_asset_root().resolve()
    try:
        folder.resolve().relative_to(root)
    except ValueError:
        raise SeriesLibraryError(
            f"series folder must be inside {root}", status=400)


def _resolve_parsed(folder: Path, doc: dict, parsed: dict, *,
                    require_locked: bool, verify: bool) -> dict:
    record, version_defaulted = _select_record(
        doc, parsed, require_locked=require_locked)
    chain: list[str] = []
    terminal, info, view, rel, absolute = _follow_file(
        folder, doc, record, parsed.get("view"), chain, require_locked)
    if verify and info.get("sha256"):
        digest, _size = _hash_file(absolute)
        if digest != info.get("sha256"):
            raise SeriesLibraryError(f"sha256 mismatch for {rel}", status=409)
    media = info.get("media_type") or _media_from_path(rel)
    requested_version = int(record["version"])
    return {
        "resolved": True,
        "kind": record.get("kind"),
        "ref": parsed.get("ref"),
        "id": record.get("id"),
        "version": requested_version,
        "view": view,
        "status": record.get("status"),
        "resolved_id": terminal.get("id"),
        "resolved_version": int(terminal.get("version") or requested_version),
        "resolved_view": view,
        "alias_chain": chain,
        "path": rel,
        "absolute_path": str(absolute),
        "payload_url": _payload_url(doc, rel, info.get("payload_url")),
        "sha256": info.get("sha256"),
        "media_type": media,
        "usage": list(info.get("usage") or []),
        "version_defaulted": version_defaulted,
    }


def _follow_file(folder, doc, record, view, chain, require_locked):
    if len(chain) > 8:
        raise SeriesLibraryError("alias cycle in asset files", status=409)
    if require_locked and record.get("status") != "locked":
        label = f"{record.get('id')}@v{record.get('version')}"
        raise SeriesLibraryError(f"{label} is not locked", status=409)
    files = record.get("files") if isinstance(record.get("files"), dict) else {}
    chosen = view or _default_view(record)
    info = files.get(chosen)
    if not isinstance(info, dict):
        label = f"{record.get('id')}@v{record.get('version')}"
        raise SeriesLibraryError(
            f"view {chosen!r} not found on {label}", status=404)
    chain.append(f"{record.get('id')}@v{int(record.get('version') or 0)}#{chosen}")
    alias = info.get("alias_of") if info.get("alias_of") else info.get("alias")
    if isinstance(alias, str) and alias.strip():
        parsed = parse_series_ref(alias)
        if parsed["kind"] != "asset":
            raise SeriesLibraryError(f"bad alias {alias}")
        target, _defaulted = _select_record(doc, parsed, require_locked=require_locked)
        return _follow_file(
            folder, doc, target, parsed.get("view"), chain, require_locked)
    rel = info.get("path")
    if not isinstance(rel, str) or not rel.strip():
        raise SeriesLibraryError(
            f"{chain[-1]} has no file path", status=404)
    absolute = _safe_join(folder, rel)
    if not absolute.is_file():
        raise SeriesLibraryError(f"file missing for {chain[-1]}: {rel}", status=404)
    return record, info, chosen, rel.replace("\\", "/"), absolute


def _default_view(record: dict) -> str:
    kind = record.get("kind")
    files = record.get("files") if isinstance(record.get("files"), dict) else {}

    def has(name: str) -> bool:
        return isinstance(files.get(name), dict)

    if kind in ("character", "look", "outfit", "camera") and has("front"):
        return "front"
    if kind == "location":
        cameras = [name for name in files if _CAMERA_VIEW_RE.match(str(name)) and has(name)]
        if len(cameras) == 1:
            return cameras[0]
        if has("empty"):
            return "empty"
    if kind == "voice" and has("voice_ref"):
        return "voice_ref"
    if kind == "style" and has("style"):
        return "style"
    if kind == "bible":
        texts = [
            name for name, info in files.items()
            if isinstance(info, dict) and _file_media(info) == "text"
        ]
        if len(texts) == 1:
            return texts[0]
    images = [
        name for name, info in files.items()
        if isinstance(info, dict) and _file_media(info) == "image"
    ]
    if len(images) == 1:
        return images[0]
    label = f"{record.get('id')}@v{record.get('version')}"
    raise SeriesLibraryError(f"view required for {label}")


def _file_media(info: dict) -> str:
    media = info.get("media_type")
    if isinstance(media, str) and media.strip():
        if media == "image":
            return "image"
        if media == "audio":
            return "audio"
        if media == "text":
            return "text"
        if media == "video":
            return "video"
    alias = info.get("alias_of") or info.get("alias")
    if isinstance(alias, str) and alias.strip():
        return "image"
    return _media_from_path(str(info.get("path") or ""))


def _media_from_path(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in _IMAGE_EXT:
        return "image"
    if ext in _AUDIO_EXT:
        return "audio"
    if ext in _TEXT_EXT:
        return "text"
    return ""


def _prepare_record(folder: Path, raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise SeriesLibraryError("asset must be a mapping")
    asset_id = str(raw.get("id") or "").strip()
    kind = str(raw.get("kind") or "").strip()
    try:
        version = int(raw.get("version"))
    except (TypeError, ValueError):
        raise SeriesLibraryError("version must be a positive integer")
    if version < 1:
        raise SeriesLibraryError("version must be a positive integer")
    _check_id_kind(asset_id, kind)
    status = str(raw.get("status") or "draft").strip()
    if status not in _ASSET_STATUSES:
        raise SeriesLibraryError("status must be draft, candidate, or locked")
    record = copy.deepcopy(raw)
    record["id"] = asset_id
    record["kind"] = kind
    record["version"] = version
    record["status"] = status
    if status == "locked":
        approver = str(raw.get("approved_by") or "").strip()
        if approver != _LOCK_APPROVER:
            raise SeriesLibraryError("approved_by must be 张景")
        record["approved_by"] = approver
    else:
        for key in _DROP_ON_NEW_VERSION:
            record.pop(key, None)
    _validate_record_extras(record)
    _normalize_name_lists(record)
    record["files"] = _prepare_files(folder, asset_id, version, raw.get("files") or {})
    return record


def _prepare_files(folder: Path, asset_id: str, version: int, files) -> dict:
    if files is None:
        files = {}
    if not isinstance(files, dict):
        raise SeriesLibraryError("files must be a mapping of view name to file info")
    prefix = f"{asset_id}/v{version}/"
    prepared = {}
    for view, info in files.items():
        if not isinstance(view, str) or _VIEW_RE.match(view) is None:
            raise SeriesLibraryError(f"invalid view name {view!r}")
        if not isinstance(info, dict):
            raise SeriesLibraryError(f"file info for {view} must be a mapping")
        item = dict(info)
        alias = item.get("alias_of") or item.get("alias")
        if isinstance(alias, str) and alias.strip():
            parsed = parse_series_ref(alias)
            if parsed["kind"] != "asset":
                raise SeriesLibraryError(f"alias {alias} is not an asset code")
            item["alias_of"] = parsed["ref"]
            item.pop("alias", None)
            prepared[view] = item
            continue
        rel = item.get("path")
        if not isinstance(rel, str) or not rel.strip():
            raise SeriesLibraryError(f"view {view} needs path or alias_of")
        rel = rel.strip().replace("\\", "/")
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise SeriesLibraryError(f"asset path escapes the series folder: {rel}")
        if not rel.startswith(prefix):
            raise SeriesLibraryError(f"path {rel} must live under {prefix}")
        absolute = _safe_join(folder, rel)
        if not absolute.is_file():
            raise SeriesLibraryError(f"file not found: {rel}", status=404)
        digest, size = _hash_file(absolute)
        given = item.get("sha256")
        if given and str(given).strip() and str(given).strip() != digest:
            raise SeriesLibraryError(f"sha256 mismatch for {rel}", status=409)
        if "bytes" in item and item.get("bytes") is not None:
            try:
                declared_bytes = int(item["bytes"])
            except (TypeError, ValueError):
                raise SeriesLibraryError(f"bytes for {rel} must be an integer")
            if declared_bytes != size:
                raise SeriesLibraryError(f"byte size mismatch for {rel}", status=409)
        item["path"] = rel
        item["sha256"] = digest
        item["bytes"] = size
        if not item.get("media_type"):
            item["media_type"] = _media_from_path(rel) or "image"
        prepared[view] = item
    return prepared


def _copy_files(folder: Path, record: dict, new_version: int) -> dict:
    asset_id = record["id"]
    copied = {}
    files = record.get("files") if isinstance(record.get("files"), dict) else {}
    for view, info in files.items():
        if not isinstance(info, dict):
            continue
        item = dict(info)
        alias = item.get("alias_of") or item.get("alias")
        if isinstance(alias, str) and alias.strip():
            parsed = parse_series_ref(alias)
            item["alias_of"] = parsed["ref"] if parsed["kind"] == "asset" else alias.strip()
            item.pop("alias", None)
            copied[view] = item
            continue
        rel = item.get("path")
        if not isinstance(rel, str) or not rel.strip():
            copied[view] = item
            continue
        new_rel = f"{asset_id}/v{new_version}/{Path(rel).name}"
        dest = _safe_join(folder, new_rel)
        if dest.exists():
            raise SeriesLibraryError(f"{new_rel} already exists", status=409)
        src = _safe_join(folder, rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        os.chmod(dest, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        digest, size = _hash_file(dest)
        item["path"] = new_rel
        item["sha256"] = digest
        item["bytes"] = size
        item.pop("payload_url", None)
        item.pop("comfytv_asset_id", None)
        item.pop("approval", None)
        copied[view] = item
    return copied


def _character_dep_refs(record: dict) -> list[str]:
    refs: list[str] = []
    wardrobe = record.get("wardrobe") or []
    if isinstance(wardrobe, list):
        for item in wardrobe:
            if not isinstance(item, dict):
                continue
            outfit = str(item.get("outfit") or "").strip()
            if outfit:
                refs.append(outfit)
            look = item.get("look")
            if isinstance(look, str) and look.strip():
                refs.append(look.strip())
    voice = record.get("voice")
    if isinstance(voice, str) and voice.strip():
        refs.append(voice.strip())
    return refs


def _resolve_dependency(doc: dict, ref: str) -> dict:
    parsed = parse_series_ref(ref)
    if parsed["kind"] != "asset":
        return {"code": ref, "status": "missing", "locked": False}
    asset_id = parsed["id"]
    if parsed.get("version") is None:
        rows = _versions(doc, asset_id)
        if not rows:
            return {"code": asset_id, "status": "missing", "locked": False}
        locked = [row for row in rows if row.get("status") == "locked"]
        chosen = max(locked or rows, key=lambda row: int(row.get("version") or 0))
    else:
        chosen = _by_version(doc, asset_id, int(parsed["version"]))
        if chosen is None:
            return {
                "code": f"{asset_id}@v{parsed['version']}",
                "status": "missing",
                "locked": False,
            }
    code = f"{chosen.get('id')}@v{int(chosen.get('version') or 0)}"
    status = str(chosen.get("status") or "")
    return {
        "code": code,
        "kind": chosen.get("kind"),
        "status": status,
        "locked": status == "locked",
    }


def legacy_id_names(value) -> list[str]:
    """Flat legacy ids from a list or a ``{view: id | [id, ...]}`` map."""
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return _dedupe_strings(value)
    if isinstance(value, dict):
        names: list[str] = []
        for item in value.values():
            names.extend(legacy_id_names(item))
        return _dedupe_strings(names)
    return []


def _dedupe_strings(raw) -> list[str]:
    seen: list[str] = []
    if not isinstance(raw, list):
        return seen
    for item in raw:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and text not in seen:
            seen.append(text)
    return seen


def _normalize_name_lists(record: dict) -> None:
    aliases = record.get("aliases")
    if isinstance(aliases, list):
        record["aliases"] = _dedupe_strings(aliases)
    legacy = record.get("legacy_ids")
    if isinstance(legacy, list):
        record["legacy_ids"] = _dedupe_strings(legacy)
    elif isinstance(legacy, dict):
        record["legacy_ids"] = _normalize_legacy_map(legacy)


def _normalize_legacy_map(raw: dict) -> dict:
    cleaned: dict = {}
    for view, item in raw.items():
        key = str(view).strip()
        if not key:
            raise SeriesLibraryError("legacy_ids view names must be strings")
        if isinstance(item, str):
            text = item.strip()
            if text:
                cleaned[key] = text
        elif isinstance(item, list):
            cleaned[key] = _dedupe_strings(item)
        else:
            raise SeriesLibraryError(
                "legacy_ids values must be a string or a list of strings")
    return cleaned


def _validate_record_extras(record: dict) -> None:
    legacy = record.get("legacy_ids")
    if legacy is not None and not isinstance(legacy, (list, dict)):
        raise SeriesLibraryError(
            "legacy_ids must be a list of strings or a view map")
    if isinstance(legacy, list) and not all(
        isinstance(item, str) and item.strip() for item in legacy
    ):
        raise SeriesLibraryError("legacy_ids must be a list of strings")
    if isinstance(legacy, dict):
        _normalize_legacy_map(legacy)
    aliases = record.get("aliases")
    if aliases is not None and (
        not isinstance(aliases, list)
        or not all(isinstance(item, str) and item.strip() for item in aliases)
    ):
        raise SeriesLibraryError("aliases must be a list of strings")
    face = record.get("face_source")
    if face is None:
        return
    if not isinstance(face, list):
        raise SeriesLibraryError("face_source must be a list")
    for item in face:
        if not isinstance(item, dict):
            raise SeriesLibraryError("face_source items must be objects")
        if item.get("provenance") not in _PROVENANCE:
            raise SeriesLibraryError(
                "face_source provenance must be self, licensed, or ai_generated")


def _alias_claims(record: dict) -> set[str]:
    names: set[str] = set()
    asset_id = str(record.get("id") or "").strip()
    if asset_id:
        names.add(asset_id)
    name_zh = str(record.get("name_zh") or "").strip()
    if name_zh:
        names.add(name_zh)
    for item in record.get("aliases") or []:
        text = str(item or "").strip()
        if text:
            names.add(text)
    for text in legacy_id_names(record.get("legacy_ids")):
        names.add(text)
    return names


def _assert_alias_unique(doc: dict, focus_id: str) -> None:
    claims: dict[str, set[str]] = {}
    for asset in doc.get("assets") or []:
        if not isinstance(asset, dict) or asset.get("kind") != "character":
            continue
        for name in _alias_claims(asset):
            claims.setdefault(name, set()).add(str(asset.get("id")))
    conflicts = [
        f"{name}: {', '.join(sorted(ids))}"
        for name, ids in sorted(claims.items())
        if len(ids) > 1 and focus_id in ids
    ]
    if conflicts:
        raise SeriesLibraryError(
            "alias conflict: " + "; ".join(conflicts),
            status=409,
            details={"conflicts": conflicts},
        )


def _declared_manifest_files(manifest: dict) -> list[dict]:
    found: list[dict] = []
    for raw in manifest.get("assets") or []:
        if not isinstance(raw, dict):
            continue
        files = raw.get("files")
        if not isinstance(files, dict):
            continue
        for info in files.values():
            if not isinstance(info, dict):
                continue
            alias = info.get("alias_of") or info.get("alias")
            if isinstance(alias, str) and alias.strip():
                continue
            path = info.get("path")
            if isinstance(path, str) and path.strip():
                found.append(info)
    return found


def _check_manifest_inventory(manifest: dict, declared: list[dict]) -> None:
    if "file_count" in manifest and manifest.get("file_count") is not None:
        try:
            expected = int(manifest["file_count"])
        except (TypeError, ValueError):
            raise SeriesLibraryError("file_count must be an integer")
        if expected != len(declared):
            raise SeriesLibraryError(
                f"manifest file_count is {expected} but {len(declared)} files were declared")
    if not manifest.get("require_sha256"):
        return
    for info in declared:
        digest = str(info.get("sha256") or "").strip().lower()
        path = info.get("path")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise SeriesLibraryError(f"sha256 required for {path}")


def _reject_excluded_manifest(manifest: dict, *, allow_excluded: bool) -> None:
    if allow_excluded:
        return
    excluded = {
        str(item) for item in (asset_store_policy().get("excluded_projects") or [])
    }
    series = manifest.get("series") if isinstance(manifest.get("series"), dict) else {}
    found = set()
    for value in (
        series.get("id"),
        series.get("project_id"),
        manifest.get("project"),
    ):
        text = str(value or "").strip()
        if text:
            found.add(text)
    hit = sorted(found & excluded)
    if hit:
        raise SeriesLibraryError(
            f"project {hit[0]} is excluded from import",
            status=409,
        )


def _check_id_kind(asset_id: str, kind: str) -> None:
    prefix = _KIND_PREFIX.get(kind)
    if prefix is None or not asset_id.startswith(prefix) or _ASSET_RE.match(asset_id) is None:
        raise SeriesLibraryError(f"id {asset_id} does not match kind {kind}")
    if kind == "camera" and "__" not in asset_id:
        raise SeriesLibraryError(
            "camera ids look like LOC_<place>__<card>")


def _select_record(doc: dict, parsed: dict, *, require_locked: bool) -> tuple[dict, bool]:
    asset_id = parsed["id"]
    rows = _versions(doc, asset_id)
    if not rows:
        raise SeriesLibraryError(f"asset {asset_id} not found", status=404)
    version_defaulted = parsed.get("version") is None
    if version_defaulted:
        pool = [row for row in rows if row.get("status") == "locked"] if require_locked else rows
        if not pool:
            raise SeriesLibraryError(f"no locked version for {asset_id}", status=409)
        record = max(pool, key=lambda row: int(row.get("version") or 0))
        return record, True
    record = _by_version(doc, asset_id, int(parsed["version"]))
    if record is None:
        raise SeriesLibraryError(
            f"asset {asset_id}@v{parsed['version']} not found", status=404)
    if require_locked and record.get("status") != "locked":
        raise SeriesLibraryError(
            f"{asset_id}@v{parsed['version']} is not locked", status=409)
    return record, False


def _tip_record(doc: dict, asset_id: str, version: int | None) -> dict:
    rows = _versions(doc, asset_id)
    if not rows:
        raise SeriesLibraryError(f"asset {asset_id} not found", status=404)
    if version is None:
        return max(rows, key=lambda row: int(row.get("version") or 0))
    found = _by_version(doc, asset_id, version)
    if found is None:
        raise SeriesLibraryError(f"asset {asset_id}@v{version} not found", status=404)
    return found


def _versions(doc: dict, asset_id: str) -> list[dict]:
    return [
        row for row in (doc.get("assets") or [])
        if isinstance(row, dict) and row.get("id") == asset_id
    ]


def _by_version(doc: dict, asset_id: str, version: int) -> dict | None:
    for row in _versions(doc, asset_id):
        try:
            current = int(row.get("version"))
        except (TypeError, ValueError):
            continue
        if current == int(version):
            return row
    return None


def _require_asset_code(code: str) -> dict:
    parsed = parse_series_ref(code)
    if parsed["kind"] != "asset":
        raise SeriesLibraryError(f"not an asset code: {code!r}")
    return parsed


def _same_record(existing: dict, incoming: dict) -> bool:
    return _canon(existing) == _canon(incoming)


def _canon(record: dict) -> dict:
    data = copy.deepcopy(record)
    files = data.get("files")
    if isinstance(files, dict):
        for info in files.values():
            if isinstance(info, dict):
                for key in _COMPUTED_FILE_KEYS:
                    info.pop(key, None)
    return data


def _matches_episode(asset: dict, episode: str) -> bool:
    episodes = asset.get("episodes")
    if episodes == episode:
        return True
    if isinstance(episodes, list) and episode in episodes:
        return True
    tags = asset.get("tags") or []
    if isinstance(tags, list) and episode in tags:
        return True
    requested = asset.get("requested_by") or []
    if isinstance(requested, list) and episode in requested:
        return True
    return False


def _payload_url(doc: dict, rel: str, explicit) -> str | None:
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    series = doc.get("series") if isinstance(doc.get("series"), dict) else {}
    sub = series.get("view_url_subfolder")
    if not isinstance(sub, str) or not sub.strip():
        return None
    rel_path = Path(rel)
    parent = rel_path.parent.as_posix()
    base = sub.strip("/")
    subfolder = base if parent in ("", ".") else f"{base}/{parent}"
    return (
        f"/view?filename={quote(rel_path.name)}"
        f"&subfolder={quote(subfolder, safe='/')}"
        f"&type=input"
    )


def _chmod_version(folder: Path, asset_id: str, version: int) -> None:
    directory = folder / asset_id / f"v{version}"
    if not directory.is_dir():
        return
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in filenames:
            os.chmod(os.path.join(dirpath, name), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    for dirpath, _dirnames, _filenames in os.walk(directory, topdown=False):
        os.chmod(dirpath, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def _hash_file(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def _safe_join(folder: Path, rel: str) -> Path:
    text = str(rel or "").strip().replace("\\", "/")
    if not text or text.startswith("/") or ".." in Path(text).parts:
        raise SeriesLibraryError(f"asset path escapes the series folder: {rel}")
    root = folder.resolve()
    candidate = (root / text).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise SeriesLibraryError(f"asset path escapes the series folder: {rel}")
    return candidate


def _as_root(root) -> Path:
    path = Path(root).expanduser()
    if not path.is_dir():
        raise SeriesLibraryError(f"series folder not found: {root}", status=404)
    return path.resolve()


def _as_root_allow_empty(root) -> Path:
    path = Path(root).expanduser()
    if path.exists() and not path.is_dir():
        raise SeriesLibraryError(f"series folder is not a directory: {root}", status=400)
    return path.resolve() if path.exists() else path.absolute()


def _load(folder: Path) -> tuple[dict, str]:
    path = folder / "library.yaml"
    if not path.is_file():
        raise SeriesLibraryError(
            f"library.yaml not found in {folder}", status=404)
    text = path.read_text(encoding="utf-8")
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SeriesLibraryError(f"library.yaml is not valid YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise SeriesLibraryError("library.yaml must be a mapping")
    if doc.get("schema_version", 1) != 1:
        raise SeriesLibraryError(
            f"unsupported schema_version {doc.get('schema_version')}")
    if "assets" not in doc:
        doc["assets"] = []
    if not isinstance(doc.get("assets"), list):
        raise SeriesLibraryError("assets must be a list")
    return doc, text


def _open_for_write(folder: Path) -> tuple[dict, str]:
    path = folder / "library.yaml"
    if path.is_file():
        doc, text = _load(folder)
        return doc, _preamble_from(text)
    return (
        {"schema_version": 1, "series": {"id": folder.name}, "assets": []},
        _DEFAULT_PREAMBLE,
    )


def _save(folder: Path, doc: dict, preamble: str) -> None:
    doc["schema_version"] = 1
    doc["updated_at"] = _now()
    path = folder / "library.yaml"
    folder.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        _snapshot(folder, path)
    text = preamble + yaml.safe_dump(
        doc,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _snapshot(folder: Path, source: Path) -> Path:
    global _SNAPSHOT_SEQ
    _SNAPSHOT_SEQ += 1
    history = folder / "library.history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
    dest = history / f"library.{stamp}-{_SNAPSHOT_SEQ}.yaml"
    shutil.copy2(source, dest)
    return dest


def _preamble_from(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            kept.append(line if line.endswith("\n") else line + "\n")
            continue
        break
    if not kept:
        return _DEFAULT_PREAMBLE
    blob = "".join(kept)
    return blob if blob.endswith("\n") else blob + "\n"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _bindings_path() -> Path:
    directory = Path(data_dir())
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "series_libraries.json"


def _read_bindings() -> dict:
    path = _bindings_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_bindings(data: dict) -> None:
    path = _bindings_path()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _peek_bound_root() -> Path | None:
    """Active series folder, or None. Does not create the user data directory."""
    path = Path(data_dir()) / "series_libraries.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    roots = data.get("roots") if isinstance(data.get("roots"), dict) else {}
    key = str(data.get("active") or "").strip()
    if not key or key not in roots:
        return None
    try:
        return _as_root(roots[key])
    except SeriesLibraryError:
        return None


def _bound_root(series: str | None) -> Path:
    data = _read_bindings()
    roots = data.get("roots") if isinstance(data.get("roots"), dict) else {}
    key = (series or data.get("active") or "")
    key = str(key).strip()
    if not key or key not in roots:
        raise SeriesLibraryError(
            "no series library bound; pass root (a folder containing library.yaml)",
            status=404,
        )
    return _as_root(roots[key])


def _read_manifest_file(path: str) -> dict:
    file = Path(path)
    if not file.is_file():
        raise SeriesLibraryError(f"manifest not found: {file}", status=404)
    text = file.read_text(encoding="utf-8")
    suffix = file.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise SeriesLibraryError(f"manifest could not be parsed: {exc}") from exc
    if not isinstance(data, dict):
        raise SeriesLibraryError("manifest must be an object")
    return data
