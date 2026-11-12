"""Script sessions stored under a ComfyTV project directory.

Layout, relative to the project root passed as ``root`` (not the series
library folder)::

    script/index.json
    script/config.json
    script/sessions/<session_id>/session.json
    script/sessions/<session_id>/import_manifest.json
    script/sessions/<session_id>/imported/<source_path>   # frozen import bytes
    script/sessions/<session_id>/.versions/<doc>/v<n>.<ext>
    script/sessions/<session_id>/current/<doc file>       # latest version
    script/sessions/<session_id>/.candidates/<id>.json
    script/sessions/<session_id>/.jobs/<id>.json

Series characters stay in ``library.yaml`` (``series_library`` / F3-a).
This module does not write that file. Official episode text and an archived
lineage are different sessions, so an archived import cannot replace E01.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path

from ..backup import data_dir
from .script_llm import (
    ScriptError,
    build_client,
    load_prompt,
    load_sections,
    resolve_llm,
)
from .script_text import (
    apply_selection,
    diff_texts,
    ensure_available,
    feedback_block,
    parse_directions,
    parse_hooks,
    protect_beats,
    shot_ids_in,
    split_marked,
    validate_outline,
)

_LOCK = threading.Lock()
_RUNNING: set[str] = set()
_APPROVER = "张景"
_DEFAULT_EXCLUDED = {"gumu-bixie"}
_SCHEMA = "comfytv/script-import.v1"
_DOC_EXT = {
    "brief": ".md",
    "adaptation": ".md",
    "outline": ".md",
    "characters": ".md",
    "hooks": ".yaml",
    "script": ".md",
    "lines": ".md",
    "review": ".md",
    "feedback": ".md",
    "directions": ".json",
    "shots": ".json",
    "storyboard": ".yaml",
    "state": ".json",
}
_JOB_ACTIONS = ("directions", "choose", "revise", "write_episode", "review")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _token(value: str, label: str) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ScriptError(f"{label} is not a safe id")
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
    if any(char not in allowed for char in text):
        raise ScriptError(f"{label} is not a safe id")
    return text


def _safe_rel(value: str) -> Path:
    raw = str(value or "").strip().replace("\\", "/")
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise ScriptError("source_path must stay inside the source directory")
    return path


def _read_json(path: Path, default):
    if not path.is_file():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return data


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    handle = os.open(path, flags)
    try:
        os.write(handle, payload)
    finally:
        os.close(handle)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _session_dir(root: Path, session_id: str) -> Path:
    return root / "script" / "sessions" / session_id


def _index_path(root: Path) -> Path:
    return root / "script" / "index.json"


def _load_index(root: Path) -> dict:
    data = _read_json(_index_path(root), {})
    if not isinstance(data, dict):
        data = {}
    data.setdefault("sessions", [])
    data.setdefault("active_by_project", {})
    return data


def _config(root: Path) -> dict:
    data = _read_json(root / "script" / "config.json", {})
    return data if isinstance(data, dict) else {}


def _excluded(root: Path) -> set[str]:
    data = _config(root)
    if "excluded_projects" in data:
        values = data.get("excluded_projects") or []
        return {str(item) for item in values}
    return set(_DEFAULT_EXCLUDED)


def _review_enabled(root: Path) -> bool:
    data = _config(root)
    if "review_enabled" in data:
        return bool(data.get("review_enabled"))
    return True


def _blank_session(*, session_id: str, project: str, episode, series, status: str,
                   stage: str, origin: str, lineage: str | None) -> dict:
    return {
        "session_id": session_id,
        "project": project,
        "series": series,
        "episode": episode,
        "lineage": lineage,
        "status": status,
        "stage": stage,
        "origin": origin,
        "storyboard_doc": None,
        "storyboard_current": False,
        "input": {"idea": "", "source_text": "", "reference_works": ""},
        "directions": [],
        "chosen_direction": None,
        "confirmed": None,
        "docs": {},
        "imports": [],
        "llm": {"provider": "deepseek", "model": "deepseek-v4-pro"},
        "turns": [],
    }


def _load_session(root: Path, session_id: str) -> dict:
    path = _session_dir(root, session_id) / "session.json"
    data = _read_json(path, None)
    if not isinstance(data, dict):
        raise ScriptError(f"session not found: {session_id}", status=404)
    return data


def _save_session(root: Path, session: dict) -> None:
    stored = json.loads(json.dumps(session))
    for doc in (stored.get("docs") or {}).values():
        if isinstance(doc, dict):
            doc.pop("text", None)
    _write_json(_session_dir(root, session["session_id"]) / "session.json", stored)


def _ext(doc: str, source_path: str | None = None) -> str:
    if doc in _DOC_EXT:
        return _DOC_EXT[doc]
    if source_path:
        suffix = Path(source_path).suffix
        if suffix:
            return suffix
    return ".md"


def _version_path(root: Path, session: dict, doc: str, version: int) -> Path:
    info = session["docs"][doc]
    return (
        _session_dir(root, session["session_id"])
        / ".versions" / doc / f"v{version}{info['ext']}"
    )


def _read_version(root: Path, session: dict, doc: str, version: int | None = None) -> str:
    info = session["docs"][doc]
    current = info["current"] if version is None else version
    path = _version_path(root, session, doc, current)
    return path.read_text(encoding="utf-8")


def _attach_text(root: Path, session: dict) -> dict:
    view = json.loads(json.dumps(session))
    for name, info in (view.get("docs") or {}).items():
        try:
            info["text"] = _read_version(root, session, name)
        except (OSError, KeyError):
            info["text"] = ""
    return view


def _remember_index(root: Path, session: dict, *, make_active: bool) -> None:
    index = _load_index(root)
    known = {row.get("id") for row in index["sessions"] if isinstance(row, dict)}
    if session["session_id"] not in known:
        index["sessions"].append({
            "id": session["session_id"],
            "project": session["project"],
            "status": session["status"],
            "lineage": session.get("lineage"),
            "episode": session.get("episode"),
        })
    else:
        for row in index["sessions"]:
            if row.get("id") == session["session_id"]:
                row["status"] = session["status"]
    if make_active:
        index["active_by_project"][session["project"]] = session["session_id"]
    _write_json(_index_path(root), index)


def _add_version(root: Path, session: dict, doc: str, payload: bytes, *,
                 by: str, model: str, note: str, source_path: str | None = None) -> dict:
    docs = session.setdefault("docs", {})
    if doc not in docs:
        docs[doc] = {
            "path": f"current/{doc}{_ext(doc, source_path)}",
            "ext": _ext(doc, source_path),
            "current": 0,
            "versions": [],
        }
    info = docs[doc]
    version = int(info["current"]) + 1
    parent = info["current"] or None
    path = _version_path(root, session, doc, version)
    if path.exists():
        raise ScriptError(f"version cannot be changed: {doc} v{version}", status=409)
    _write_new_bytes(path, payload)
    current = _session_dir(root, session["session_id"]) / info["path"]
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_bytes(payload)
    record = {
        "v": version,
        "sha256": _sha(payload),
        "parent": parent,
        "by": by,
        "model": model,
        "at": _now(),
        "note": note,
    }
    info["versions"].append(record)
    info["current"] = version
    if doc == "outline" and version > 1 and session.get("confirmed"):
        session["confirmed"] = None
        if session.get("stage") in ("confirmed", "episode_script"):
            session["stage"] = "outline"
    return record


def _reject_archived(session: dict) -> None:
    if session.get("status") == "archived":
        raise ScriptError("archived session cannot be edited", status=409)


def _usage(llm) -> dict | None:
    usage = getattr(llm, "last_usage", None)
    return usage if isinstance(usage, dict) else None


def _model_name(session: dict) -> str:
    llm = session.get("llm") if isinstance(session.get("llm"), dict) else {}
    return str(llm.get("model") or "deepseek-v4-pro")


def _public_llm(root: Path, request: dict | None) -> dict:
    public = resolve_llm(root, request if isinstance(request, dict) else {})["public"]
    return {
        "provider": public["provider"],
        "model": public["model"],
        "base_url": public["base_url"],
    }


def render_brief(root: Path, idea: str, source_text: str, reference: str) -> str:
    parts = [piece.strip() for piece in (idea, source_text, reference) if piece and piece.strip()]
    blob = "\n\n".join(parts) if parts else "未指定"
    template = load_prompt("brief", root)
    values = {
        "{{input}}": blob,
        "{{watch}}": "未指定",
        "{{drama}}": "未指定",
        "{{production}}": "未指定",
        "{{generatable}}": "未指定",
    }
    text = template
    for key, value in values.items():
        text = text.replace(key, value)
    if not text.endswith("\n"):
        text += "\n"
    return text


def start_session(root, *, project: str, episode: str | None = None, idea: str = "",
                  source_text: str = "", reference_works: str = "", series: str | None = None,
                  llm: dict | None = None) -> dict:
    folder = Path(root)
    project_id = _token(project, "project")
    if project_id in _excluded(folder):
        raise ScriptError(f"project {project_id} is excluded")
    session_id = "sw_" + uuid.uuid4().hex[:12]
    session = _blank_session(
        session_id=session_id,
        project=project_id,
        episode=episode,
        series=series,
        status="draft",
        stage="brief",
        origin="writer",
        lineage=episode,
    )
    session["input"] = {
        "idea": idea or "",
        "source_text": source_text or "",
        "reference_works": reference_works or "",
    }
    session["llm"] = _public_llm(folder, llm)
    with _LOCK:
        brief = render_brief(folder, idea or "", source_text or "", reference_works or "")
        _add_version(
            folder, session, "brief", brief.encode("utf-8"),
            by="user", model="", note="start",
        )
        _save_session(folder, session)
        index = _load_index(folder)
        active = index.get("active_by_project", {}).get(project_id)
        make_active = True
        if active:
            try:
                current = _load_session(folder, active)
            except ScriptError:
                current = None
            if current and current.get("origin") == "import" and current.get("status") == "active":
                make_active = False
                index = _load_index(folder)
                index["writing_session"] = session_id
                _write_json(_index_path(folder), index)
        _remember_index(folder, session, make_active=make_active)
        return _attach_text(folder, session)


def _receipt(rows: list[dict]) -> str:
    lines = sorted(f"{row['sha256']}  {row['source_path']}" for row in rows)
    return _sha("\n".join(lines).encode("utf-8"))


def _manifest_files(manifest: dict) -> list[dict]:
    if manifest.get("schema") != _SCHEMA:
        raise ScriptError("manifest schema must be comfytv/script-import.v1")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ScriptError("manifest files must be a list")
    if "file_count" not in manifest or manifest.get("file_count") != len(files):
        raise ScriptError("file_count does not match files")
    project = manifest.get("project")
    if not project:
        raise ScriptError("manifest project is required")
    status = manifest.get("status")
    if status not in ("active", "archived"):
        raise ScriptError("manifest status must be active or archived")
    return files


def import_scripts(root, manifest: dict, *, source_dir) -> dict:
    if not isinstance(manifest, dict):
        raise ScriptError("manifest must be an object")
    folder = Path(root)
    files = _manifest_files(manifest)
    project = _token(str(manifest.get("project")), "project")
    if project in _excluded(folder):
        raise ScriptError(f"project {project} is excluded")
    status = manifest.get("status")
    source = Path(source_dir).resolve()
    if not source.is_dir():
        raise ScriptError("source_dir is not a directory", status=404)
    payloads = []
    seen_docs: set[str] = set()
    seen_paths: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ScriptError("each manifest file must be an object")
        rel = _safe_rel(str(entry.get("source_path") or ""))
        rel_key = rel.as_posix()
        if rel_key in seen_paths:
            raise ScriptError(f"manifest lists {rel_key} more than once")
        seen_paths.add(rel_key)
        if status == "archived" and rel_key.startswith("episodes/"):
            raise ScriptError("archived import cannot use an episodes/ path")
        doc = str(entry.get("doc") or _infer_doc(rel)).strip()
        doc = _token(doc, "doc")
        if doc in seen_docs:
            raise ScriptError(f"manifest lists {doc} more than once")
        seen_docs.add(doc)
        file_path = (source / rel).resolve()
        if source != file_path and source not in file_path.parents:
            raise ScriptError("source_path must stay inside the source directory")
        if not file_path.is_file():
            raise ScriptError(f"import file not found: {rel.as_posix()}", status=404)
        raw = file_path.read_bytes()
        digest = _sha(raw)
        expected = str(entry.get("sha256") or "")
        if digest != expected:
            raise ScriptError(f"sha256 mismatch for {rel.as_posix()}")
        try:
            size = int(entry.get("bytes"))
        except (TypeError, ValueError):
            raise ScriptError(f"bytes mismatch for {rel.as_posix()}")
        if size != len(raw):
            raise ScriptError(f"bytes mismatch for {rel.as_posix()}")
        payloads.append({
            "doc": doc,
            "source_path": rel.as_posix(),
            "sha256": digest,
            "bytes": len(raw),
            "raw": raw,
        })
    session_id = _token(
        str(manifest.get("session_id") or f"{project}-{manifest.get('lineage') or manifest.get('episode') or 'import'}"),
        "session_id",
    )
    with _LOCK:
        index = _load_index(folder)
        for row in index["sessions"]:
            if not isinstance(row, dict) or row.get("id") == session_id:
                continue
            other = _load_session(folder, row["id"])
            claimed = {item.get("source_path") for item in other.get("imports") or []}
            overlap = claimed.intersection(item["source_path"] for item in payloads)
            if overlap:
                raise ScriptError(
                    "source_path already imported: " + ", ".join(sorted(overlap)),
                    status=409,
                )
        created = not (_session_dir(folder, session_id) / "session.json").is_file()
        if created:
            session = _blank_session(
                session_id=session_id,
                project=project,
                episode=manifest.get("episode"),
                series=manifest.get("series"),
                status=status,
                stage="episode_script" if "script" in seen_docs else "brief",
                origin="import",
                lineage=manifest.get("lineage"),
            )
            session["llm"] = _public_llm(folder, None)
        else:
            session = _load_session(folder, session_id)
            if session.get("project") != project:
                raise ScriptError("session belongs to a different project", status=409)
            if session.get("status") != status:
                raise ScriptError("session status does not match the manifest", status=409)
            if (session.get("lineage") or None) != (manifest.get("lineage") or None):
                raise ScriptError("session lineage does not match the manifest", status=409)
            if (session.get("episode") or None) != (manifest.get("episode") or None):
                raise ScriptError("session episode does not match the manifest", status=409)
        existing = {item["source_path"]: item for item in session.get("imports") or []}
        plan = []
        skipped = 0
        for item in payloads:
            prior = existing.get(item["source_path"])
            doc_info = (session.get("docs") or {}).get(item["doc"])
            if prior is None and doc_info is None:
                plan.append(item)
                continue
            prior_sha = prior["sha256"] if prior else (doc_info["versions"][-1]["sha256"])
            if prior_sha == item["sha256"]:
                skipped += 1
                continue
            raise ScriptError(
                f"imported version cannot be changed: {item['source_path']}",
                status=409,
            )
        for item in plan:
            _write_new_bytes(
                _session_dir(folder, session_id) / "imported" / item["source_path"],
                item["raw"],
            )
            _add_version(
                folder, session, item["doc"], item["raw"],
                by="user", model="", note=f"import {item['source_path']}",
                source_path=item["source_path"],
            )
            session["imports"].append({
                "doc": item["doc"],
                "source_path": item["source_path"],
                "sha256": item["sha256"],
                "bytes": item["bytes"],
                "version": 1,
            })
        if "shots" in seen_docs and status == "active":
            session["storyboard_doc"] = "shots"
            session["storyboard_current"] = True
        if status == "archived":
            session["status"] = "archived"
            session["storyboard_current"] = False
            if "storyboard" in seen_docs:
                session["storyboard_doc"] = "storyboard"
        session["lineage"] = manifest.get("lineage")
        _save_session(folder, session)
        _write_json(_session_dir(folder, session_id) / "import_manifest.json", manifest)
        _remember_index(folder, session, make_active=status == "active")
    shot_ids: list[str] = []
    for item in payloads:
        if item["doc"] == "shots":
            shot_ids = shot_ids_in(item["raw"])
    rows = [{"sha256": item["sha256"], "source_path": item["source_path"]} for item in payloads]
    return {
        "session_id": session_id,
        "project": project,
        "status": status,
        "file_count": len(payloads),
        "added": len(plan),
        "skipped": skipped,
        "shot_count": len(shot_ids),
        "shot_ids": shot_ids,
        "receipt_sha256": _receipt(rows),
    }


def _infer_doc(rel: Path) -> str:
    stem = rel.stem
    if stem in {"script", "script_v2"}:
        return "script"
    return stem


def verify_imports(root, session_id: str) -> dict:
    folder = Path(root)
    with _LOCK:
        session = _load_session(folder, _token(session_id, "session_id"))
        rows = []
        for item in session.get("imports") or []:
            path = _session_dir(folder, session["session_id"]) / "imported" / item["source_path"]
            raw = path.read_bytes()
            digest = _sha(raw)
            if digest != item["sha256"] or len(raw) != int(item["bytes"]):
                raise ScriptError(
                    f"sha256 mismatch for {item['source_path']}",
                    status=409,
                )
            version = _version_path(folder, session, item["doc"], int(item["version"]))
            if _sha(version.read_bytes()) != item["sha256"]:
                raise ScriptError(
                    f"sha256 mismatch for {item['source_path']}",
                    status=409,
                )
            rows.append({"sha256": digest, "source_path": item["source_path"]})
        return {
            "ok": True,
            "file_count": len(rows),
            "receipt_sha256": _receipt(rows),
        }


def current_storyboard(root, *, project: str) -> dict:
    folder = Path(root)
    project_id = _token(project, "project")
    with _LOCK:
        index = _load_index(folder)
        session_id = (index.get("active_by_project") or {}).get(project_id)
        if not session_id:
            raise ScriptError("no active storyboard session", status=404)
        session = _load_session(folder, session_id)
        if not session.get("storyboard_current"):
            raise ScriptError("active session has no current storyboard", status=404)
        doc = session.get("storyboard_doc") or "shots"
        text = _read_version(folder, session, doc)
        ids = shot_ids_in(text.encode("utf-8"))
        return {
            "session_id": session_id,
            "doc": doc,
            "shot_ids": ids,
            "text": text,
        }


def get_session(root, session_id: str) -> dict:
    folder = Path(root)
    with _LOCK:
        session = _load_session(folder, _token(session_id, "session_id"))
        return _attach_text(folder, session)


def _require_direction_count(n: int) -> int:
    if n < 2 or n > 6:
        raise ScriptError("n must be from 2 to 6")
    return n


def run_directions(root, session_id: str, llm, *, n: int = 4, extra: str = "") -> dict:
    folder = Path(root)
    count = _require_direction_count(int(n))
    with _LOCK:
        session = _load_session(folder, session_id)
        _reject_archived(session)
        brief = _read_version(folder, session, "brief")
        model = _model_name(session)
    system = load_prompt("directions", folder)
    user = f"task: directions\nn: {count}\nextra: {extra or ''}\n---\n{brief}"
    rows = parse_directions(llm.complete(system=system, user=user, model=model), count)
    with _LOCK:
        session = _load_session(folder, session_id)
        session["directions"] = rows
        session["stage"] = "directions"
        session["turns"].append({
            "at": _now(), "by": "llm", "doc": "directions", "instruction": extra or "",
            "result_version": None, "outcome": "accepted",
        })
        _save_session(folder, session)
    result = {"directions": rows}
    usage = _usage(llm)
    if usage:
        result["usage"] = usage
    return result


def run_choose(root, session_id: str, llm, *, direction_id: str, note: str = "") -> dict:
    folder = Path(root)
    with _LOCK:
        session = _load_session(folder, session_id)
        _reject_archived(session)
        chosen = next(
            (row for row in session.get("directions") or [] if row.get("id") == direction_id),
            None,
        )
        if chosen is None:
            raise ScriptError(f"unknown direction {direction_id}")
        brief = _read_version(folder, session, "brief")
        model = _model_name(session)
    sections = load_sections(folder)
    heading_lines = "\n".join(f"- {name}" for name in sections)
    system = load_prompt("choose", folder)
    user = (
        f"task: choose\ndirection_id: {direction_id}\nnote: {note or ''}\n---\n"
        f"{heading_lines}\n\n{brief}\n\n{json.dumps(chosen, ensure_ascii=False)}"
    )
    parts = split_marked(
        llm.complete(system=system, user=user, model=model),
        ["OUTLINE", "CHARACTERS", "HOOKS"],
    )
    validate_outline(parts["OUTLINE"], sections)
    if not parts["CHARACTERS"].strip():
        raise ScriptError("characters.md is empty")
    hooks = parse_hooks(parts["HOOKS"])
    outline = parts["OUTLINE"] if parts["OUTLINE"].endswith("\n") else parts["OUTLINE"] + "\n"
    characters = parts["CHARACTERS"] if parts["CHARACTERS"].endswith("\n") else parts["CHARACTERS"] + "\n"
    with _LOCK:
        session = _load_session(folder, session_id)
        _add_version(folder, session, "outline", outline.encode("utf-8"),
                     by="llm", model=model, note=note or f"choose {direction_id}")
        _add_version(folder, session, "characters", characters.encode("utf-8"),
                     by="llm", model=model, note=note or f"choose {direction_id}")
        _add_version(folder, session, "hooks", hooks.encode("utf-8"),
                     by="llm", model=model, note=note or f"choose {direction_id}")
        session["chosen_direction"] = direction_id
        session["stage"] = "outline"
        session["turns"].append({
            "at": _now(), "by": "llm", "doc": "outline", "instruction": note or "",
            "result_version": session["docs"]["outline"]["current"], "outcome": "accepted",
        })
        _save_session(folder, session)
    return {"docs": {"outline": outline, "characters": characters, "hooks": hooks}}


def run_revise(root, session_id: str, llm, *, doc: str, instruction: str,
               selection: dict | None = None, candidate_id: str | None = None) -> dict:
    folder = Path(root)
    doc_name = _token(doc, "doc")
    with _LOCK:
        session = _load_session(folder, session_id)
        _reject_archived(session)
        if candidate_id:
            base_text, parent_version = _candidate_base(folder, session, candidate_id, doc_name)
        else:
            if doc_name not in session.get("docs", {}):
                raise ScriptError(f"document not found: {doc_name}", status=404)
            base_text = _read_version(folder, session, doc_name)
            parent_version = session["docs"][doc_name]["current"]
        model = _model_name(session)
    system = load_prompt("revise", folder)
    if selection:
        scope = (
            "selection: " + json.dumps(selection, ensure_ascii=False)
            + "\nReturn only the replacement for that selection."
        )
    else:
        scope = "selection: whole document\nReturn the whole document."
    user = f"task: revise\ndoc: {doc_name}\ninstruction: {instruction}\n{scope}\n---\n{base_text}"
    model_text = llm.complete(system=system, user=user, model=model)
    if selection:
        updated, changed = apply_selection(base_text, selection, model_text)
    else:
        updated = protect_beats(base_text, ensure_available(model_text))
        changed = False
    if doc_name == "outline":
        validate_outline(updated, load_sections(folder))
    candidate = {
        "id": uuid.uuid4().hex,
        "doc": doc_name,
        "parent_version": parent_version,
        "content": updated,
        "selection": selection,
        "sha256_outside_changed": changed,
        "model": model,
        "at": _now(),
        "instruction": instruction,
    }
    with _LOCK:
        session = _load_session(folder, session_id)
        _reject_archived(session)
        path = _session_dir(folder, session_id) / ".candidates" / f"{candidate['id']}.json"
        _write_json(path, candidate)
        session["turns"].append({
            "at": candidate["at"], "by": "llm", "doc": doc_name,
            "instruction": instruction, "selection": selection,
            "result_version": None,
            "outcome": "continued" if candidate_id else "accepted",
        })
        _save_session(folder, session)
    return {
        "candidate_id": candidate["id"],
        "text": updated,
        "sha256_outside_changed": changed,
    }


def _candidate_base(root: Path, session: dict, candidate_id: str, doc_name: str) -> tuple[str, int]:
    path = _session_dir(root, session["session_id"]) / ".candidates" / f"{candidate_id}.json"
    data = _read_json(path, None)
    if not isinstance(data, dict):
        raise ScriptError(f"candidate not found: {candidate_id}", status=404)
    if data.get("doc") != doc_name:
        raise ScriptError("candidate belongs to a different document")
    return str(data.get("content") or ""), int(data.get("parent_version") or 0)


def accept_candidate(root, session_id: str, candidate_id: str) -> dict:
    folder = Path(root)
    with _LOCK:
        session = _load_session(folder, session_id)
        _reject_archived(session)
        path = _session_dir(folder, session_id) / ".candidates" / f"{candidate_id}.json"
        candidate = _read_json(path, None)
        if not isinstance(candidate, dict):
            raise ScriptError(f"candidate not found: {candidate_id}", status=404)
        doc = candidate["doc"]
        parent = int(candidate.get("parent_version") or 0)
        if session["docs"][doc]["current"] != parent:
            raise ScriptError("candidate is stale", status=409)
        content = str(candidate.get("content") or "")
        protect_beats(_read_version(folder, session, doc, parent), content)
        record = _add_version(
            folder, session, doc, content.encode("utf-8"),
            by="llm", model=str(candidate.get("model") or ""),
            note="accept",
        )
        session["turns"].append({
            "at": _now(), "by": "llm", "doc": doc, "instruction": candidate.get("instruction") or "",
            "result_version": record["v"], "outcome": "accepted",
        })
        _save_session(folder, session)
        return {"version": record["v"], "doc": doc, "by": "llm"}


def save_manual(root, session_id: str, *, doc: str, content: str) -> dict:
    folder = Path(root)
    doc_name = _token(doc, "doc")
    with _LOCK:
        session = _load_session(folder, session_id)
        _reject_archived(session)
        if doc_name not in session.get("docs", {}):
            raise ScriptError(f"document not found: {doc_name}", status=404)
        previous = _read_version(folder, session, doc_name)
        updated = protect_beats(previous, content)
        if doc_name == "outline":
            validate_outline(updated, load_sections(folder))
        record = _add_version(
            folder, session, doc_name, updated.encode("utf-8"),
            by="user", model="", note="manual",
        )
        block = feedback_block(doc_name, previous, updated, record["at"])
        prior_feedback = ""
        if "feedback" in session["docs"]:
            prior_feedback = _read_version(folder, session, "feedback")
        _add_version(
            folder, session, "feedback", (prior_feedback + block).encode("utf-8"),
            by="user", model="", note=f"manual {doc_name}",
        )
        session["turns"].append({
            "at": record["at"], "by": "user", "doc": doc_name, "instruction": "",
            "result_version": record["v"], "outcome": "manual",
        })
        _save_session(folder, session)
        return {"version": record["v"], "by": "user", "doc": doc_name}


def confirm_session(root, session_id: str, *, approved_by: str) -> dict:
    approver = str(approved_by or "").strip()
    if approver != _APPROVER:
        raise ScriptError("approved_by must be 张景")
    folder = Path(root)
    with _LOCK:
        session = _load_session(folder, session_id)
        _reject_archived(session)
        if session.get("stage") != "outline":
            raise ScriptError("stage must be outline before confirm")
        if "outline" not in session.get("docs", {}):
            raise ScriptError("outline is missing")
        session["confirmed"] = {"by": approver, "at": _now()}
        session["stage"] = "confirmed"
        session["turns"].append({
            "at": session["confirmed"]["at"], "by": approver, "doc": "outline",
            "instruction": "", "result_version": session["docs"]["outline"]["current"],
            "outcome": "accepted",
        })
        _save_session(folder, session)
        return _attach_text(folder, session)


def run_episode(root, session_id: str, llm, *, episode: str) -> dict:
    folder = Path(root)
    with _LOCK:
        session = _load_session(folder, session_id)
        _reject_archived(session)
        if session.get("stage") != "confirmed":
            raise ScriptError("write_episode requires stage confirmed")
        outline = _read_version(folder, session, "outline") if "outline" in session["docs"] else ""
        model = _model_name(session)
    system = load_prompt("episode", folder)
    user = f"task: episode\nepisode: {episode}\n---\n{outline}"
    parts = split_marked(
        llm.complete(system=system, user=user, model=model),
        ["SCRIPT", "LINES"],
    )
    lines = parts["LINES"] if parts["LINES"].endswith("\n") else parts["LINES"] + "\n"
    with _LOCK:
        session = _load_session(folder, session_id)
        if session.get("stage") != "confirmed":
            raise ScriptError("write_episode requires stage confirmed")
        previous = ""
        if "script" in session.get("docs", {}):
            previous = _read_version(folder, session, "script")
        script = protect_beats(previous, parts["SCRIPT"])
        _add_version(folder, session, "script", script.encode("utf-8"),
                     by="llm", model=model, note=f"episode {episode}")
        _add_version(folder, session, "lines", lines.encode("utf-8"),
                     by="llm", model=model, note=f"episode {episode}")
        session["episode"] = episode
        session["stage"] = "episode_script"
        session["turns"].append({
            "at": _now(), "by": "llm", "doc": "script", "instruction": episode,
            "result_version": session["docs"]["script"]["current"], "outcome": "accepted",
        })
        _save_session(folder, session)
    return {"docs": {"script": script, "lines": lines}}


def run_review(root, session_id: str, llm, *, doc: str = "script") -> dict:
    folder = Path(root)
    if not _review_enabled(folder):
        raise ScriptError("review is disabled")
    doc_name = _token(doc, "doc")
    with _LOCK:
        session = _load_session(folder, session_id)
        _reject_archived(session)
        if doc_name not in session.get("docs", {}):
            raise ScriptError(f"document not found: {doc_name}", status=404)
        source = _read_version(folder, session, doc_name)
        model = _model_name(session)
    system = load_prompt("review", folder)
    user = f"task: review\ndoc: {doc_name}\n---\n{source}"
    text = ensure_available(llm.complete(system=system, user=user, model=model))
    if not text.endswith("\n"):
        text += "\n"
    with _LOCK:
        session = _load_session(folder, session_id)
        _add_version(folder, session, "review", text.encode("utf-8"),
                     by="llm", model=model, note=f"review {doc_name}")
        session["turns"].append({
            "at": _now(), "by": "llm", "doc": "review", "instruction": doc_name,
            "result_version": session["docs"]["review"]["current"], "outcome": "accepted",
        })
        _save_session(folder, session)
    return {"text": text}


def diff_doc(root, session_id: str, *, doc: str, a: int, b: int) -> dict:
    folder = Path(root)
    doc_name = _token(doc, "doc")
    with _LOCK:
        session = _load_session(folder, session_id)
        left = _read_version(folder, session, doc_name, int(a))
        right = _read_version(folder, session, doc_name, int(b))
    report = diff_texts(left, right)
    report["doc"] = doc_name
    report["a"] = int(a)
    report["b"] = int(b)
    return report


def revert_doc(root, session_id: str, *, doc: str, to: int) -> dict:
    folder = Path(root)
    doc_name = _token(doc, "doc")
    with _LOCK:
        session = _load_session(folder, session_id)
        _reject_archived(session)
        target = int(to)
        payload = _version_path(folder, session, doc_name, target).read_bytes()
        record = _add_version(
            folder, session, doc_name, payload,
            by="user", model="", note=f"revert to v{target}",
        )
        session["turns"].append({
            "at": record["at"], "by": "user", "doc": doc_name,
            "instruction": f"revert to v{target}",
            "result_version": record["v"], "outcome": "accepted",
        })
        _save_session(folder, session)
        return {"version": record["v"], "doc": doc_name}


def enqueue(root, session_id: str, action: str, payload: dict) -> dict:
    if action not in _JOB_ACTIONS:
        raise ScriptError(f"unknown job action {action}")
    folder = Path(root)
    job = {
        "job_id": uuid.uuid4().hex,
        "action": action,
        "status": "queued",
        "payload": payload,
        "result": None,
        "error": None,
    }
    with _LOCK:
        _load_session(folder, session_id)
        path = _session_dir(folder, session_id) / ".jobs" / f"{job['job_id']}.json"
        _write_json(path, job)
    return {"job_id": job["job_id"], "status": "queued"}


def _job_path(root: Path, session_id: str, job_id: str) -> Path:
    return _session_dir(root, session_id) / ".jobs" / f"{job_id}.json"


def job_status(root, session_id: str, job_id: str) -> dict:
    path = _job_path(Path(root), session_id, job_id)
    data = _read_json(path, None)
    if not isinstance(data, dict):
        raise ScriptError(f"job not found: {job_id}", status=404)
    return data


def _run_job(root: Path, session_id: str, action: str, payload: dict, llm) -> dict:
    if action == "directions":
        return run_directions(
            root, session_id, llm,
            n=int(payload.get("n") or 4),
            extra=str(payload.get("extra") or ""),
        )
    if action == "choose":
        return run_choose(
            root, session_id, llm,
            direction_id=str(payload.get("direction_id") or ""),
            note=str(payload.get("note") or ""),
        )
    if action == "revise":
        return run_revise(
            root, session_id, llm,
            doc=str(payload.get("doc") or ""),
            instruction=str(payload.get("instruction") or ""),
            selection=payload.get("selection") if isinstance(payload.get("selection"), dict) else None,
            candidate_id=payload.get("candidate_id"),
        )
    if action == "write_episode":
        return run_episode(root, session_id, llm, episode=str(payload.get("episode") or ""))
    if action == "review":
        return run_review(root, session_id, llm, doc=str(payload.get("doc") or "script"))
    raise ScriptError(f"unknown job action {action}")


def _claim_queued(job_dir: Path) -> list[tuple[Path, dict]]:
    if not job_dir.is_dir():
        return []
    claimed: list[tuple[Path, dict]] = []
    with _LOCK:
        for path in sorted(job_dir.glob("*.json")):
            job = _read_json(path, None)
            if not isinstance(job, dict) or job.get("status") != "queued":
                continue
            job["status"] = "running"
            job["started_at"] = _now()
            _write_json(path, job)
            claimed.append((path, job))
    return claimed


def _has_queued(job_dir: Path) -> bool:
    if not job_dir.is_dir():
        return False
    for path in job_dir.glob("*.json"):
        job = _read_json(path, None)
        if isinstance(job, dict) and job.get("status") == "queued":
            return True
    return False


def pump(root, llm, *, session_id: str) -> list[dict]:
    folder = Path(root)
    session_key = _token(session_id, "session_id")
    job_dir = _session_dir(folder, session_key) / ".jobs"
    finished = []
    while True:
        claimed = _claim_queued(job_dir)
        if not claimed:
            break
        for path, job in claimed:
            try:
                result = _run_job(
                    folder, session_key, job["action"], job.get("payload") or {}, llm,
                )
                usage = _usage(llm)
                if usage and isinstance(result, dict) and "usage" not in result:
                    result = dict(result)
                    result["usage"] = usage
                job["result"] = result
                job["status"] = "done"
                job["error"] = None
            except ScriptError as exc:
                job["status"] = "error"
                job["result"] = None
                job["error"] = str(exc)
            except Exception:
                job["status"] = "error"
                job["result"] = None
                job["error"] = "script writer job failed"
            job["finished_at"] = _now()
            _write_json(path, job)
            finished.append(job)
    return finished


def _root_from_args(args: dict) -> Path:
    if args.get("root"):
        return Path(str(args["root"]))
    project = args.get("project")
    if not project and isinstance(args.get("manifest"), dict):
        project = args["manifest"].get("project")
    if not project:
        raise ScriptError("root is required")
    path = Path(data_dir()) / "projects" / _token(str(project), "project")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _maybe_thread(root: Path, session_id: str) -> None:
    if os.environ.get("COMFYTV_TESTING") == "1":
        return
    with _LOCK:
        if session_id in _RUNNING:
            return
        _RUNNING.add(session_id)
    threading.Thread(
        target=_pump_live, args=(Path(root), session_id), daemon=True,
    ).start()


def _pump_live(root: Path, session_id: str) -> None:
    job_dir = _session_dir(root, session_id) / ".jobs"
    try:
        try:
            session = _load_session(root, session_id)
            client = build_client(root, session.get("llm") if isinstance(session.get("llm"), dict) else {})
        except ScriptError as exc:
            _fail_queued(root, session_id, str(exc))
            return
        pump(root, client, session_id=session_id)
    finally:
        with _LOCK:
            more = _has_queued(job_dir)
            if not more:
                _RUNNING.discard(session_id)
        if more:
            _pump_live(root, session_id)


def _fail_queued(root: Path, session_id: str, message: str) -> None:
    job_dir = _session_dir(root, session_id) / ".jobs"
    if not job_dir.is_dir():
        return
    for path in job_dir.glob("*.json"):
        job = _read_json(path, None)
        if isinstance(job, dict) and job.get("status") == "queued":
            job["status"] = "error"
            job["error"] = message
            _write_json(path, job)


def dispatch(args: dict) -> dict:
    if not isinstance(args, dict):
        raise ScriptError("arguments must be an object")
    action = str(args.get("action") or "")
    if action == "to_shots":
        raise ScriptError("to_shots is not implemented in script_writer core")
    root = _root_from_args(args)
    if action == "start":
        return start_session(
            root,
            project=str(args.get("project") or ""),
            episode=args.get("episode"),
            idea=str(args.get("idea") or ""),
            source_text=str(args.get("source_text") or ""),
            reference_works=str(args.get("reference_works") or ""),
            series=args.get("series"),
            llm=args.get("llm") if isinstance(args.get("llm"), dict) else None,
        )
    if action == "import":
        manifest = args.get("manifest")
        if not isinstance(manifest, dict):
            raise ScriptError("manifest must be an object")
        source_dir = args.get("source_dir")
        if not source_dir:
            raise ScriptError("source_dir is required")
        return import_scripts(root, manifest, source_dir=source_dir)
    session_id = str(args.get("session_id") or "")
    if action == "get":
        return get_session(root, session_id)
    if action == "verify":
        return verify_imports(root, session_id)
    if action == "current_storyboard":
        return current_storyboard(root, project=str(args.get("project") or ""))
    if action == "status":
        return job_status(root, session_id, str(args.get("job_id") or ""))
    if action == "accept":
        return accept_candidate(root, session_id, str(args.get("candidate_id") or ""))
    if action == "save_manual":
        return save_manual(
            root, session_id,
            doc=str(args.get("doc") or ""),
            content=str(args.get("content") or ""),
        )
    if action == "confirm":
        return confirm_session(root, session_id, approved_by=str(args.get("approved_by") or ""))
    if action == "diff":
        return diff_doc(
            root, session_id,
            doc=str(args.get("doc") or ""),
            a=int(args.get("a")),
            b=int(args.get("b")),
        )
    if action == "revert":
        return revert_doc(
            root, session_id,
            doc=str(args.get("doc") or ""),
            to=int(args.get("to")),
        )
    if action in _JOB_ACTIONS:
        payload = {
            "n": args.get("n", 4),
            "extra": args.get("extra") or "",
            "direction_id": args.get("direction_id") or "",
            "note": args.get("note") or "",
            "doc": args.get("doc") or "",
            "instruction": args.get("instruction") or "",
            "selection": args.get("selection") if isinstance(args.get("selection"), dict) else None,
            "candidate_id": args.get("candidate_id"),
            "episode": args.get("episode") or "",
        }
        queued = enqueue(root, session_id, action, payload)
        _maybe_thread(root, session_id)
        return queued
    raise ScriptError(f"unknown action {action}")
