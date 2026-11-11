"""Character studio on top of the series library.

Records, aliases, lock checks, and studio_history live in ``library.yaml``.
Image bytes live under ``<series>/<ID>/v<n>/``. Still images are assembled as
MiniMax H3 Image graphs (image VAE, ``<Picture N>``). Voice auditions are
assembled for the existing ``comfytv_s2pro`` node (``ComfyTV.S2Pro``).
Generation is queued by the caller; this module does not submit prompts and
does not call ``/interrupt``, ``/free``, or ``cancel_stage``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from datetime import datetime
from pathlib import Path

import yaml

from .series_library import (
    SeriesLibraryError,
    _as_root,
    character_dependencies,
    get_series_asset,
    new_series_asset_version,
    parse_series_ref,
    read_series_document,
    resolve_series_ref,
    update_draft_series_asset,
)

_TEMPLATES = Path(__file__).with_name("character_studio_templates")
_DEFAULTS_PATH = Path(__file__).with_name("character_studio_defaults.yaml")
_DEFAULTS_CACHE: dict | None = None
_LEGACY_ID = re.compile(r"^P(?P<num>\d+)_(?P<slug>[A-Za-z0-9_]+)$")
_PICTURE = re.compile(r"<Picture (\d+)>")
_PROVENANCE = {"self", "licensed", "ai_generated"}
_MUTABLE = {"draft", "candidate"}
_REFINE_OPS = {"face": "refine_face", "hair": "refine_hair", "outfit": "refine_outfit"}


def load_studio_config(doc: dict | None) -> dict:
    """Defaults from disk, then ``comfytv.character_studio`` on the library."""
    config = _merge(_defaults(), {})
    if not isinstance(doc, dict):
        return config
    block = doc.get("comfytv")
    if isinstance(block, dict) and isinstance(block.get("character_studio"), dict):
        config = _merge(config, block["character_studio"])
    return config


def video_reference_views(record: dict, config: dict | None = None) -> list[str]:
    """Views that may enter a video reference slot. ``sheet`` stays out unless configured."""
    options = config or _defaults()
    files = record.get("files") if isinstance(record.get("files"), dict) else {}
    names = [str(item) for item in (options.get("video_ref_views") or ["front", "headshot"])]
    if options.get("sheet_in_video_ref"):
        if "sheet" not in names:
            names.append("sheet")
    else:
        names = [name for name in names if name != "sheet"]
    return [name for name in names if isinstance(files.get(name), dict)]


def resolve_character_name(root, name: str) -> dict:
    """Map a character id, Chinese name, alias, ``legacy_ids`` entry, or ``P01_`` name.

    Does not rewrite ``library.yaml``. ``P01_xiaolongnu`` is not ``CHR_xiaolongnv``.
    """
    text = str(name or "").strip()
    if len(text) >= 2 and text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    if not text:
        raise SeriesLibraryError("name is required")
    folder = _as_root(root)
    doc = read_series_document(folder)
    config = load_studio_config(doc)
    characters = _characters(doc)
    parsed = parse_series_ref(text)
    if parsed["kind"] == "asset" and str(parsed["id"]).startswith("CHR_"):
        record = _version_of(characters, parsed)
        return _resolved(record, "id")

    owners: dict[str, list[str]] = {}
    for record in characters:
        reasons: list[str] = []
        if str(record.get("name_zh") or "").strip() == text:
            reasons.append("name_zh")
        if text in _strings(record.get("aliases")):
            reasons.append("alias")
        if text in _strings(record.get("legacy_ids")):
            reasons.append("legacy_id")
        if reasons:
            owners.setdefault(str(record.get("id")), []).extend(reasons)
    if config.get("accept_legacy_p_ids", True):
        match = _LEGACY_ID.fullmatch(text)
        if match:
            candidate = f"CHR_{match.group('slug')}"
            if any(str(row.get("id")) == candidate for row in characters):
                owners.setdefault(candidate, []).append("legacy_pattern")
    ids = sorted(owners)
    if len(ids) > 1:
        raise SeriesLibraryError(
            "alias conflict: " + ", ".join(ids),
            status=409,
            details={"conflicts": ids, "name": text},
        )
    if not ids:
        raise SeriesLibraryError(f"no character matches {text!r}", status=404)
    rows = [row for row in characters if str(row.get("id")) == ids[0]]
    return _resolved(_prefer_locked(rows), owners[ids[0]][0])


def get_character(root, code: str) -> dict:
    found = resolve_character_name(root, code)
    record = get_series_asset(root, found["code"])
    if record.get("kind") != "character":
        raise SeriesLibraryError(f"{found['code']} is not a character")
    doc = read_series_document(root)
    config = load_studio_config(doc)
    version = int(record["version"])
    video_refs = [
        f"{record['id']}@v{version}#{view}"
        for view in video_reference_views(record, config)
    ]
    return {
        "asset": record,
        "dependencies": character_dependencies(root, found["code"]),
        "video_refs": video_refs,
        "sheet_in_video_ref": bool(config.get("sheet_in_video_ref")),
        "ref": found["ref"],
        "matched_by": found["matched_by"],
    }


def import_face(root, code: str, files, provenance: str, note: str | None = None,
                *, by: str = "character_studio") -> dict:
    if provenance not in _PROVENANCE:
        raise SeriesLibraryError(
            "provenance must be self, licensed, or ai_generated")
    if not isinstance(files, list) or not files:
        raise SeriesLibraryError("files is required")
    record = _ensure_mutable(root, code)
    folder = _as_root(root)
    face_source = [dict(item) for item in (record.get("face_source") or []) if isinstance(item, dict)]
    file_entries: dict[str, dict] = {}
    copied: list[str] = []
    notes: list[str] = []
    start = len(face_source)
    for offset, item in enumerate(files, start=1):
        path_text, item_note = _file_arg(item)
        source = Path(path_text).expanduser()
        if not source.is_file():
            raise SeriesLibraryError(f"file not found: {path_text}", status=404)
        suffix = source.suffix.lower() or ".png"
        view = f"face_{start + offset:02d}"
        rel = f"{record['id']}/v{int(record['version'])}/{view}{suffix}"
        dest = _safe_dest(folder, rel)
        if source.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            _writable(dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        file_entries[view] = {
            "path": rel,
            "media_type": "image",
            "usage": ["face_source", "review"],
        }
        written_note = item_note or (note or "")
        face_source.append({
            "file": rel,
            "sha256": digest,
            "provenance": provenance,
            "note": written_note,
        })
        copied.append(rel)
        notes.append(written_note)
    blob = "\n".join([provenance, note or "", *copied])
    entry = _history(
        op="import_face",
        inputs=[provenance, note or "", *copied],
        prompt_hash=_hash_text(blob),
        seed=0,
        seeds=[0],
        workflow="import",
        outputs=copied,
        by=by,
    )
    updated = _write(folder, record, {
        "files": file_entries,
        "face_source": face_source,
        "studio_history": _with_history(record, entry),
    })
    return _planned(updated, entry, None)


def compose_outfit(root, code: str, face: str, outfit_body: str, *,
                   batch: int | None = None, seed: int | None = None,
                   image_1: str | None = None, by: str = "character_studio") -> dict:
    record = _ensure_mutable(root, code)
    folder = _as_root(root)
    doc = read_series_document(folder)
    config = load_studio_config(doc)
    which = str(image_1 or config.get("image_1_default") or "outfit_body")
    if which not in ("face", "outfit_body"):
        raise SeriesLibraryError("image_1 must be face or outfit_body")
    count = _batch_count(config, batch, kind="compose")
    seeds = _seed_run(seed, count)
    face_rel, face_copied = _stage_image(folder, record, _image_source(folder, face), "src_face")
    outfit_rel, outfit_copied = _stage_image(
        folder, record, _image_source(folder, outfit_body), "src_outfit")
    ordered = (
        [("outfit_body", outfit_rel), ("face", face_rel)]
        if which == "outfit_body"
        else [("face", face_rel), ("outfit_body", outfit_rel)]
    )
    slots = {role: f"<Picture {index}>" for index, (role, _rel) in enumerate(ordered, start=1)}
    text = _render(folder, "compose_outfit", {
        "face_picture": slots["face"],
        "outfit_picture": slots["outfit_body"],
    })
    width, height = _canvas(config, str(config.get("image_aspect") or "16:9"))
    prefix = _prefix(record, "compose_outfit")
    prompt, outputs = build_h3_image_workflow(
        doc, ordered, [text], seeds, width, height, prefix, config)
    entry = _history(
        op="compose_outfit",
        inputs=[f"face={face}", f"outfit_body={outfit_body}", f"image_1={which}"],
        prompt_hash=_hash_text(text),
        seed=seeds[0],
        seeds=seeds,
        workflow="h3_image",
        outputs=outputs,
        by=by,
    )
    files = {}
    if face_copied:
        files["src_face"] = _image_file(face_rel, ["studio_input"])
    if outfit_copied:
        files["src_outfit"] = _image_file(outfit_rel, ["studio_input"])
    changes: dict = {"studio_history": _with_history(record, entry)}
    if files:
        changes["files"] = files
    updated = _write(folder, record, changes)
    return _planned(updated, entry, prompt)


def make_sheet(root, code: str, *, source: str | None = None, seed: int | None = None,
               by: str = "character_studio") -> dict:
    record = _ensure_mutable(root, code)
    folder = _as_root(root)
    doc = read_series_document(folder)
    config = load_studio_config(doc)
    view = source or "front"
    rel = _material_rel(folder, record, view)
    text = _render(folder, "make_sheet", {
        "proportion": str(config.get("sheet_proportion") or "二八身材比例"),
        "source_picture": "<Picture 1>",
        "aspect": str(config.get("sheet_aspect") or "16:9"),
    })
    seeds = _seed_run(seed, _batch_count(config, None, kind="sheet"))
    width, height = _canvas(config, str(config.get("sheet_aspect") or "16:9"))
    prompt, outputs = build_h3_image_workflow(
        doc, [("front", rel)], [text], seeds, width, height, _prefix(record, "sheet"), config)
    entry = _history(
        op="sheet",
        inputs=[f"from={view}", rel],
        prompt_hash=_hash_text(text),
        seed=seeds[0],
        seeds=seeds,
        workflow="h3_image",
        outputs=outputs,
        by=by,
    )
    updated = _write(folder, record, {"studio_history": _with_history(record, entry)})
    return _planned(updated, entry, prompt)


def derive_views(root, code: str, views: list | None = None, *,
                 seed: int | None = None, by: str = "character_studio") -> dict:
    record = _ensure_mutable(root, code)
    folder = _as_root(root)
    doc = read_series_document(folder)
    config = load_studio_config(doc)
    requested = [str(item) for item in (views or config.get("derive_views") or [])]
    if not requested:
        raise SeriesLibraryError("views is required")
    labels = config.get("view_labels") or {}
    unknown = [view for view in requested if view not in labels]
    if unknown:
        raise SeriesLibraryError("unknown views: " + ", ".join(unknown))
    rel = _material_rel(folder, record, "front")
    instructions = config.get("view_instructions") or {}
    texts = [
        _render(folder, "derive_view", {
            "source_picture": "<Picture 1>",
            "view_label": labels[view],
            "instruction": str(instructions.get(view) or ""),
        })
        for view in requested
    ]
    seeds = _seed_run(seed, len(texts))
    width, height = _canvas(config, str(config.get("image_aspect") or "16:9"))
    prompt, outputs = build_h3_image_workflow(
        doc, [("front", rel)], texts, seeds, width, height,
        _prefix(record, "derive_view"), config)
    entry = _history(
        op="derive_view",
        inputs=[f"from=front", *requested],
        prompt_hash=_hash_text("\n".join(texts)),
        seed=seeds[0],
        seeds=seeds,
        workflow="h3_image",
        outputs=outputs,
        by=by,
    )
    updated = _write(folder, record, {"studio_history": _with_history(record, entry)})
    return _planned(updated, entry, prompt)


def refine(root, code: str, part: str, instruction: str, *,
           seed: int | None = None, by: str = "character_studio") -> dict:
    if part not in _REFINE_OPS:
        raise SeriesLibraryError("part must be face, hair, or outfit")
    text_instruction = str(instruction or "").strip()
    if not text_instruction:
        raise SeriesLibraryError("instruction is required")
    record = _ensure_mutable(root, code)
    folder = _as_root(root)
    doc = read_series_document(folder)
    config = load_studio_config(doc)
    root_rel, files_update = _pin_refine_root(folder, record)
    rendered = _render(folder, "refine", {
        "source_picture": "<Picture 1>",
        "part_label": str((config.get("part_labels") or {}).get(part) or part),
        "instruction": text_instruction,
    })
    seeds = _seed_run(seed, _batch_count(config, None, kind="refine"))
    width, height = _canvas(config, str(config.get("image_aspect") or "16:9"))
    prompt, outputs = build_h3_image_workflow(
        doc, [("root", root_rel)], [rendered], seeds, width, height,
        _prefix(record, _REFINE_OPS[part]), config)
    entry = _history(
        op=_REFINE_OPS[part],
        inputs=[f"part={part}", text_instruction, root_rel],
        prompt_hash=_hash_text(rendered),
        seed=seeds[0],
        seeds=seeds,
        workflow="h3_image",
        outputs=outputs,
        by=by,
    )
    changes: dict = {"studio_history": _with_history(record, entry)}
    if files_update:
        changes["files"] = files_update
    if part in ("face", "hair"):
        changes["needs_rederive"] = list(config.get("rederive_views") or [])
    updated = _write(folder, record, changes)
    return _planned(updated, entry, prompt)


def voice_audition(root, voice_code: str, text: str, seeds: list, *,
                   desc_voice_en: str | None = None,
                   by: str = "character_studio") -> dict:
    line = str(text or "").strip()
    if not line:
        raise SeriesLibraryError("text is required")
    if not isinstance(seeds, list) or not seeds:
        raise SeriesLibraryError("seeds is required")
    record = _ensure_mutable(root, voice_code)
    if record.get("kind") != "voice":
        raise SeriesLibraryError("voice_code must be a VOX_ asset")
    folder = _as_root(root)
    doc = read_series_document(folder)
    config = load_studio_config(doc)
    low = int(config.get("audition_seeds_min") or 3)
    high = int(config.get("audition_seeds_max") or 5)
    if not low <= len(seeds) <= high:
        raise SeriesLibraryError(f"voice audition needs {low}-{high} seeds")
    audio_rel = _material_rel(folder, record, "voice_ref")
    reference_text = _read_text_view(folder, record, "voice_ref_txt")
    params = _s2_params(folder, record, config)
    seed_values = [int(item) for item in seeds]
    prompt, outputs = build_s2pro_workflow(
        doc, audio_rel, reference_text, line, seed_values, params,
        _prefix(record, "voice_audition"), config)
    entry = _history(
        op="voice_audition",
        inputs=[line, audio_rel],
        prompt_hash=_hash_text(f"{line}\n{reference_text}"),
        seed=seed_values[0],
        seeds=seed_values,
        workflow="s2pro",
        outputs=outputs,
        by=by,
    )
    audition = [dict(item) for item in (record.get("audition") or []) if isinstance(item, dict)]
    for seed, output in zip(seed_values, outputs):
        audition.append({
            "seed": seed,
            "file": output,
            "prompt_hash": entry["prompt_hash"],
            "text": line,
            "at": entry["at"],
        })
    changes: dict = {
        "studio_history": _with_history(record, entry),
        "audition": audition,
    }
    if desc_voice_en:
        changes["desc_voice_en"] = str(desc_voice_en)
    updated = _write(folder, record, changes)
    return _planned(updated, entry, prompt)


def bind_voice(root, code: str, voice_code: str, *,
               by: str = "character_studio") -> dict:
    record = _ensure_mutable(root, code)
    if record.get("kind") != "character":
        raise SeriesLibraryError("code must be a character")
    voice = get_series_asset(root, voice_code)
    if voice.get("kind") != "voice":
        raise SeriesLibraryError("voice_code must be a VOX_ asset")
    ref = f"{voice['id']}@v{int(voice['version'])}"
    entry = _history(
        op="bind_voice",
        inputs=[voice_code],
        prompt_hash=_hash_text(ref),
        seed=0,
        seeds=[0],
        workflow="bind",
        outputs=[ref],
        by=by,
    )
    updated = _write(_as_root(root), record, {
        "voice": ref,
        "studio_history": _with_history(record, entry),
    })
    return _planned(updated, entry, None)


def pick_candidate(root, code: str, op_id: str, index: int, as_view: str, *,
                   source: str | None = None, by: str = "character_studio") -> dict:
    record = get_series_asset(root, code)
    if record.get("status") not in _MUTABLE:
        raise SeriesLibraryError("pick writes a draft or candidate version", status=409)
    view = str(as_view or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_]+", view):
        raise SeriesLibraryError(f"invalid view name {as_view!r}")
    history = [dict(item) for item in (record.get("studio_history") or []) if isinstance(item, dict)]
    entry = next((item for item in history if item.get("op_id") == op_id), None)
    if entry is None:
        raise SeriesLibraryError(f"unknown op_id {op_id}", status=404)
    outputs = [str(item) for item in (entry.get("outputs") or [])]
    chosen_index = int(index)
    if chosen_index < 1 or chosen_index > max(len(outputs), 1 if source else 0):
        raise SeriesLibraryError("index is outside the candidate list")
    chosen = str(source or (outputs[chosen_index - 1] if outputs else ""))
    source_path = Path(chosen).expanduser()
    if not chosen or not source_path.is_file():
        raise SeriesLibraryError(
            f"candidate file is not ready: {chosen or '<missing>'}",
            status=404,
        )
    folder = _as_root(root)
    config = load_studio_config(read_series_document(folder))
    suffix = source_path.suffix.lower() or ".png"
    rel = f"{record['id']}/v{int(record['version'])}/{view}{suffix}"
    dest = _safe_dest(folder, rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != dest.resolve():
        shutil.copyfile(source_path, dest)
        _writable(dest)
    entry["picked"] = rel
    entry["outputs"] = outputs
    needs = list(record.get("needs_rederive") or [])
    if view in needs:
        needs = [item for item in needs if item != view]
    updated = _write(folder, record, {
        "files": {view: _image_file(rel, _usage_for(view, config))},
        "studio_history": history,
        "needs_rederive": needs,
    })
    stored = updated["files"][view]
    return {
        "code": _code(updated),
        "op_id": op_id,
        "view": view,
        "path": stored["path"],
        "sha256": stored["sha256"],
        "asset": updated,
    }


def diff_characters(root, a: str, b: str) -> dict:
    left = get_series_asset(root, a)
    right = get_series_asset(root, b)
    fields = []
    for key in (
        "status", "name_zh", "aliases", "legacy_ids", "identity",
        "wardrobe", "voice", "needs_rederive", "face_source", "desc_voice_en",
    ):
        if left.get(key) != right.get(key):
            fields.append({"field": key, "a": left.get(key), "b": right.get(key)})
    names = set((left.get("files") or {})) | set((right.get("files") or {}))
    files = []
    for name in sorted(names):
        sha_a = _file_sha(left, name)
        sha_b = _file_sha(right, name)
        if sha_a != sha_b:
            files.append({"view": name, "a": sha_a, "b": sha_b})
    return {
        "a": _code(left),
        "b": _code(right),
        "fields": fields,
        "files": files,
        "urls": {"a": _file_urls(left), "b": _file_urls(right)},
    }


def attach_job(root, code: str, op_id: str, job_id: str) -> dict:
    record = get_series_asset(root, code)
    history = [dict(item) for item in (record.get("studio_history") or []) if isinstance(item, dict)]
    found = False
    for entry in history:
        if entry.get("op_id") == op_id:
            entry["prompt_id"] = str(job_id)
            found = True
    if not found:
        raise SeriesLibraryError(f"unknown op_id {op_id}", status=404)
    return update_draft_series_asset(root, _code(record), {"studio_history": history})


def build_h3_image_workflow(doc: dict, images: list[tuple[str, str]], texts: list[str],
                            seeds: list[int], width: int, height: int,
                            filename_prefix: str, config: dict) -> tuple[dict, list[str]]:
    """API prompt for MiniMax H3 Image. ``image_1`` is the picture being edited."""
    vae_name = str(config.get("image_vae") or "")
    if not vae_name or vae_name == str(config.get("video_vae") or ""):
        raise SeriesLibraryError(
            "still images must use the H3 image VAE, not the video VAE")
    if not images or not texts or not seeds:
        raise SeriesLibraryError("image workflow needs images, prompt text, and seeds")
    prompt: dict[str, dict] = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": config["unet"],
            "weight_dtype": config.get("weight_dtype") or "default",
        }},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": config["clip"],
            "type": config.get("clip_type") or "minimax",
        }},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
    }
    next_id = 4
    load_ids: list[str] = []
    for _role, rel in images:
        node_id = str(next_id)
        prompt[node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": _load_name(doc, rel)},
        }
        load_ids.append(node_id)
        next_id += 1
    cond_ids: list[str] = []
    for text in texts:
        _assert_pictures(text, len(images))
        inputs: dict = {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": text,
            "width": int(width),
            "height": int(height),
        }
        for index, load_id in enumerate(load_ids, start=1):
            inputs[f"image_{index}"] = [load_id, 0]
        node_id = str(next_id)
        prompt[node_id] = {
            "class_type": "MiniMaxH3ImageConditioning",
            "inputs": inputs,
        }
        cond_ids.append(node_id)
        next_id += 1
    if len(texts) == 1:
        pairs = [(cond_ids[0], int(seed)) for seed in seeds]
    else:
        pairs = [
            (cond_id, int(seeds[index] if index < len(seeds) else seeds[0]))
            for index, cond_id in enumerate(cond_ids)
        ]
    outputs: list[str] = []
    for index, (cond_id, seed) in enumerate(pairs):
        sampler = str(next_id)
        prompt[sampler] = {"class_type": "KSampler", "inputs": {
            "model": ["1", 0],
            "seed": seed,
            "steps": int(config.get("steps") or 6),
            "cfg": float(config.get("cfg") or 1),
            "sampler_name": config.get("sampler") or "euler",
            "scheduler": config.get("scheduler") or "simple",
            "positive": [cond_id, 0],
            "negative": [cond_id, 0],
            "latent_image": [cond_id, 1],
            "denoise": float(config.get("denoise") or 1),
        }}
        next_id += 1
        decode = str(next_id)
        prompt[decode] = {"class_type": "VAEDecode", "inputs": {
            "samples": [sampler, 0],
            "vae": ["3", 0],
        }}
        next_id += 1
        save = str(next_id)
        prefix = f"{filename_prefix}_{seed}" if len(texts) == 1 else f"{filename_prefix}_{index}_{seed}"
        prompt[save] = {"class_type": "SaveImage", "inputs": {
            "images": [decode, 0],
            "filename_prefix": prefix,
        }}
        outputs.append(prefix)
        next_id += 1
    return prompt, outputs


def build_s2pro_workflow(doc: dict, audio_rel: str, reference_text: str, line: str,
                         seeds: list[int], params: dict, filename_prefix: str,
                         config: dict) -> tuple[dict, list[str]]:
    """API prompt for the installed comfytv_s2pro node. One LoadAudio, one node per seed."""
    voice = config.get("voice") or {}
    names = voice.get("inputs") or {}
    class_type = str(voice.get("class_type") or "ComfyTV.S2Pro")
    prompt: dict[str, dict] = {
        "1": {"class_type": "LoadAudio", "inputs": {"audio": _load_name(doc, audio_rel)}},
    }
    outputs: list[str] = []
    next_id = 2
    for seed in seeds:
        node_id = str(next_id)
        prompt[node_id] = {"class_type": class_type, "inputs": {
            names.get("reference_audio", "reference_audio"): ["1", 0],
            names.get("reference_text", "reference_text"): reference_text,
            names.get("text", "text"): line,
            names.get("seed", "seed"): int(seed),
            names.get("temperature", "temperature"): params["temperature"],
            names.get("top_p", "top_p"): params["top_p"],
            names.get("chunk_length", "chunk_length"): params["chunk_length"],
        }}
        next_id += 1
        save = str(next_id)
        prefix = f"{filename_prefix}_{int(seed)}"
        prompt[save] = {"class_type": "SaveAudio", "inputs": {
            "audio": [node_id, 0],
            "filename_prefix": prefix,
        }}
        outputs.append(prefix)
        next_id += 1
    return prompt, outputs


def _defaults() -> dict:
    global _DEFAULTS_CACHE
    if _DEFAULTS_CACHE is None:
        loaded = yaml.safe_load(_DEFAULTS_PATH.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise SeriesLibraryError("character_studio_defaults.yaml must be a mapping")
        _DEFAULTS_CACHE = loaded
    return _DEFAULTS_CACHE


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _characters(doc: dict) -> list[dict]:
    return [
        row for row in (doc.get("assets") or [])
        if isinstance(row, dict) and row.get("kind") == "character" and row.get("id")
    ]


def _strings(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _version_of(characters: list[dict], parsed: dict) -> dict:
    rows = [row for row in characters if row.get("id") == parsed["id"]]
    if not rows:
        raise SeriesLibraryError(f"asset {parsed['id']} not found", status=404)
    if parsed.get("version") is None:
        return _prefer_locked(rows)
    for row in rows:
        if int(row.get("version") or 0) == int(parsed["version"]):
            return row
    raise SeriesLibraryError(
        f"asset {parsed['id']}@v{parsed['version']} not found", status=404)


def _prefer_locked(rows: list[dict]) -> dict:
    locked = [row for row in rows if row.get("status") == "locked"]
    pool = locked or rows
    return max(pool, key=lambda row: int(row.get("version") or 0))


def _resolved(record: dict, matched_by: str) -> dict:
    version = int(record["version"])
    code = f"{record['id']}@v{version}"
    return {
        "resolved": True,
        "id": record["id"],
        "version": version,
        "status": record.get("status"),
        "code": code,
        "ref": f"{code}#front",
        "matched_by": matched_by,
    }


def _ensure_mutable(root, code: str) -> dict:
    parsed = parse_series_ref(str(code))
    if parsed["kind"] != "asset":
        found = resolve_character_name(root, code)
        record = get_series_asset(root, found["code"])
    elif parsed.get("version") is None and str(parsed["id"]).startswith("CHR_"):
        found = resolve_character_name(root, parsed["id"])
        record = get_series_asset(root, found["code"])
    else:
        lookup = str(code) if parsed.get("version") is not None else str(parsed["id"])
        record = get_series_asset(root, lookup)
    if record.get("status") in _MUTABLE:
        return record
    if record.get("status") == "locked":
        return new_series_asset_version(root, _code(record))
    raise SeriesLibraryError(f"cannot edit status {record.get('status')}")


def _write(folder: Path, record: dict, changes: dict) -> dict:
    return update_draft_series_asset(folder, _code(record), changes)


def _planned(record: dict, entry: dict, prompt: dict | None) -> dict:
    result = {
        "code": _code(record),
        "op_id": entry["op_id"],
        "asset": record,
        "studio_history": entry,
    }
    if prompt is not None:
        result["prompt"] = prompt
    return result


def _with_history(record: dict, entry: dict) -> list:
    history = [dict(item) for item in (record.get("studio_history") or []) if isinstance(item, dict)]
    history.append(entry)
    return history


def _history(*, op: str, inputs: list, prompt_hash: str, seed: int, seeds: list[int],
             workflow: str, outputs: list, by: str) -> dict:
    return {
        "op_id": f"{op}-{uuid.uuid4().hex[:12]}",
        "op": op,
        "inputs": [str(item) for item in inputs],
        "prompt_hash": prompt_hash,
        "seed": int(seed),
        "seeds": [int(item) for item in seeds],
        "workflow": workflow,
        "prompt_id": "",
        "outputs": [str(item) for item in outputs],
        "picked": None,
        "at": _now(),
        "by": by,
    }


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _code(record: dict) -> str:
    return f"{record['id']}@v{int(record['version'])}"


def _prefix(record: dict, op: str) -> str:
    return f"comfytv/character_studio/{record['id']}/v{int(record['version'])}/{op}"


def _batch_count(config: dict, requested: int | None, *, kind: str) -> int:
    if kind == "compose":
        default = int(config.get("batch_default") or 2)
        low = int(config.get("batch_min") or 2)
        high = int(config.get("batch_max") or 4)
    elif kind == "sheet":
        default = int(config.get("sheet_batch") or 1)
        low, high = 1, int(config.get("batch_max") or 4)
    elif kind == "refine":
        default = int(config.get("refine_batch") or 1)
        low, high = 1, int(config.get("batch_max") or 4)
    else:
        default = 1
        low, high = 1, int(config.get("batch_max") or 4)
    count = default if requested is None else int(requested)
    if count < low or count > high:
        raise SeriesLibraryError(f"batch must be {low}-{high}")
    return count


def _seed_run(seed: int | None, count: int) -> list[int]:
    base = 1 if seed is None else int(seed)
    return [base + offset for offset in range(count)]


def _canvas(config: dict, aspect: str) -> tuple[int, int]:
    widths = config.get("widths") or {}
    pair = widths.get(aspect)
    if not isinstance(pair, list) or len(pair) != 2:
        raise SeriesLibraryError(f"unknown aspect {aspect}")
    width, height = int(pair[0]), int(pair[1])
    if width % 32 or height % 32:
        raise SeriesLibraryError("canvas edges must be multiples of 32")
    return width, height


def _render(folder: Path, name: str, fields: dict) -> str:
    custom = folder / "character_studio" / "templates" / f"{name}.txt"
    path = custom if custom.is_file() else _TEMPLATES / f"{name}.txt"
    try:
        rendered = path.read_text(encoding="utf-8").format(**fields)
    except KeyError as exc:
        raise SeriesLibraryError(f"template {name} is missing {exc}") from exc
    return rendered.strip()


def _assert_pictures(text: str, image_count: int) -> None:
    found = {int(item) for item in _PICTURE.findall(text)}
    expected = set(range(1, image_count + 1))
    if found != expected:
        raise SeriesLibraryError(
            f"prompt pictures {sorted(found)} do not match image inputs {sorted(expected)}")


def _image_source(folder: Path, spec: str) -> Path:
    text = str(spec or "").strip()
    if not text:
        raise SeriesLibraryError("image is required")
    parsed = parse_series_ref(text)
    if parsed["kind"] == "asset":
        resolved = resolve_series_ref(folder, text, require_locked=False)
        return Path(resolved["absolute_path"])
    path = Path(text).expanduser()
    if not path.is_file():
        raise SeriesLibraryError(f"image not found: {text}", status=404)
    return path.resolve()


def _stage_image(folder: Path, record: dict, source: Path, name: str) -> tuple[str, bool]:
    existing = _library_rel(folder, source)
    if existing is not None:
        return existing, False
    suffix = source.suffix.lower() or ".png"
    rel = f"{record['id']}/v{int(record['version'])}/{name}{suffix}"
    dest = _safe_dest(folder, rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    _writable(dest)
    return rel, True


def _pin_refine_root(folder: Path, record: dict) -> tuple[str, dict]:
    files = record.get("files") if isinstance(record.get("files"), dict) else {}
    current = files.get("root")
    if isinstance(current, dict) and current.get("path"):
        path = folder / str(current["path"])
        if path.is_file():
            return str(current["path"]).replace("\\", "/"), {}
    front = _material_rel(folder, record, "front")
    rel = f"{record['id']}/v{int(record['version'])}/root.png"
    dest = _safe_dest(folder, rel)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / front, dest)
        _writable(dest)
    return rel, {"root": _image_file(rel, ["studio_input"])}


def _material_rel(folder: Path, record: dict, view: str) -> str:
    resolved = resolve_series_ref(
        folder, f"{_code(record)}#{view}", require_locked=False)
    absolute = Path(resolved["absolute_path"]).resolve()
    try:
        return absolute.relative_to(folder.resolve()).as_posix()
    except ValueError:
        raise SeriesLibraryError(f"view {view} is outside the series folder")


def _library_rel(folder: Path, source: Path) -> str | None:
    try:
        return source.resolve().relative_to(folder.resolve()).as_posix()
    except ValueError:
        return None


def _load_name(doc: dict, rel: str) -> str:
    series = doc.get("series") if isinstance(doc.get("series"), dict) else {}
    sub = series.get("view_url_subfolder")
    if isinstance(sub, str) and sub.strip():
        return f"{sub.strip('/')}/{rel}"
    return rel


def _safe_dest(folder: Path, rel: str) -> Path:
    if rel.startswith("/") or ".." in Path(rel).parts:
        raise SeriesLibraryError(f"asset path escapes the series folder: {rel}")
    return folder / rel


def _writable(path: Path) -> None:
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)


def _image_file(rel: str, usage: list[str]) -> dict:
    return {"path": rel, "media_type": "image", "usage": list(usage)}


def _usage_for(view: str, config: dict) -> list[str]:
    if view == "sheet" and not config.get("sheet_in_video_ref"):
        return ["review"]
    if view in ("front", "headshot"):
        return ["identity_video", "image_ref", "review"]
    if view == "sheet":
        return ["identity_video", "image_ref", "review"]
    return ["image_ref", "review"]


def _file_arg(item) -> tuple[str, str]:
    if isinstance(item, str):
        return item, ""
    if isinstance(item, dict) and item.get("path"):
        return str(item["path"]), str(item.get("note") or "")
    raise SeriesLibraryError("each file must be a path or {path, note}")


def _read_text_view(folder: Path, record: dict, view: str) -> str:
    info = (record.get("files") or {}).get(view)
    if not isinstance(info, dict) or not info.get("path"):
        return ""
    path = folder / str(info["path"])
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _s2_params(folder: Path, record: dict, config: dict) -> dict:
    voice = config.get("voice") or {}
    params = {
        "temperature": voice.get("temperature", 0.8),
        "top_p": voice.get("top_p", 0.8),
        "chunk_length": voice.get("chunk_length", 200),
    }
    info = (record.get("files") or {}).get("s2_params")
    if isinstance(info, dict) and info.get("path"):
        path = folder / str(info["path"])
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SeriesLibraryError(f"s2_params is not valid JSON: {exc}") from exc
            if isinstance(loaded, dict):
                for key in ("temperature", "top_p", "chunk_length"):
                    if key in loaded:
                        params[key] = loaded[key]
    return params


def _file_sha(record: dict, view: str):
    info = (record.get("files") or {}).get(view)
    if not isinstance(info, dict):
        return None
    return info.get("sha256")


def _file_urls(record: dict) -> dict:
    urls = {}
    for name, info in (record.get("files") or {}).items():
        if not isinstance(info, dict):
            continue
        urls[name] = info.get("payload_url") or info.get("path")
    return urls
