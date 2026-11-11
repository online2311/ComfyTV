"""Server-side ComfyUI graph used when no ComfyTV page is mirroring.

The document is a UI-format workflow (nodes, links, groups) plus a named
``widgets`` map so stage tools can set fields without LiteGraph. Runs convert
it to the API prompt ComfyUI's queue accepts.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import uuid

from .. import storage
from ..nodes.stages import STAGE_META
from ..nodes.stages.common.shot_document import load_shots_document

_log = logging.getLogger(__name__)

_KIND_OUTPUTS = {
    "text": [("text", "STRING")],
    "image": [("image", "IMAGE")],
    "image-batch": [("images", "IMAGE"), ("image", "IMAGE")],
    "image-picker": [("image", "IMAGE")],
    "video": [("video", "VIDEO")],
    "audio": [("audio", "AUDIO")],
    "audio-picker": [("audio", "AUDIO")],
}

_ASSET_GROUP = {"image": "images", "video": "videos", "audio": "audio"}
_ASSET_ITEM = {"image": "image", "video": "video", "audio": "audio"}

_REVIEW_COLORS = {
    "approved": ("#2e7d32", "#1e3320"),
    "review": ("#b58900", "#332c14"),
    "archived": ("#555555", "#262626"),
}
_NODE_MODES = {"always": 0, "mute": 2, "bypass": 4}


def project_id_of(payload: dict | None) -> str:
    raw = (payload or {}).get("project_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return "default"


def empty_document() -> dict:
    return {
        "version": 1,
        "last_node_id": 0,
        "last_link_id": 0,
        "nodes": [],
        "links": [],
        "groups": [],
    }


def _as_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_document(doc: dict) -> dict:
    if not isinstance(doc, dict):
        raise ValueError("graph document must be an object")
    out = empty_document()
    out["nodes"] = [n for n in (doc.get("nodes") or []) if isinstance(n, dict)]
    out["links"] = [l for l in (doc.get("links") or []) if isinstance(l, (list, tuple)) and len(l) >= 5]
    out["groups"] = [g for g in (doc.get("groups") or []) if isinstance(g, dict)]
    ids = [i for i in (_as_int(n.get("id")) for n in out["nodes"]) if i is not None]
    lids = [i for i in (_as_int(l[0]) for l in out["links"]) if i is not None]
    out["last_node_id"] = max([_as_int(doc.get("last_node_id")) or 0, max(ids, default=0)])
    out["last_link_id"] = max([_as_int(doc.get("last_link_id")) or 0, max(lids, default=0)])
    for node in out["nodes"]:
        node.setdefault("widgets", {})
        node.setdefault("inputs", [])
        node.setdefault("outputs", [])
        node.setdefault("properties", {})
        node.setdefault("flags", {})
        if not isinstance(node["widgets"], dict):
            node["widgets"] = {}
    return out


def load_document(project_id: str) -> dict:
    row = storage.get_headless_graph(project_id)
    if not row:
        return empty_document()
    return normalize_document(row.get("document") or {})


def save_document(project_id: str, doc: dict) -> dict:
    doc = normalize_document(doc)
    storage.save_headless_graph(project_id, doc)
    return doc


def _node_by_id(doc: dict, ref) -> dict | None:
    want = str(ref)
    for node in doc["nodes"]:
        if str(node.get("id")) == want:
            return node
    return None


def find_node(doc: dict, ref: str) -> dict | None:
    ref = str(ref)
    exact_uid = None
    prefix = None
    for node in doc["nodes"]:
        uid = str((node.get("properties") or {}).get("comfytv_uid") or "")
        if uid and uid == ref:
            exact_uid = node
            break
        if uid and len(ref) >= 8 and uid.startswith(ref) and prefix is None:
            prefix = node
    if exact_uid is not None:
        return exact_uid
    by_id = _node_by_id(doc, ref)
    if by_id is not None:
        return by_id
    return prefix


def _require(doc: dict, ref, what: str = "node") -> dict:
    node = find_node(doc, str(ref))
    if node is None:
        raise ValueError(f"{what} {ref!r} not found on the headless graph")
    return node


def _link_by_id(doc: dict, link_id):
    for link in doc["links"]:
        if link[0] == link_id:
            return link
    return None


def _remove_link(doc: dict, link_id) -> None:
    link = _link_by_id(doc, link_id)
    doc["links"] = [l for l in doc["links"] if l[0] != link_id]
    if link is None:
        return
    src = _node_by_id(doc, link[1])
    dst = _node_by_id(doc, link[3])
    if src is not None:
        for out in src.get("outputs") or []:
            out["links"] = [x for x in (out.get("links") or []) if x != link_id]
    if dst is not None:
        for inp in dst.get("inputs") or []:
            if inp.get("link") == link_id:
                inp["link"] = None


def _slot_index(slots: list, ref, side: str) -> int:
    if ref is None:
        raise ValueError(f"{side} slot is required")
    as_int = _as_int(ref)
    if as_int is not None and str(as_int) == str(ref).strip():
        if as_int < 0 or as_int >= len(slots):
            raise ValueError(
                f"{side} slot index {as_int} out of range (0..{max(len(slots) - 1, 0)})"
            )
        return as_int
    name = str(ref)
    for i, slot in enumerate(slots):
        if str(slot.get("name") or "") == name:
            return i
    names = ", ".join(str(s.get("name") or "") for s in slots) or "(none)"
    raise ValueError(f"no {side} slot {name!r}; slots: {names}")


def _compatible(out_type, in_type) -> bool:
    if in_type in (None, "", "*") or out_type in (None, "", "*"):
        return True
    out_set = {p.strip() for p in str(out_type).split(",") if p.strip()}
    in_set = {p.strip() for p in str(in_type).split(",") if p.strip()}
    return bool(out_set & in_set)


def _default_outputs(stage_class: str) -> list[dict]:
    name = stage_class.removeprefix("ComfyTV.")
    kind = (STAGE_META.get(name) or {}).get("kind")
    pairs = _KIND_OUTPUTS.get(kind) or [("output", "*")]
    return [{"name": n, "type": t, "links": []} for n, t in pairs]


def _stage_node(stage_class: str, node_id: int, payload: dict, project_id: str) -> dict:
    widgets = {
        "force_run_token": 1,
        "project_id": project_id,
        "parent_output_id": 0,
        "custom_params": "{}",
    }
    if payload.get("prompt") is not None:
        widgets["main_prompt"] = str(payload["prompt"])
    if payload.get("workflow") is not None:
        widgets["workflow"] = str(payload["workflow"])
    extra = payload.get("widgets")
    if isinstance(extra, dict):
        widgets.update(extra)
    pos = payload.get("pos")
    if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
        pos = [40, 40 + 160 * (node_id - 1)]
    title = payload.get("title")
    return {
        "id": node_id,
        "type": stage_class,
        "pos": [float(pos[0]), float(pos[1])],
        "size": [400, 220],
        "title": str(title) if title is not None else stage_class,
        "mode": 0,
        "flags": {},
        "properties": {
            "comfytv_uid": uuid.uuid4().hex,
            "comfytv_server": "",
            "asset_refs": list(payload.get("asset_refs") or []),
        },
        "widgets": widgets,
        "inputs": [],
        "outputs": _default_outputs(stage_class),
    }


def add_stage(payload: dict) -> dict:
    project_id = project_id_of(payload)
    doc = load_document(project_id)
    doc["last_node_id"] = int(doc["last_node_id"]) + 1
    node = _stage_node(str(payload["node_class"]), doc["last_node_id"], payload, project_id)
    doc["nodes"].append(node)
    save_document(project_id, doc)
    return {
        "graph_node_id": str(node["id"]),
        "uid": node["properties"]["comfytv_uid"],
        "headless": True,
    }


def _apply_stage_fields(node: dict, payload: dict) -> list[str]:
    updated: list[str] = []
    widgets = node.setdefault("widgets", {})
    props = node.setdefault("properties", {})
    if payload.get("workflow") is not None:
        _validate_stage_workflow(node, str(payload["workflow"]))
        widgets["workflow"] = str(payload["workflow"])
        updated.append("workflow")
    if payload.get("prompt") is not None:
        widgets["main_prompt"] = str(payload["prompt"])
        updated.append("prompt")
    if payload.get("title") is not None:
        node["title"] = str(payload["title"])
        updated.append("title")
    if isinstance(payload.get("widgets"), dict):
        for name, value in payload["widgets"].items():
            widgets[str(name)] = value
            updated.append(f"widgets.{name}")
    if payload.get("server") is not None:
        raw = str(payload["server"])
        props["comfytv_server"] = "" if raw.lower() in ("", "local") else raw
        updated.append("server")
    if payload.get("asset_refs") is not None:
        props["asset_refs"] = list(payload["asset_refs"])
        updated.append("asset_refs")
    if payload.get("media_order") is not None:
        props["media_order"] = payload["media_order"]
        updated.append("media_order")
    return updated


def _validate_stage_workflow(node: dict, label: str) -> None:
    # Deferred: api.mcp_tools.stages imports the command bus, which imports
    # this module while the package is still loading.
    from .mcp_tools.stages import _validate_workflow_label
    _validate_workflow_label(str(node.get("type") or ""), label)


def set_stage(payload: dict) -> dict:
    project_id = project_id_of(payload)
    doc = load_document(project_id)
    node = _require(doc, payload.get("node"), "stage")
    updated = _apply_stage_fields(node, payload)
    save_document(project_id, doc)
    return {
        "graph_node_id": str(node["id"]),
        "uid": (node.get("properties") or {}).get("comfytv_uid"),
        "updated": updated,
        "headless": True,
    }


def _connect(doc: dict, src: dict, dst: dict, from_slot, to_slot) -> dict:
    outputs = src.setdefault("outputs", [])
    if not outputs:
        raise ValueError(f"node {src.get('id')} has no outputs")
    from_idx = 0 if from_slot is None else _slot_index(outputs, from_slot, "output")
    out = outputs[from_idx]
    inputs = dst.setdefault("inputs", [])
    if to_slot is None:
        to_idx = -1
        for i, inp in enumerate(inputs):
            if inp.get("link") is not None:
                continue
            if _compatible(out.get("type"), inp.get("type")):
                to_idx = i
                break
        if to_idx < 0:
            raise ValueError(
                f"no free input on node {dst.get('id')} compatible with "
                f"type {out.get('type')}"
            )
    else:
        try:
            to_idx = _slot_index(inputs, to_slot, "input")
        except ValueError:
            if _as_int(to_slot) is not None and str(_as_int(to_slot)) == str(to_slot).strip():
                raise
            inputs.append({
                "name": str(to_slot),
                "type": out.get("type") or "*",
                "link": None,
            })
            to_idx = len(inputs) - 1
        inp = inputs[to_idx]
        if not _compatible(out.get("type"), inp.get("type")):
            raise ValueError(
                f"input {inp.get('name')!r} type {inp.get('type')} is not "
                f"compatible with output type {out.get('type')}"
            )
    existing = inputs[to_idx].get("link")
    if existing is not None:
        _remove_link(doc, existing)
    doc["last_link_id"] = int(doc["last_link_id"]) + 1
    link_id = doc["last_link_id"]
    doc["links"].append([
        link_id, src["id"], from_idx, dst["id"], to_idx, out.get("type") or "*",
    ])
    inputs[to_idx]["link"] = link_id
    out.setdefault("links", []).append(link_id)
    input_name = str(inputs[to_idx].get("name") or to_idx)
    result = {
        "from": str(src["id"]),
        "to": str(dst["id"]),
        "input": input_name,
        "headless": True,
    }
    kind = next((k for k in ("image", "video", "audio") if k in input_name), None)
    if kind:
        linked = [
            inp for inp in inputs
            if inp.get("link") is not None and kind in str(inp.get("name") or "")
        ]
        position = linked.index(inputs[to_idx]) + 1
        result["position"] = position
        result["mention"] = f"@{kind}_{position}"
    return result


def connect_stages(payload: dict) -> dict:
    project_id = project_id_of(payload)
    doc = load_document(project_id)
    src = _require(doc, payload.get("from_node"), "from_node")
    dst = _require(doc, payload.get("to_node"), "to_node")
    result = _connect(doc, src, dst, payload.get("from_slot"), payload.get("to_slot"))
    save_document(project_id, doc)
    return result


def remove_stage(payload: dict) -> dict:
    project_id = project_id_of(payload)
    doc = load_document(project_id)
    node = _require(doc, payload.get("node"), "stage")
    removed_id = node["id"]
    for link in list(doc["links"]):
        if link[1] == removed_id or link[3] == removed_id:
            _remove_link(doc, link[0])
    doc["nodes"] = [n for n in doc["nodes"] if n.get("id") != removed_id]
    save_document(project_id, doc)
    return {
        "removed": True,
        "graph_node_id": str(removed_id),
        "uid": (node.get("properties") or {}).get("comfytv_uid"),
        "headless": True,
    }


def _slim(doc: dict, node: dict) -> dict:
    links = {l[0]: l for l in doc["links"]}
    inputs = []
    for inp in node.get("inputs") or []:
        link = links.get(inp.get("link"))
        item = {
            "name": str(inp.get("name") or ""),
            "type": str(inp.get("type") or ""),
            "connected": inp.get("link") is not None,
        }
        if link is not None:
            item["from_node"] = str(link[1])
            item["from_slot"] = int(link[2])
        inputs.append(item)
    outputs = []
    for out in node.get("outputs") or []:
        targets = []
        for lid in out.get("links") or []:
            link = links.get(lid)
            if link is not None:
                targets.append(str(link[3]))
        outputs.append({
            "name": str(out.get("name") or ""),
            "type": str(out.get("type") or ""),
            "to_nodes": targets,
        })
    props = node.get("properties") or {}
    type_name = str(node.get("type") or "")
    return {
        "node_id": str(node.get("id")),
        "graph_node_id": str(node.get("id")),
        "type": type_name,
        "node_class": type_name,
        "title": str(node.get("title") or ""),
        "mode": int(node.get("mode") or 0),
        "widgets": dict(node.get("widgets") or {}),
        "inputs": inputs,
        "outputs": outputs,
        "pos": node.get("pos"),
        "uid": props.get("comfytv_uid"),
        "asset_refs": list(props.get("asset_refs") or []),
        **({"is_stage": True} if type_name.startswith("ComfyTV.") else {}),
    }


def get_stage(payload: dict) -> dict:
    project_id = project_id_of(payload)
    doc = load_document(project_id)
    node = _require(doc, payload.get("node"), "stage")
    result = _slim(doc, node)
    result["project_id"] = project_id
    result["headless"] = True
    result["running"] = False
    return result


def graph_get(payload: dict) -> dict:
    project_id = project_id_of(payload)
    doc = load_document(project_id)
    nodes = [_slim(doc, n) for n in doc["nodes"]]
    return {
        "node_count": len(nodes),
        "nodes": nodes,
        "project_id": project_id,
        "headless": True,
    }


def _apply_op(doc: dict, op: dict) -> dict:
    name = str(op.get("op") or "")
    if name == "add_node":
        type_name = str(op.get("type") or "")
        if not type_name:
            raise ValueError("add_node needs a type (a node class name)")
        doc["last_node_id"] = int(doc["last_node_id"]) + 1
        node_id = doc["last_node_id"]
        pos = op.get("pos") if isinstance(op.get("pos"), (list, tuple)) else [40, 40]
        node = {
            "id": node_id,
            "type": type_name,
            "pos": [float(pos[0]), float(pos[1])],
            "size": [200, 100],
            "title": str(op["title"]) if op.get("title") is not None else type_name,
            "mode": 0,
            "flags": {},
            "properties": {},
            "widgets": dict(op.get("widgets") or {}) if isinstance(op.get("widgets"), dict) else {},
            "inputs": [],
            "outputs": [{"name": "output", "type": "*", "links": []}],
        }
        if type_name.startswith("ComfyTV."):
            node["properties"]["comfytv_uid"] = uuid.uuid4().hex
            node["outputs"] = _default_outputs(type_name)
        doc["nodes"].append(node)
        return {"op": name, "node_id": str(node_id), "type": type_name}
    if name == "remove_node":
        node = _require(doc, op.get("node"))
        removed = node["id"]
        for link in list(doc["links"]):
            if link[1] == removed or link[3] == removed:
                _remove_link(doc, link[0])
        doc["nodes"] = [n for n in doc["nodes"] if n.get("id") != removed]
        return {"op": name, "node_id": str(removed)}
    if name == "set_widget":
        node = _require(doc, op.get("node"))
        key = str(op.get("name") or "")
        if not key:
            raise ValueError("set_widget needs a name")
        node.setdefault("widgets", {})[key] = op.get("value")
        return {"op": name, "node_id": str(node["id"]), "name": key}
    if name == "set_title":
        node = _require(doc, op.get("node"))
        node["title"] = str(op.get("title") or "")
        return {"op": name, "node_id": str(node["id"])}
    if name == "move":
        items = op.get("nodes") if isinstance(op.get("nodes"), list) else [op]
        moved = []
        for item in items:
            node = _require(doc, item.get("node"))
            pos = item.get("pos")
            if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
                raise ValueError(f"node {node['id']} needs pos [x, y]")
            node["pos"] = [float(pos[0]), float(pos[1])]
            moved.append(str(node["id"]))
        return {"op": name, "node_ids": moved}
    if name == "connect":
        src = _require(doc, op.get("from_node"))
        dst = _require(doc, op.get("to_node"))
        result = _connect(doc, src, dst, op.get("from_slot"), op.get("to_slot"))
        result["op"] = name
        return result
    if name == "disconnect":
        node = _require(doc, op.get("node"))
        idx = _slot_index(node.get("inputs") or [], op.get("input"), "input")
        link_id = (node["inputs"][idx] or {}).get("link")
        if link_id is not None:
            _remove_link(doc, link_id)
        return {"op": name, "node_id": str(node["id"]), "input": str(op.get("input"))}
    if name == "set_mode":
        node = _require(doc, op.get("node"))
        mode = str(op.get("mode") or "")
        if mode not in _NODE_MODES:
            raise ValueError(f"invalid mode {mode!r} — valid: always, mute, bypass")
        node["mode"] = _NODE_MODES[mode]
        return {"op": name, "node_id": str(node["id"]), "mode": mode}
    if name == "clone":
        node = _require(doc, op.get("node"))
        doc["last_node_id"] = int(doc["last_node_id"]) + 1
        copy_node = copy.deepcopy(node)
        copy_node["id"] = doc["last_node_id"]
        pos = op.get("pos")
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            copy_node["pos"] = [float(pos[0]), float(pos[1])]
        else:
            origin = node.get("pos") or [0, 0]
            copy_node["pos"] = [float(origin[0]) + 40, float(origin[1]) + 40]
        for inp in copy_node.get("inputs") or []:
            inp["link"] = None
        for out in copy_node.get("outputs") or []:
            out["links"] = []
        props = copy_node.setdefault("properties", {})
        if props.get("comfytv_uid"):
            props["comfytv_uid"] = uuid.uuid4().hex
        doc["nodes"].append(copy_node)
        return {
            "op": name,
            "node_id": str(copy_node["id"]),
            "cloned_from": str(node["id"]),
        }
    if name == "set_color":
        node = _require(doc, op.get("node"))
        if "color" in op:
            node["color"] = str(op["color"]) if op["color"] else None
        if "bgcolor" in op:
            node["bgcolor"] = str(op["bgcolor"]) if op["bgcolor"] else None
        return {"op": name, "node_id": str(node["id"])}
    if name == "set_review":
        node = _require(doc, op.get("node"))
        state = str(op.get("state") or "")
        if state not in ("approved", "review", "archived", ""):
            raise ValueError("set_review state must be approved, review, archived or ''")
        props = node.setdefault("properties", {})
        if state:
            props["comfytv_review"] = state
            color, bg = _REVIEW_COLORS[state]
            node["color"] = color
            node["bgcolor"] = bg
        else:
            props.pop("comfytv_review", None)
            node["color"] = None
            node["bgcolor"] = None
        return {"op": name, "node_id": str(node["id"]), "state": state}
    if name == "create_group":
        ids = op.get("nodes")
        if not isinstance(ids, list) or not ids:
            raise ValueError("create_group needs nodes (a non-empty array of node ids)")
        members = [_require(doc, i) for i in ids]
        xs = [float((n.get("pos") or [0, 0])[0]) for n in members]
        ys = [float((n.get("pos") or [0, 0])[1]) for n in members]
        group = {
            "title": str(op.get("title") or "Group"),
            "color": op.get("color"),
            "nodes": [str(n["id"]) for n in members],
            "pos": [min(xs) - 24, min(ys) - 54],
        }
        doc["groups"].append(group)
        return {"op": name, "title": group["title"], "node_count": len(members)}
    if name == "collapse":
        node = _require(doc, op.get("node"))
        flags = node.setdefault("flags", {})
        flags["collapsed"] = op.get("collapsed") is not False
        return {"op": name, "node_id": str(node["id"]), "collapsed": flags["collapsed"]}
    if name == "pin":
        node = _require(doc, op.get("node"))
        flags = node.setdefault("flags", {})
        flags["pinned"] = op.get("pinned") is not False
        return {"op": name, "node_id": str(node["id"]), "pinned": flags["pinned"]}
    if name in ("convert_to_subgraph", "unpack_subgraph"):
        raise ValueError(
            f"{name} needs the live LiteGraph canvas — open a ComfyTV page "
            "for subgraph packing, or keep the headless graph flat"
        )
    raise ValueError(f"unknown op {name!r}")


def graph_edit(payload: dict) -> dict:
    project_id = project_id_of(payload)
    doc = load_document(project_id)
    ops = payload.get("ops") or []
    applied = []
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            raise ValueError(f"ops[{i}] must be an object")
        try:
            applied.append(_apply_op(doc, op))
            save_document(project_id, doc)
        except ValueError as e:
            prefix = f"ops[{i}] ({op.get('op') or '?'}): "
            message = str(e)
            if applied:
                message += f" — {len(applied)} earlier op(s) already applied"
            if message.startswith("ops["):
                raise
            raise ValueError(prefix + message) from e
    return {"applied": applied, "headless": True, "project_id": project_id}


def _levels(doc: dict) -> dict[str, int]:
    deps: dict[str, list[str]] = {str(n["id"]): [] for n in doc["nodes"]}
    for link in doc["links"]:
        src, dst = str(link[1]), str(link[3])
        if dst in deps and src in deps:
            deps[dst].append(src)
    level: dict[str, int] = {}

    def walk(nid: str, stack: tuple[str, ...]) -> int:
        if nid in level:
            return level[nid]
        if nid in stack:
            return 0
        parents = deps.get(nid) or []
        level[nid] = 0 if not parents else 1 + max(walk(p, stack + (nid,)) for p in parents)
        return level[nid]

    for nid in deps:
        walk(nid, ())
    return level


def arrange_canvas(payload: dict) -> dict:
    project_id = project_id_of(payload)
    doc = load_document(project_id)
    nodes = doc["nodes"]
    if not nodes:
        return {"arranged": 0, "headless": True}
    try:
        margin = float(payload.get("margin") if payload.get("margin") is not None else 100)
    except (TypeError, ValueError):
        raise ValueError("margin must be a number")
    margin = max(20.0, min(400.0, margin))
    layout = payload.get("layout") or "horizontal"
    if layout == "grid":
        columns = payload.get("columns")
        cols = int(columns) if columns else math.ceil(math.sqrt(len(nodes)))
        cols = max(1, min(12, cols))
        pitch_x = max(float((n.get("size") or [200])[0]) for n in nodes) + margin
        row_h = max(float((n.get("size") or [200, 100])[1]) for n in nodes) + margin + 30
        for i, node in enumerate(nodes):
            node["pos"] = [margin + (i % cols) * pitch_x, margin + (i // cols) * row_h]
    else:
        levels = _levels(doc)
        buckets: dict[int, list] = {}
        for node in nodes:
            buckets.setdefault(levels.get(str(node["id"]), 0), []).append(node)
        pitch_x = max(float((n.get("size") or [200])[0]) for n in nodes) + margin
        pitch_y = max(float((n.get("size") or [200, 100])[1]) for n in nodes) + margin + 30
        vertical = layout == "vertical"
        for level, members in buckets.items():
            for j, node in enumerate(members):
                if vertical:
                    node["pos"] = [margin + j * pitch_x, margin + level * pitch_y]
                else:
                    node["pos"] = [margin + level * pitch_x, margin + j * pitch_y]
    save_document(project_id, doc)
    return {"arranged": len(nodes), "layout": layout, "headless": True}


def _inject_assets(inputs: dict, refs) -> None:
    counts = {"image": 0, "video": 0, "audio": 0}
    for ref in refs or []:
        if not isinstance(ref, dict) or "asset_id" not in ref:
            continue
        try:
            aid = int(ref["asset_id"])
        except (TypeError, ValueError):
            continue
        asset = storage.get_asset(aid)
        if not asset:
            continue
        kind = ref.get("type") or asset.get("media_type") or "image"
        if kind not in counts:
            continue
        counts[kind] += 1
        group = _ASSET_GROUP[kind]
        bucket = inputs.get(group)
        if not isinstance(bucket, dict):
            bucket = {}
        bucket[f"{_ASSET_ITEM[kind]}_{counts[kind]}"] = asset.get("payload_url") or ""
        inputs[group] = bucket


def _is_link(value, node_id: str, slot: int | None = None) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    if isinstance(value[1], bool) or not isinstance(value[1], int):
        return False
    if str(value[0]) != str(node_id):
        return False
    return slot is None or int(value[1]) == slot


def graph_to_prompt(doc: dict) -> dict:
    """API prompt for the headless document.

    Named widgets become inputs. Links become ``[node_id, slot]``. Asset
    references are injected as autogrow groups (``images.image_1`` and so on).
    A UI workflow that only has ``widgets_values`` is converted with ComfyUI's
    node schemas when those schemas are loaded.
    """
    doc = normalize_document(doc)
    needs_schema = any(
        n.get("widgets_values") and not (n.get("widgets") or {})
        for n in doc["nodes"]
    )
    if needs_schema:
        converted = _try_schema_convert(doc)
        if converted is not None:
            return converted
        raise ValueError(
            "could not convert this UI workflow to an API prompt "
            "(node schemas are not loaded). Pass graph_run prompt={...} "
            "in API format, or build the graph with add_stage / graph_edit."
        )
    link_by_id = {l[0]: l for l in doc["links"]}
    api: dict[str, dict] = {}
    for node in doc["nodes"]:
        if int(node.get("mode") or 0) == 2:
            continue
        inputs = dict(node.get("widgets") or {})
        _inject_assets(inputs, (node.get("properties") or {}).get("asset_refs"))
        for inp in node.get("inputs") or []:
            link = link_by_id.get(inp.get("link"))
            if link is None:
                continue
            if int(node.get("mode") or 0) == 2:
                continue
            src_id = str(link[1])
            src = _node_by_id(doc, src_id)
            if src is not None and int(src.get("mode") or 0) == 2:
                continue
            inputs[str(inp.get("name") or "input")] = [src_id, int(link[2])]
        api[str(node["id"])] = {
            "class_type": str(node.get("type") or ""),
            "inputs": inputs,
            "_meta": {"title": str(node.get("title") or "")},
        }
    return api


def _try_schema_convert(doc: dict) -> dict | None:
    try:
        from ..runners.vendor.workflow_to_api import convert_ui_to_api
        from ..runners.workflow_db.convert import build_object_info
        info = build_object_info()
    except Exception as e:
        _log.warning("[ComfyTV/headless] node schemas unavailable: %s", type(e).__name__)
        return None
    if not info:
        return None
    try:
        return convert_ui_to_api(doc, info)
    except Exception as e:
        _log.warning(
            "[ComfyTV/headless] UI workflow conversion failed: %s",
            type(e).__name__,
        )
        return None


def ancestor_ids(doc: dict, target_id: str) -> set[str]:
    incoming: dict[str, list[str]] = {}
    muted = {
        str(n["id"]) for n in doc["nodes"] if int(n.get("mode") or 0) == 2
    }
    for link in doc["links"]:
        src, dst = str(link[1]), str(link[3])
        if src in muted:
            continue
        incoming.setdefault(dst, []).append(src)
    keep: set[str] = set()
    stack = [str(target_id)]
    while stack:
        cur = stack.pop()
        if cur in keep or cur in muted:
            continue
        keep.add(cur)
        stack.extend(incoming.get(cur, []))
    return keep


def adopt_workflow(project_id: str, workflow: dict) -> dict:
    if not isinstance(workflow, dict):
        raise ValueError("workflow must be an object")
    if "nodes" not in workflow or "links" not in workflow:
        raise ValueError(
            "workflow must be a ComfyUI UI graph with nodes and links; "
            "an API prompt belongs in graph_run's prompt argument"
        )
    doc = normalize_document(workflow)
    save_document(project_id, doc)
    return doc


def _storyboard_document(node: dict) -> dict:
    raw = (node.get("widgets") or {}).get("storyboard_data")
    if isinstance(raw, dict):
        document = raw
    elif not raw:
        document = {"shots": []}
    else:
        try:
            document = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("storyboard_data is not valid JSON") from exc
    if not isinstance(document, dict) or not isinstance(document.get("shots"), list):
        raise ValueError("storyboard_data is not a shot document")
    return document


def _require_storyboard(doc: dict, ref) -> dict:
    node = _require(doc, ref, "stage")
    type_name = str(node.get("type") or "")
    if type_name not in ("ComfyTV.StoryboardStage", "StoryboardStage"):
        raise ValueError(
            f"stage {ref!r} is {type_name or 'unknown'}, not a StoryboardStage"
        )
    return node


def storyboard_get(payload: dict) -> dict:
    project_id = project_id_of(payload)
    doc = load_document(project_id)
    node = _require_storyboard(doc, payload.get("node"))
    document = _storyboard_document(node)
    shots = document.get("shots") or []
    return {
        "document": document,
        "shots": len(shots),
        "episode": document.get("episode"),
        "series": document.get("series"),
        "graph_node_id": str(node["id"]),
        "headless": True,
    }


def storyboard_set(payload: dict) -> dict:
    project_id = project_id_of(payload)
    doc = load_document(project_id)
    node = _require_storyboard(doc, payload.get("node"))
    try:
        document = load_shots_document(payload.get("document"))
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    node.setdefault("widgets", {})["storyboard_data"] = json.dumps(
        document, ensure_ascii=False,
    )
    save_document(project_id, doc)
    shots = document.get("shots") or []
    return {
        "ok": True,
        "shots": len(shots),
        "episode": document.get("episode"),
        "series": document.get("series"),
        "graph_node_id": str(node["id"]),
        "headless": True,
    }


def local_server_or_raise(node: dict) -> None:
    server = str((node.get("properties") or {}).get("comfytv_server") or "")
    if server and server.lower() not in ("", "local"):
        raise ValueError(
            "headless runs execute on this machine's ComfyUI prompt queue; "
            f"stage server {server!r} is a remote machine. Set server='local', "
            "or run that stage from an open ComfyTV page."
        )
