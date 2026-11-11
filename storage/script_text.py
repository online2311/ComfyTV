"""Beat ids, selection splicing, outline checks, and line diffs."""

from __future__ import annotations

import difflib
import hashlib
import json
import re

import yaml

from .script_llm import ScriptError

_BEAT_LINE = re.compile(
    r"^(?P<indent>\s*)-\s+(?:(?P<id>(?:E|X)\d+S\d+_b\d+)\s+)?\[(?P<kind>动作|台词|独白)"
)
_HEADER = re.compile(r"^##\s+((?:E|X)\d+S\d+)\b")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_QUOTE = re.compile(r"[“\"「『][^”\"」』\n]{1,200}[”\"」』]")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_available(text: str) -> str:
    if text is None or not str(text).strip():
        raise ScriptError("model returned an empty response", status=502)
    stripped = str(text).lstrip()
    if stripped == "UNAVAILABLE" or stripped.startswith("UNAVAILABLE"):
        raise ScriptError("model marked the request unavailable", status=409)
    return str(text)


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        spans.append((cursor, cursor + len(line), line))
        cursor += len(line)
    if text and not text.endswith("\n") and spans:
        return spans
    return spans


def _beat_id(line: str) -> str | None:
    match = _BEAT_LINE.match(line)
    if match is None:
        return None
    return match.group("id")


def _is_unnumbered(line: str) -> bool:
    match = _BEAT_LINE.match(line)
    return match is not None and not match.group("id")


def beat_ids(text: str) -> list[str]:
    found = []
    for _start, _end, line in _line_spans(text):
        beat_id = _beat_id(line)
        if beat_id:
            found.append(beat_id)
    return found


def _scene_of(line: str) -> str | None:
    match = _HEADER.match(line.strip("\n"))
    if match:
        return match.group(1)
    beat_id = _beat_id(line)
    if beat_id:
        return beat_id.rsplit("_b", 1)[0]
    return None


def _last_scene(text: str) -> str | None:
    scene = None
    for _start, _end, line in _line_spans(text):
        found = _scene_of(line)
        if found:
            scene = found
    return scene


def _span(text: str, selected: list[str]) -> tuple[str, str, str]:
    wanted = set(selected)
    hits: list[tuple[int, int, str]] = []
    for start, end, line in _line_spans(text):
        beat_id = _beat_id(line)
        if beat_id in wanted:
            hits.append((start, end, beat_id))
    found = {beat_id for _start, _end, beat_id in hits}
    missing = [beat_id for beat_id in selected if beat_id not in found]
    if missing:
        raise ScriptError(
            "beat id cannot be changed or removed: " + ", ".join(missing)
        )
    start = min(item[0] for item in hits)
    end = max(item[1] for item in hits)
    return text[:start], text[start:end], text[end:]


def _next_id(scene: str, existing: set[str]) -> str:
    width = 2
    numbers = []
    prefix = f"{scene}_b"
    for beat_id in existing:
        if not beat_id.startswith(prefix):
            continue
        suffix = beat_id[len(prefix):]
        if suffix.isdigit():
            numbers.append(int(suffix))
            width = max(width, len(suffix))
    number = (max(numbers) if numbers else 0) + 1
    return f"{scene}_b{number:0{width}d}"


def _insert_id(line: str, beat_id: str) -> str:
    return re.sub(r"^(\s*-\s+)", rf"\1{beat_id} ", line, count=1)


def _assign_middle(prefix: str, mid: str, suffix: str) -> str:
    if not mid.endswith("\n"):
        mid += "\n"
    scene = _last_scene(prefix) or "E01S01"
    existing = set(beat_ids(prefix) + beat_ids(mid) + beat_ids(suffix))
    pieces = []
    for line in mid.splitlines(keepends=True):
        if _is_unnumbered(line):
            scene = _last_scene(prefix + "".join(pieces)) or scene
            beat_id = _next_id(scene, existing)
            existing.add(beat_id)
            pieces.append(_insert_id(line, beat_id))
            continue
        found = _beat_id(line)
        if found:
            scene = found.rsplit("_b", 1)[0]
        pieces.append(line)
    return prefix + "".join(pieces) + suffix


def _require_old_ids(before: str, after: str) -> None:
    old = beat_ids(before)
    new = beat_ids(after)
    if len(new) != len(set(new)):
        raise ScriptError("duplicate beat id")
    missing = [beat_id for beat_id in old if beat_id not in set(new)]
    if missing:
        raise ScriptError(
            "beat id cannot be changed or removed: " + ", ".join(missing)
        )


def protect_beats(before: str, after: str) -> str:
    """Keep every existing beat id. Number a new ``[动作|台词|独白]`` line."""
    if not beat_ids(before) and not any(_is_unnumbered(line) for line in after.splitlines(keepends=True)):
        return after
    prefix = ""
    suffix = ""
    mid = after
    assigned = _assign_middle(prefix, mid, suffix) if any(
        _is_unnumbered(line) for line in after.splitlines(keepends=True)
    ) else after
    _require_old_ids(before, assigned)
    return assigned


def apply_selection(original: str, selection: dict, model_text: str) -> tuple[str, bool]:
    """Splice a rewrite. Returns the new text and whether the outside hash moved."""
    model_text = ensure_available(model_text)
    beat_list = selection.get("beat_ids") if isinstance(selection, dict) else None
    if beat_list:
        return _apply_beats(original, [str(item) for item in beat_list], model_text)
    if isinstance(selection, dict) and "char_start" in selection:
        return _apply_chars(
            original,
            int(selection["char_start"]),
            int(selection["char_end"]),
            model_text,
        )
    raise ScriptError("selection needs beat_ids or char_start/char_end")


def _apply_beats(original: str, selected: list[str], model_text: str) -> tuple[str, bool]:
    if not selected:
        raise ScriptError("selection beat_ids is empty")
    prefix, _mid, suffix = _span(original, selected)
    outside_ids = [beat_id for beat_id in beat_ids(original) if beat_id not in set(selected)]
    model_ids = set(beat_ids(model_text))
    if any(beat_id in model_ids for beat_id in outside_ids):
        new_prefix, new_mid, new_suffix = _span(model_text, selected)
        if new_prefix != prefix or new_suffix != suffix:
            raise ScriptError("selection changed text outside the selection")
        mid = new_mid
    else:
        mid = model_text
    updated = _assign_middle(prefix, mid, suffix)
    _require_old_ids(original, updated)
    if not updated.startswith(prefix) or not updated.endswith(suffix):
        raise ScriptError("selection changed text outside the selection")
    return updated, False


def _apply_chars(original: str, start: int, end: int, model_text: str) -> tuple[str, bool]:
    if start < 0 or end < start or end > len(original):
        raise ScriptError("selection character range is outside the document")
    prefix = original[:start]
    suffix = original[end:]
    outside_ids = set(beat_ids(prefix + suffix))
    if outside_ids and any(beat_id in set(beat_ids(model_text)) for beat_id in outside_ids):
        if not (model_text.startswith(prefix) and (not suffix or model_text.endswith(suffix))):
            raise ScriptError("selection changed text outside the selection")
        updated = model_text
    else:
        updated = prefix + model_text + suffix
    if not (updated.startswith(prefix) and updated.endswith(suffix)):
        raise ScriptError("selection changed text outside the selection")
    protected = protect_beats(original, updated)
    if not (protected.startswith(prefix) and protected.endswith(suffix)):
        raise ScriptError("selection changed text outside the selection")
    return protected, False


def _axes(row: dict) -> tuple[str, str, str]:
    def norm(value) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    return (
        norm(row.get("genre") or row.get("logline")),
        norm(row.get("conflict")),
        norm(row.get("ending") or row.get("selling_points")),
    )


def parse_directions(text: str, n: int) -> list[dict]:
    raw = ensure_available(text).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("[")
    end = raw.rfind("]")
    if start < 0 or end < start:
        raise ScriptError("directions were not a JSON array")
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ScriptError("directions were not a JSON array") from exc
    if not isinstance(data, list):
        raise ScriptError("directions were not a JSON array")
    if len(data) != n:
        raise ScriptError(f"directions returned {len(data)}, expected {n}")
    rows = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ScriptError("each direction must be an object")
        for key in ("logline", "conflict", "selling_points", "risks"):
            if not str(item.get(key) or "").strip():
                raise ScriptError(f"direction {index} is missing {key}")
        characters = item.get("characters")
        if not isinstance(characters, list) or not characters:
            raise ScriptError(f"direction {index} is missing characters")
        row = dict(item)
        row["id"] = str(item.get("id") or f"d{index}")
        row["characters"] = [str(name) for name in characters]
        rows.append(row)
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            diffs = sum(
                one != two
                for one, two in zip(_axes(rows[left]), _axes(rows[right]))
            )
            if diffs < 2:
                raise ScriptError("directions are not distinct enough")
    return rows


def _heading_map(text: str) -> list[tuple[int, str, int, int]]:
    """Return (level, title, line_start, line_end_exclusive) for markdown headings."""
    lines = text.splitlines()
    found = []
    for index, line in enumerate(lines):
        match = _HEADING.match(line)
        if match:
            found.append((len(match.group(1)), match.group(2).strip(), index))
    spans = []
    for position, (level, title, index) in enumerate(found):
        end = len(lines)
        for later_level, _title, later_index in found[position + 1:]:
            if later_level <= level:
                end = later_index
                break
        spans.append((level, title, index, end))
    return spans


def validate_outline(text: str, sections: list[str]) -> None:
    spans = _heading_map(text)
    titles = {title for _level, title, _start, _end in spans}
    for name in sections:
        if name not in titles:
            raise ScriptError(f"outline is missing heading {name}")
    lines = text.splitlines()
    for _level, title, start, end in spans:
        if "梗概" not in title:
            continue
        body = "\n".join(lines[start:end])
        if _QUOTE.search(body):
            raise ScriptError("梗概里出现引号对白")


def split_marked(text: str, marks: list[str]) -> dict[str, str]:
    body = ensure_available(text)
    pattern = re.compile(r"^<<<([A-Za-z0-9_]+)>>>\s*$", re.MULTILINE)
    matches = list(pattern.finditer(body))
    if not matches:
        raise ScriptError("model output is missing section markers")
    found: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        found[name] = body[start:end].strip("\n")
    missing = [mark for mark in marks if not found.get(mark, "").strip()]
    if missing:
        raise ScriptError("model output is missing " + ", ".join(missing))
    return {mark: found[mark].strip() + "\n" for mark in marks}


def parse_hooks(text: str) -> str:
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ScriptError("hooks.yaml could not be parsed") from exc
    if loaded is None:
        raise ScriptError("hooks.yaml is empty")
    return text if text.endswith("\n") else text + "\n"


def diff_texts(a_text: str, b_text: str) -> dict:
    a_lines = _numbered_line_map(a_text)
    b_lines = _numbered_line_map(b_text)
    hunks = []
    for beat_id, line in a_lines:
        other = dict(b_lines).get(beat_id)
        if other is None:
            hunks.append({"op": "delete", "beat_id": beat_id, "a": line, "b": ""})
        elif other == line:
            hunks.append({"op": "equal", "beat_id": beat_id, "a": line, "b": other})
        else:
            hunks.append({"op": "replace", "beat_id": beat_id, "a": line, "b": other})
    known = {beat_id for beat_id, _line in a_lines}
    for beat_id, line in b_lines:
        if beat_id not in known:
            hunks.append({"op": "insert", "beat_id": beat_id, "a": "", "b": line})
    unified = "".join(difflib.unified_diff(
        a_text.splitlines(keepends=True),
        b_text.splitlines(keepends=True),
        fromfile="a",
        tofile="b",
    ))
    return {"hunks": hunks, "unified": unified}


def _numbered_line_map(text: str) -> list[tuple[str, str]]:
    rows = []
    for _start, _end, line in _line_spans(text):
        beat_id = _beat_id(line)
        if beat_id:
            rows.append((beat_id, line.strip("\n")))
    return rows


def feedback_block(doc: str, old: str, new: str, at: str) -> str:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    rows = [f"## {at} {doc}", ""]
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        left = " / ".join(old_lines[i1:i2]) or "（空）"
        right = " / ".join(new_lines[j1:j2]) or "（空）"
        rows.append(f"- {left} → {right}")
    if len(rows) == 2:
        rows.append("- （无文本差异）")
    return "\n".join(rows) + "\n"


def shot_ids_in(raw: bytes) -> list[str]:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    shots = data.get("shots") if isinstance(data, dict) else None
    if not isinstance(shots, list):
        return []
    found = []
    for shot in shots:
        if isinstance(shot, dict) and shot.get("shot_id"):
            found.append(str(shot["shot_id"]))
    return found
