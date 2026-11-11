"""Align a script's dialogue onto a Whisper timeline.

The characters written into the SRT are the script's. The cue times come from
Whisper. Matching is a character-level edit distance (Needleman-Wunsch, the
monotonic form of DTW that also allows dropped and extra characters).
Punctuation and full-width forms are ignored while matching, so a homophone
such as 练/炼 is a one-character substitution and the cue still says 练.

A script sentence with no aligned characters is paced from the neighboring
cues and marked ``interpolated`` in the report. Sentences under the confidence
threshold are listed separately; their times still come from Whisper.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

from .text_overlay import wrap_text
from .whisper_asr import segments_to_srt

# Same pace the shot document uses when it checks that a line fits the shot.
CHAR_PER_SECOND = 4.0
DEFAULT_MAX_LINE_CELLS = 32
DEFAULT_LOW_CONFIDENCE = 0.6

# Exact matches must outrank substitutions, and a substitution must outrank
# deleting the script character and inserting the Whisper one. Otherwise a
# homophone loses its timestamp, and a sentence Whisper never spoke steals
# the next sentence's audio. When those scores tie, the smaller index drift
# wins: a trailing extra stays trailing, and a repeated character stays with
# the earlier sentence.
_MATCH = 2
_MISMATCH = -1
_GAP = -1

_TIME_RE = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def subtitle_cells(text: str) -> float:
    """Cell width used by the subtitle line breaker. A CJK cell counts as two."""
    return float(sum(2 if ord(ch) > 0x2E7F else 1 for ch in text))


@dataclass
class _Line:
    shot_id: str
    line_id: str
    speaker: str
    text: str


@dataclass
class _Tok:
    kind: str
    original: str
    key: str
    start: float
    end: float


@dataclass
class _Piece:
    line_index: int
    key: str
    original: str
    at: int


def align_subtitles(
    whisper,
    script,
    *,
    max_line_cells: int = DEFAULT_MAX_LINE_CELLS,
    low_confidence: float = DEFAULT_LOW_CONFIDENCE,
    min_cue_s: float = 0.3,
) -> dict:
    """Return ``{"srt", "text", "report"}``.

    ``whisper`` is a segment list (word times when present, otherwise the
    segment span), an SRT string, or a per-shot map. ``script`` is a shots
    document, a list of lines, or plain text with one sentence per line.
    """
    lines = _parse_script(script)
    cells = _coerce_cells(max_line_cells)
    threshold = _coerce_threshold(low_confidence)
    if not lines:
        return _pack([], cells, threshold)
    mode, payload = _parse_whisper(whisper)
    if mode == "flat":
        rows = _align_lines(lines, _tokens_from_segments(payload))
        _fill_gaps(rows, min_cue_s)
    else:
        segmap = {shot_id: segments for shot_id, segments in payload}
        rows = []
        timeline_end = 0.0
        for shot_id, group in _line_groups(lines):
            part = _align_lines(group, _tokens_from_segments(segmap.get(shot_id, [])))
            _fill_gaps(part, min_cue_s)
            part, timeline_end = _place_group(part, timeline_end)
            rows.extend(part)
    for row in rows:
        row["cue_text"] = wrap_text(row["text"], subtitle_cells, cells) if row["text"] else ""
        if row["end"] < row["start"]:
            row["end"] = row["start"]
    return _pack(rows, cells, threshold)


def _pack(rows: list[dict], cells: int, threshold: float) -> dict:
    low = []
    interpolated = []
    for row in rows:
        if row["status"] == "interpolated":
            interpolated.append(row)
        elif row["confidence"] < threshold:
            row["note"] = "对齐置信度低于阈值"
            low.append(row)
    report = {
        "aligner": "character_edit_distance",
        "max_line_cells": cells,
        "low_confidence_threshold": threshold,
        "summary": _summary(low, interpolated),
        "sentences": rows,
        "low_confidence": low,
        "interpolated": interpolated,
    }
    srt = segments_to_srt([
        {"start": row["start"], "end": row["end"], "text": row["cue_text"]}
        for row in rows
    ])
    text = "\n".join(row["text"] for row in rows)
    return {"srt": srt, "text": text, "report": report}


def _summary(low: list[dict], interpolated: list[dict]) -> str:
    if low:
        low_part = f"低置信度 {len(low)} 句：" + "、".join(_label(row) for row in low)
    else:
        low_part = "低置信度 0 句"
    if interpolated:
        missed = f"未识别并插值 {len(interpolated)} 句：" + "、".join(
            _label(row) for row in interpolated
        )
    else:
        missed = "未识别并插值 0 句"
    return f"{low_part}。{missed}。"


def _label(row: dict) -> str:
    if row["shot_id"] and row["line_id"]:
        return f"{row['shot_id']}/{row['line_id']}"
    return row["line_id"] or row["text"]


def _align_lines(lines: list[_Line], tokens: list[_Tok]) -> list[dict]:
    pieces: list[_Piece] = []
    content_chars = [0] * len(lines)
    for line_index, line in enumerate(lines):
        for at, ch in enumerate(line.text):
            for key in _keys(ch):
                pieces.append(_Piece(line_index, key, ch, at))
                content_chars[line_index] += 1
    content = [(index, tok) for index, tok in enumerate(tokens) if tok.kind == "content"]
    ops = _edit_ops([piece.key for piece in pieces], [tok.key for _, tok in content])

    buckets = [
        {"exact": 0, "subs": [], "missing": [], "extra": [], "wis": [], "heard_at": []}
        for _ in lines
    ]
    inserts: list[int] = []
    for kind, script_index, whisper_index in ops:
        if script_index is not None:
            piece = pieces[script_index]
            bucket = buckets[piece.line_index]
            if kind == "eq":
                bucket["exact"] += 1
                bucket["wis"].append(whisper_index)
                bucket["heard_at"].append(whisper_index)
            elif kind == "sub":
                bucket["subs"].append({
                    "script": piece.original,
                    "whisper": content[whisper_index][1].original,
                    "at": piece.at,
                })
                bucket["wis"].append(whisper_index)
                bucket["heard_at"].append(whisper_index)
            else:
                bucket["missing"].append({"script": piece.original, "at": piece.at})
        elif whisper_index is not None:
            inserts.append(whisper_index)
    for whisper_index in inserts:
        for bucket in buckets:
            if bucket["wis"] and min(bucket["wis"]) < whisper_index < max(bucket["wis"]):
                bucket["extra"].append({
                    "whisper": content[whisper_index][1].original,
                })
                bucket["heard_at"].append(whisper_index)
                break

    rows = []
    for line, count, bucket in zip(lines, content_chars, buckets):
        heard = "".join(
            content[index][1].original for index in sorted(bucket["heard_at"])
        )
        if bucket["wis"]:
            first = content[min(bucket["wis"])][1]
            last_token, last = content[max(bucket["wis"])]
            end = last.end
            cursor = last_token + 1
            while cursor < len(tokens) and tokens[cursor].kind == "punct":
                end = max(end, tokens[cursor].end)
                cursor += 1
            start, end = _r3(first.start), _r3(end)
            status = "aligned"
            mark = ""
            note = ""
            confidence = round(bucket["exact"] / count, 4) if count else 1.0
        else:
            start = end = None
            status = "interpolated"
            mark = "interpolated"
            note = "剧本有此句，Whisper 未识别，时间由相邻句插值"
            confidence = 0.0
        rows.append({
            "shot_id": line.shot_id,
            "line_id": line.line_id,
            "speaker": line.speaker,
            "text": line.text,
            "heard": heard,
            "start": start,
            "end": end,
            "status": status,
            "mark": mark,
            "note": note,
            "confidence": confidence,
            "content_chars": count,
            "exact": bucket["exact"],
            "substitutions": bucket["subs"],
            "missing": bucket["missing"],
            "extra": bucket["extra"],
        })
    return rows


def _edit_ops(script: list[str], whisper: list[str]) -> list[tuple[str, int | None, int | None]]:
    n, m = len(script), len(whisper)
    # (score, exact matches, negative index drift). Drift only breaks ties.
    state = [[(0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    move = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        state[i][0] = (_GAP * i, 0, 0)
        move[i][0] = 1
    for j in range(1, m + 1):
        state[0][j] = (_GAP * j, 0, 0)
        move[0][j] = 2
    for i in range(1, n + 1):
        key = script[i - 1]
        for j in range(1, m + 1):
            equal = key == whisper[j - 1]
            prev_score, prev_exact, prev_drift = state[i - 1][j - 1]
            diag = (
                prev_score + (_MATCH if equal else _MISMATCH),
                prev_exact + (1 if equal else 0),
                prev_drift - abs((i - 1) - (j - 1)),
            )
            up_score, up_exact, up_drift = state[i - 1][j]
            up = (up_score + _GAP, up_exact, up_drift)
            left_score, left_exact, left_drift = state[i][j - 1]
            left = (left_score + _GAP, left_exact, left_drift)
            best = diag
            how = 0
            if up > best:
                best = up
                how = 1
            if left > best:
                best = left
                how = 2
            state[i][j] = best
            move[i][j] = how

    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i or j:
        how = move[i][j]
        if how == 0 and i and j:
            kind = "eq" if script[i - 1] == whisper[j - 1] else "sub"
            ops.append((kind, i - 1, j - 1))
            i -= 1
            j -= 1
        elif how == 1 and i:
            ops.append(("del", i - 1, None))
            i -= 1
        elif j:
            ops.append(("ins", None, j - 1))
            j -= 1
        else:
            break
    ops.reverse()
    return ops


def _fill_gaps(rows: list[dict], min_cue_s: float) -> None:
    index = 0
    count = len(rows)
    while index < count:
        if rows[index]["start"] is not None:
            index += 1
            continue
        end = index
        while end < count and rows[end]["start"] is None:
            end += 1
        prev_end = rows[index - 1]["end"] if index else None
        next_start = rows[end]["start"] if end < count else None
        _place_gap(rows[index:end], prev_end, next_start, min_cue_s)
        index = end


def _place_gap(rows: list[dict], prev_end, next_start, min_cue_s: float) -> None:
    if prev_end is not None and next_start is not None:
        weights = [row["content_chars"] if row["content_chars"] else 1 for row in rows]
        total = float(sum(weights))
        span = max(0.0, next_start - prev_end)
        cursor = float(prev_end)
        last = len(rows) - 1
        for index, (row, weight) in enumerate(zip(rows, weights)):
            row["start"] = _r3(cursor)
            if index == last:
                row["end"] = _r3(prev_end + span)
            else:
                cursor += span * weight / total
                row["end"] = _r3(cursor)
        return
    if prev_end is None and next_start is not None:
        durations = [_synthetic(row, min_cue_s) for row in rows]
        budget = sum(durations)
        if next_start <= 0 or budget <= 0:
            _lay(rows, 0.0, [0.0 for _ in rows])
        elif budget > next_start:
            scale = next_start / budget
            _lay(rows, 0.0, [item * scale for item in durations])
        else:
            _lay(rows, next_start - budget, durations)
        return
    origin = 0.0 if prev_end is None else float(prev_end)
    _lay(rows, origin, [_synthetic(row, min_cue_s) for row in rows])


def _synthetic(row: dict, min_cue_s: float) -> float:
    weight = row["content_chars"] if row["content_chars"] else 1
    return max(min_cue_s, weight / CHAR_PER_SECOND)


def _lay(rows: list[dict], cursor: float, durations: list[float]) -> None:
    for row, duration in zip(rows, durations):
        row["start"] = _r3(cursor)
        cursor += duration
        row["end"] = _r3(cursor)


def _place_group(rows: list[dict], timeline_end: float) -> tuple[list[dict], float]:
    """Shift a shot-local clock so it starts after the previous shot.

    A shot whose first cue already begins at or after the previous shot's end
    is left on that absolute timeline.
    """
    if not rows:
        return rows, timeline_end
    local_start = min(row["start"] for row in rows)
    local_end = max(row["end"] for row in rows)
    if timeline_end > 0 and local_start < timeline_end - 1e-6:
        offset = timeline_end
        for row in rows:
            row["start"] = _r3(row["start"] + offset)
            row["end"] = _r3(row["end"] + offset)
        local_end = max(row["end"] for row in rows)
    return rows, max(timeline_end, local_end)


def _line_groups(lines: list[_Line]) -> list[tuple[str, list[_Line]]]:
    groups: list[tuple[str, list[_Line]]] = []
    current = None
    bucket: list[_Line] = []
    for line in lines:
        if current is None:
            current = line.shot_id
        if line.shot_id != current and bucket:
            groups.append((current, bucket))
            bucket = []
            current = line.shot_id
        bucket.append(line)
    if bucket:
        groups.append((current if current is not None else "", bucket))
    return groups


def _parse_script(script) -> list[_Line]:
    if isinstance(script, str):
        raw = script.strip().lstrip("\ufeff")
        if not raw:
            return []
        if raw[0] in "{[":
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("script JSON is invalid") from exc
        else:
            return _lines_from_items(
                [line.strip() for line in raw.splitlines() if line.strip()],
                "",
            )
    else:
        data = script
    if data is None:
        return []
    if isinstance(data, list):
        return _lines_from_items(data, "")
    if isinstance(data, dict):
        shots = data.get("shots")
        if isinstance(shots, list):
            lines: list[_Line] = []
            for shot in shots:
                if not isinstance(shot, dict):
                    raise ValueError("script shots must be objects")
                dialogue = shot.get("dialogue") or []
                if not isinstance(dialogue, list):
                    raise ValueError("script dialogue must be a list")
                lines.extend(_lines_from_items(dialogue, str(shot.get("shot_id") or "")))
            return lines
        if isinstance(data.get("dialogue"), list):
            return _lines_from_items(data["dialogue"], str(data.get("shot_id") or ""))
        if isinstance(data.get("lines"), list):
            return _lines_from_items(data["lines"], str(data.get("shot_id") or ""))
    raise ValueError("script must be shots JSON, a list of lines, or one sentence per line")


def _lines_from_items(items, shot_id: str) -> list[_Line]:
    lines = []
    seq = 0
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            speaker = ""
            line_id = ""
            item_shot = shot_id
        elif isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            speaker = str(item.get("speaker") or "")
            line_id = str(item.get("line_id") or "")
            item_shot = str(item.get("shot_id") or shot_id)
        else:
            continue
        if not text:
            continue
        seq += 1
        lines.append(_Line(
            item_shot,
            line_id or f"L{seq:03d}",
            speaker,
            text,
        ))
    return lines


def _parse_whisper(whisper):
    if isinstance(whisper, str):
        raw = whisper.strip()
        if not raw:
            return "flat", []
        if "-->" in raw:
            return "flat", _parse_srt(raw)
        if raw[0] in "{[":
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("whisper timeline JSON is invalid") from exc
            return _parse_whisper_data(data)
        raise ValueError("whisper timeline must be segments JSON or SRT")
    return _parse_whisper_data(whisper)


def _parse_whisper_data(data):
    if data is None:
        return "flat", []
    if isinstance(data, list):
        if data and all(
            isinstance(item, dict) and str(item.get("shot_id") or "").strip()
            for item in data
        ):
            return "by_shot", _group_segments(data)
        if any(not isinstance(item, dict) for item in data):
            raise ValueError("whisper timeline must be a list of segments")
        return "flat", data
    if isinstance(data, dict):
        if isinstance(data.get("segments"), list):
            return _parse_whisper_data(data["segments"])
        shots = data.get("shots")
        if isinstance(shots, list):
            groups = []
            for shot in shots:
                if not isinstance(shot, dict):
                    raise ValueError("whisper shots must be objects")
                segments = shot.get("segments") or []
                if not isinstance(segments, list):
                    raise ValueError("whisper shot segments must be a list")
                groups.append((str(shot.get("shot_id") or ""), segments))
            return "by_shot", groups
        if data and all(isinstance(value, list) for value in data.values()):
            return "by_shot", [(str(key), value) for key, value in data.items()]
    raise ValueError("whisper timeline must be segments JSON or SRT")


def _group_segments(segments: list) -> list[tuple[str, list]]:
    order: list[str] = []
    buckets: dict[str, list] = {}
    for segment in segments:
        shot_id = str(segment.get("shot_id") or "")
        if shot_id not in buckets:
            buckets[shot_id] = []
            order.append(shot_id)
        buckets[shot_id].append(segment)
    return [(shot_id, buckets[shot_id]) for shot_id in order]


def _parse_srt(text: str) -> list[dict]:
    segments = []
    for block in re.split(r"\n\s*\n", text.strip()):
        start = end = None
        body: list[str] = []
        for line in block.splitlines():
            match = _TIME_RE.search(line)
            if match:
                start = _stamp(match, 1)
                end = _stamp(match, 5)
            elif start is not None and not line.strip().isdigit():
                body.append(line)
        if start is None:
            continue
        segments.append({
            "start": start,
            "end": end if end is not None else start,
            "text": "\n".join(body).strip(),
        })
    return segments


def _stamp(match, group: int) -> float:
    hours = int(match.group(group))
    minutes = int(match.group(group + 1))
    seconds = int(match.group(group + 2))
    millis = int(match.group(group + 3).ljust(3, "0")[:3])
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def _tokens_from_segments(segments) -> list[_Tok]:
    tokens: list[_Tok] = []
    for segment in segments or []:
        if isinstance(segment, dict):
            tokens.extend(_segment_tokens(segment))
    return tokens


def _segment_tokens(segment: dict) -> list[_Tok]:
    start = _as_float(segment.get("start"))
    if start is None:
        start = 0.0
    end = _as_float(segment.get("end"))
    if end is None or end < start:
        end = start
    words = segment.get("words")
    usable = []
    if isinstance(words, list):
        for word in words:
            if isinstance(word, dict) and str(word.get("text") or "").strip():
                usable.append(word)
    if usable and all(_as_float(word.get("start")) is not None for word in usable):
        tokens: list[_Tok] = []
        for word in usable:
            word_start = _as_float(word.get("start"))
            if word_start is None:
                word_start = start
            word_end = _as_float(word.get("end"))
            if word_end is None or word_end < word_start:
                word_end = word_start
            tokens.extend(_explode(str(word.get("text") or ""), word_start, word_end))
        return tokens
    if usable:
        text = "".join(str(word.get("text") or "") for word in usable)
    else:
        text = str(segment.get("text") or "")
    return _explode(text, start, end)


def _explode(text: str, start: float, end: float) -> list[_Tok]:
    chars = list(text.strip())
    if not chars:
        return []
    if end < start:
        end = start
    span = end - start
    count = len(chars)
    tokens: list[_Tok] = []
    for index, ch in enumerate(chars):
        t0 = start + span * index / count
        t1 = start + span * (index + 1) / count
        keys = _keys(ch)
        if not keys:
            tokens.append(_Tok("punct", ch, "", t0, t1))
            continue
        piece = (t1 - t0) / len(keys)
        for key_index, key in enumerate(keys):
            tokens.append(_Tok(
                "content",
                ch if key_index == 0 else key,
                key,
                t0 + piece * key_index,
                t0 + piece * (key_index + 1),
            ))
    return tokens


def _keys(ch: str) -> list[str]:
    if not _is_content(ch):
        return []
    folded = unicodedata.normalize("NFKC", ch).casefold()
    return [item for item in folded if _is_content(item)]


def _is_content(ch: str) -> bool:
    if not ch or ch.isspace():
        return False
    return unicodedata.category(ch)[0] not in {"P", "Z", "C", "S"}


def _as_float(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _r3(value: float) -> float:
    return round(float(value), 3)


def _coerce_cells(value) -> int:
    try:
        cells = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_LINE_CELLS
    if cells < 1:
        return 1
    return cells


def _coerce_threshold(value) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return DEFAULT_LOW_CONFIDENCE
    if threshold < 0:
        return 0.0
    if threshold > 1:
        return 1.0
    return threshold
