"""Audio, subtitle, and model-path helpers for local Whisper.

No model weights are loaded here. Resampling uses scipy when it is installed
and a numpy interpolator otherwise, so CI (numpy only) and Spark (scipy)
share one result shape.
"""
from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
from pathlib import Path

import folder_paths
import numpy as np

try:
    import av
except ImportError:
    av = None

try:
    import soundfile as sf
except ImportError:
    sf = None

# scipy is optional. Importing it while this module loads pulls the whole
# scientific stack into every ComfyUI stage import. Load it on first resample.
_resample_poly = None
_resample_poly_checked = False


def _load_resample_poly():
    global _resample_poly, _resample_poly_checked
    if _resample_poly_checked:
        return _resample_poly
    _resample_poly_checked = True
    try:
        from scipy.signal import resample_poly as fn
    except ImportError:
        _resample_poly = None
    else:
        _resample_poly = fn
    return _resample_poly


TARGET_SR = 16000
ENV_MODEL_DIR = "COMFYTV_WHISPER_MODEL"
FALLBACK_MODEL_DIR = "/home/nodecloud/whisper-v3/models/whisper-large-v3"

LANGUAGES = (
    "auto", "zh", "en", "ja", "ko", "yue", "fr", "de", "es",
    "pt", "ru", "it", "th", "vi", "id", "ar",
)

_TIMESTAMP = re.compile(r"<\|(\d+(?:\.\d+)?)\|>")
_SOUNDFILE_EXT = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def resample_audio(samples, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample a mono waveform to ``target_sr``."""
    audio = np.asarray(samples, dtype=np.float64).reshape(-1)
    orig_sr = int(orig_sr)
    target_sr = int(target_sr)
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"sample rates must be positive, got {orig_sr} -> {target_sr}")
    if audio.size == 0 or orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    resample_poly = _load_resample_poly()
    if resample_poly is not None:
        divisor = math.gcd(orig_sr, target_sr)
        converted = resample_poly(audio, target_sr // divisor, orig_sr // divisor)
        return np.asarray(converted, dtype=np.float32)
    duration = audio.size / float(orig_sr)
    count = max(1, int(round(duration * target_sr)))
    if audio.size == 1:
        return np.full(count, float(audio[0]), dtype=np.float32)
    src_t = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    dst_t = np.linspace(0.0, duration, num=count, endpoint=False)
    return np.interp(dst_t, src_t, audio).astype(np.float32)


def _to_numpy(wave) -> np.ndarray:
    if hasattr(wave, "detach"):
        wave = wave.detach()
        if hasattr(wave, "float"):
            wave = wave.float()
        if hasattr(wave, "cpu"):
            wave = wave.cpu()
        if hasattr(wave, "numpy"):
            wave = wave.numpy()
    return np.asarray(wave, dtype=np.float32)


def _to_mono(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 1:
        return audio
    if audio.ndim != 2:
        raise RuntimeError(f"unexpected waveform shape {audio.shape}")
    # Planar (channels, time) when the leading axis is a small channel count.
    if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
        return audio.mean(axis=0)
    return audio.mean(axis=-1)


def audio_dict_to_mono16k(audio: dict) -> np.ndarray:
    """ComfyUI AUDIO dict ``{waveform, sample_rate}`` -> mono float32 at 16 kHz."""
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise RuntimeError("AUDIO input must be a dict with 'waveform' and 'sample_rate'.")
    rate = audio.get("sample_rate", audio.get("sampler_rate"))
    if not rate:
        raise RuntimeError("AUDIO input is missing sample_rate.")
    return resample_audio(_to_mono(_to_numpy(audio["waveform"])), int(rate), TARGET_SR)


def _frame_mono(frame) -> tuple[np.ndarray, int]:
    arr = np.asarray(frame.to_ndarray())
    if np.issubdtype(arr.dtype, np.integer):
        peak = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / peak
    else:
        arr = arr.astype(np.float32)
    return _to_mono(arr), int(frame.sample_rate)


def _decode_av(path: str) -> np.ndarray:
    if av is None:
        raise RuntimeError(
            f"cannot read {path}: PyAV is not installed and soundfile could not open it"
        )
    container = av.open(path)
    try:
        if not any(stream.type == "audio" for stream in container.streams):
            raise RuntimeError(f"no audio stream in {path}")
        pieces = []
        rate = None
        for frame in container.decode(audio=0):
            mono, rate = _frame_mono(frame)
            if mono.size:
                pieces.append(mono)
        if not pieces or rate is None:
            raise RuntimeError(f"no audio frames in {path}")
        return resample_audio(np.concatenate(pieces), rate, TARGET_SR)
    finally:
        container.close()


def _decode_soundfile(path: str) -> np.ndarray:
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    return resample_audio(np.asarray(data, dtype=np.float32).mean(axis=1), int(rate), TARGET_SR)


def load_media_mono16k(path: str) -> np.ndarray:
    """Read an audio or video file and return mono float32 audio at 16 kHz."""
    ext = Path(path).suffix.lower()
    if sf is not None and ext in _SOUNDFILE_EXT:
        try:
            return _decode_soundfile(path)
        except Exception:
            pass
    return _decode_av(path)


def _view_url_to_annotated(url: str) -> str:
    params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
    filename = params.get("filename", "")
    subfolder = params.get("subfolder", "")
    type_ = (params.get("type") or "output").lower()
    if not filename:
        raise RuntimeError(f"media URL has no filename: {url}")
    for suffix, kind in ((" [output]", "output"), (" [input]", "input"), (" [temp]", "temp")):
        if filename.endswith(suffix):
            filename = filename[: -len(suffix)]
            type_ = kind
            break
    rel = f"{subfolder}/{filename}" if subfolder else filename
    return f"{rel} [{type_}]"


def _annotated_to_path(annotated: str) -> str:
    getter = getattr(folder_paths, "get_annotated_filepath", None)
    if getter is not None:
        found = getter(annotated)
        if not found or not Path(found).is_file():
            raise RuntimeError(f"media file not found: {annotated}")
        return str(Path(found).resolve())
    if " [" not in annotated or not annotated.endswith("]"):
        raise RuntimeError(f"media file not found: {annotated}")
    name, _, type_part = annotated.rpartition(" [")
    directory = folder_paths.get_directory_by_type(type_part[:-1])
    path = Path(directory) / name
    if not path.is_file():
        raise RuntimeError(f"media file not found: {annotated}")
    return str(path.resolve())


def resolve_media_path(media: str) -> str:
    """Resolve a filesystem path, ``/view?`` URL, or ``name [output]`` annotation."""
    media = (media or "").strip()
    if not media:
        raise RuntimeError("media path is empty")
    if media.startswith("/view?"):
        return _annotated_to_path(_view_url_to_annotated(media))
    if media.endswith("]") and " [" in media:
        return _annotated_to_path(media)
    path = Path(media)
    if path.is_file():
        return str(path.resolve())
    raise RuntimeError(f"media file not found: {media}")


def _is_snapshot(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _folder_paths(kind: str) -> list[str]:
    getter = getattr(folder_paths, "get_folder_paths", None)
    if getter is not None:
        try:
            return [str(item) for item in getter(kind)]
        except Exception:
            return []
    mapping = getattr(folder_paths, "folder_names_and_paths", None)
    if isinstance(mapping, dict) and kind in mapping:
        entry = mapping[kind]
        paths = entry[0] if isinstance(entry, (list, tuple)) else entry
        return [str(item) for item in paths]
    return []


def _candidate_snapshots() -> list[Path]:
    found: list[Path] = []
    for kind in ("whisper", "stt"):
        for folder in _folder_paths(kind):
            root = Path(folder)
            if _is_snapshot(root):
                found.append(root)
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if child.is_dir() and _is_snapshot(child):
                    found.append(child)
    return found


def resolve_model_dir(explicit: str | None = None) -> str:
    """Find a local Hugging Face Whisper snapshot. Never downloads.

    Order: explicit widget path, ``COMFYTV_WHISPER_MODEL``, a snapshot under
    the ComfyUI ``whisper`` then ``stt`` model folders (preferring a directory
    named ``large-v3``), then ``FALLBACK_MODEL_DIR`` when that directory exists.
    """
    explicit = (explicit or "").strip()
    if explicit:
        path = Path(explicit)
        if not _is_snapshot(path):
            raise RuntimeError(
                f"Whisper model_dir {explicit} is not a local Hugging Face "
                "snapshot (missing config.json). This node never downloads weights."
            )
        return str(path.resolve())

    env = os.environ.get(ENV_MODEL_DIR, "").strip()
    if env:
        path = Path(env)
        if not _is_snapshot(path):
            raise RuntimeError(
                f"{ENV_MODEL_DIR}={env} is not a local Hugging Face snapshot "
                "(missing config.json). This node never downloads weights."
            )
        return str(path.resolve())

    candidates = _candidate_snapshots()
    for path in candidates:
        if "large-v3" in path.name:
            return str(path.resolve())
    if candidates:
        return str(candidates[0].resolve())

    fallback = Path(FALLBACK_MODEL_DIR)
    if _is_snapshot(fallback):
        return str(fallback.resolve())
    raise RuntimeError(
        "Whisper large-v3 snapshot not found. Set COMFYTV_WHISPER_MODEL to the "
        "local Hugging Face snapshot, or place it in models/whisper or models/stt "
        "(a directory containing config.json). "
        f"Documented fallback: {FALLBACK_MODEL_DIR}. "
        "This node never downloads weights."
    )


def register_whisper_model_folders() -> None:
    """Register ``models/whisper`` and ``models/stt`` with ComfyUI folder_paths."""
    adder = getattr(folder_paths, "add_model_folder_path", None)
    mapping = getattr(folder_paths, "folder_names_and_paths", None)
    if adder is None and not isinstance(mapping, dict):
        return
    models_dir = getattr(folder_paths, "models_dir", None)
    if not models_dir:
        base = getattr(folder_paths, "base_path", None)
        if not base:
            return
        models_dir = os.path.join(str(base), "models")
    for kind in ("whisper", "stt"):
        path = os.path.join(str(models_dir), kind)
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            pass
        if path in _folder_paths(kind):
            continue
        if adder is not None:
            adder(kind, path)
            continue
        current, exts = mapping.get(kind, ([], set()))
        mapping[kind] = (list(current) + [path], exts)


def _srt_time(seconds: float) -> str:
    total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    blocks = []
    index = 1
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") if seg.get("end") is not None else start)
        if end < start:
            end = start
        blocks.append(f"{index}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}")
        index += 1
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3400 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
    )


def join_transcript(segments: list[dict]) -> str:
    parts = [str(seg.get("text") or "").strip() for seg in segments]
    parts = [part for part in parts if part]
    if not parts:
        return ""
    out = parts[0]
    for part in parts[1:]:
        if _is_cjk(out[-1]) or _is_cjk(part[0]):
            out += part
        else:
            out += " " + part
    return out


def segments_to_json(segments: list[dict]) -> str:
    clean = []
    for seg in segments:
        item = {
            "start": round(float(seg.get("start") or 0.0), 3),
            "end": round(float(seg.get("end") or 0.0), 3),
            "text": str(seg.get("text") or "").strip(),
        }
        words = seg.get("words") or []
        if words:
            item["words"] = [
                {
                    "start": round(float(word.get("start") or 0.0), 3),
                    "end": round(float(word.get("end") or 0.0), 3),
                    "text": str(word.get("text") or "").strip(),
                }
                for word in words
                if str(word.get("text") or "").strip()
            ]
        clean.append(item)
    return json.dumps(clean, ensure_ascii=False)


def pack_transcript(segments: list[dict]) -> tuple[str, str, str]:
    return segments_to_srt(segments), join_transcript(segments), segments_to_json(segments)


# Half-width sentence marks Whisper emits inside Chinese transcripts.
# Decimal points are handled separately so "3.5" stays "3.5".
_ZH_PUNCT = str.maketrans({
    ",": "，",
    "?": "？",
    "!": "！",
    ":": "：",
    ";": "；",
    "(": "（",
    ")": "）",
})


def normalize_zh_punctuation(text: str) -> str:
    """Map half-width sentence punctuation to full-width Chinese forms.

    ``.`` between digits stays a decimal point. ``...`` becomes ``……``.
    SRT timestamp lines are not passed here; ``finish_transcript`` rewrites
    cue and word text before the SRT is built, so ``00:00:06,100`` stays.
    """
    if not text:
        return text
    text = text.replace("...", "……")
    text = text.translate(_ZH_PUNCT)
    if "." not in text:
        return text
    chars = list(text)
    for index, ch in enumerate(chars):
        if ch != ".":
            continue
        prev = chars[index - 1] if index else ""
        nxt = chars[index + 1] if index + 1 < len(chars) else ""
        if prev.isdigit() and nxt.isdigit():
            continue
        chars[index] = "。"
    return "".join(chars)


def _normalize_segment_punctuation(segments: list[dict]) -> list[dict]:
    normalized = []
    for seg in segments:
        item = dict(seg)
        item["text"] = normalize_zh_punctuation(str(item.get("text") or ""))
        words = item.get("words")
        if words:
            item["words"] = [
                {**word, "text": normalize_zh_punctuation(str(word.get("text") or ""))}
                for word in words
            ]
        normalized.append(item)
    return normalized


def parse_whisper_timestamp_text(decoded: str) -> list[dict]:
    """Turn ``<|0.00|>text<|1.20|>`` decoder text into segments."""
    if not decoded:
        return []
    marks = list(_TIMESTAMP.finditer(decoded))
    if len(marks) < 2:
        return []
    segments = []
    for index, mark in enumerate(marks[:-1]):
        start = float(mark.group(1))
        end = float(marks[index + 1].group(1))
        text = decoded[mark.end():marks[index + 1].start()].strip()
        if not text:
            continue
        if end < start:
            end = start
        segments.append({"start": start, "end": end, "text": text})
    return segments


def group_token_words(pieces: list[tuple[str, float, float]]) -> list[dict]:
    """Group decoded token pieces. CJK stays one token per word; Latin joins on spaces."""
    words: list[dict] = []
    buf = ""
    start = None
    end = None

    def flush() -> None:
        nonlocal buf, start, end
        text = buf.strip()
        if text and start is not None:
            words.append({
                "start": round(float(start), 3),
                "end": round(float(end if end is not None else start), 3),
                "text": text,
            })
        buf = ""
        start = None
        end = None

    for text, t0, t1 in pieces:
        if not text:
            continue
        if buf and (text[0].isspace() or _is_cjk(text[0]) or _is_cjk(buf[-1])):
            flush()
        if start is None:
            start = float(t0)
        buf += text
        end = float(t1)
    flush()
    return words


def _as_id_list(value) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, int):
        return [value]
    if value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(item) for item in value]


def _as_float_list(value) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(item) for item in value]


def _as_float(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _decode(tokenizer, ids, *, timestamps: bool = False) -> str:
    kwargs = {"skip_special_tokens": True}
    if timestamps:
        kwargs["decode_with_timestamps"] = True
    try:
        return tokenizer.decode(ids, **kwargs)
    except TypeError:
        return tokenizer.decode(ids)


def _clamp_word_times(words: list[dict], segment_start: float, segment_end: float) -> list[dict]:
    """Pull word timestamps into ``[segment_start, segment_end]``.

    Token times can land past the segment boundary (observed: a word end of
    6.20 s inside a segment that ends at 6.10 s). The segment end stays the
    SRT cue so the next cue does not overlap; the word is clamped instead of
    extending the segment.
    """
    lo = float(segment_start)
    hi = float(segment_end)
    if hi < lo:
        hi = lo
    clamped = []
    for word in words:
        start = min(max(float(word.get("start") or 0.0), lo), hi)
        end_value = word.get("end")
        end = float(end_value if end_value is not None else start)
        end = min(max(end, lo), hi)
        if end < start:
            end = start
        item = dict(word)
        item["start"] = round(start, 3)
        item["end"] = round(end, 3)
        clamped.append(item)
    return clamped


def _token_window(index: int, times: list[float], segment_end: float) -> tuple[float, float]:
    """One token's ``[start, end]``. The end is the next timestamp, or the segment end."""
    start = float(times[index])
    if index + 1 < len(times):
        end = float(times[index + 1])
    else:
        end = max(start, float(segment_end))
    if end < start:
        end = start
    return start, end


def _pieces_from_tokens(
    tokenizer, ids: list[int], times: list[float], segment_end: float,
) -> list[tuple[str, float, float]]:
    """Decode tokens, merging consecutive byte-level fragments into UTF-8.

    A Whisper byte-level token that is only part of a Chinese character
    decodes to U+FFFD. Consecutive fragments are merged until the span
    decodes without U+FFFD, then emitted as one piece. The piece uses the
    first token's start and the last token's end. A fragment that never
    completes is dropped so a word is never U+FFFD.
    """
    pieces: list[tuple[str, float, float]] = []
    count = len(ids)
    index = 0
    while index < count:
        single = _decode(tokenizer, [ids[index]])
        if single == "":
            index += 1
            continue
        last = index
        text = single
        if "\ufffd" in text:
            while "\ufffd" in text and last + 1 < count:
                nxt = _decode(tokenizer, [ids[last + 1]])
                if not nxt or "\ufffd" not in nxt:
                    break
                last += 1
                text = _decode(tokenizer, ids[index:last + 1])
            if "\ufffd" in text:
                index = last + 1
                continue
        start, _ = _token_window(index, times, segment_end)
        _, end = _token_window(last, times, segment_end)
        if end < start:
            end = start
        pieces.append((text, start, end))
        index = last + 1
    return pieces


def _words_for_tokens(
    tokenizer, tokens, token_times, segment_end: float, segment_start: float = 0.0,
) -> list[dict]:
    ids = _as_id_list(tokens)
    times = _as_float_list(token_times)
    if len(times) > len(ids) > 0:
        times = times[-len(ids):]
    elif len(times) < len(ids):
        ids = ids[:len(times)]
    pieces = _pieces_from_tokens(tokenizer, ids, times, segment_end)
    return _clamp_word_times(
        group_token_words(pieces), segment_start, segment_end,
    )


def _segment_rows(outputs) -> list[dict] | None:
    if not isinstance(outputs, dict):
        return None
    segs = outputs.get("segments")
    if not segs:
        return None
    first = segs[0]
    if isinstance(first, list):
        return list(first)
    if isinstance(first, dict):
        return list(segs)
    return None


def segments_from_generate_output(outputs, tokenizer, *, timestamps: str, duration_s: float) -> list[dict]:
    """Turn ``model.generate`` output into ``[{start, end, text, words?}]``."""
    rows = _segment_rows(outputs)
    if rows is not None:
        decoded = []
        for row in rows:
            start = _as_float(row.get("start") or 0.0)
            end = _as_float(row.get("end") if row.get("end") is not None else start)
            if end < start:
                end = start
            if row.get("tokens") is not None and tokenizer is not None:
                text = _decode(tokenizer, _as_id_list(row.get("tokens"))).strip()
            else:
                text = str(row.get("text") or "").strip()
            if not text:
                continue
            item = {"start": start, "end": end, "text": text}
            if timestamps == "word" and row.get("token_timestamps") is not None and tokenizer is not None:
                words = _words_for_tokens(
                    tokenizer, row.get("tokens"), row.get("token_timestamps"), end,
                    segment_start=start,
                )
                if words:
                    item["words"] = words
            decoded.append(item)
        return decoded

    if tokenizer is None:
        return []
    ids = _as_id_list(outputs.get("sequences") if isinstance(outputs, dict) else outputs)
    if not ids:
        return []
    marked = _decode(tokenizer, ids, timestamps=True)
    parsed = parse_whisper_timestamp_text(marked)
    if parsed:
        return parsed
    text = _decode(tokenizer, ids).strip()
    if not text:
        return []
    return [{"start": 0.0, "end": float(duration_s), "text": text}]


def finalize_segments(segments: list[dict], *, timestamps: str, duration_s: float) -> list[dict]:
    """Collapse timestamped cues when the caller asked for plain text only."""
    if timestamps != "none":
        return segments
    text = join_transcript(segments)
    if not text:
        return []
    return [{"start": 0.0, "end": round(float(duration_s), 3), "text": text}]


def finish_transcript(
    segments: list[dict],
    *,
    timestamps: str,
    duration_s: float,
    normalize_punctuation: bool = False,
) -> tuple[str, str, str]:
    """Build ``(srt, text, segments_json)``.

    ``normalize_punctuation`` rewrites half-width sentence punctuation in cue
    and word text to full-width Chinese forms before the SRT is rendered.
    """
    if normalize_punctuation:
        segments = _normalize_segment_punctuation(segments)
    segments = finalize_segments(
        segments, timestamps=timestamps, duration_s=duration_s,
    )
    return pack_transcript(segments)


def srt_output_stem(media: str) -> str:
    """Stem for ``output/comfytv/<name>.srt``.

    Uses the media filename from a ``/view`` URL, ``name [output]`` annotation,
    or filesystem path. An audio-only run with an empty media string uses
    ``whisper``.
    """
    raw = (media or "").strip()
    if not raw:
        return "whisper"
    name = raw
    if raw.startswith("/view?") or raw.startswith("http://") or raw.startswith("https://"):
        parsed = urllib.parse.urlparse(raw if "://" in raw else f"http://local{raw}")
        params = dict(urllib.parse.parse_qsl(parsed.query))
        name = params.get("filename") or Path(parsed.path).name
    elif raw.endswith("]") and " [" in raw:
        name = raw[:raw.rfind(" [")]
    stem = Path(Path(name).name).stem.replace("\x00", "").strip()
    if stem in {"", ".", ".."}:
        return "whisper"
    return stem


def write_output_srt(media: str, srt: str) -> str:
    """Write ``<name>.srt`` under ComfyUI ``output/comfytv`` and return the path.

    The same media name overwrites the previous file. The returned path is absolute.
    """
    root = (Path(folder_paths.get_output_directory()) / "comfytv").resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = (root / f"{srt_output_stem(media)}.srt").resolve()
    if path.parent != root:
        path = root / "whisper.srt"
    path.write_text(srt or "", encoding="utf-8")
    return str(path)
