"""Score one shot against locked character refs and reshoot below threshold.

Reshoots are submitted one at a time. Each submission reads GET /queue and
waits until that queue is empty. This module does not cancel other work.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import server
from PIL import Image, ImageDraw

from .consistency_backend import (
    ConsistencyBackendUnavailable,
    SimilarityBackend,
    build_backend,
)
from ..storage.series_library import (
    SeriesLibraryError,
    image_urls_for_shot,
    parse_series_ref,
    resolve_root_arg,
    resolve_series_ref,
)
from .local_comfy import prepare_workflow
from .base import RunnerContext
from ._media_paths import fresh_output_path, path_to_view_url, view_url_to_path

DEFAULT_THRESHOLD = 0.70
DEFAULT_MAX_RESHOOTS = 2
DEFAULT_FRAME_COUNT = 4
DEFAULT_QUEUE_TIMEOUT_S = 3600.0
DEFAULT_HISTORY_TIMEOUT_S = 7200.0
DEFAULT_POLL_S = 2.0
_IDENTITY_VIEWS = ("headshot", "front")
_SEED_MAX = 2_147_483_647


class ConsistencyRescoreError(RuntimeError):
    pass


class ConsistencyQueueBusy(ConsistencyRescoreError):
    def __init__(self, message: str, *, report: dict | None = None):
        super().__init__(message)
        self.report = report or {}


@dataclass
class ReshootJob:
    seed: int
    shot: dict
    prompt: str
    images: list
    workflow: str
    options: dict = field(default_factory=dict)
    previous_video: str = ""


def cache_key(
    *,
    video_url: str,
    shot_index: int,
    threshold: float,
    max_reshoots: int,
    frame_count: int,
    seed: int,
    backend: str,
    workflow: str,
    series_root: str,
    force_run_token: int,
) -> str:
    payload = {
        "video_url": video_url or "",
        "shot_index": int(shot_index or 1),
        "threshold": float(threshold),
        "max_reshoots": int(max_reshoots),
        "frame_count": int(frame_count),
        "seed": int(seed),
        "backend": backend or "auto",
        "workflow": workflow or "",
        "series_root": str(series_root or ""),
        "force_run_token": int(force_run_token or 0),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def next_seed(seed: int, step: int) -> int:
    base = int(seed)
    if base < 0:
        base = 0
    value = base + int(step)
    if value > _SEED_MAX:
        value = value % _SEED_MAX
    if value <= 0 or value == int(seed):
        value = 1 if base >= _SEED_MAX else base + 1
    return int(value)


def locked_character_refs(shot: dict, root) -> list[dict]:
    characters = shot.get("characters") if isinstance(shot, dict) else None
    if not isinstance(characters, list) or not characters:
        raise ConsistencyRescoreError("shot has no characters to score")
    refs: list[dict] = []
    for character in characters:
        code = character if isinstance(character, str) else None
        if isinstance(character, dict):
            code = character.get("id")
        text = str(code or "").strip()
        if not text:
            continue
        if text.startswith("NEW:") or text.startswith("episode:"):
            raise ConsistencyRescoreError(
                f"character {text} is not a locked library asset"
            )
        parsed = parse_series_ref(text)
        if parsed.get("kind") != "asset" or not str(parsed.get("id") or "").startswith("CHR_"):
            raise ConsistencyRescoreError(f"character {text} is not a CHR_ asset")
        refs.append(_resolve_identity(root, parsed))
    if not refs:
        raise ConsistencyRescoreError("shot has no characters to score")
    return refs


def _resolve_identity(root, parsed: dict) -> dict:
    version = parsed.get("version")
    ident = parsed["id"] if version is None else f"{parsed['id']}@v{version}"
    last_error: Exception | None = None
    for view in _IDENTITY_VIEWS:
        try:
            resolved = resolve_series_ref(
                root, f"{ident}#{view}", require_locked=True,
            )
        except SeriesLibraryError as exc:
            if "not locked" in str(exc):
                raise ConsistencyRescoreError(str(exc)) from exc
            last_error = exc
            continue
        if resolved.get("media_type") != "image":
            last_error = ConsistencyRescoreError(
                f"{ident}#{view} is not an image"
            )
            continue
        return resolved
    if last_error is not None:
        raise ConsistencyRescoreError(str(last_error))
    raise ConsistencyRescoreError(f"no locked identity view for {ident}")


def score_frames(frames: list[dict], refs: list[dict], scorer) -> tuple[float, list[dict]]:
    per_character: list[dict] = []
    columns: list[list[float]] = [[] for _ in frames]
    for ref in refs:
        best: float | None = None
        for index, frame in enumerate(frames):
            value = float(scorer.score_pair(frame["path"], ref["absolute_path"]))
            columns[index].append(value)
            best = value if best is None else max(best, value)
        per_character.append({
            "id": ref.get("id"),
            "code": f"{ref.get('id')}@v{ref.get('version')}",
            "view": ref.get("view"),
            "score": 0.0 if best is None else best,
            "path": ref.get("absolute_path"),
        })
    for index, frame in enumerate(frames):
        frame["score"] = min(columns[index]) if columns[index] else 0.0
    if not per_character:
        raise ConsistencyRescoreError("shot has no characters to score")
    shot_score = min(row["score"] for row in per_character)
    finish = getattr(scorer, "finish_take", None)
    if finish is not None:
        finish()
    return shot_score, per_character


def frame_times(duration_s: float, count: int) -> list[float]:
    count = int(count)
    if count < 1:
        raise ConsistencyRescoreError("frame_count must be >= 1")
    span = max(float(duration_s) - 0.05, 0.0)
    if count == 1:
        return [span / 2.0]
    return [span * index / (count - 1) for index in range(count)]


def extract_shot_frames(
    video_url: str,
    count: int,
    *,
    duration_s: float | None = None,
    source_path: str | None = None,
    run: Callable | None = None,
    output_path: Callable | None = None,
    to_url: Callable | None = None,
) -> list[dict]:
    runner = run or subprocess.run
    source = source_path
    if source is None:
        found = view_url_to_path(video_url)
        if found is None:
            raise ConsistencyRescoreError(f"cannot resolve video {video_url}")
        source = str(found)
    if duration_s is None:
        duration_s = probe_duration(source, run=runner)
    maker = output_path or (lambda: fresh_output_path(".png", subfolder="comfytv/frames"))
    publish = to_url or path_to_view_url
    frames = []
    for moment in frame_times(duration_s, count):
        dest = Path(maker())
        dest.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-y", "-ss", f"{moment:.3f}", "-i", source,
            "-frames:v", "1", str(dest),
        ]
        result = runner(command, check=False, capture_output=True)
        if getattr(result, "returncode", 0) != 0:
            raise ConsistencyRescoreError(
                f"ffmpeg could not extract a frame at {moment:.3f}s"
            )
        frames.append({
            "t": float(moment),
            "path": str(dest),
            "image_url": publish(dest),
        })
    return frames


def probe_duration(path: str, *, run: Callable) -> float:
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", path,
    ]
    result = run(command, check=False, capture_output=True, text=True)
    if getattr(result, "returncode", 0) != 0:
        raise ConsistencyRescoreError(f"ffprobe could not read duration of {path}")
    try:
        return float((result.stdout or "0").strip())
    except (TypeError, ValueError) as exc:
        raise ConsistencyRescoreError(f"ffprobe returned no duration for {path}") from exc


def take_label(take: dict) -> str:
    return (
        f"take {take['take']}  seed {take['seed']}  "
        f"score {float(take['score']):.3f}"
    )


def render_contact_sheet(takes: list[dict], dest: Path) -> None:
    if not takes:
        raise ConsistencyRescoreError("contact sheet has no takes")
    thumb = 160
    label_h = 20
    pad = 8
    cols = max(len(take.get("frames") or []) for take in takes)
    cols = max(cols, 1)
    width = pad + cols * (thumb + pad)
    row_h = label_h + thumb + pad
    height = pad + len(takes) * row_h
    sheet = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for row, take in enumerate(takes):
        top = pad + row * row_h
        draw.text((pad, top), take_label(take), fill=(230, 230, 230))
        for col, frame in enumerate(take.get("frames") or []):
            left = pad + col * (thumb + pad)
            cell_top = top + label_h
            path = frame.get("path")
            if path and Path(path).is_file():
                image = Image.open(path).convert("RGB")
                image = image.resize((thumb, thumb))
                sheet.paste(image, (left, cell_top))
            draw.text(
                (left + 4, cell_top + thumb - 14),
                f"{float(frame.get('score') or 0):.2f}",
                fill=(255, 220, 80),
            )
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)


def default_contact_sheet(takes: list[dict]) -> str:
    dest = fresh_output_path(".png", subfolder="comfytv/consistency")
    render_contact_sheet(takes, dest)
    return path_to_view_url(dest)


def apply_selection(report: dict, selected_index: int) -> dict:
    ranked = sorted(
        report.get("takes") or [],
        key=lambda row: (-float(row.get("score") or 0), int(row.get("take") or 0)),
    )
    index = int(selected_index or 1)
    if not ranked:
        index = 1
        video = ""
    else:
        if index < 1:
            index = 1
        if index > len(ranked):
            index = len(ranked)
        video = ranked[index - 1].get("video_url") or ""
    updated = dict(report)
    updated["takes"] = ranked
    updated["images"] = _pick_images(ranked)
    updated["selected_index"] = index
    updated["selected_video"] = video
    updated["stop"] = "pick_take"
    return updated


def _pick_images(ranked: list[dict]) -> list[dict]:
    rows = []
    for index, take in enumerate(ranked, start=1):
        frames = take.get("frames") or []
        url = ""
        if frames and isinstance(frames[0], dict):
            url = str(frames[0].get("image_url") or "")
        rows.append({
            "index": index,
            "label": take_label(take),
            "image_url": url,
            "score": take.get("score"),
            "video_url": take.get("video_url") or "",
        })
    return rows


def queue_is_idle(snapshot: dict) -> bool:
    return not (snapshot.get("running") or snapshot.get("pending"))


def normalize_queue(payload: dict) -> dict:
    return {
        "running": _queue_ids(payload.get("queue_running")),
        "pending": _queue_ids(payload.get("queue_pending")),
    }


def _queue_ids(items) -> list[str]:
    found = []
    for item in items or []:
        if isinstance(item, (list, tuple)) and len(item) > 1:
            found.append(str(item[1]))
        elif isinstance(item, str):
            found.append(item)
    return found


def fetch_queue(base_url: str, *, timeout: float = 30.0) -> dict:
    payload = _request_json(base_url.rstrip("/") + "/queue", timeout=timeout)
    return normalize_queue(payload if isinstance(payload, dict) else {})


def post_prompt(base_url: str, prompt: dict, *, timeout: float = 60.0) -> str:
    payload = _request_json(
        base_url.rstrip("/") + "/prompt",
        body={"prompt": prompt},
        timeout=timeout,
    )
    prompt_id = payload.get("prompt_id") if isinstance(payload, dict) else None
    if not prompt_id:
        raise ConsistencyRescoreError("POST /prompt returned no prompt_id")
    return str(prompt_id)


def fetch_history(base_url: str, prompt_id: str, *, timeout: float = 30.0) -> dict | None:
    payload = _request_json(
        base_url.rstrip("/") + "/history/" + urllib.parse.quote(prompt_id),
        timeout=timeout,
    )
    if not isinstance(payload, dict):
        return None
    entry = payload.get(prompt_id)
    return entry if isinstance(entry, dict) else None


def video_url_from_history(entry: dict) -> str:
    outputs = entry.get("outputs") or {}
    for node_out in outputs.values():
        if not isinstance(node_out, dict):
            continue
        for key in ("videos", "gifs", "video"):
            items = node_out.get(key)
            if isinstance(items, dict):
                items = [items]
            if not items:
                continue
            first = items[0] if isinstance(items, list) else None
            if not isinstance(first, dict):
                continue
            filename = str(first.get("filename") or "")
            if not filename:
                continue
            query = [("filename", filename)]
            subfolder = str(first.get("subfolder") or "")
            if subfolder:
                query.append(("subfolder", subfolder))
            query.append(("type", str(first.get("type") or "output")))
            return "/view?" + urllib.parse.urlencode(query)
    raise ConsistencyRescoreError("reshoot finished without a video file")


def await_idle(
    reader: Callable[[], dict],
    *,
    timeout_s: float,
    poll_s: float = DEFAULT_POLL_S,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    allow_running: bool = False,
) -> dict:
    pause = sleep or time.sleep
    now = clock or time.monotonic
    start = now()
    last = {"running": [], "pending": []}
    while True:
        last = reader() or {}
        running = list(last.get("running") or [])
        pending = list(last.get("pending") or [])
        if not pending and (allow_running or not running):
            return {"running": running, "pending": pending}
        if now() - start >= float(timeout_s):
            raise ConsistencyQueueBusy(
                "GET /queue stayed busy "
                f"(running={running}, pending={pending}). "
                "The reshoot was not submitted.",
            )
        pause(poll_s)


def submit_reshoot_prompt(
    base_url: str,
    prompt: dict,
    *,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    queue_timeout_s: float = DEFAULT_QUEUE_TIMEOUT_S,
    history_timeout_s: float = DEFAULT_HISTORY_TIMEOUT_S,
    poll_s: float = DEFAULT_POLL_S,
) -> str:
    await_idle(
        lambda: fetch_queue(base_url),
        timeout_s=queue_timeout_s,
        poll_s=poll_s,
        sleep=sleep,
        clock=clock,
    )
    latest = fetch_queue(base_url)
    if not queue_is_idle(latest):
        raise ConsistencyQueueBusy(
            "GET /queue filled before the reshoot was submitted.",
        )
    prompt_id = post_prompt(base_url, prompt)
    return _wait_for_video(
        base_url,
        prompt_id,
        timeout_s=history_timeout_s,
        poll_s=poll_s,
        sleep=sleep,
        clock=clock,
    )


def _wait_for_video(
    base_url: str,
    prompt_id: str,
    *,
    timeout_s: float,
    poll_s: float,
    sleep: Callable[[float], None] | None,
    clock: Callable[[], float] | None,
) -> str:
    pause = sleep or time.sleep
    now = clock or time.monotonic
    start = now()
    while True:
        entry = fetch_history(base_url, prompt_id)
        if entry:
            status = (entry.get("status") or {}).get("status_str")
            if status == "error":
                raise ConsistencyRescoreError(
                    f"reshoot prompt {prompt_id} failed"
                )
            if status == "success" or entry.get("outputs"):
                return video_url_from_history(entry)
        if now() - start >= float(timeout_s):
            raise ConsistencyRescoreError(
                f"reshoot prompt {prompt_id} did not finish"
            )
        pause(poll_s)


def _request_json(url: str, *, body: dict | None = None, timeout: float = 30.0) -> dict:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw) if raw else {}
    if not isinstance(parsed, dict):
        raise ConsistencyRescoreError(f"unexpected JSON from {url}")
    return parsed


def discover_base_url(explicit: str | None) -> str:
    text = str(explicit or "").strip()
    if text:
        return text.rstrip("/")
    port = getattr(getattr(server, "PromptServer", None), "instance", None)
    port_value = getattr(port, "port", None)
    address = getattr(port, "address", None) or "127.0.0.1"
    if not port_value:
        raise ConsistencyRescoreError(
            "ComfyUI base URL is unknown, so GET /queue cannot run. "
            "Set comfy_base_url on the stage."
        )
    host = "127.0.0.1" if address in ("", "0.0.0.0") else address
    return f"http://{host}:{port_value}"


class QueueVideoSubmitter:
    """Submit one video prompt after GET /queue reports idle."""

    def __init__(self, base_url: str, *, queue_timeout_s: float, history_timeout_s: float,
                 sleep=None, clock=None, poll_s: float = DEFAULT_POLL_S):
        self.base_url = base_url
        self.queue_timeout_s = queue_timeout_s
        self.history_timeout_s = history_timeout_s
        self.sleep = sleep
        self.clock = clock
        self.poll_s = poll_s

    def submit(self, job: ReshootJob) -> str:
        prompt = build_reshoot_prompt(job)
        return submit_reshoot_prompt(
            self.base_url,
            prompt,
            sleep=self.sleep,
            clock=self.clock,
            queue_timeout_s=self.queue_timeout_s,
            history_timeout_s=self.history_timeout_s,
            poll_s=self.poll_s,
        )


def build_reshoot_prompt(job: ReshootJob) -> dict:
    if not str(job.workflow or "").strip():
        raise ConsistencyRescoreError(
            "reshoot needs a video workflow label"
        )
    options = dict(job.options or {})
    options["seed"] = int(job.seed)
    context = RunnerContext(
        kind="video",
        main_prompt=job.prompt or "",
        upstream={
            "images": list(job.images or []),
            "texts": [],
            "videos": [],
            "audio": [],
        },
        options=options,
    )
    workflow, _meta = prepare_workflow(f"video/{job.workflow}", {"video"}, context)
    return workflow


def coerce_video_url(payload) -> str:
    if isinstance(payload, str):
        text = payload.strip()
        if not text.startswith("{"):
            if not text:
                raise ConsistencyRescoreError("reshoot runner returned no video")
            return text
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text
    if isinstance(payload, dict):
        for key in ("video_url", "url"):
            if payload.get(key):
                return str(payload[key])
        for bucket in ("videos", "images"):
            items = payload.get(bucket)
            if isinstance(items, list) and items and isinstance(items[0], dict):
                url = items[0].get("video_url") or items[0].get("image_url") or items[0].get("url")
                if url:
                    return str(url)
    raise ConsistencyRescoreError("reshoot runner returned no video")


async def run_consistency_rescore(
    *,
    video_url: str,
    series_root,
    shot: dict | None = None,
    storyboard=None,
    shot_index: int = 1,
    scorer: SimilarityBackend | None = None,
    backend: str = "auto",
    reader: Callable[[], dict] | None = None,
    submitter=None,
    base_url: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    max_reshoots: int = DEFAULT_MAX_RESHOOTS,
    frame_count: int = DEFAULT_FRAME_COUNT,
    seed: int = -1,
    workflow: str = "",
    prompt: str = "",
    images: list | None = None,
    options: dict | None = None,
    extract_frames: Callable | None = None,
    write_contact_sheet: Callable | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    queue_timeout_s: float = DEFAULT_QUEUE_TIMEOUT_S,
    history_timeout_s: float = DEFAULT_HISTORY_TIMEOUT_S,
    poll_s: float = DEFAULT_POLL_S,
    selected_index: int = 1,
    allow_running: bool = False,
) -> dict:
    if not str(video_url or "").strip():
        raise ConsistencyRescoreError("consistency rescore needs a shot video")
    series_root = _require_root(series_root)
    limit = int(max_reshoots)
    if limit < 0:
        raise ConsistencyRescoreError("max_reshoots must be >= 0")
    gate = float(threshold)
    if gate < 0.0 or gate > 1.0:
        raise ConsistencyRescoreError("threshold must be between 0 and 1")
    chosen = shot if isinstance(shot, dict) else _shot_from_storyboard(storyboard, shot_index)
    refs = locked_character_refs(chosen, series_root)
    active = scorer if scorer is not None else build_backend(backend)
    extract = extract_frames or extract_shot_frames
    sheet_writer = write_contact_sheet or default_contact_sheet
    queue_reader = reader
    video_submitter = submitter
    motion = _shot_prompt(chosen, prompt)
    wired = list(images or [])
    ref_images = wired or image_urls_for_shot(chosen, series_root)
    takes: list[dict] = []
    current = video_url
    reshoots = 0
    while True:
        frames = [dict(frame) for frame in extract(current, int(frame_count))]
        shot_score, per_character = score_frames(frames, refs, active)
        take_no = len(takes) + 1
        take_seed = int(seed) if take_no == 1 else next_seed(int(seed), reshoots)
        takes.append({
            "take": take_no,
            "seed": take_seed,
            "score": shot_score,
            "passed": shot_score >= gate,
            "video_url": current,
            "frames": frames,
            "characters": per_character,
        })
        if shot_score >= gate or reshoots >= limit:
            break
        reshoots += 1
        new_seed = next_seed(int(seed), reshoots)
        _wait_for_reshoot(
            queue_reader, base_url, queue_timeout_s, poll_s, sleep, clock, takes,
            allow_running=allow_running,
        )
        if video_submitter is None:
            video_submitter = QueueVideoSubmitter(
                discover_base_url(base_url),
                queue_timeout_s=queue_timeout_s,
                history_timeout_s=history_timeout_s,
                sleep=sleep,
                clock=clock,
                poll_s=poll_s,
            )
        current = await _call_submitter(video_submitter, ReshootJob(
            seed=new_seed,
            shot=chosen,
            prompt=motion,
            images=ref_images,
            workflow=workflow,
            options=dict(options or {}),
            previous_video=current,
        ))
    report = {
        "stop": "pick_take",
        "status": "awaiting_pick",
        "shot_id": chosen.get("shot_id") or chosen.get("shot_no"),
        "shot_index": int(shot_index or 1),
        "threshold": gate,
        "max_reshoots": limit,
        "backend": getattr(active, "name", backend),
        "reshoots": reshoots,
        "takes": list(takes),
        "takes_in_order": list(takes),
        "contact_sheet": sheet_writer(_ranked(takes)),
    }
    return apply_selection(report, selected_index)


async def _call_submitter(submitter, job: ReshootJob) -> str:
    submitted = submitter.submit(job)
    if inspect.isawaitable(submitted):
        submitted = await submitted
    return coerce_video_url(submitted) if not isinstance(submitted, str) else submitted


def _wait_for_reshoot(reader, base_url, timeout_s, poll_s, sleep, clock, takes,
                      allow_running: bool = False) -> dict:
    if reader is None:
        resolved = discover_base_url(base_url)
        reader = lambda: fetch_queue(resolved)
    try:
        return await_idle(
            reader,
            timeout_s=timeout_s,
            poll_s=poll_s,
            sleep=sleep,
            clock=clock,
            allow_running=allow_running,
        )
    except ConsistencyQueueBusy as exc:
        scored = ", ".join(
            f"take {row['take']}={float(row['score']):.3f}" for row in takes
        )
        partial = {
            "stop": "pick_take",
            "status": "queue_busy",
            "takes_in_order": list(takes),
            "takes": _ranked(takes),
        }
        raise ConsistencyQueueBusy(
            f"{exc} Scored so far: {scored}.",
            report=partial,
        ) from exc


def _require_root(series_root):
    if series_root:
        return series_root
    try:
        return resolve_root_arg(None, None)
    except SeriesLibraryError as exc:
        raise ConsistencyRescoreError(str(exc)) from exc


def _ranked(takes: list[dict]) -> list[dict]:
    return sorted(
        takes,
        key=lambda row: (-float(row.get("score") or 0), int(row.get("take") or 0)),
    )


def _shot_prompt(shot: dict, prompt: str) -> str:
    text = str(prompt or "").strip()
    if text:
        return text
    motion = shot.get("motion_prompt") or shot.get("motion_desc") or ""
    return motion.strip() if isinstance(motion, str) else ""


def _shot_from_storyboard(storyboard, shot_index: int) -> dict:
    if isinstance(storyboard, dict) and "characters" in storyboard and "shots" not in storyboard:
        return storyboard
    document: Any = storyboard
    if isinstance(storyboard, str):
        if not storyboard.strip():
            raise ConsistencyRescoreError("storyboard is empty")
        try:
            document = json.loads(storyboard)
        except json.JSONDecodeError as exc:
            raise ConsistencyRescoreError("storyboard is not JSON") from exc
    shots = [
        shot for shot in ((document or {}).get("shots") or [])
        if isinstance(shot, dict)
    ]
    index = int(shot_index or 1)
    if index < 1 or index > len(shots):
        raise ConsistencyRescoreError(
            f"shot_index {index} is outside 1..{len(shots) or 0}"
        )
    return shots[index - 1]


__all__ = [
    "ConsistencyQueueBusy",
    "ConsistencyRescoreError",
    "DEFAULT_FRAME_COUNT",
    "DEFAULT_MAX_RESHOOTS",
    "DEFAULT_THRESHOLD",
    "ReshootJob",
    "apply_selection",
    "cache_key",
    "extract_shot_frames",
    "locked_character_refs",
    "render_contact_sheet",
    "run_consistency_rescore",
    "submit_reshoot_prompt",
    "take_label",
]
