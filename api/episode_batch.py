"""Submit a planned episode as one VideoStage per shot, strictly one at a time.

Each submit checks ComfyUI's prompt queue — the same list ``GET /queue``
returns — and waits until nothing is running or pending. The next shot is
not queued until that prompt has finished. Completed shots are recorded in
a state file and skipped on the next call. This module never interrupts
the queue and never frees memory.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from aiohttp import web

from .. import storage
from ..backup import data_dir
from ..nodes.stages.common.episode_video import EpisodeBatchError, plan_episode_videos
from ..nodes.stages.common.shot_document import load_shots_document
from . import headless_graph as graph
from . import headless_queue as queue
from ._common import routes
from .headless_dispatch import page_is_live

_log = logging.getLogger(__name__)

_LOCKS: dict[str, asyncio.Lock] = {}
_ACTIVE = {"queued", "submitting", "running"}
_TERMINAL = {"done", "error", "cancelled", "interrupted"}


def _lock_for(path: str) -> asyncio.Lock:
    lock = _LOCKS.get(path)
    if lock is None:
        lock = asyncio.Lock()
        _LOCKS[path] = lock
    return lock


def _state_path(args: dict, episode: str) -> Path:
    raw = args.get("state_path")
    if isinstance(raw, str) and raw.strip():
        return Path(raw.strip())
    project = graph.project_id_of(args)
    safe_episode = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(episode or "episode"))
    return Path(data_dir()) / "episode_batches" / f"{project}_{safe_episode}.json"


def _load_state(path: Path) -> dict:
    if not path.is_file():
        return {"completed": {}, "nodes": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EpisodeBatchError(f"cannot read state file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise EpisodeBatchError(f"state file is not an object: {path}")
    if not isinstance(data.get("completed"), dict):
        data["completed"] = {}
    if not isinstance(data.get("nodes"), dict):
        data["nodes"] = {}
    return data


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


async def _load_document(args: dict) -> dict:
    if args.get("document") is not None:
        try:
            return load_shots_document(args["document"])
        except ValueError as exc:
            raise EpisodeBatchError(str(exc)) from exc
    raw_path = args.get("document_path")
    if isinstance(raw_path, str) and raw_path.strip():
        path = Path(raw_path.strip())
        if not path.is_file():
            raise EpisodeBatchError(f"shots file not found: {path}")
        try:
            return load_shots_document(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EpisodeBatchError(f"cannot read shots file {path}: {exc}") from exc
    node = args.get("node")
    if node:
        # Imported here: mcp_tools loads this module while its package initializes.
        from .mcp_tools._shared import submit_or_headless
        payload = {"node": str(node)}
        if args.get("project_id"):
            payload["project_id"] = args["project_id"]
        try:
            result = await submit_or_headless("storyboard_get", payload)
        except ValueError as exc:
            raise EpisodeBatchError(str(exc)) from exc
        document = result.get("document") if isinstance(result, dict) else None
        try:
            return load_shots_document(document)
        except ValueError as exc:
            raise EpisodeBatchError(str(exc)) from exc
    raise EpisodeBatchError(
        "pass document, document_path, or node (a StoryboardStage id)"
    )


def _plan_kwargs(args: dict, work: Path) -> dict:
    return {
        "library_root": args.get("library_root"),
        "frames_root": args.get("frames_root"),
        "base_seed": args.get("base_seed"),
        "shots": args["shots"] if "shots" in args else None,
        "workflows": args.get("workflows"),
        "audio": args.get("audio"),
        "resolution": str(args.get("resolution") or "720P"),
        "aspect_ratio": str(args.get("aspect_ratio") or "16:9"),
        "work_dir": str(work),
        "require_locked": bool(args.get("require_locked", True)),
    }


async def run_episode_batch(args: dict) -> dict:
    if not isinstance(args, dict):
        raise EpisodeBatchError("episode batch arguments must be an object")
    action = str(args.get("action") or "plan")
    if action not in ("plan", "run"):
        raise EpisodeBatchError("action must be plan or run")
    document = await _load_document(args)
    state_path = _state_path(args, str(document.get("episode") or ""))
    work = Path(args["work_dir"]) if isinstance(args.get("work_dir"), str) and args.get("work_dir") else state_path.parent / "dialogue"
    try:
        plan = plan_episode_videos(document, **_plan_kwargs(args, work))
    except EpisodeBatchError:
        raise
    except ValueError as exc:
        raise EpisodeBatchError(str(exc)) from exc
    plan["state_path"] = str(state_path)
    if action == "plan":
        _log.info(
            "[ComfyTV/episode] plan %s shots=%s ok=%s",
            plan.get("episode"), plan.get("shot_count"), plan.get("ok"),
        )
        return plan
    return await _run_plan(args, plan, state_path)


async def _run_plan(args: dict, plan: dict, state_path: Path) -> dict:
    project_id = graph.project_id_of(args)
    if page_is_live(project_id):
        raise EpisodeBatchError(
            "episode batch submits through the headless queue. "
            "Close the ComfyTV page, then call action run again."
        )
    if not plan.get("ok"):
        preview = "; ".join(plan.get("errors") or [])[:1200]
        raise EpisodeBatchError(
            "refusing to submit; the plan has errors. "
            + preview
        )
    rerun = _rerun_set(args.get("rerun"), plan)
    max_shots = _max_shots(args.get("max_shots"))
    lock = _lock_for(str(state_path))
    if lock.locked():
        raise EpisodeBatchError(
            f"episode batch is already running for {state_path}"
        )
    await lock.acquire()
    try:
        return await _submit_serial(
            plan,
            project_id=project_id,
            state_path=state_path,
            rerun=rerun,
            max_shots=max_shots,
        )
    finally:
        lock.release()


def _rerun_set(value, plan: dict) -> set[str]:
    if value is True:
        return set(plan.get("selected") or [])
    if value is None or value is False:
        return set()
    if not isinstance(value, list):
        raise EpisodeBatchError("rerun must be true or a list of shot_id")
    return {str(item) for item in value}


def _max_shots(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EpisodeBatchError("max_shots must be a positive integer")
    return value


async def _submit_serial(
    plan: dict,
    *,
    project_id: str,
    state_path: Path,
    rerun: set[str],
    max_shots: int | None,
) -> dict:
    # Imported here: mcp_tools.stages pulls the command bus, which imports this
    # module while the mcp_tools package is still initializing.
    from .mcp_tools.stages import _add_stage, _set_stage

    runner = queue.get_queue()
    state = _load_state(state_path)
    results: list[dict] = []
    submitted = 0
    for shot in plan["shots"]:
        shot_id = str(shot["shot_id"])
        node_id = state["nodes"].get(shot_id)
        if _skip_completed(state, state_path, project_id, shot_id, node_id, rerun):
            done = state["completed"].get(shot_id) or {}
            results.append({
                "shot_id": shot_id,
                "status": "skipped",
                "job_id": done.get("job_id"),
                "graph_node_id": node_id or done.get("graph_node_id"),
            })
            continue
        if max_shots is not None and submitted >= max_shots:
            results.append({"shot_id": shot_id, "status": "pending"})
            continue
        await _wait_until_idle(runner)
        if _skip_completed(state, state_path, project_id, shot_id, node_id, rerun):
            done = state["completed"].get(shot_id) or {}
            results.append({
                "shot_id": shot_id,
                "status": "skipped",
                "job_id": done.get("job_id"),
                "graph_node_id": node_id or done.get("graph_node_id"),
            })
            continue
        if _queue_busy(runner):
            raise EpisodeBatchError(
                f"{shot_id}: GET /queue was not empty at submit; the shot was not submitted"
            )
        node_id = await _ensure_node(
            shot, project_id=project_id, node_id=node_id,
            add_stage=_add_stage, set_stage=_set_stage,
        )
        state["nodes"][shot_id] = node_id
        _save_state(state_path, state)
        existing = None
        if shot_id not in rerun:
            existing = storage.latest_headless_job_for_stage(project_id, node_id)
        if existing and existing["status"] in _ACTIVE:
            row = await _wait_job(runner, existing["id"])
        else:
            if _queue_busy(runner):
                await _wait_until_idle(runner)
            queued = await queue.enqueue_stage({
                "node": node_id,
                "project_id": project_id,
            })
            if shot_id in rerun:
                state["completed"].pop(shot_id, None)
                _save_state(state_path, state)
            row = await _wait_job(runner, queued["job_id"])
        if row["status"] != "done":
            _save_state(state_path, state)
            detail = (row.get("error_text") or row["status"])[:500]
            raise EpisodeBatchError(
                f"{shot_id} failed: {detail}. "
                "Completed shots stay completed; call action run again to continue. "
                f"State: {state_path}"
            )
        finished = {
            "shot_id": shot_id,
            "status": "done",
            "job_id": row["id"],
            "graph_node_id": node_id,
            "outputs": _outputs(row),
        }
        state["completed"][shot_id] = {
            "job_id": row["id"],
            "graph_node_id": node_id,
            "outputs": finished["outputs"],
        }
        _save_state(state_path, state)
        results.append(finished)
        submitted += 1
        _log.info("[ComfyTV/episode] finished %s job %s", shot_id, row["id"])
    return {
        "ok": True,
        "dry_run": False,
        "episode": plan.get("episode"),
        "shot_count": plan.get("shot_count"),
        "submitted": submitted,
        "results": results,
        "state_path": str(state_path),
        "warnings": plan.get("warnings") or [],
    }


def _outputs(row: dict | None) -> list:
    if not row or not row.get("outputs_json"):
        return []
    try:
        parsed = json.loads(row["outputs_json"])
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _skip_completed(
    state: dict, state_path: Path, project_id: str, shot_id: str,
    node_id: str | None, rerun: set[str],
) -> bool:
    if shot_id in rerun:
        return False
    if shot_id in state["completed"]:
        return True
    if not node_id:
        return False
    row = storage.latest_headless_job_for_stage(project_id, str(node_id))
    if not row or row["status"] != "done":
        return False
    state["completed"][shot_id] = {
        "job_id": row["id"],
        "graph_node_id": str(node_id),
        "outputs": _outputs(row),
    }
    _save_state(state_path, state)
    return True


def _queue_busy(runner) -> bool:
    try:
        running, pending = runner.client.queue_ids()
    except Exception as exc:
        raise EpisodeBatchError(
            "could not read GET /queue; the next shot was not submitted"
        ) from exc
    return bool(running or pending)


async def _wait_until_idle(runner) -> None:
    while _queue_busy(runner):
        await runner.tick()
        await _pause(runner)


async def _wait_job(runner, job_id: str) -> dict:
    while True:
        await runner.tick()
        row = storage.get_headless_job(job_id)
        if row is None:
            raise EpisodeBatchError(f"job {job_id} disappeared before it finished")
        if row["status"] in _TERMINAL:
            return row
        await _pause(runner)


async def _pause(runner) -> None:
    delay = float(getattr(runner, "poll_s", 0.05) or 0.05)
    await asyncio.sleep(min(0.05, delay))


async def _ensure_node(
    shot: dict, *, project_id: str, node_id: str | None, add_stage, set_stage,
) -> str:
    stage = shot.get("stage") or {}
    payload = {
        "project_id": project_id,
        "node_class": "ComfyTV.VideoStage",
        "title": stage.get("title") or shot["shot_id"],
        "workflow": stage.get("workflow"),
        "prompt": stage.get("prompt") or "",
        "widgets": stage.get("widgets") or {},
    }
    if node_id:
        _clear_batch_media(project_id, str(node_id))
        payload = dict(payload)
        payload.pop("node_class", None)
        payload["node"] = str(node_id)
        try:
            await set_stage(payload)
        except ValueError as exc:
            raise EpisodeBatchError(f"{shot['shot_id']}: {exc}") from exc
        return str(node_id)
    try:
        added = await add_stage(payload)
    except ValueError as exc:
        raise EpisodeBatchError(f"{shot['shot_id']}: {exc}") from exc
    return str(added["graph_node_id"])


_MEDIA_PREFIXES = ("images.image", "videos.video", "audio.audio")


def _clear_batch_media(project_id: str, node_id: str) -> None:
    """Drop image, video, and audio widgets this batch wrote on an earlier run.

    ``set_stage`` only overlays named widgets, so a shorter reference list or
    a shot that no longer has dialogue would otherwise keep the old slots.
    """
    doc = graph.load_document(project_id)
    node = graph.find_node(doc, node_id)
    if node is None:
        return
    widgets = node.get("widgets")
    if not isinstance(widgets, dict):
        return
    for key in list(widgets):
        if key.startswith(_MEDIA_PREFIXES):
            widgets.pop(key, None)
    graph.save_document(project_id, doc)


@routes.post("/comfytv/episode_batch")
async def episode_batch_route(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "body must be a JSON object"}, status=400)
    if not isinstance(data, dict):
        return web.json_response({"error": "body must be a JSON object"}, status=400)
    try:
        return web.json_response(await run_episode_batch(data))
    except EpisodeBatchError as exc:
        return web.json_response({"error": str(exc)}, status=400)
