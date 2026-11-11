"""Route an MCP write to the open page, or to the headless graph and queue."""

from __future__ import annotations

import time

from .canvas_state import STALE_AFTER_S, get_mirror_entry, mirror_summary
from . import headless_graph as graph
from . import headless_queue as queue

_PAGE_ONLY = frozenset({
    "scene_get", "scene_edit", "scene_capture", "scene_record",
    "layer_get", "layer_edit", "layer_capture",
    "director_get", "director_edit",
    "canvas_focus",
})


def _entry_live(entry: dict | None) -> bool:
    if not entry:
        return False
    if time.time() - entry["received_at"] > STALE_AFTER_S:
        return False
    if entry.get("ws_connected") is False:
        return False
    return True


def page_is_live(project_id: str | None = None) -> bool:
    """True when an open ComfyTV page should keep executing writes.

    A hidden-but-connected tab still counts. A stale mirror, or a tab whose
    websocket is disconnected, does not — those writes go to the headless graph.
    """
    if isinstance(project_id, str) and project_id:
        return _entry_live(get_mirror_entry(project_id))
    if _entry_live(get_mirror_entry(None)):
        return True
    for row in mirror_summary():
        if row.get("stale"):
            continue
        if row.get("tab_ws_connected") is False:
            continue
        return True
    return False


def _page_only(action: str) -> dict:
    raise ValueError(
        f"{action} drives a mounted editor or the live viewport, so it needs "
        "an open ComfyTV page. Stage and graph edits, and run_stage / "
        "graph_run, work with no page through the headless graph and queue."
    )


def _canvas_command(payload: dict) -> dict:
    command = str(payload.get("command") or "")
    if command in ("Comfy.Interrupt", "Comfy.ClearPendingTasks"):
        raise ValueError(
            f"{command} is not issued by the headless queue. A prompt already "
            "in ComfyUI is left running. cancel_stage drops a job that has "
            "not been submitted yet."
        )
    if command == "Comfy.SaveWorkflow":
        project_id = graph.project_id_of(payload)
        graph.save_document(project_id, graph.load_document(project_id))
        return {"executed": command, "headless": True}
    raise ValueError(
        f"{command} changes the live view and needs an open ComfyTV page"
    )


async def execute_headless(action: str, payload: dict) -> dict:
    if action == "canvas_command":
        return _canvas_command(payload)
    if action in _PAGE_ONLY:
        _page_only(action)
    if action == "add_stage":
        return graph.add_stage(payload)
    if action == "set_stage":
        return graph.set_stage(payload)
    if action == "connect_stages":
        return graph.connect_stages(payload)
    if action == "remove_stage":
        return graph.remove_stage(payload)
    if action == "get_stage":
        return graph.get_stage(payload)
    if action == "graph_get":
        return graph.graph_get(payload)
    if action == "graph_edit":
        return graph.graph_edit(payload)
    if action == "arrange_canvas":
        return graph.arrange_canvas(payload)
    if action == "run_stage":
        return await queue.enqueue_stage(payload)
    if action == "graph_run":
        return await queue.enqueue_graph(payload)
    if action == "cancel_stage":
        return queue.cancel_queued_stage(payload)
    raise ValueError(f"no headless handler for {action!r}")
