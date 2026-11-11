import contextvars
import logging
import time

from aiohttp import web
from server import PromptServer

from .. import storage
from ..bot.mcp_server import McpServer, rpc_error
from ._common import routes
from .capabilities import VERSION
from .mcp_tools import TOOLS

_log = logging.getLogger(__name__)

BOT_CHAT_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "comfytv_bot_chat", default="")

_BROADCAST_THROTTLE_S = 60.0

_last_activity: float | None = None
_last_broadcast = 0.0


def _mark_activity() -> None:
    global _last_activity, _last_broadcast
    now = time.time()
    _last_activity = now
    if now - _last_broadcast < _BROADCAST_THROTTLE_S:
        return
    _last_broadcast = now
    try:
        PromptServer.instance.send_sync("comfytv-mcp-activity", {})
    except Exception:
        _log.exception("[ComfyTV/mcp] activity broadcast failed")


def _reset_activity() -> None:
    global _last_activity, _last_broadcast
    _last_activity = None
    _last_broadcast = 0.0

INSTRUCTIONS = (
    "Window into ComfyTV, the canvas-app layer running inside this ComfyUI. "
    "Call server_info first. Read tools report state (projects, canvas, "
    "workflows, outputs, assets, jobs, errors). Write tools (add_stage, "
    "set_stage, connect_stages, run_stage, graph_edit, graph_run) drive the "
    "canvas. When a ComfyTV page is open and mirroring, that page executes "
    "them. With no page open they run server-side: edits land in the "
    "persisted headless graph, and run_stage / graph_run return a job_id "
    "immediately. Jobs run one at a time on ComfyUI's own prompt queue "
    "(nothing is submitted while GET /queue is non-empty). Poll queue_status; "
    "do not block on a fixed timeout. resume_queue continues unfinished "
    "stages after a restart. The live canvas snapshot is still mirrored from "
    "an open page — retry get_canvas after ~10 seconds before concluding no "
    "page is open. Scene, director, and layer editors, and viewport commands, "
    "still need a page. Typical headless loop: add_stage/set_stage or "
    "graph_run {prompt} -> queue_status until done. For machine-level "
    "operations (installing nodes, downloading models) pair this server with "
    "the official comfy-mcp server. Never interrupt or free the ComfyUI queue "
    "from these tools."
)


def _tool_visible(name: str) -> bool:
    if name in ("ask_user", "remember"):
        return bool(BOT_CHAT_ID.get())
    if name != "skill":
        return True
    from .. import skill_store
    return skill_store.skills_enabled()


def _prompts_list() -> list[dict]:
    from .. import skill_store
    return [
        {
            "name": s["name"],
            "title": s.get("display_name") or s["name"],
            "description": s["description"],
        }
        for s in skill_store.enabled_skills()
    ]


def _prompts_get(name: str) -> dict | None:
    from .. import skill_store
    entry = skill_store.find_enabled(name)
    if entry is None:
        return None
    return {
        "description": entry["description"],
        "messages": [{
            "role": "user",
            "content": {"type": "text", "text": skill_store.read_skill(name)},
        }],
    }


SERVER = McpServer(name="comfytv-mcp", title="ComfyTV", version=VERSION, instructions=INSTRUCTIONS,
                   tools=TOOLS, tool_visible=_tool_visible, prompts=_prompts_list, get_prompt=_prompts_get)


@routes.get("/comfytv/mcp_activity")
async def mcp_activity(_request: web.Request) -> web.Response:
    return web.json_response({
        "active": _last_activity is not None,
        "last_seen": _last_activity,
    })


def _mcp_enabled() -> bool:
    try:
        return bool(storage.get_setting("enable-mcp"))
    except Exception:
        _log.exception("[ComfyTV/mcp] enable-mcp lookup failed")
        return False


@routes.post("/comfytv/mcp")
async def mcp_post(request: web.Request) -> web.Response:
    if not _mcp_enabled():
        return web.json_response(
            rpc_error(None, -32000,
                      "ComfyTV MCP is disabled — enable it in the ComfyTV "
                      "sidebar under Settings"),
            status=403,
        )
    _mark_activity()
    chat_id = str(request.query.get("bot_chat") or "")
    token = BOT_CHAT_ID.set(chat_id if chat_id and storage.get_bot_chat(chat_id) else "")
    try:
        return await SERVER.respond(request)
    finally:
        BOT_CHAT_ID.reset(token)


@routes.get("/comfytv/mcp")
async def mcp_get(_request: web.Request) -> web.Response:
    return web.json_response(
        {"error": "SSE streams are not supported; POST JSON-RPC messages instead"},
        status=405,
    )


@routes.delete("/comfytv/mcp")
async def mcp_delete(_request: web.Request) -> web.Response:
    return web.Response(status=200)
