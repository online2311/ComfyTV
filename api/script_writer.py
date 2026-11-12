"""HTTP access to script sessions stored under a project directory."""

from aiohttp import web

from ..storage.script_writer import ScriptError, dispatch
from ._common import routes


@routes.post("/comfytv/script_writer")
async def script_writer_post(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    try:
        return web.json_response(dispatch(payload))
    except ScriptError as exc:
        return web.json_response({"error": str(exc)}, status=exc.http_status)
