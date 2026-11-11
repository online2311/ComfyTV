"""HTTP access to segment reshoot plans, splices, and version pointers."""

from aiohttp import web

from ..storage.segment_reshoot import (
    ReshootError,
    adopt,
    list_reshoots,
    plan,
    rollback,
    run,
    status,
)
from ._common import routes


def _error(exc: ReshootError) -> web.Response:
    return web.json_response({"error": str(exc)}, status=exc.http_status)


async def _payload(request: web.Request) -> dict:
    if request.can_read_body:
        try:
            data = await request.json()
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _query_args(request: web.Request) -> dict:
    args = {}
    for key in ("project_id", "shot_id", "reshoot_id"):
        value = request.query.get(key)
        if value:
            args[key] = value
    return args


@routes.post("/comfytv/segment_reshoot/plan")
async def plan_reshoot(request: web.Request) -> web.Response:
    try:
        return web.json_response(plan(await _payload(request)))
    except ReshootError as exc:
        return _error(exc)


@routes.post("/comfytv/segment_reshoot/run")
async def run_reshoot(request: web.Request) -> web.Response:
    try:
        return web.json_response(run(await _payload(request)))
    except ReshootError as exc:
        return _error(exc)


@routes.get("/comfytv/segment_reshoot/status")
async def reshoot_status(request: web.Request) -> web.Response:
    try:
        return web.json_response(status(_query_args(request)))
    except ReshootError as exc:
        return _error(exc)


@routes.get("/comfytv/segment_reshoot/list")
async def reshoot_list(request: web.Request) -> web.Response:
    try:
        return web.json_response(list_reshoots(_query_args(request)))
    except ReshootError as exc:
        return _error(exc)


@routes.post("/comfytv/segment_reshoot/adopt")
async def adopt_reshoot(request: web.Request) -> web.Response:
    try:
        return web.json_response(adopt(await _payload(request)))
    except ReshootError as exc:
        return _error(exc)


@routes.post("/comfytv/segment_reshoot/rollback")
async def rollback_reshoot(request: web.Request) -> web.Response:
    try:
        return web.json_response(rollback(await _payload(request)))
    except ReshootError as exc:
        return _error(exc)
