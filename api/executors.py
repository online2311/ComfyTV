"""HTTP API for the executor registry.

Token-bearing routes take the secret from the JSON body or an
``Authorization: Bearer`` header and never write it to the log.
"""

from __future__ import annotations

import logging

from aiohttp import web

from .. import storage
from ..storage.executors import ExecutorConflict, JoinRejected
from ._common import routes

_log = logging.getLogger(__name__)


async def _body(request: web.Request) -> dict | None:
    try:
        body = await request.json()
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    return body


def _bearer(request: web.Request) -> str:
    header = request.headers.get("Authorization", "")
    if header[:7].lower() == "bearer ":
        return header[7:].strip()
    return ""


def _error(message: str, status: int) -> web.Response:
    return web.json_response({"error": message}, status=status)


@routes.get("/comfytv/executors")
async def list_executors(_request: web.Request) -> web.Response:
    return web.json_response({
        "executors": storage.list_executors(ensure_default=True),
    })


@routes.post("/comfytv/executors/join-token")
async def configure_join_token(request: web.Request) -> web.Response:
    body = await _body(request)
    if body is None or "token" not in body:
        return _error("join token must be 16 to 256 characters", 400)
    token = body.pop("token")
    try:
        storage.configure_join_token(token)
    except ValueError:
        return _error("join token must be 16 to 256 characters", 400)
    return web.json_response({"configured": True})


@routes.post("/comfytv/executors/join-tokens")
async def issue_join_token(_request: web.Request) -> web.Response:
    return web.json_response({"token": storage.issue_join_token()}, status=201)


@routes.post("/comfytv/executors/join")
async def join_executor(request: web.Request) -> web.Response:
    body = await _body(request)
    if body is None:
        return _error("invalid JSON body", 400)
    token = _bearer(request) or str(body.pop("join_token", "") or "")
    body.pop("join_token", None)
    try:
        row = storage.join_executor(
            token=token,
            name=body.get("name") or "",
            base_url=body.get("base_url") or "",
            gpu_class=body.get("gpu_class") or "",
            labels=body.get("labels"),
            capabilities=body.get("capabilities"),
            max_concurrency=body.get("max_concurrency", 1),
            provider=body.get("provider"),
        )
    except JoinRejected:
        return _error("join token rejected", 401)
    except ExecutorConflict as exc:
        return _error(str(exc), 409)
    except ValueError as exc:
        return _error(str(exc), 400)
    _log.info("[ComfyTV/executors] join accepted id=%s", row["id"])
    return web.json_response({"executor": row}, status=201)


@routes.post("/comfytv/executors")
async def register_executor(request: web.Request) -> web.Response:
    body = await _body(request)
    if body is None:
        return _error("invalid JSON body", 400)
    if not body.get("name") or not body.get("base_url"):
        return _error("name and base_url are required", 400)
    try:
        row = storage.register_executor(
            name=body["name"],
            base_url=body["base_url"],
            kind=body.get("kind") or "permanent",
            gpu_class=body.get("gpu_class") or "",
            labels=body.get("labels"),
            capabilities=body.get("capabilities"),
            max_concurrency=body.get("max_concurrency", 1),
            provider=body.get("provider"),
            enabled=True if "enabled" not in body else body["enabled"],
        )
    except ExecutorConflict as exc:
        return _error(str(exc), 409)
    except ValueError as exc:
        return _error(str(exc), 400)
    return web.json_response({"executor": row}, status=201)


@routes.get("/comfytv/executors/{executor_id}")
async def get_executor(request: web.Request) -> web.Response:
    row = storage.get_executor(request.match_info["executor_id"])
    if row is None:
        return _error("executor not found", 404)
    return web.json_response({"executor": row})


@routes.patch("/comfytv/executors/{executor_id}")
async def update_executor(request: web.Request) -> web.Response:
    body = await _body(request)
    if body is None:
        return _error("invalid JSON body", 400)
    try:
        row = storage.update_executor(request.match_info["executor_id"], body)
    except ExecutorConflict as exc:
        return _error(str(exc), 409)
    except ValueError as exc:
        return _error(str(exc), 400)
    if row is None:
        return _error("executor not found", 404)
    return web.json_response({"executor": row})


@routes.post("/comfytv/executors/{executor_id}/drain")
async def drain_executor(request: web.Request) -> web.Response:
    body = await _body(request)
    if body is None:
        return _error("invalid JSON body", 400)
    draining = True if "draining" not in body else body["draining"]
    try:
        row = storage.drain_executor(
            request.match_info["executor_id"], draining=draining,
        )
    except ValueError as exc:
        return _error(str(exc), 400)
    if row is None:
        return _error("executor not found", 404)
    return web.json_response({"executor": row})


@routes.delete("/comfytv/executors/{executor_id}")
async def remove_executor(request: web.Request) -> web.Response:
    if not storage.remove_executor(request.match_info["executor_id"]):
        return _error("executor not found", 404)
    return web.json_response({"ok": True})
