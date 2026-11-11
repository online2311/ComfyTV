"""MCP tool for the executor registry.

Scheduling is intentionally not here. ``accepts_work`` and ``gpu_class`` are
what the next scheduler reads. Join tokens are accepted on ``action='join'``
and are not echoed back.
"""

from __future__ import annotations

from ... import storage
from ...storage.executors import ExecutorConflict, JoinRejected


async def _executors(args: dict) -> dict:
    action = args.get("action") or "list"
    if action == "list":
        return {"executors": storage.list_executors(ensure_default=True)}
    if action == "get":
        executor_id = str(args.get("id") or "")
        row = storage.get_executor(executor_id)
        if row is None:
            raise ValueError("executor not found")
        return {"executor": row}
    if action == "register":
        if not args.get("name") or not args.get("base_url"):
            raise ValueError("name and base_url are required")
        try:
            row = storage.register_executor(
                name=args["name"],
                base_url=args["base_url"],
                kind=args.get("kind") or "permanent",
                gpu_class=args.get("gpu_class") or "",
                labels=args.get("labels"),
                capabilities=args.get("capabilities"),
                max_concurrency=args.get("max_concurrency", 1),
                provider=args.get("provider"),
                enabled=True if "enabled" not in args else args["enabled"],
            )
        except ExecutorConflict as exc:
            raise ValueError(str(exc)) from None
        return {"executor": row}
    if action == "update":
        executor_id = str(args.get("id") or "")
        changes = {
            key: args[key]
            for key in (
                "name", "kind", "base_url", "gpu_class", "labels", "capabilities",
                "max_concurrency", "enabled", "draining", "provider",
            )
            if key in args
        }
        try:
            row = storage.update_executor(executor_id, changes)
        except ExecutorConflict as exc:
            raise ValueError(str(exc)) from None
        if row is None:
            raise ValueError("executor not found")
        return {"executor": row}
    if action == "drain":
        executor_id = str(args.get("id") or "")
        draining = True if "draining" not in args else args["draining"]
        row = storage.drain_executor(executor_id, draining=draining)
        if row is None:
            raise ValueError("executor not found")
        return {"executor": row}
    if action == "remove":
        executor_id = str(args.get("id") or "")
        if not storage.remove_executor(executor_id):
            raise ValueError("executor not found")
        return {"ok": True}
    if action == "join":
        token = args.get("join_token") if isinstance(args.get("join_token"), str) else ""
        try:
            row = storage.join_executor(
                token=token,
                name=args.get("name") or "",
                base_url=args.get("base_url") or "",
                gpu_class=args.get("gpu_class") or "",
                labels=args.get("labels"),
                capabilities=args.get("capabilities"),
                max_concurrency=args.get("max_concurrency", 1),
                provider=args.get("provider"),
            )
        except JoinRejected:
            raise ValueError("join token rejected") from None
        except ExecutorConflict as exc:
            raise ValueError(str(exc)) from None
        return {"executor": row}
    raise ValueError(
        "unknown action "
        f"{action!r} (use list, get, register, update, drain, remove, or join)"
    )


TOOLS: dict[str, dict] = {
    "executors": {
        "description": (
            "GPU executor registry for this controller. action='list' or 'get' "
            "reads hosts. 'register', 'update', 'drain', and 'remove' manage "
            "them. 'join' is how an ephemeral executor registers itself at boot "
            "with a one-time join token; the token is not returned. "
            "accepts_work is false while disabled, draining, unseen, "
            "unreachable, or while running+pending has reached max_concurrency "
            "(default 1). idle is true only when the latest probe saw an empty "
            "ComfyUI queue, including prompts other people submitted. "
            "idle_since is when that empty stretch started. This tool does not "
            "dispatch work and does not interrupt or free a queue. "
            "gpu_class is a free-form label such as gb10-128g or rtx5090-32g."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list", "get", "register", "update", "drain", "remove", "join",
                    ],
                },
                "id": {"type": "string"},
                "name": {"type": "string"},
                "kind": {"type": "string", "enum": ["permanent", "ephemeral"]},
                "base_url": {"type": "string"},
                "gpu_class": {"type": "string"},
                "labels": {"type": "object"},
                "capabilities": {"type": "object"},
                "max_concurrency": {"type": "integer", "minimum": 1, "maximum": 32},
                "enabled": {"type": "boolean"},
                "draining": {"type": "boolean"},
                "provider": {"type": "object"},
                "join_token": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "handler": _executors,
    },
}
