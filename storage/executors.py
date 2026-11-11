"""Persisted executor registry and one-time join tokens.

The next scheduler should call ``list_executors`` and send a stage only to a
row whose ``accepts_work`` is true and whose ``gpu_class`` matches the stage.
``idle_since`` is the start of the current empty-queue stretch; a later
RunPod lifecycle can reclaim an ephemeral executor that has been idle for
10–15 minutes. This module does not submit prompts and does not call
interrupt or free.

Join tokens are configured on the controller (environment variable
``COMFYTV_EXECUTOR_JOIN_TOKEN`` or ``configure_join_token``). Only the SHA-256
digest is written. ``absorb_configured_join_token`` removes the variable from
the process environment after hashing it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlsplit

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from server import PromptServer

from .. import db
from ..db import Executor, ExecutorJoinToken

_log = logging.getLogger(__name__)

ENV_JOIN_TOKEN = "COMFYTV_EXECUTOR_JOIN_TOKEN"
_TOKEN_PREFIX = b"comfytv-executor-join-v1\x00"
_MIN_TOKEN = 16
_MAX_TOKEN = 256
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_KINDS = ("permanent", "ephemeral")
_JSON_CAP = 16384
_UPDATABLE = {
    "name", "kind", "base_url", "gpu_class", "labels", "capabilities",
    "max_concurrency", "enabled", "draining", "provider",
}


class JoinRejected(Exception):
    def __init__(self) -> None:
        super().__init__("join token rejected")


class ExecutorConflict(Exception):
    def __init__(self, message: str = "executor name already in use") -> None:
        super().__init__(message)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _loads(raw: Optional[str], fallback):
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def token_digest(token: str) -> str:
    return hashlib.sha256(_TOKEN_PREFIX + token.encode("utf-8")).hexdigest()


def _check_token_shape(token: str) -> None:
    if not isinstance(token, str) or len(token) < _MIN_TOKEN or len(token) > _MAX_TOKEN:
        raise ValueError("join token must be 16 to 256 characters")


def configure_join_token(token: str) -> None:
    """Store a controller-configured one-time token. Plaintext is discarded."""
    _check_token_shape(token)
    digest = token_digest(token)
    with db.get_session() as s:
        previous = s.execute(
            select(ExecutorJoinToken).where(ExecutorJoinToken.origin == "configured")
        ).scalars().all()
        for row in previous:
            if row.token_hash != digest:
                s.delete(row)
        if s.get(ExecutorJoinToken, digest) is None:
            s.add(ExecutorJoinToken(token_hash=digest, origin="configured"))
        s.commit()
    _log.info("[ComfyTV/executors] join token configured")


def absorb_configured_join_token() -> None:
    """Hash ``COMFYTV_EXECUTOR_JOIN_TOKEN`` and drop it from the environment."""
    raw = os.environ.pop(ENV_JOIN_TOKEN, None)
    if not raw:
        return
    try:
        configure_join_token(raw)
    finally:
        del raw


def issue_join_token() -> str:
    """Mint a one-time token. The caller sees the plaintext once."""
    token = secrets.token_urlsafe(32)
    digest = token_digest(token)
    with db.get_session() as s:
        s.add(ExecutorJoinToken(token_hash=digest, origin="issued"))
        s.commit()
    _log.info("[ComfyTV/executors] issued one-time join token")
    return token


def local_comfy_base_url() -> str:
    port = 8188
    instance = getattr(PromptServer, "instance", None)
    raw = getattr(instance, "port", None) if instance is not None else None
    if raw:
        try:
            port = int(raw)
        except (TypeError, ValueError):
            port = 8188
    return f"http://127.0.0.1:{port}"


def normalize_base_url(url: str) -> str:
    text = str(url or "").strip()
    parsed = urlsplit(text)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("base_url must be an http or https URL")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not include credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not include a query or fragment")
    parts = [part.lower() for part in parsed.path.split("/") if part]
    if any(part in {"interrupt", "free"} for part in parts):
        raise ValueError("base_url must not target interrupt or free")
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port else ""
    path = "/" + "/".join(part for part in parsed.path.split("/") if part)
    if path == "/":
        path = ""
    return f"{parsed.scheme}://{host}{port}{path}"


def _name(value: str) -> str:
    text = str(value or "").strip()
    if not _NAME.match(text):
        raise ValueError(
            "name must be 1-64 characters of letters, digits, '.', '_' or '-'"
        )
    return text


def _kind(value: str) -> str:
    text = str(value or "").strip()
    if text not in _KINDS:
        raise ValueError("kind must be 'permanent' or 'ephemeral'")
    return text


def _gpu_class(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) > 64 or any(ord(ch) < 32 for ch in text):
        raise ValueError("gpu_class must be at most 64 plain characters")
    return text


def _concurrency(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_concurrency must be an integer")
    if value < 1 or value > 32:
        raise ValueError("max_concurrency must be from 1 to 32")
    return value


def _object(value: Any, field: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    try:
        encoded = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be JSON serializable") from None
    if len(encoded) > _JSON_CAP:
        raise ValueError(f"{field} is too large")
    loaded = json.loads(encoded)
    if not isinstance(loaded, dict):
        raise ValueError(f"{field} must be a JSON object")
    return loaded


def _flag(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _is_idle(row: Executor) -> bool:
    return (
        row.health == "reachable"
        and bool(row.queue_known)
        and int(row.running_count or 0) == 0
        and int(row.pending_count or 0) == 0
    )


def _admission(row: Executor) -> tuple[bool, Optional[str]]:
    if not row.enabled:
        return False, "disabled"
    if row.draining:
        return False, "draining"
    if row.health == "unknown":
        return False, "unknown"
    if row.health != "reachable":
        return False, "unreachable"
    if not row.queue_known:
        return False, "queue_unknown"
    inflight = int(row.running_count or 0) + int(row.pending_count or 0)
    if inflight >= int(row.max_concurrency or 1):
        return False, "at_capacity"
    return True, None


def _to_dict(row: Executor) -> dict:
    accepts, reason = _admission(row)
    gpu = _loads(row.gpu_json, None)
    return {
        "id": row.id,
        "name": row.name,
        "kind": row.kind,
        "base_url": row.base_url,
        "gpu_class": row.gpu_class or "",
        "labels": _loads(row.labels_json, {}),
        "capabilities": _loads(row.capabilities_json, {}),
        "max_concurrency": int(row.max_concurrency or 1),
        "enabled": bool(row.enabled),
        "draining": bool(row.draining),
        "is_default": bool(row.is_default),
        "last_seen": _iso(row.last_seen),
        "health": row.health or "unknown",
        "health_error": row.health_error,
        "running_count": int(row.running_count or 0),
        "pending_count": int(row.pending_count or 0),
        "queue_known": bool(row.queue_known),
        "idle": _is_idle(row),
        "idle_since": _iso(row.idle_since),
        "gpu": gpu if isinstance(gpu, dict) else None,
        "provider": _loads(row.provider_json, {}),
        "accepts_work": accepts,
        "admission_reason": reason,
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
    }


def _ordered(session):
    return session.execute(
        select(Executor).order_by(Executor.created_at, Executor.name)
    ).scalars().all()


def list_executors(*, ensure_default: bool = False) -> list[dict]:
    if ensure_default:
        ensure_default_executor()
    with db.get_session() as s:
        return [_to_dict(row) for row in _ordered(s)]


def get_executor(executor_id: str) -> Optional[dict]:
    with db.get_session() as s:
        row = s.get(Executor, executor_id)
        return _to_dict(row) if row else None


def ensure_default_executor() -> Optional[dict]:
    """Seed the in-process ComfyUI when the registry is empty.

    An existing row is left alone, so registering GPU hosts does not invent a
    second default. Deleting the last row and listing again restores ``local``,
    which is the host the headless worker already submits to.
    """
    absorb_configured_join_token()
    with db.get_session() as s:
        if s.execute(select(func.count()).select_from(Executor)).scalar():
            return None
        row = Executor(
            id=uuid.uuid4().hex,
            name="local",
            kind="permanent",
            base_url=local_comfy_base_url(),
            gpu_class="",
            labels_json=json.dumps({"role": "local"}),
            capabilities_json="{}",
            max_concurrency=1,
            enabled=True,
            draining=False,
            is_default=True,
            health="unknown",
            running_count=0,
            pending_count=0,
            queue_known=False,
            provider_json="{}",
        )
        s.add(row)
        try:
            s.commit()
        except IntegrityError:
            s.rollback()
            return None
        return _to_dict(row)


def register_executor(
    *,
    name: str,
    base_url: str,
    kind: str = "permanent",
    gpu_class: str = "",
    labels: Optional[dict] = None,
    capabilities: Optional[dict] = None,
    max_concurrency: int = 1,
    provider: Optional[dict] = None,
    enabled: bool = True,
    is_default: bool = False,
) -> dict:
    fields = _identity(
        name=name,
        base_url=base_url,
        kind=kind,
        gpu_class=gpu_class,
        labels=labels,
        capabilities=capabilities,
        max_concurrency=max_concurrency,
        provider=provider,
    )
    enabled = _flag(enabled, "enabled")
    with db.get_session() as s:
        taken = s.execute(
            select(Executor.id).where(Executor.name == fields["name"])
        ).scalar_one_or_none()
        if taken is not None:
            raise ExecutorConflict()
        row = Executor(
            id=uuid.uuid4().hex,
            name=fields["name"],
            kind=fields["kind"],
            base_url=fields["base_url"],
            gpu_class=fields["gpu_class"],
            labels_json=json.dumps(fields["labels"], ensure_ascii=False),
            capabilities_json=json.dumps(fields["capabilities"], ensure_ascii=False),
            max_concurrency=fields["max_concurrency"],
            enabled=enabled,
            draining=False,
            is_default=bool(is_default),
            health="unknown",
            running_count=0,
            pending_count=0,
            queue_known=False,
            provider_json=json.dumps(fields["provider"], ensure_ascii=False),
        )
        s.add(row)
        try:
            s.commit()
        except IntegrityError:
            s.rollback()
            raise ExecutorConflict() from None
        return _to_dict(row)


def _identity(**fields) -> dict:
    return {
        "name": _name(fields["name"]),
        "base_url": normalize_base_url(fields["base_url"]),
        "kind": _kind(fields.get("kind") or "permanent"),
        "gpu_class": _gpu_class(fields.get("gpu_class")),
        "labels": _object(fields.get("labels"), "labels"),
        "capabilities": _object(fields.get("capabilities"), "capabilities"),
        "max_concurrency": _concurrency(fields.get("max_concurrency", 1)),
        "provider": _object(fields.get("provider"), "provider"),
    }


def _apply_identity(row: Executor, fields: dict) -> None:
    url_changed = row.base_url != fields["base_url"]
    row.name = fields["name"]
    row.kind = fields["kind"]
    row.base_url = fields["base_url"]
    row.gpu_class = fields["gpu_class"]
    row.labels_json = json.dumps(fields["labels"], ensure_ascii=False)
    row.capabilities_json = json.dumps(fields["capabilities"], ensure_ascii=False)
    row.max_concurrency = fields["max_concurrency"]
    row.provider_json = json.dumps(fields["provider"], ensure_ascii=False)
    if url_changed:
        row.health = "unknown"
        row.health_error = None
        row.queue_known = False
        row.running_count = 0
        row.pending_count = 0
        row.idle_since = None
        row.last_seen = None
        row.gpu_json = None


def update_executor(executor_id: str, changes: dict) -> Optional[dict]:
    if not isinstance(changes, dict):
        raise ValueError("changes must be an object")
    unknown = set(changes) - _UPDATABLE
    if unknown:
        raise ValueError(f"unknown executor fields: {sorted(unknown)}")
    with db.get_session() as s:
        row = s.get(Executor, executor_id)
        if row is None:
            return None
        merged = {
            "name": row.name,
            "base_url": row.base_url,
            "kind": row.kind,
            "gpu_class": row.gpu_class,
            "labels": _loads(row.labels_json, {}),
            "capabilities": _loads(row.capabilities_json, {}),
            "max_concurrency": int(row.max_concurrency or 1),
            "provider": _loads(row.provider_json, {}),
        }
        for key in ("name", "base_url", "kind", "gpu_class", "labels",
                    "capabilities", "max_concurrency", "provider"):
            if key in changes:
                merged[key] = changes[key]
        fields = _identity(**merged)
        if fields["name"] != row.name:
            taken = s.execute(
                select(Executor.id)
                .where(Executor.name == fields["name"])
                .where(Executor.id != executor_id)
            ).scalar_one_or_none()
            if taken is not None:
                raise ExecutorConflict()
        _apply_identity(row, fields)
        if "enabled" in changes:
            row.enabled = _flag(changes["enabled"], "enabled")
        if "draining" in changes:
            row.draining = _flag(changes["draining"], "draining")
        row.updated_at = _now()
        try:
            s.commit()
        except IntegrityError:
            s.rollback()
            raise ExecutorConflict() from None
        return _to_dict(row)


def drain_executor(executor_id: str, draining: bool = True) -> Optional[dict]:
    """Stop admitting new work. The current ComfyUI queue is left running."""
    return update_executor(executor_id, {"draining": draining})


def remove_executor(executor_id: str) -> bool:
    with db.get_session() as s:
        row = s.get(Executor, executor_id)
        if row is None:
            return False
        s.delete(row)
        s.commit()
        return True


def join_executor(
    *,
    token: str,
    name: str,
    base_url: str,
    gpu_class: str = "",
    labels: Optional[dict] = None,
    capabilities: Optional[dict] = None,
    max_concurrency: int = 1,
    provider: Optional[dict] = None,
) -> dict:
    """Register or refresh an ephemeral executor. Consumes a valid token."""
    absorb_configured_join_token()
    if not isinstance(token, str) or len(token) < _MIN_TOKEN or len(token) > _MAX_TOKEN:
        raise JoinRejected()
    digest = token_digest(token)
    fields = _identity(
        name=name,
        base_url=base_url,
        kind="ephemeral",
        gpu_class=gpu_class,
        labels=labels,
        capabilities=capabilities,
        max_concurrency=max_concurrency,
        provider=provider,
    )
    with db.get_session() as s:
        token_row = s.get(ExecutorJoinToken, digest)
        if token_row is None or not hmac.compare_digest(token_row.token_hash, digest):
            raise JoinRejected()
        existing = s.execute(
            select(Executor).where(Executor.name == fields["name"])
        ).scalar_one_or_none()
        if existing is not None and existing.kind != "ephemeral":
            raise ExecutorConflict("that name belongs to a permanent executor")
        s.delete(token_row)
        if existing is None:
            row = Executor(
                id=uuid.uuid4().hex,
                enabled=True,
                draining=False,
                is_default=False,
                health="unknown",
                running_count=0,
                pending_count=0,
                queue_known=False,
            )
            _apply_identity(row, fields)
            s.add(row)
        else:
            row = existing
            row.enabled = True
            _apply_identity(row, fields)
        try:
            s.commit()
        except IntegrityError:
            s.rollback()
            raise ExecutorConflict() from None
        _log.info("[ComfyTV/executors] ephemeral executor %s joined", row.name)
        return _to_dict(row)


def record_observation(
    executor_id: str,
    *,
    health: str,
    running_count: int,
    pending_count: int,
    queue_known: bool,
    queue_empty: bool,
    gpu: Optional[dict],
    error: Optional[str],
    now: datetime,
) -> Optional[dict]:
    """Write one probe. Idle starts only when the shared empty check passed."""
    if health not in ("reachable", "unreachable"):
        raise ValueError("health must be reachable or unreachable")
    with db.get_session() as s:
        row = s.get(Executor, executor_id)
        if row is None:
            return None
        previous_health = row.health
        previous_idle = _is_idle(row)
        row.health = health
        row.health_error = error
        row.running_count = int(running_count)
        row.pending_count = int(pending_count)
        row.queue_known = bool(queue_known)
        if health == "reachable":
            row.last_seen = now
        if gpu is not None:
            row.gpu_json = json.dumps(gpu, ensure_ascii=False)
        idle = health == "reachable" and queue_known and queue_empty
        if idle:
            if row.idle_since is None:
                row.idle_since = now
        else:
            row.idle_since = None
        row.updated_at = now
        s.commit()
        if previous_health != row.health or previous_idle != idle:
            _log.info(
                "[ComfyTV/executors] %s health %s -> %s idle=%s running=%s pending=%s",
                row.name, previous_health, row.health, idle,
                row.running_count, row.pending_count,
            )
        return _to_dict(row)
