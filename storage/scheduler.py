"""Persistence for shot x stage schedule units and shot affinity.

This module does not open a connection to an executor.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func, select

from .. import db
from ..db import HeadlessJob, ScheduleAffinity, ScheduleUnit
from .executors import foreign_executors_registered

_ACTIVE = ("submitting", "running", "cancel-requested")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _loads(raw, fallback):
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _plain(value: Any, field: str, limit: int) -> str:
    text = str(value or "").strip()
    if not text or len(text) > limit or any(ord(ch) < 32 for ch in text):
        raise ValueError(f"{field} must be 1-{limit} plain characters")
    return text


def _gpu_class(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) > 64 or any(ord(ch) < 32 for ch in text):
        raise ValueError("gpu_class must be at most 64 plain characters")
    return text


def _prompt(value: Any) -> dict:
    if not isinstance(value, dict) or not value:
        raise ValueError("prompt must be a non-empty API-format object")
    encoded = json.dumps(value, ensure_ascii=False)
    if len(encoded) > 2_000_000:
        raise ValueError("prompt is too large")
    return value


def _unit_to_dict(row: ScheduleUnit) -> dict:
    return {
        "id": row.id,
        "seq": row.seq,
        "project_id": row.project_id,
        "shot_id": row.shot_id,
        "stage": row.stage,
        "gpu_class": row.gpu_class or "",
        "prompt_json": row.prompt_json or "{}",
        "partial_json": row.partial_json,
        "status": row.status,
        "executor_id": row.executor_id,
        "avoid_executor_id": row.avoid_executor_id,
        "prompt_id": row.prompt_id,
        "attempt": int(row.attempt or 0),
        "attempt_id": row.attempt_id,
        "client_id": row.client_id,
        "post_started": bool(row.post_started),
        "discard_outputs": bool(row.discard_outputs),
        "error_text": row.error_text,
        "waiting_reason": row.waiting_reason,
        "outputs_json": row.outputs_json,
        "output_id": row.output_id,
        "headless_job_id": row.headless_job_id,
        "submitted_at": _aware(row.submitted_at),
        "created_at": _aware(row.created_at),
        "updated_at": _aware(row.updated_at),
    }


def _affinity_to_dict(row: ScheduleAffinity) -> dict:
    return {
        "project_id": row.project_id,
        "shot_id": row.shot_id,
        "gpu_class": row.gpu_class or "",
        "executor_id": row.executor_id or "",
        "updated_at": _iso(row.updated_at),
    }


def _apply(row: ScheduleUnit, fields: dict) -> None:
    if "status" in fields and fields["status"] is not None:
        row.status = fields["status"]
    if fields.get("clear_executor"):
        row.executor_id = None
    elif "executor_id" in fields:
        row.executor_id = fields["executor_id"] or None
    if "avoid_executor_id" in fields:
        row.avoid_executor_id = fields["avoid_executor_id"] or None
    if fields.get("clear_prompt"):
        row.prompt_id = None
    elif "prompt_id" in fields:
        row.prompt_id = fields["prompt_id"] or None
    if "attempt" in fields and fields["attempt"] is not None:
        row.attempt = int(fields["attempt"])
    if fields.get("clear_attempt"):
        row.attempt_id = None
        row.client_id = None
    else:
        if "attempt_id" in fields:
            row.attempt_id = fields["attempt_id"] or None
        if "client_id" in fields:
            row.client_id = fields["client_id"] or None
    if "post_started" in fields:
        row.post_started = bool(fields["post_started"])
    if "discard_outputs" in fields:
        row.discard_outputs = bool(fields["discard_outputs"])
    if "error_text" in fields:
        text = fields["error_text"]
        row.error_text = None if text is None else str(text)[:4000]
    if "waiting_reason" in fields:
        row.waiting_reason = fields["waiting_reason"] or None
    if "outputs_json" in fields:
        row.outputs_json = fields["outputs_json"]
    if "output_id" in fields:
        row.output_id = fields["output_id"]
    if "submitted_at" in fields:
        row.submitted_at = _aware(fields["submitted_at"])


def create_schedule_unit(
    *,
    project_id: str,
    shot_id: str,
    stage: str,
    prompt: dict,
    gpu_class: str = "",
    partial: Optional[list] = None,
    headless_job_id: Optional[str] = None,
) -> dict:
    project_id = _plain(project_id, "project_id", 128)
    shot_id = _plain(shot_id, "shot_id", 128)
    stage = _plain(stage, "stage", 64)
    gpu_class = _gpu_class(gpu_class)
    prompt = _prompt(prompt)
    with db.get_session() as s:
        current = s.execute(select(func.max(ScheduleUnit.seq))).scalar()
        row = ScheduleUnit(
            id=uuid.uuid4().hex,
            seq=int(current or 0) + 1,
            project_id=project_id,
            shot_id=shot_id,
            stage=stage,
            gpu_class=gpu_class,
            prompt_json=json.dumps(prompt, ensure_ascii=False),
            partial_json=None if partial is None else json.dumps(list(partial)),
            status="queued",
            headless_job_id=headless_job_id,
            attempt=0,
        )
        s.add(row)
        s.commit()
        return _unit_to_dict(row)


def get_schedule_unit(unit_id: str) -> Optional[dict]:
    with db.get_session() as s:
        row = s.get(ScheduleUnit, unit_id)
        return _unit_to_dict(row) if row else None


def unit_for_headless(job_id: str) -> Optional[dict]:
    with db.get_session() as s:
        row = s.execute(
            select(ScheduleUnit).where(ScheduleUnit.headless_job_id == job_id)
        ).scalar_one_or_none()
        return _unit_to_dict(row) if row else None


def list_schedule_units(
    *,
    status: Optional[str] = None,
    statuses: Optional[tuple[str, ...]] = None,
    limit: int = 200,
) -> list[dict]:
    with db.get_session() as s:
        q = select(ScheduleUnit)
        if status:
            q = q.where(ScheduleUnit.status == status)
        elif statuses:
            q = q.where(ScheduleUnit.status.in_(tuple(statuses)))
        q = q.order_by(ScheduleUnit.seq.asc()).limit(max(1, min(int(limit), 200)))
        return [_unit_to_dict(row) for row in s.execute(q).scalars().all()]


def claim_schedule_unit(unit_id: str, *, expect: str, become: str) -> Optional[dict]:
    with db.get_session() as s:
        row = s.get(ScheduleUnit, unit_id)
        if row is None or row.status != expect:
            return None
        row.status = become
        row.updated_at = _now()
        s.commit()
        return _unit_to_dict(row)


def update_schedule_unit(unit_id: str, **fields) -> Optional[dict]:
    with db.get_session() as s:
        row = s.get(ScheduleUnit, unit_id)
        if row is None:
            return None
        _apply(row, fields)
        row.updated_at = _now()
        s.commit()
        return _unit_to_dict(row)


def executor_loads() -> dict[str, int]:
    """In-flight units per executor. Queued units are not counted."""
    counts: dict[str, int] = {}
    for unit in list_schedule_units(statuses=_ACTIVE):
        executor_id = unit.get("executor_id")
        if executor_id:
            counts[executor_id] = counts.get(executor_id, 0) + 1
    return counts


def load_breakdown() -> dict[str, dict[str, int]]:
    blank = {"submitting": 0, "running": 0, "cancel_requested": 0}
    found: dict[str, dict[str, int]] = {}
    for unit in list_schedule_units(statuses=_ACTIVE):
        executor_id = unit.get("executor_id")
        if not executor_id:
            continue
        bucket = found.setdefault(executor_id, dict(blank))
        status = unit["status"]
        if status == "cancel-requested":
            bucket["cancel_requested"] += 1
        elif status in bucket:
            bucket[status] += 1
    return found


def get_affinity(project_id: str, shot_id: str) -> Optional[dict]:
    with db.get_session() as s:
        row = s.get(ScheduleAffinity, (project_id, shot_id))
        return _affinity_to_dict(row) if row else None


def lock_affinity(project_id: str, shot_id: str, gpu_class: str, executor_id: str) -> dict:
    """Record the shot's class on first placement. Later calls keep both fields."""
    with db.get_session() as s:
        row = s.get(ScheduleAffinity, (project_id, shot_id))
        if row is None:
            row = ScheduleAffinity(
                project_id=project_id,
                shot_id=shot_id,
                gpu_class=gpu_class or "",
                executor_id=executor_id or "",
            )
            s.add(row)
        row.updated_at = _now()
        s.commit()
        return _affinity_to_dict(row)


def list_affinity(limit: int = 200) -> list[dict]:
    with db.get_session() as s:
        rows = s.execute(
            select(ScheduleAffinity).order_by(ScheduleAffinity.updated_at.desc()).limit(limit)
        ).scalars().all()
        return [_affinity_to_dict(row) for row in rows]


def _shot_of(prompt: dict, job_id: str) -> str:
    if isinstance(prompt, dict):
        for node in prompt.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs")
            if isinstance(inputs, dict):
                shot = inputs.get("shot_id")
                if isinstance(shot, str) and shot.strip():
                    return shot.strip()[:128]
    return f"job-{job_id}"[:128]


def _gpu_of(prompt: dict) -> str:
    if not isinstance(prompt, dict):
        return ""
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        gpu = inputs.get("gpu_class")
        if isinstance(gpu, str):
            try:
                return _gpu_class(gpu)
            except ValueError:
                return ""
    return ""


def adopt_headless_jobs() -> list[str]:
    """Copy queued headless jobs into schedule units when other hosts exist.

    The headless row becomes ``delegated`` so the in-process worker does not
    submit it as well. A single default executor leaves the jobs untouched.
    """
    if not foreign_executors_registered():
        return []
    created: list[str] = []
    with db.get_session() as s:
        jobs = s.execute(
            select(HeadlessJob)
            .where(HeadlessJob.status.in_(("queued", "delegated")))
            .order_by(HeadlessJob.seq.asc())
            .limit(50)
        ).scalars().all()
        for job in jobs:
            existing = s.execute(
                select(ScheduleUnit.id).where(ScheduleUnit.headless_job_id == job.id)
            ).scalar_one_or_none()
            if existing is not None:
                continue
            prompt = _loads(job.prompt_json, {})
            if not isinstance(prompt, dict) or not prompt:
                continue
            current = s.execute(select(func.max(ScheduleUnit.seq))).scalar()
            row = ScheduleUnit(
                id=uuid.uuid4().hex,
                seq=int(current or 0) + 1,
                project_id=(job.project_id or "default")[:128],
                shot_id=_shot_of(prompt, job.id),
                stage=(job.stage_ref or job.kind or "graph")[:64] or "graph",
                gpu_class=_gpu_of(prompt),
                prompt_json=job.prompt_json or "{}",
                partial_json=job.partial_json,
                status="queued",
                headless_job_id=job.id,
                attempt=0,
            )
            s.add(row)
            job.status = "delegated"
            job.updated_at = _now()
            created.append(row.id)
        if created:
            s.commit()
    return created
