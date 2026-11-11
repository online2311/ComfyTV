import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func, or_, select

from .. import db
from ..db import HeadlessGraph, HeadlessJob

_UNSET = object()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def utcnow() -> datetime:
    return _now()


def parse_utc_timestamp(value) -> datetime:
    """Read a job timestamp. Naive values are UTC, matching rows stored before offsets were emitted."""
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_utc_timestamp(value) -> str | None:
    """ISO-8601 with a numeric offset. Empty values stay null."""
    if value is None or value == "":
        return None
    return parse_utc_timestamp(value).isoformat()


def _loads(raw, fallback):
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def _graph_to_dict(row: HeadlessGraph) -> dict:
    return {
        "project_id": row.project_id,
        "document": _loads(row.document_json, {}),
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _job_to_dict(row: HeadlessJob) -> dict:
    return {
        "id": row.id,
        "seq": row.seq,
        "project_id": row.project_id,
        "kind": row.kind,
        "status": row.status,
        "prompt_json": row.prompt_json or "{}",
        "partial_json": row.partial_json,
        "prompt_id": row.prompt_id,
        "stage_ref": row.stage_ref,
        "stage_uid": row.stage_uid,
        "error_text": row.error_text,
        "outputs_json": row.outputs_json,
        "progress_json": row.progress_json,
        "completed_json": row.completed_json,
        "attempt": row.attempt or 1,
        "created_at": format_utc_timestamp(row.created_at),
        "updated_at": format_utc_timestamp(row.updated_at),
        "started_at": format_utc_timestamp(row.started_at),
        "finished_at": format_utc_timestamp(row.finished_at),
    }


def get_headless_graph(project_id: str) -> Optional[dict]:
    with db.get_session() as s:
        row = s.get(HeadlessGraph, project_id)
        return _graph_to_dict(row) if row else None


def save_headless_graph(project_id: str, document: dict) -> dict:
    payload = json.dumps(document, ensure_ascii=False)
    with db.get_session() as s:
        row = s.get(HeadlessGraph, project_id)
        if row is None:
            row = HeadlessGraph(project_id=project_id, document_json=payload)
            s.add(row)
        else:
            row.document_json = payload
        row.updated_at = _now()
        s.commit()
        return _graph_to_dict(row)


def create_headless_job(
    *,
    project_id: str,
    kind: str,
    prompt: dict,
    partial: Optional[list] = None,
    stage_ref: Optional[str] = None,
    stage_uid: Optional[str] = None,
    completed: Optional[dict] = None,
    attempt: int = 1,
) -> dict:
    with db.get_session() as s:
        current = s.execute(select(func.max(HeadlessJob.seq))).scalar()
        row = HeadlessJob(
            id=uuid.uuid4().hex,
            seq=int(current or 0) + 1,
            project_id=project_id or "default",
            kind=kind,
            status="queued",
            prompt_json=json.dumps(prompt, ensure_ascii=False),
            partial_json=None if partial is None else json.dumps(list(partial)),
            stage_ref=stage_ref,
            stage_uid=stage_uid,
            completed_json=None if not completed else json.dumps(completed),
            attempt=int(attempt or 1),
        )
        s.add(row)
        s.commit()
        return _job_to_dict(row)


def update_headless_job(job_id: str, **fields) -> Optional[dict]:
    allowed = {
        "status", "prompt_id", "error_text", "outputs_json", "progress_json",
        "completed_json", "partial_json", "attempt", "clear_prompt_id",
        "started_at", "finished_at", "clear_started_at", "clear_finished_at",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"unknown headless job fields: {sorted(unknown)}")
    with db.get_session() as s:
        row = s.get(HeadlessJob, job_id)
        if row is None:
            return None
        if "status" in fields and fields["status"] is not None:
            row.status = fields["status"]
        if fields.get("clear_prompt_id"):
            row.prompt_id = None
        elif "prompt_id" in fields:
            row.prompt_id = fields["prompt_id"] or None
        if "error_text" in fields:
            row.error_text = fields["error_text"]
        if "outputs_json" in fields:
            row.outputs_json = fields["outputs_json"]
        if "progress_json" in fields:
            row.progress_json = fields["progress_json"]
        if "completed_json" in fields:
            row.completed_json = fields["completed_json"]
        if "partial_json" in fields:
            row.partial_json = fields["partial_json"]
        if "attempt" in fields and fields["attempt"] is not None:
            row.attempt = int(fields["attempt"])
        if fields.get("clear_started_at"):
            row.started_at = None
        elif "started_at" in fields:
            row.started_at = fields["started_at"]
        if fields.get("clear_finished_at"):
            row.finished_at = None
        elif "finished_at" in fields:
            row.finished_at = fields["finished_at"]
        row.updated_at = _now()
        s.commit()
        return _job_to_dict(row)


def get_headless_job(job_id: str) -> Optional[dict]:
    with db.get_session() as s:
        row = s.get(HeadlessJob, job_id)
        return _job_to_dict(row) if row else None


def headless_job_by_prompt(prompt_id: str) -> Optional[dict]:
    with db.get_session() as s:
        row = s.execute(
            select(HeadlessJob)
            .where(HeadlessJob.prompt_id == prompt_id)
            .order_by(HeadlessJob.seq.desc())
            .limit(1)
        ).scalar_one_or_none()
        return _job_to_dict(row) if row else None


def claim_headless_job(job_id: str, *, expect: str, become: str,
                       prompt_id: Any = _UNSET,
                       error_text: Any = _UNSET,
                       started_at: Any = _UNSET,
                       finished_at: Any = _UNSET,
                       clear_started_at: bool = False,
                       clear_finished_at: bool = False) -> Optional[dict]:
    """Move a job from ``expect`` to ``become`` in one transaction.

    Returns None when the row is missing or no longer in ``expect``, so a
    cancel and a submit cannot both win.
    """
    with db.get_session() as s:
        row = s.get(HeadlessJob, job_id)
        if row is None or row.status != expect:
            return None
        row.status = become
        if prompt_id is not _UNSET:
            row.prompt_id = prompt_id or None
        if error_text is not _UNSET:
            row.error_text = error_text
        if clear_started_at:
            row.started_at = None
        elif started_at is not _UNSET:
            row.started_at = started_at
        if clear_finished_at:
            row.finished_at = None
        elif finished_at is not _UNSET:
            row.finished_at = finished_at
        row.updated_at = _now()
        s.commit()
        return _job_to_dict(row)


def next_queued_headless_job() -> Optional[dict]:
    with db.get_session() as s:
        row = s.execute(
            select(HeadlessJob)
            .where(HeadlessJob.status == "queued")
            .order_by(HeadlessJob.seq.asc())
            .limit(1)
        ).scalar_one_or_none()
        return _job_to_dict(row) if row else None


def list_headless_jobs(
    status: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    with db.get_session() as s:
        q = select(HeadlessJob)
        if status:
            q = q.where(HeadlessJob.status == status)
        if project_id:
            q = q.where(HeadlessJob.project_id == project_id)
        q = q.order_by(HeadlessJob.seq.desc()).limit(max(1, min(int(limit), 200)))
        return [_job_to_dict(row) for row in s.execute(q).scalars().all()]


def latest_headless_job_for_stage(project_id: str, ref: str) -> Optional[dict]:
    ref = str(ref)
    with db.get_session() as s:
        conds = [
            HeadlessJob.stage_ref == ref,
            HeadlessJob.stage_uid == ref,
        ]
        if len(ref) >= 8:
            conds.append(HeadlessJob.stage_uid.startswith(ref))
            conds.append(HeadlessJob.stage_ref.startswith(ref))
        row = s.execute(
            select(HeadlessJob)
            .where(HeadlessJob.project_id == project_id)
            .where(HeadlessJob.kind == "stage")
            .where(or_(*conds))
            .order_by(HeadlessJob.seq.desc())
            .limit(1)
        ).scalar_one_or_none()
        return _job_to_dict(row) if row else None
