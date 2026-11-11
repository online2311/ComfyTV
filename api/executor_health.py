"""Read-only health probes for registered executors.

Each pass GETs ``/system_stats`` and ``/queue``. It never posts, and it never
calls interrupt or free. An executor is idle only when ``queue_is_empty``
says both lists are empty. ``idle_since`` then stays put until a later probe
sees work or loses the host, which is what a RunPod reaper will measure.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from urllib.parse import urlsplit

import aiohttp

from .. import storage
from .queue_state import queue_is_empty

_log = logging.getLogger(__name__)

HEALTH_INTERVAL_S = 15.0
_PROBE_TIMEOUT_S = 4.0
_PROBE_CONCURRENCY = 8


class ProbeError(Exception):
    pass


def _assert_probe(method: str, url: str) -> None:
    if method != "GET":
        raise ProbeError("probe is read-only")
    path = urlsplit(url).path.rstrip("/")
    if not (path.endswith("/system_stats") or path.endswith("/queue")):
        raise ProbeError("probe is read-only")


async def _http_get(session: aiohttp.ClientSession, method: str, url: str):
    _assert_probe(method, url)
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                return resp.status, None
            try:
                body = await resp.json(content_type=None)
            except (aiohttp.ContentTypeError, ValueError):
                return resp.status, None
            return resp.status, body
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise ProbeError(type(exc).__name__) from None


def _gpu_snapshot(stats) -> dict:
    if not isinstance(stats, dict):
        return {"comfyui_version": "", "devices": []}
    system = stats.get("system") if isinstance(stats.get("system"), dict) else {}
    devices = []
    for device in stats.get("devices") or []:
        if not isinstance(device, dict):
            continue
        devices.append({
            "name": device.get("name") or "",
            "type": device.get("type") or "",
            "index": device.get("index"),
            "vram_total": device.get("vram_total"),
            "vram_free": device.get("vram_free"),
        })
    return {
        "comfyui_version": system.get("comfyui_version") or "",
        "devices": devices,
    }


def _parse_queue(body) -> tuple[list, list] | None:
    if not isinstance(body, dict):
        return None
    running = body.get("queue_running")
    pending = body.get("queue_pending")
    if not isinstance(running, list) or not isinstance(pending, list):
        return None
    return running, pending


def _failure_reason(exc: Exception) -> str:
    if isinstance(exc, ProbeError):
        text = str(exc)
        if "://" in text:
            return type(exc).__name__
        return text
    return type(exc).__name__


class ExecutorHealthChecker:
    def __init__(self, fetch=None, clock=None, interval_s: float = HEALTH_INTERVAL_S):
        self._fetch = fetch
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.interval_s = interval_s
        self._task: asyncio.Task | None = None

    def ensure_started(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._run(), name="comfytv-executor-health")

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self.check_all()
            except asyncio.CancelledError:
                raise
            except Exception:
                _log.exception("[ComfyTV/executors] health check failed")
            await asyncio.sleep(self.interval_s)

    async def check_all(self) -> list[dict]:
        rows = storage.list_executors(ensure_default=True)
        if self._fetch is not None:
            return await self._check_rows(self._fetch, rows)
        timeout = aiohttp.ClientTimeout(total=_PROBE_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async def fetch(method: str, url: str):
                return await _http_get(session, method, url)
            return await self._check_rows(fetch, rows)

    async def _check_rows(self, fetch, rows: list[dict]) -> list[dict]:
        sem = asyncio.Semaphore(_PROBE_CONCURRENCY)

        async def one(row: dict) -> dict:
            if not row.get("enabled", True):
                return row
            async with sem:
                return await self._check_one(fetch, row)

        return list(await asyncio.gather(*(one(row) for row in rows)))

    async def _check_one(self, fetch, row: dict) -> dict:
        observed_url = row["base_url"]
        generation = int(row.get("obs_generation") or 0)
        observation = await self._probe(fetch, observed_url)
        updated = storage.record_observation(
            row["id"],
            observed_url=observed_url,
            generation=generation,
            now=self._clock(),
            **observation,
        )
        return updated or row

    async def _probe(self, fetch, base_url: str) -> dict:
        base = str(base_url).rstrip("/")
        stats, queue = await asyncio.gather(
            self._read(fetch, base + "/system_stats"),
            self._read(fetch, base + "/queue"),
        )
        stats_ok, stats_body, stats_err = stats
        queue_ok, queue_body, queue_err = queue
        if not stats_ok or not queue_ok:
            _log.info(
                "[ComfyTV/executors] probe unreachable (%s)",
                stats_err or queue_err or "unreachable",
            )
            return {
                "health": "unreachable",
                "running_count": 0,
                "pending_count": 0,
                "queue_known": False,
                "queue_empty": False,
                "gpu": _gpu_snapshot(stats_body) if stats_ok else None,
                "error": stats_err or queue_err or "unreachable",
            }
        parsed = _parse_queue(queue_body)
        gpu = _gpu_snapshot(stats_body)
        if parsed is None:
            return {
                "health": "reachable",
                "running_count": 0,
                "pending_count": 0,
                "queue_known": False,
                "queue_empty": False,
                "gpu": gpu,
                "error": None,
            }
        running, pending = parsed
        return {
            "health": "reachable",
            "running_count": len(running),
            "pending_count": len(pending),
            "queue_known": True,
            "queue_empty": queue_is_empty(running, pending),
            "gpu": gpu,
            "error": None,
        }

    async def _read(self, fetch, url: str) -> tuple[bool, object, str | None]:
        try:
            status, body = await fetch("GET", url)
        except Exception as exc:
            return False, None, _failure_reason(exc)
        if status != 200:
            return False, None, f"HTTP {status}"
        return True, body, None


_CHECKER: ExecutorHealthChecker | None = None


def get_checker() -> ExecutorHealthChecker:
    global _CHECKER
    if _CHECKER is None:
        _CHECKER = ExecutorHealthChecker()
    return _CHECKER


def ensure_started() -> None:
    storage.absorb_configured_join_token()
    get_checker().ensure_started()
