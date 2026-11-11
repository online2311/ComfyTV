"""HTTP access to a series folder's library.yaml index."""

from aiohttp import web

from ..storage.series_library import (
    SeriesLibraryError,
    bind_series_library,
    get_series_asset,
    import_series_manifest,
    list_series_assets,
    lock_series_asset,
    new_series_asset_version,
    register_series_asset,
    resolve_root_arg,
    resolve_series_ref,
    resolve_series_shot,
)
from ._common import routes


def _error(exc: SeriesLibraryError) -> web.Response:
    return web.json_response({"error": str(exc)}, status=exc.http_status)


async def _payload(request: web.Request) -> dict:
    if request.can_read_body:
        try:
            data = await request.json()
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _folder(request: web.Request, payload: dict):
    root = payload.get("root") or request.query.get("root")
    series = payload.get("series") or request.query.get("series")
    return resolve_root_arg(root, series)


@routes.get("/comfytv/series_library")
async def list_library(request: web.Request) -> web.Response:
    try:
        folder = _folder(request, {})
        return web.json_response(list_series_assets(
            folder,
            kind=request.query.get("kind") or None,
            status=request.query.get("status") or None,
            episode=request.query.get("episode") or None,
        ))
    except SeriesLibraryError as exc:
        return _error(exc)


@routes.get("/comfytv/series_library/asset")
async def get_library_asset(request: web.Request) -> web.Response:
    try:
        folder = _folder(request, {})
        code = request.query.get("code") or ""
        if not code:
            raise SeriesLibraryError("code is required")
        return web.json_response({"asset": get_series_asset(folder, code)})
    except SeriesLibraryError as exc:
        return _error(exc)


@routes.post("/comfytv/series_library/bind")
async def bind_library(request: web.Request) -> web.Response:
    try:
        payload = await _payload(request)
        root = payload.get("root")
        if not root:
            raise SeriesLibraryError("root is required")
        return web.json_response(bind_series_library(root))
    except SeriesLibraryError as exc:
        return _error(exc)


@routes.post("/comfytv/series_library/register")
async def register_library_asset(request: web.Request) -> web.Response:
    try:
        payload = await _payload(request)
        folder = _folder(request, payload)
        asset = payload.get("asset")
        if not isinstance(asset, dict):
            raise SeriesLibraryError("asset object is required")
        return web.json_response({"asset": register_series_asset(folder, asset)})
    except SeriesLibraryError as exc:
        return _error(exc)


@routes.post("/comfytv/series_library/lock")
async def lock_library_asset(request: web.Request) -> web.Response:
    try:
        payload = await _payload(request)
        folder = _folder(request, payload)
        code = str(payload.get("code") or "")
        if not code:
            raise SeriesLibraryError("code is required")
        asset = lock_series_asset(
            folder,
            code,
            approved_by=str(payload.get("approved_by") or ""),
            approved_at=payload.get("approved_at"),
            approval_ref=payload.get("approval_ref"),
        )
        return web.json_response({"asset": asset})
    except SeriesLibraryError as exc:
        return _error(exc)


@routes.post("/comfytv/series_library/new_version")
async def new_library_version(request: web.Request) -> web.Response:
    try:
        payload = await _payload(request)
        folder = _folder(request, payload)
        code = str(payload.get("code") or "")
        if not code:
            raise SeriesLibraryError("code is required")
        files = payload.get("files") if isinstance(payload.get("files"), dict) else None
        return web.json_response({
            "asset": new_series_asset_version(folder, code, files=files),
        })
    except SeriesLibraryError as exc:
        return _error(exc)


@routes.post("/comfytv/series_library/resolve")
async def resolve_library(request: web.Request) -> web.Response:
    try:
        payload = await _payload(request)
        folder = _folder(request, payload)
        require_locked = payload.get("require_locked", True)
        if not isinstance(require_locked, bool):
            require_locked = True
        if isinstance(payload.get("shot"), dict):
            return web.json_response(resolve_series_shot(
                folder, payload["shot"], require_locked=require_locked))
        ref = payload.get("ref")
        if not ref:
            raise SeriesLibraryError("ref or shot is required")
        return web.json_response({
            "resolved": resolve_series_ref(
                folder, str(ref), require_locked=require_locked,
                verify=bool(payload.get("verify")),
            ),
        })
    except SeriesLibraryError as exc:
        return _error(exc)


@routes.post("/comfytv/series_library/import")
async def import_library(request: web.Request) -> web.Response:
    try:
        payload = await _payload(request)
        folder = _folder(request, payload)
        manifest = payload.get("manifest") if isinstance(payload.get("manifest"), dict) else None
        manifest_path = payload.get("manifest_path")
        return web.json_response(import_series_manifest(
            folder,
            manifest,
            manifest_path=str(manifest_path) if manifest_path else None,
        ))
    except SeriesLibraryError as exc:
        return _error(exc)
