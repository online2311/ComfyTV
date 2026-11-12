"""Root directories for ComfyTV project and series files.

Records and series libraries live under the ComfyUI input directory.
Finished media lives under the ComfyUI output directory. The database and
its backups stay in ``backup.data_dir`` and are not read here.

``COMFYTV_ASSET_ROOT`` replaces ``<input>/comfytv``.
``COMFYTV_OUTPUT_ROOT`` replaces ``<output>/comfytv``.
"""

from __future__ import annotations

import os
from pathlib import Path

import folder_paths


def _override(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return Path(os.path.expanduser(raw))


def comfytv_asset_root() -> Path:
    """``<input>/comfytv``, or ``COMFYTV_ASSET_ROOT`` when that env var is set."""
    override = _override("COMFYTV_ASSET_ROOT")
    if override is not None:
        return override
    return Path(folder_paths.get_input_directory()) / "comfytv"


def project_dir(project: str) -> Path:
    """``<asset root>/projects/<project>``."""
    return comfytv_asset_root() / "projects" / project


def series_dir(series: str) -> Path:
    """``<asset root>/series/<series>``."""
    return comfytv_asset_root() / "series" / series


def comfytv_output_root() -> Path:
    """``<output>/comfytv``, or ``COMFYTV_OUTPUT_ROOT`` when that env var is set."""
    override = _override("COMFYTV_OUTPUT_ROOT")
    if override is not None:
        return override
    return Path(folder_paths.get_output_directory()) / "comfytv"


def project_output_dir(project: str) -> Path:
    """``<output root>/projects/<project>``."""
    return comfytv_output_root() / "projects" / project
