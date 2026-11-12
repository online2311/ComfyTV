"""Root directory for ComfyTV project and series files on disk.

The database and its backups stay in ``backup.data_dir`` (the ComfyUI user
directory). Video-generation assets use the ComfyUI input directory so series
libraries and per-project records share one tree:

``<input>/comfytv/projects/<project_id>/``
``<input>/comfytv/series/<series_id>/``

``COMFYTV_ASSET_ROOT`` replaces the ``<input>/comfytv`` folder when it is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import folder_paths


def comfytv_asset_root() -> Path:
    """``<input>/comfytv``, or ``COMFYTV_ASSET_ROOT`` when that env var is set."""
    override = os.environ.get("COMFYTV_ASSET_ROOT", "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path(folder_paths.get_input_directory()) / "comfytv"


def project_dir(project: str) -> Path:
    """Directory for one project's records, under the asset root."""
    return comfytv_asset_root() / "projects" / project
