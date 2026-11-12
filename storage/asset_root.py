"""ComfyTV asset root under ComfyUI's input directory.

Series libraries live at ``<root>/series/<series_id>``. Project files,
including scripts, live at ``<root>/projects/<project>``. ``COMFYTV_ASSET_ROOT``
replaces ``<ComfyUI input>/comfytv`` when it is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import folder_paths

_TOKEN_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")


def comfytv_asset_root() -> Path:
    override = os.environ.get("COMFYTV_ASSET_ROOT", "").strip()
    if override:
        return Path(override)
    return Path(folder_paths.get_input_directory()) / "comfytv"


def project_dir(project: str) -> Path:
    return comfytv_asset_root() / "projects" / _token(project)


def _token(project: str) -> str:
    text = str(project or "").strip()
    if not text or text in {".", ".."} or any(char not in _TOKEN_CHARS for char in text):
        raise ValueError("project is not a safe id")
    return text
