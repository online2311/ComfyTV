"""Asset root shared by series libraries and project directories."""

from __future__ import annotations

import pytest


def test_asset_root_follows_the_env_override(tmp_path, monkeypatch):
    from ComfyTV.storage.asset_root import comfytv_asset_root, project_dir

    monkeypatch.setenv("COMFYTV_ASSET_ROOT", str(tmp_path / "assets"))
    assert comfytv_asset_root() == tmp_path / "assets"
    assert project_dir("gumu-shuangxiu") == tmp_path / "assets" / "projects" / "gumu-shuangxiu"


def test_asset_root_uses_the_comfyui_input_directory(tmp_path, monkeypatch):
    import folder_paths
    from ComfyTV.storage.asset_root import comfytv_asset_root

    monkeypatch.delenv("COMFYTV_ASSET_ROOT", raising=False)
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path / "input"))
    assert comfytv_asset_root() == tmp_path / "input" / "comfytv"


def test_project_dir_rejects_a_path(monkeypatch, tmp_path):
    from ComfyTV.storage.asset_root import project_dir

    monkeypatch.setenv("COMFYTV_ASSET_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="safe id"):
        project_dir("../series")
