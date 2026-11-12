"""Asset root shared by series libraries and project directories."""

from __future__ import annotations


def test_asset_root_follows_the_env_override(tmp_path, monkeypatch):
    from ComfyTV.storage.asset_root import (
        comfytv_asset_root,
        project_dir,
        series_dir,
    )

    monkeypatch.setenv("COMFYTV_ASSET_ROOT", str(tmp_path / "assets"))
    assert comfytv_asset_root() == tmp_path / "assets"
    assert project_dir("gumu-shuangxiu") == (
        tmp_path / "assets" / "projects" / "gumu-shuangxiu"
    )
    assert series_dir("gumu-sx") == tmp_path / "assets" / "series" / "gumu-sx"


def test_asset_root_uses_the_comfyui_input_directory(tmp_path, monkeypatch):
    import folder_paths
    from ComfyTV.storage.asset_root import comfytv_asset_root

    monkeypatch.delenv("COMFYTV_ASSET_ROOT", raising=False)
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path / "input"))
    assert comfytv_asset_root() == tmp_path / "input" / "comfytv"


def test_output_root_follows_output_dir_or_env(tmp_path, monkeypatch):
    import folder_paths
    from ComfyTV.storage.asset_root import comfytv_output_root, project_output_dir

    monkeypatch.delenv("COMFYTV_OUTPUT_ROOT", raising=False)
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path / "output"))
    assert comfytv_output_root() == tmp_path / "output" / "comfytv"
    assert project_output_dir("gumu-shuangxiu") == (
        tmp_path / "output" / "comfytv" / "projects" / "gumu-shuangxiu"
    )
    monkeypatch.setenv("COMFYTV_OUTPUT_ROOT", str(tmp_path / "renders"))
    assert comfytv_output_root() == tmp_path / "renders"
    assert project_output_dir("demo") == tmp_path / "renders" / "projects" / "demo"
