"""Shared ComfyTV asset and output roots."""

from __future__ import annotations

import pytest
import yaml


def test_asset_root_uses_the_input_directory(tmp_path, monkeypatch):
    import folder_paths
    from ComfyTV.storage.asset_root import (
        comfytv_asset_root,
        comfytv_output_root,
        project_dir,
        project_output_dir,
        series_dir,
    )

    monkeypatch.delenv("COMFYTV_ASSET_ROOT", raising=False)
    monkeypatch.delenv("COMFYTV_OUTPUT_ROOT", raising=False)
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path / "input"))
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path / "output"))

    assert comfytv_asset_root() == tmp_path / "input" / "comfytv"
    assert series_dir("gumu-sx") == tmp_path / "input" / "comfytv" / "series" / "gumu-sx"
    assert project_dir("gumu-shuangxiu") == (
        tmp_path / "input" / "comfytv" / "projects" / "gumu-shuangxiu"
    )
    assert comfytv_output_root() == tmp_path / "output" / "comfytv"
    assert project_output_dir("gumu-shuangxiu") == (
        tmp_path / "output" / "comfytv" / "projects" / "gumu-shuangxiu"
    )


def test_env_overrides_replace_the_comfytv_folder(tmp_path, monkeypatch):
    from ComfyTV.storage.asset_root import comfytv_asset_root, comfytv_output_root, series_dir

    monkeypatch.setenv("COMFYTV_ASSET_ROOT", str(tmp_path / "assets"))
    monkeypatch.setenv("COMFYTV_OUTPUT_ROOT", str(tmp_path / "renders"))
    assert comfytv_asset_root() == tmp_path / "assets"
    assert series_dir("gumu-sx") == tmp_path / "assets" / "series" / "gumu-sx"
    assert comfytv_output_root() == tmp_path / "renders"


def test_resolve_root_defaults_to_series_dir_and_follows_spark_root(tmp_path, monkeypatch):
    import folder_paths
    from ComfyTV.storage.series_library import SeriesLibraryError, resolve_root_arg

    monkeypatch.delenv("COMFYTV_ASSET_ROOT", raising=False)
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path / "input"))
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(tmp_path / "user"))
    series = tmp_path / "input" / "comfytv" / "series" / "gumu-sx"
    series.mkdir(parents=True)
    (series / "library.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "series": {"id": "gumu-sx"}, "assets": []}),
        encoding="utf-8",
    )
    assert resolve_root_arg(series="gumu-sx") == series.resolve()

    spark = tmp_path / "input" / "comfytv" / "series" / "spark-copy"
    spark.mkdir()
    (spark / "library.yaml").write_text("schema_version: 1\nseries: {id: gumu-sx}\nassets: []\n", encoding="utf-8")
    box = tmp_path / "box" / "gumu-sx"
    box.mkdir(parents=True)
    (box / "library.yaml").write_text(
        yaml.safe_dump({
            "schema_version": 1,
            "series": {"id": "gumu-sx", "spark_root": str(spark)},
            "assets": [],
        }),
        encoding="utf-8",
    )
    assert resolve_root_arg(root=str(box)) == spark.resolve()

    outside = tmp_path / "elsewhere" / "lib"
    outside.mkdir(parents=True)
    (outside / "library.yaml").write_text(
        "schema_version: 1\nseries: {id: other}\nassets: []\n",
        encoding="utf-8",
    )
    with pytest.raises(SeriesLibraryError, match="inside"):
        resolve_root_arg(root=str(outside))

    monkeypatch.setenv("COMFYTV_ASSET_ROOT", str(tmp_path / "custom"))
    assert resolve_root_arg(root=str(outside)) == outside.resolve()


def test_legacy_id_map_round_trips_and_resolves(tmp_path):
    from ComfyTV.storage.character_studio import resolve_character_name
    from ComfyTV.storage.series_library import get_series_asset, import_series_manifest

    root = tmp_path / "gumu-sx"
    body = b"front-map"
    rel = "CHR_xiaolongnv/v1/front.png"
    path = root / rel
    path.parent.mkdir(parents=True)
    path.write_bytes(body)
    legacy = {
        "front": "P01_xiaolongnv_base_outfitA_front_v3.png",
        "side": ["P01_xiaolongnv_side", "P01_xiaolongnv_profile"],
    }
    imported = import_series_manifest(root, {
        "schema_version": 1,
        "series": {"id": "gumu-sx"},
        "assets": [{
            "id": "CHR_xiaolongnv",
            "version": 1,
            "kind": "character",
            "status": "draft",
            "name_zh": "小龙女",
            "legacy_ids": legacy,
            "files": {"front": {"path": rel, "media_type": "image"}},
        }],
    })
    assert imported["added"] == ["CHR_xiaolongnv@v1"]
    stored = get_series_asset(root, "CHR_xiaolongnv@v1")["legacy_ids"]
    assert stored == legacy
    assert resolve_character_name(root, "P01_xiaolongnv_side")["id"] == "CHR_xiaolongnv"
    assert resolve_character_name(
        root, "P01_xiaolongnv_base_outfitA_front_v3.png",
    )["matched_by"] == "legacy_id"
