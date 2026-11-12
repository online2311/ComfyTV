"""Project records live under the ComfyUI input tree, not the user data dir."""

from pathlib import Path

import folder_paths
import pytest
import yaml


def test_default_root_is_the_input_comfytv_folder(monkeypatch):
    monkeypatch.delenv("COMFYTV_ASSET_ROOT", raising=False)
    monkeypatch.delenv("COMFYTV_OUTPUT_ROOT", raising=False)
    from ComfyTV.storage.asset_root import (
        comfytv_asset_root,
        comfytv_output_root,
        project_dir,
        project_output_dir,
        series_dir,
    )

    root = comfytv_asset_root()
    assert root == Path(folder_paths.get_input_directory()) / "comfytv"
    assert project_dir("idol-mv-hit") == root / "projects" / "idol-mv-hit"
    assert series_dir("gumu-sx") == root / "series" / "gumu-sx"
    output = comfytv_output_root()
    assert output == Path(folder_paths.get_output_directory()) / "comfytv"
    assert project_output_dir("idol-mv-hit") == output / "projects" / "idol-mv-hit"


def test_env_replaces_the_input_and_output_folders(monkeypatch, tmp_path):
    assets = tmp_path / "assets"
    media = tmp_path / "media"
    monkeypatch.setenv("COMFYTV_ASSET_ROOT", str(assets))
    monkeypatch.setenv("COMFYTV_OUTPUT_ROOT", str(media))
    from ComfyTV.storage.asset_root import (
        comfytv_asset_root,
        comfytv_output_root,
        project_dir,
        project_output_dir,
        series_dir,
    )

    assert comfytv_asset_root() == assets
    assert project_dir("gumu") == assets / "projects" / "gumu"
    assert series_dir("gumu-sx") == assets / "series" / "gumu-sx"
    assert comfytv_output_root() == media
    assert project_output_dir("gumu") == media / "projects" / "gumu"


def test_reshoot_records_are_not_written_beside_the_database(reset_db, monkeypatch, tmp_path):
    from ComfyTV.backup import data_dir
    from ComfyTV.storage.segment_reshoot import plan

    root = tmp_path / "asset-root"
    monkeypatch.setenv("COMFYTV_ASSET_ROOT", str(root))
    source = Path(folder_paths.get_output_directory()) / "comfytv" / "reshoot-src" / "root-src.mp4"
    if not source.exists():
        from test_segment_reshoot import _write_clip
        _write_clip(source, seconds=2)
    audio = tmp_path / "tone.wav"
    from test_segment_reshoot import _write_tone
    _write_tone(audio, 0.4, 440)
    from ComfyTV.runners import media

    planned = plan({
        "project_id": "root-check",
        "mode": "audio_only",
        "previews": False,
        "source": {
            "url": media.path_to_view_url(source),
            "shot_id": "E01S01_01",
            "take": 1,
        },
        "t_start": 0.2,
        "t_end": 0.7,
        "audio": str(audio),
    })
    takes = root / "projects" / "root-check" / "reshoots" / "E01S01_01" / "takes.json"
    assert takes.is_file()
    assert planned["storage"]["records_dir"].endswith(
        "projects/root-check/reshoots/E01S01_01"
    )
    assert Path(data_dir()) not in takes.parents
    assert not (Path(data_dir()) / "projects" / "root-check").exists()


def test_resolve_root_defaults_to_series_dir_and_follows_spark_root(tmp_path, monkeypatch):
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
    (spark / "library.yaml").write_text(
        "schema_version: 1\nseries: {id: gumu-sx}\nassets: []\n",
        encoding="utf-8",
    )
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
    with pytest.raises(SeriesLibraryError, match="inside"):
        resolve_root_arg(root=str(outside))
    allowed = tmp_path / "custom" / "series" / "gumu-sx"
    allowed.mkdir(parents=True)
    (allowed / "library.yaml").write_text(
        "schema_version: 1\nseries: {id: gumu-sx}\nassets: []\n",
        encoding="utf-8",
    )
    assert resolve_root_arg(root=str(allowed)) == allowed.resolve()


def test_series_name_and_spark_root_cannot_leave_the_asset_root(tmp_path, monkeypatch):
    from ComfyTV.storage.series_library import SeriesLibraryError, resolve_root_arg

    monkeypatch.delenv("COMFYTV_ASSET_ROOT", raising=False)
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path / "input"))
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(tmp_path / "user"))
    for name in ("../projects/x", "../../etc", "", "/etc", "gumu/sx"):
        with pytest.raises(SeriesLibraryError, match="not safe"):
            resolve_root_arg(series=name)

    series = tmp_path / "input" / "comfytv" / "series" / "gumu-sx"
    series.mkdir(parents=True)
    outside = tmp_path / "etc"
    outside.mkdir()
    (series / "library.yaml").write_text(
        yaml.safe_dump({
            "schema_version": 1,
            "series": {"id": "gumu-sx", "spark_root": str(outside)},
            "assets": [],
        }),
        encoding="utf-8",
    )
    with pytest.raises(SeriesLibraryError, match="inside"):
        resolve_root_arg(series="gumu-sx")


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
