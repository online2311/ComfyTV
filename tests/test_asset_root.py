"""Project records live under the ComfyUI input tree, not the user data dir."""

from pathlib import Path

import folder_paths


def test_default_root_is_the_input_comfytv_folder(monkeypatch):
    monkeypatch.delenv("COMFYTV_ASSET_ROOT", raising=False)
    from ComfyTV.storage.asset_root import comfytv_asset_root, project_dir

    root = comfytv_asset_root()
    assert root == Path(folder_paths.get_input_directory()) / "comfytv"
    assert project_dir("idol-mv-hit") == root / "projects" / "idol-mv-hit"


def test_env_replaces_the_input_folder(monkeypatch, tmp_path):
    override = tmp_path / "assets"
    monkeypatch.setenv("COMFYTV_ASSET_ROOT", str(override))
    from ComfyTV.storage.asset_root import comfytv_asset_root, project_dir

    assert comfytv_asset_root() == override
    assert project_dir("gumu") == override / "projects" / "gumu"


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
