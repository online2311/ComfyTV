"""Character studio records, alias resolution, and workflow assembly."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path

import pytest
import yaml

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "series_library" / "library.yaml"
IMAGE_VAE = "minimax_h3_image_vae_fp16.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"


def _dump(root: Path, doc: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "library.yaml").write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _write(root: Path, rel: str, body: bytes) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def _library(tmp_path: Path) -> Path:
    root = tmp_path / "gumu-sx"
    front = "CHR_xiaolongnv/v1/front.png"
    wav = "VOX_xiaolongnv/v1/voice_ref.wav"
    txt = "VOX_xiaolongnv/v1/voice_ref.txt"
    _write(root, front, b"front-bytes")
    _write(root, wav, b"RIFFvoice")
    (root / txt).write_text("参考句。", encoding="utf-8")
    _dump(root, {
        "schema_version": 1,
        "series": {
            "id": "gumu-sx",
            "code": "SER_gumu_sx",
            "project_id": "gumu-shuangxiu",
            "view_url_subfolder": "comfytv/series/gumu-sx",
        },
        "assets": [
            {
                "id": "CHR_xiaolongnv",
                "version": 1,
                "kind": "character",
                "status": "draft",
                "name_zh": "小龙女",
                "aliases": ["龙女"],
                "legacy_ids": ["P01_xiaolongnv_base_outfitA_front_v3.png"],
                "files": {
                    "front": {
                        "path": front,
                        "media_type": "image",
                        "usage": ["identity_video", "image_ref", "review"],
                    },
                },
            },
            {
                "id": "VOX_xiaolongnv",
                "version": 1,
                "kind": "voice",
                "status": "draft",
                "name_zh": "小龙女音色",
                "files": {
                    "voice_ref": {"path": wav, "media_type": "audio", "usage": ["voice_clone"]},
                    "voice_ref_txt": {"path": txt, "media_type": "text", "usage": ["text"]},
                },
            },
        ],
    })
    return root


def _index_only(tmp_path: Path) -> Path:
    root = tmp_path / "gumu-sx"
    root.mkdir()
    (root / "library.yaml").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    return root


def _conditioning(prompt: dict) -> list[dict]:
    return [
        node for node in prompt.values()
        if node.get("class_type") == "MiniMaxH3ImageConditioning"
    ]


def _history_ok(entry: dict, *, workflow: str) -> None:
    assert entry["inputs"]
    assert isinstance(entry["seed"], int)
    assert len(entry["prompt_hash"]) == 64
    assert entry["outputs"]
    assert entry["workflow"] == workflow
    assert entry["op_id"]


class TestAssetStorePolicy:
    def test_gumu_paths_and_excluded_project(self):
        from ComfyTV.storage.series_library import asset_store_policy

        policy = asset_store_policy()
        assert policy["truth"] == "comfytv"
        assert policy["box_is_backup"] is True
        project = policy["projects"]["gumu-shuangxiu"]
        assert project["series_id"] == "gumu-sx"
        assert project["series_code"] == "SER_gumu_sx"
        assert policy["series_root_template"] == (
            "/home/nodecloud/ComfyUI/input/comfytv/series/{series_id}"
        )
        assert policy["project_root_template"] == (
            "/home/nodecloud/ComfyUI/input/comfytv/projects/{project_id}"
        )
        assert "gumu-bixie" in policy["excluded_projects"]
        assert "candidate" in policy["statuses"]
        assert policy["lock_approver"] == "张景"


class TestAliasAndLegacy:
    def test_hit_miss_conflict_and_p01_without_rewriting(self, tmp_path):
        from ComfyTV.storage.series_library import SeriesLibraryError
        from ComfyTV.storage.character_studio import resolve_character_name

        root = _index_only(tmp_path)
        before = (root / "library.yaml").read_bytes()
        hit = resolve_character_name(root, "小龙女")
        assert hit["ref"] == "CHR_xiaolongnv@v1#front"
        assert hit["status"] == "locked"
        assert hit["matched_by"] == "name_zh"
        assert resolve_character_name(root, "P01_xiaolongnv")["id"] == "CHR_xiaolongnv"
        assert resolve_character_name(root, "CHR_xiaolongnv")["code"] == "CHR_xiaolongnv@v1"
        with pytest.raises(SeriesLibraryError, match="no character"):
            resolve_character_name(root, "P01_xiaolongnu")
        with pytest.raises(SeriesLibraryError, match="no character"):
            resolve_character_name(root, "不存在的人")
        assert (root / "library.yaml").read_bytes() == before

        doc = yaml.safe_load((root / "library.yaml").read_text(encoding="utf-8"))
        doc["assets"].append({
            "id": "CHR_longnv",
            "version": 1,
            "kind": "character",
            "status": "draft",
            "name_zh": "龙女乙",
            "aliases": ["小龙女"],
            "files": {},
        })
        _dump(root, doc)
        with pytest.raises(SeriesLibraryError, match="alias conflict") as conflict:
            resolve_character_name(root, "小龙女")
        assert set(conflict.value.details["conflicts"]) == {"CHR_xiaolongnv", "CHR_longnv"}

    def test_explicit_legacy_id_survives_when_pattern_is_off(self, tmp_path):
        from ComfyTV.storage.character_studio import resolve_character_name
        from ComfyTV.storage.series_library import SeriesLibraryError

        root = _library(tmp_path)
        doc = yaml.safe_load((root / "library.yaml").read_text(encoding="utf-8"))
        doc["comfytv"] = {"character_studio": {"accept_legacy_p_ids": False}}
        doc["assets"][0]["legacy_ids"].append("P01_xiaolongnv")
        _dump(root, doc)
        found = resolve_character_name(root, "P01_xiaolongnv")
        assert found["id"] == "CHR_xiaolongnv"
        assert found["matched_by"] == "legacy_id"
        assert found["ref"].endswith("#front")
        stored = yaml.safe_load((root / "library.yaml").read_text(encoding="utf-8"))
        assert stored["assets"][0]["id"] == "CHR_xiaolongnv"
        with pytest.raises(SeriesLibraryError):
            resolve_character_name(root, "P02_yangguo")

    def test_register_rejects_duplicate_alias(self, tmp_path):
        from ComfyTV.storage.series_library import (
            SeriesLibraryError,
            get_series_asset,
            register_series_asset,
        )

        root = _library(tmp_path)
        rel = "CHR_longnv/v1/front.png"
        _write(root, rel, b"other")
        with pytest.raises(SeriesLibraryError, match="alias conflict"):
            register_series_asset(root, {
                "id": "CHR_longnv",
                "version": 1,
                "kind": "character",
                "status": "draft",
                "name_zh": "另一个",
                "aliases": ["小龙女"],
                "files": {"front": {"path": rel, "media_type": "image"}},
            })
        with pytest.raises(SeriesLibraryError):
            get_series_asset(root, "CHR_longnv@v1")


class TestCandidateImport:
    def _manifest(self, body: bytes, **asset_overrides) -> dict:
        digest = hashlib.sha256(body).hexdigest()
        asset = {
            "id": "CHR_yangguo",
            "version": 1,
            "kind": "character",
            "status": "candidate",
            "name_zh": "杨过",
            "legacy_ids": ["P02_yangguo"],
            "picked_by": "视频制作助手",
            "approved_by": "视频制作助手",
            "files": {
                "front": {
                    "path": "CHR_yangguo/v1/front.png",
                    "sha256": digest,
                    "bytes": len(body),
                    "media_type": "image",
                    "usage": ["identity_video", "image_ref", "review"],
                },
            },
        }
        asset.update(asset_overrides)
        return {
            "schema_version": 1,
            "series": {"id": "gumu-sx", "project_id": "gumu-shuangxiu"},
            "file_count": 1,
            "require_sha256": True,
            "assets": [asset],
        }

    def test_candidate_stays_unlocked_and_keeps_legacy_ids(self, tmp_path):
        from ComfyTV.storage.series_library import (
            SeriesLibraryError,
            get_series_asset,
            import_series_manifest,
            lock_series_asset,
        )

        root = tmp_path / "gumu-sx"
        body = b"yangguo-picked"
        _write(root, "CHR_yangguo/v1/front.png", body)
        imported = import_series_manifest(root, self._manifest(body))
        assert imported["added"] == ["CHR_yangguo@v1"]
        assert imported["file_count"] == 1
        assert imported["sha256_checked"] == 1
        asset = get_series_asset(root, "CHR_yangguo@v1")
        assert asset["status"] == "candidate"
        assert "approved_by" not in asset
        assert asset["picked_by"] == "视频制作助手"
        assert asset["legacy_ids"] == ["P02_yangguo"]
        assert asset["files"]["front"]["sha256"] == hashlib.sha256(body).hexdigest()
        assert (root / "CHR_yangguo/v1/front.png").stat().st_mode & stat.S_IWUSR

        with pytest.raises(SeriesLibraryError, match="张景"):
            lock_series_asset(root, "CHR_yangguo@v1", approved_by="视频制作助手")
        locked = lock_series_asset(root, "CHR_yangguo@v1", approved_by="张景")
        assert locked["status"] == "locked"
        assert locked["approved_by"] == "张景"
        assert (root / "CHR_yangguo/v1/front.png").stat().st_mode & stat.S_IWUSR == 0

    def test_manifest_sha_and_file_count_fail_before_writing_the_index(self, tmp_path):
        from ComfyTV.storage.series_library import SeriesLibraryError, import_series_manifest

        root = tmp_path / "fresh"
        body = b"bytes"
        _write(root, "CHR_yangguo/v1/front.png", body)
        bad_count = self._manifest(body)
        bad_count["file_count"] = 4
        with pytest.raises(SeriesLibraryError, match="file_count"):
            import_series_manifest(root, bad_count)
        assert not (root / "library.yaml").exists()

        missing_sha = self._manifest(body)
        missing_sha["assets"][0]["files"]["front"].pop("sha256")
        with pytest.raises(SeriesLibraryError, match="sha256 required"):
            import_series_manifest(root, missing_sha)
        assert not (root / "library.yaml").exists()

        wrong_sha = self._manifest(body)
        wrong_sha["assets"][0]["files"]["front"]["sha256"] = "a" * 64
        with pytest.raises(SeriesLibraryError, match="sha256 mismatch"):
            import_series_manifest(root, wrong_sha)
        assert not (root / "library.yaml").exists()

        wrong_bytes = self._manifest(body)
        wrong_bytes["assets"][0]["files"]["front"]["bytes"] = 1
        with pytest.raises(SeriesLibraryError, match="byte size"):
            import_series_manifest(root, wrong_bytes)
        assert not (root / "library.yaml").exists()

    def test_locked_import_requires_zhangjing_and_bixie_is_excluded(self, tmp_path):
        from ComfyTV.storage.series_library import SeriesLibraryError, import_series_manifest

        root = tmp_path / "gumu-sx"
        body = b"locked-face"
        _write(root, "CHR_yangguo/v1/front.png", body)
        manifest = self._manifest(body, status="locked", approved_by="视频制作助手")
        with pytest.raises(SeriesLibraryError, match="张景"):
            import_series_manifest(root, manifest)
        assert not (root / "library.yaml").exists()

        excluded = self._manifest(body)
        excluded["project"] = "gumu-bixie"
        with pytest.raises(SeriesLibraryError, match="gumu-bixie"):
            import_series_manifest(root, excluded)
        excluded_series = self._manifest(body)
        excluded_series["series"] = {"id": "gumu-bixie"}
        with pytest.raises(SeriesLibraryError, match="excluded"):
            import_series_manifest(tmp_path / "other", excluded_series)

    def test_alias_only_files_do_not_count_toward_file_count(self, tmp_path):
        from ComfyTV.storage.series_library import SeriesLibraryError, import_series_manifest

        root = tmp_path / "gumu-sx"
        body = b"front"
        _write(root, "CHR_yangguo/v1/front.png", body)
        manifest = self._manifest(body)
        manifest["assets"][0]["files"]["headshot"] = {"alias_of": "CHR_yangguo@v1#front"}
        manifest["file_count"] = 2
        with pytest.raises(SeriesLibraryError, match="file_count"):
            import_series_manifest(root, manifest)
        manifest["file_count"] = 1
        imported = import_series_manifest(root, manifest)
        assert imported["file_count"] == 1


class TestLockDependencies:
    def test_lock_lists_unlocked_dependencies_then_accepts_zhangjing(self, tmp_path):
        from ComfyTV.storage.character_studio import get_character
        from ComfyTV.storage.series_library import (
            SeriesLibraryError,
            lock_series_asset,
            register_series_asset,
            update_draft_series_asset,
        )

        root = _library(tmp_path)
        outfit = "OUT_baiyi/v1/front.png"
        _write(root, outfit, b"robe")
        register_series_asset(root, {
            "id": "OUT_baiyi",
            "version": 1,
            "kind": "outfit",
            "status": "draft",
            "files": {"front": {"path": outfit, "media_type": "image"}},
        })
        update_draft_series_asset(root, "CHR_xiaolongnv@v1", {
            "wardrobe": [{"outfit": "OUT_baiyi@v1", "look": None, "default": True}],
            "voice": "VOX_xiaolongnv@v1",
        })
        report = get_character(root, "小龙女")["dependencies"]
        assert {row["code"] for row in report} == {"OUT_baiyi@v1", "VOX_xiaolongnv@v1"}
        assert any(row["status"] == "draft" and not row["locked"] for row in report)
        with pytest.raises(SeriesLibraryError, match="OUT_baiyi@v1"):
            lock_series_asset(root, "CHR_xiaolongnv@v1", approved_by="张景")
        lock_series_asset(root, "OUT_baiyi@v1", approved_by="张景")
        lock_series_asset(root, "VOX_xiaolongnv@v1", approved_by="张景")
        locked = lock_series_asset(root, "CHR_xiaolongnv@v1", approved_by="张景")
        assert locked["status"] == "locked"


class TestStudioHistoryAndWorkflows:
    def test_every_op_records_history_and_h3_wiring(self, tmp_path):
        from ComfyTV.storage.character_studio import (
            bind_voice,
            compose_outfit,
            derive_views,
            diff_characters,
            get_character,
            import_face,
            load_studio_config,
            make_sheet,
            refine,
            video_reference_views,
            voice_audition,
        )
        from ComfyTV.storage.series_library import (
            SeriesLibraryError,
            get_series_asset,
            lock_series_asset,
            read_series_document,
        )

        root = _library(tmp_path)
        face = tmp_path / "face.png"
        face.write_bytes(b"face-source")
        imported = import_face(
            root, "CHR_xiaolongnv@v1", [str(face)], "ai_generated", note="studio",
        )
        _history_ok(imported["studio_history"], workflow="import")
        assert imported["asset"]["face_source"][0]["provenance"] == "ai_generated"
        assert imported["asset"]["face_source"][0]["sha256"] == hashlib.sha256(b"face-source").hexdigest()

        clothes = tmp_path / "clothes.png"
        clothes.write_bytes(b"clothes")
        composed = compose_outfit(
            root, imported["code"], "CHR_xiaolongnv@v1#front", str(clothes),
            batch=2, seed=11,
        )
        entry = composed["studio_history"]
        _history_ok(entry, workflow="h3_image")
        assert entry["seed"] == 11
        assert entry["op"] == "compose_outfit"
        prompt = composed["prompt"]
        conds = _conditioning(prompt)
        assert len(conds) == 1
        text = conds[0]["inputs"]["prompt"]
        assert text.startswith("<Picture 2>中的角色穿上<Picture 1>的衣服")
        assert "保持<Picture 1>的身材" in text
        assert entry["prompt_hash"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
        image_1 = prompt[conds[0]["inputs"]["image_1"][0]]
        image_2 = prompt[conds[0]["inputs"]["image_2"][0]]
        assert image_1["inputs"]["image"].endswith("src_outfit.png")
        assert image_1["inputs"]["image"].startswith("comfytv/series/gumu-sx/")
        assert image_2["inputs"]["image"].endswith("front.png")
        assert {int(item) for item in re.findall(r"<Picture (\d+)>", text)} == {1, 2}
        vae_id = conds[0]["inputs"]["vae"][0]
        assert prompt[vae_id]["class_type"] == "VAELoader"
        assert prompt[vae_id]["inputs"]["vae_name"] == IMAGE_VAE
        decoded = json.dumps(prompt)
        assert VIDEO_VAE not in decoded
        assert "MiniMaxH3ImageToVideo" not in decoded
        assert "Lora" not in decoded
        assert len([node for node in prompt.values() if node["class_type"] == "SaveImage"]) == 2
        assert [node["inputs"]["seed"] for node in prompt.values() if node["class_type"] == "KSampler"] == [11, 12]

        swapped = compose_outfit(
            root, composed["code"], "CHR_xiaolongnv@v1#front", str(clothes),
            batch=2, seed=3, image_1="face",
        )
        swapped_text = _conditioning(swapped["prompt"])[0]["inputs"]["prompt"]
        assert swapped_text.startswith("<Picture 1>中的角色穿上<Picture 2>的衣服")

        with pytest.raises(SeriesLibraryError, match="batch"):
            compose_outfit(
                root, composed["code"], "CHR_xiaolongnv@v1#front", str(clothes), batch=1,
            )

        sheet = make_sheet(root, composed["code"], seed=5)
        sheet_text = _conditioning(sheet["prompt"])[0]["inputs"]["prompt"]
        assert "二八身材比例" in sheet_text
        assert "<Picture 1>" in sheet_text
        _history_ok(sheet["studio_history"], workflow="h3_image")
        assert VIDEO_VAE not in json.dumps(sheet["prompt"])

        derived = derive_views(root, composed["code"], ["headshot", "side"], seed=8)
        derived_conds = _conditioning(derived["prompt"])
        assert len(derived_conds) == 2
        for cond in derived_conds:
            assert "<Picture 1>" in cond["inputs"]["prompt"]
            assert "<Picture 2>" not in cond["inputs"]["prompt"]
            assert cond["inputs"]["image_1"][0] in derived["prompt"]
            loaded = derived["prompt"][cond["inputs"]["image_1"][0]]["inputs"]["image"]
            assert loaded.endswith("front.png")
        _history_ok(derived["studio_history"], workflow="h3_image")
        assert derived["studio_history"]["prompt_hash"] == hashlib.sha256(
            "\n".join(cond["inputs"]["prompt"] for cond in derived_conds).encode("utf-8")
        ).hexdigest()

        audition = voice_audition(
            root, "VOX_xiaolongnv@v1", "月色真好。", [7, 8, 9], desc_voice_en="cool",
        )
        _history_ok(audition["studio_history"], workflow="s2pro")
        assert audition["studio_history"]["seed"] == 7
        classes = [node["class_type"] for node in audition["prompt"].values()]
        assert classes.count("ComfyTV.S2Pro") == 3
        assert "MiniMaxH3ImageConditioning" not in classes
        spoken = [
            node["inputs"]["seed"]
            for node in audition["prompt"].values()
            if node["class_type"] == "ComfyTV.S2Pro"
        ]
        assert spoken == [7, 8, 9]
        assert get_series_asset(root, audition["code"])["desc_voice_en"] == "cool"
        with pytest.raises(SeriesLibraryError, match="3-5"):
            voice_audition(root, audition["code"], "再一句。", [1, 2])

        bound = bind_voice(root, composed["code"], "VOX_xiaolongnv@v1")
        _history_ok(bound["studio_history"], workflow="bind")
        assert bound["asset"]["voice"] == "VOX_xiaolongnv@v1"

        profile = get_character(root, "龙女")
        assert profile["ref"] == "CHR_xiaolongnv@v1#front"
        assert all("#sheet" not in ref for ref in profile["video_refs"])
        assert profile["sheet_in_video_ref"] is False
        ops = [item["op"] for item in profile["asset"]["studio_history"]]
        assert ops[:5] == [
            "import_face", "compose_outfit", "compose_outfit", "sheet", "derive_view",
        ]

        original = (root / "CHR_xiaolongnv/v1/front.png").read_bytes()
        lock_series_asset(root, "VOX_xiaolongnv@v1", approved_by="张景")
        locked = lock_series_asset(root, "CHR_xiaolongnv@v1", approved_by="张景")
        assert locked["files"]["front"]["sha256"]
        refined = refine(root, "CHR_xiaolongnv@v1", "hair", "改成高马尾", seed=21)
        assert refined["code"] == "CHR_xiaolongnv@v2"
        assert refined["asset"]["status"] == "draft"
        assert (root / "CHR_xiaolongnv/v1/front.png").read_bytes() == original
        untouched = get_series_asset(root, "CHR_xiaolongnv@v1")
        assert untouched["files"]["front"]["sha256"] == locked["files"]["front"]["sha256"]
        assert untouched.get("studio_history") == locked.get("studio_history")
        _history_ok(refined["studio_history"], workflow="h3_image")
        assert refined["studio_history"]["op"] == "refine_hair"
        assert refined["asset"]["needs_rederive"] == [
            "headshot", "three_quarter", "side", "back",
        ]
        root_image = _conditioning(refined["prompt"])[0]["inputs"]["image_1"]
        assert refined["prompt"][root_image[0]]["inputs"]["image"].endswith("root.png")
        assert (root / "CHR_xiaolongnv/v2/root.png").read_bytes() == original
        difference = diff_characters(root, "CHR_xiaolongnv@v1", refined["code"])
        assert any(row["field"] == "status" for row in difference["fields"])

        config = load_studio_config(read_series_document(root))
        assert "sheet" not in video_reference_views(refined["asset"], config)

    def test_refine_keeps_the_same_root_and_sheet_stays_out_of_video_slots(self, tmp_path):
        from ComfyTV.storage.character_studio import (
            load_studio_config,
            pick_candidate,
            refine,
            video_reference_views,
        )
        from ComfyTV.storage.series_library import (
            get_series_asset,
            read_series_document,
            update_draft_series_asset,
        )

        root = _library(tmp_path)
        original = (root / "CHR_xiaolongnv/v1/front.png").read_bytes()
        first = refine(root, "CHR_xiaolongnv@v1", "face", "眉心一点红", seed=4)
        chosen = tmp_path / "chosen.png"
        chosen.write_bytes(b"new-front")
        record = get_series_asset(root, first["code"])
        history = record["studio_history"]
        history[-1]["outputs"] = [str(chosen)]
        update_draft_series_asset(root, first["code"], {"studio_history": history})
        picked = pick_candidate(root, first["code"], history[-1]["op_id"], 1, "front")
        assert picked["sha256"] == hashlib.sha256(b"new-front").hexdigest()
        assert (root / "CHR_xiaolongnv/v1/front.png").read_bytes() == b"new-front"
        assert (root / "CHR_xiaolongnv/v1/root.png").read_bytes() == original

        second = refine(root, first["code"], "face", "再改一次眉", seed=6)
        image = _conditioning(second["prompt"])[0]["inputs"]["image_1"]
        assert second["prompt"][image[0]]["inputs"]["image"].endswith("root.png")
        assert (root / "CHR_xiaolongnv/v1/root.png").read_bytes() == original

        sheet_bytes = b"sheet-card"
        sheet_file = tmp_path / "sheet.png"
        sheet_file.write_bytes(sheet_bytes)
        history = get_series_asset(root, first["code"])["studio_history"]
        history[-1]["outputs"] = [str(sheet_file)]
        update_draft_series_asset(root, first["code"], {"studio_history": history})
        saved = pick_candidate(root, first["code"], history[-1]["op_id"], 1, "sheet")
        assert saved["asset"]["files"]["sheet"]["usage"] == ["review"]
        config = load_studio_config(read_series_document(root))
        assert "sheet" not in video_reference_views(saved["asset"], config)

        opened = read_series_document(root)
        opened.setdefault("comfytv", {})["character_studio"] = {"sheet_in_video_ref": True}
        _dump(root, opened)
        enabled = load_studio_config(read_series_document(root))
        fresh = get_series_asset(root, first["code"])
        assert "sheet" in video_reference_views(fresh, enabled)


class TestMcpQueue:
    async def test_compose_returns_job_id_without_blocking(self, tmp_path, monkeypatch):
        import ComfyTV.api.mcp_tools.character_studio as tool
        from ComfyTV.storage.series_library import get_series_asset

        root = _library(tmp_path)
        clothes = tmp_path / "clothes.png"
        clothes.write_bytes(b"clothes")
        seen = {}

        async def fake_enqueue(args):
            seen["prompt"] = args["prompt"]
            return {"job_id": "job-9", "status": "queued"}

        monkeypatch.setattr(tool, "enqueue_graph", fake_enqueue)
        out = await tool._character_studio({
            "action": "compose_outfit",
            "root": str(root),
            "code": "CHR_xiaolongnv@v1",
            "face": "CHR_xiaolongnv@v1#front",
            "outfit_body": str(clothes),
            "batch": 2,
            "seed": 4,
        })
        assert out["job_id"] == "job-9"
        assert out["status"] == "queued"
        assert seen["prompt"]["3"]["inputs"]["vae_name"] == IMAGE_VAE
        stored = get_series_asset(root, out["code"])
        assert stored["studio_history"][-1]["prompt_id"] == "job-9"

        from ComfyTV.storage.series_library import SeriesLibraryError

        with pytest.raises(SeriesLibraryError, match="no character"):
            await tool._character_studio({
                "action": "resolve_alias",
                "root": str(root),
                "name": "没有这个人",
            })
