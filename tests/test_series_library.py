"""Series library.yaml index: read the production format, lock, version, resolve."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "series_library" / "library.yaml"
MANIFEST = Path(__file__).resolve().parent / "fixtures" / "series_library" / "import_manifest.json"


def _user_dir(tmp_path, monkeypatch) -> Path:
    user = tmp_path / "user" / "default"
    user.mkdir(parents=True, exist_ok=True)
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(user))
    return user


def materialize(dest: Path, source: Path = FIXTURE) -> Path:
    """Copy library.yaml and write one stub file per recorded path."""
    dest.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8")
    (dest / "library.yaml").write_text(text, encoding="utf-8")
    doc = yaml.safe_load(text)
    for asset in doc["assets"]:
        for view, info in (asset.get("files") or {}).items():
            rel = (info or {}).get("path")
            if not rel:
                continue
            path = dest / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{asset['id']}:{view}".encode())
    return dest


def _write_stub(root: Path, rel: str, body: bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


class TestParseRef:
    def test_code_with_version_and_view(self):
        from ComfyTV.storage.series_library import parse_series_ref

        parsed = parse_series_ref("  {CHR_xiaolongnv@v1#front}  ")
        assert parsed["kind"] == "asset"
        assert parsed["id"] == "CHR_xiaolongnv"
        assert parsed["version"] == 1
        assert parsed["view"] == "front"

    def test_episode_and_new_and_camera_card(self):
        from ComfyTV.storage.series_library import parse_series_ref

        episode = parse_series_ref("episode:E01S01_01/first.png")
        assert episode["kind"] == "episode"
        assert episode["path"] == "E01S01_01/first.png"
        new = parse_series_ref("NEW:LOC_linzhaoying_shishi")
        assert new["kind"] == "unresolved"
        assert new["proposed_id"] == "LOC_linzhaoying_shishi"
        camera = parse_series_ref("LOC_huacong__W_quan@v2#empty")
        assert camera["id"] == "LOC_huacong__W_quan"
        assert camera["version"] == 2
        assert camera["view"] == "empty"

    def test_bare_placeholder_is_not_an_asset_code(self):
        from ComfyTV.storage.series_library import parse_series_ref

        assert parse_series_ref("{LOC}")["kind"] == "invalid"
        assert parse_series_ref("小龙女")["kind"] == "invalid"


class TestReadFixture:
    def test_yaml_anchors_load_as_shared_approval(self, tmp_path):
        from ComfyTV.storage.series_library import get_series_asset

        root = tmp_path / "anchored"
        (root / "CHR_a" / "v1").mkdir(parents=True)
        (root / "CHR_a" / "v1" / "front.png").write_bytes(b"a")
        (root / "library.yaml").write_text(
            "schema_version: 1\n"
            "series: {id: demo}\n"
            "assets:\n"
            "- id: CHR_a\n"
            "  version: 1\n"
            "  kind: character\n"
            "  status: locked\n"
            "  approved_by: 张景\n"
            "  files:\n"
            "    front:\n"
            "      path: CHR_a/v1/front.png\n"
            "      approval: &id001\n"
            "        by: 张景\n"
            "    headshot:\n"
            "      path: CHR_a/v1/front.png\n"
            "      approval: *id001\n",
            encoding="utf-8",
        )
        got = get_series_asset(root, "CHR_a@v1")
        assert got["files"]["front"]["approval"] == {"by": "张景"}
        assert got["files"]["headshot"]["approval"] == got["files"]["front"]["approval"]

    def test_lists_locked_codes_without_rewriting(self, tmp_path):
        from ComfyTV.storage.series_library import list_series_assets

        root = materialize(tmp_path / "gumu-sx")
        before = (root / "library.yaml").read_bytes()
        listed = list_series_assets(root)
        assert listed["series"]["id"] == "gumu-sx"
        codes = {(row["id"], row["kind"], row["status"]) for row in listed["assets"]}
        assert ("CHR_xiaolongnv", "character", "locked") in codes
        assert ("LOOK_xiaolongnv_baiyi_opened", "look", "locked") in codes
        assert ("LOC_gumu_shishi", "location", "locked") in codes
        assert ("PRP_zhutai", "prop", "locked") in codes
        assert ("STY_01", "style", "locked") in codes
        assert ("VOX_xiaolongnv", "voice", "locked") in codes
        assert ("OUT_xiaolongnv_baiyi", "outfit", "locked") in codes
        assert (root / "library.yaml").read_bytes() == before
        assert not (root / "library.history").exists()

    def test_get_keeps_desc_and_views(self, tmp_path):
        from ComfyTV.storage.series_library import get_series_asset

        root = materialize(tmp_path / "gumu-sx")
        got = get_series_asset(root, "CHR_xiaolongnv@v1")
        assert got["name_zh"] == "小龙女"
        assert "jet-black hair" in got["desc_en"]
        assert "front" in got["files"]
        assert "expr_cold" in got["files"]
        assert got["files"]["doc_desc"]["path"].endswith("desc.md")


class TestResolve:
    def test_version_view_alias_and_defaults(self, tmp_path):
        from ComfyTV.storage.series_library import resolve_series_ref

        root = materialize(tmp_path / "gumu-sx")
        front = resolve_series_ref(root, "CHR_xiaolongnv@v1#front")
        assert front["resolved"] is True
        assert front["absolute_path"].endswith("CHR_xiaolongnv/v1/front.png")
        assert (root / "CHR_xiaolongnv/v1/front.png").read_bytes() == b"CHR_xiaolongnv:front"
        assert "comfytv/series/gumu-sx/CHR_xiaolongnv/v1" in front["payload_url"]
        assert front["usage"] == ["identity_video", "image_ref", "review"]

        defaulted = resolve_series_ref(root, "CHR_xiaolongnv@v1")
        assert defaulted["view"] == "front"
        assert defaulted["absolute_path"] == front["absolute_path"]

        latest = resolve_series_ref(root, "{CHR_xiaolongnv}")
        assert latest["version"] == 1
        assert latest["version_defaulted"] is True

        alias = resolve_series_ref(root, "OUT_xiaolongnv_baiyi@v1#front")
        assert alias["resolved_id"] == "CHR_xiaolongnv"
        assert alias["absolute_path"] == front["absolute_path"]
        assert alias["alias_chain"][-1].startswith("CHR_xiaolongnv@v1#front")

        empty = resolve_series_ref(root, "LOC_gumu_shishi@v1")
        assert empty["view"] == "empty"
        assert empty["absolute_path"].endswith("empty.png")

        style = resolve_series_ref(root, "STY_01")
        assert style["view"] == "style"
        voice = resolve_series_ref(root, "VOX_xiaolongnv@v1")
        assert voice["view"] == "voice_ref"
        assert voice["media_type"] == "audio"

        lit = resolve_series_ref(root, "PRP_zhutai@v1#lit")
        assert lit["view"] == "lit"

    def test_prop_without_view_is_ambiguous(self, tmp_path):
        from ComfyTV.storage.series_library import SeriesLibraryError, resolve_series_ref

        root = materialize(tmp_path / "gumu-sx")
        with pytest.raises(SeriesLibraryError, match="view required"):
            resolve_series_ref(root, "PRP_zhutai@v1")

    def test_episode_and_new_do_not_invent_files(self, tmp_path):
        from ComfyTV.storage.series_library import resolve_series_ref

        root = materialize(tmp_path / "gumu-sx")
        episode = resolve_series_ref(root, "episode:E01S01_01/first.png")
        assert episode["resolved"] is False
        assert episode["kind"] == "episode"
        new = resolve_series_ref(root, "NEW:LOC_linzhaoying_shishi")
        assert new["resolved"] is False
        assert new["kind"] == "unresolved"

    def test_missing_code_and_path_escape(self, tmp_path):
        from ComfyTV.storage.series_library import SeriesLibraryError, resolve_series_ref

        root = materialize(tmp_path / "gumu-sx")
        with pytest.raises(SeriesLibraryError) as missing:
            resolve_series_ref(root, "CHR_nobody@v1#front")
        assert missing.value.http_status == 404
        with pytest.raises(SeriesLibraryError):
            resolve_series_ref(root, "CHR_xiaolongnv@v1#not_a_view")


class TestRegisterLockVersion:
    def _draft_prop(self, root: Path) -> None:
        _write_stub(root, "PRP_huitu/v1/ref.png", b"rabbit")

    def test_register_lock_is_immutable_and_new_version_copies(self, tmp_path):
        from ComfyTV.storage.series_library import (
            SeriesLibraryError,
            get_series_asset,
            lock_series_asset,
            new_series_asset_version,
            register_series_asset,
            resolve_series_ref,
        )

        root = materialize(tmp_path / "gumu-sx")
        self._draft_prop(root)
        original_front = (root / "CHR_xiaolongnv/v1/front.png").read_bytes()
        original_yaml = (root / "library.yaml").read_bytes()

        with pytest.raises(SeriesLibraryError, match="already"):
            register_series_asset(root, {
                "id": "CHR_xiaolongnv",
                "version": 1,
                "kind": "character",
                "status": "draft",
                "files": {"front": {"path": "CHR_xiaolongnv/v1/front.png"}},
            })
        assert (root / "library.yaml").read_bytes() == original_yaml

        registered = register_series_asset(root, {
            "id": "PRP_huitu",
            "version": 1,
            "kind": "prop",
            "status": "draft",
            "name_zh": "灰兔",
            "desc_en": "A small grey rabbit.",
            "files": {
                "ref": {
                    "path": "PRP_huitu/v1/ref.png",
                    "media_type": "image",
                    "usage": ["prop_video", "image_ref"],
                },
            },
        })
        assert registered["status"] == "draft"
        assert registered["files"]["ref"]["sha256"]
        assert len(list((root / "library.history").glob("library.*.yaml"))) == 1

        with pytest.raises(SeriesLibraryError, match="approved_by"):
            lock_series_asset(root, "PRP_huitu@v1", approved_by="")

        locked = lock_series_asset(
            root, "PRP_huitu@v1", approved_by="张景", approval_ref="test-lock",
        )
        assert locked["status"] == "locked"
        assert locked["approved_by"] == "张景"
        mode = (root / "PRP_huitu/v1/ref.png").stat().st_mode
        assert mode & stat.S_IWUSR == 0

        with pytest.raises(SeriesLibraryError, match="locked"):
            lock_series_asset(root, "PRP_huitu@v1", approved_by="张景")
        with pytest.raises(SeriesLibraryError, match="already"):
            register_series_asset(root, {
                "id": "PRP_huitu",
                "version": 1,
                "kind": "prop",
                "desc_en": "changed",
                "files": {"ref": {"path": "PRP_huitu/v1/ref.png"}},
            })

        v1_bytes = (root / "PRP_huitu/v1/ref.png").read_bytes()
        created = new_series_asset_version(root, "PRP_huitu@v1")
        assert created["version"] == 2
        assert created["status"] == "draft"
        assert created["supersedes"] == "PRP_huitu@v1"
        assert "approved_by" not in created
        assert (root / "PRP_huitu/v2/ref.png").read_bytes() == v1_bytes
        assert (root / "PRP_huitu/v2/ref.png").stat().st_mode & stat.S_IWUSR
        assert get_series_asset(root, "PRP_huitu@v1")["status"] == "locked"
        assert get_series_asset(root, "PRP_huitu@v1")["desc_en"] == "A small grey rabbit."
        assert (root / "PRP_huitu/v1/ref.png").read_bytes() == v1_bytes
        assert (root / "CHR_xiaolongnv/v1/front.png").read_bytes() == original_front

        still_v1 = resolve_series_ref(root, "PRP_huitu")
        assert still_v1["version"] == 1
        with pytest.raises(SeriesLibraryError, match="locked"):
            resolve_series_ref(root, "PRP_huitu@v2")
        draft = resolve_series_ref(root, "PRP_huitu@v2", require_locked=False)
        assert draft["version"] == 2

        history = sorted((root / "library.history").glob("library.*.yaml"))
        assert history[0].read_bytes() == original_yaml
        reloaded = yaml.safe_load((root / "library.yaml").read_text(encoding="utf-8"))
        assert reloaded["schema_version"] == 1
        assert reloaded["series"]["id"] == "gumu-sx"
        ids = [(row["id"], row["version"]) for row in reloaded["assets"]]
        assert ("CHR_xiaolongnv", 1) in ids
        assert ("PRP_huitu", 1) in ids
        assert ("PRP_huitu", 2) in ids

    def test_register_rejects_bad_id_kind_and_escaping_path(self, tmp_path):
        from ComfyTV.storage.series_library import SeriesLibraryError, register_series_asset

        root = materialize(tmp_path / "gumu-sx")
        with pytest.raises(SeriesLibraryError):
            register_series_asset(root, {
                "id": "CHR_bad",
                "version": 1,
                "kind": "prop",
                "files": {},
            })
        outside = tmp_path / "outside.png"
        outside.write_bytes(b"nope")
        with pytest.raises(SeriesLibraryError, match="escapes"):
            register_series_asset(root, {
                "id": "PRP_escape",
                "version": 1,
                "kind": "prop",
                "files": {"ref": {"path": "../outside.png"}},
            })


class TestImportManifest:
    def test_import_adds_new_id_and_refuses_locked_edits(self, tmp_path):
        from ComfyTV.storage.series_library import (
            SeriesLibraryError,
            get_series_asset,
            import_series_manifest,
        )

        root = materialize(tmp_path / "gumu-sx")
        _write_stub(root, "CHR_yangguo/v1/front.png", b"yangguo-front")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        imported = import_series_manifest(root, manifest)
        assert imported["added"] == ["CHR_yangguo@v1"]
        assert get_series_asset(root, "CHR_yangguo@v1")["status"] == "draft"
        assert get_series_asset(root, "CHR_yangguo@v1")["files"]["front"]["sha256"]

        again = import_series_manifest(root, manifest)
        assert again["added"] == []
        assert "CHR_yangguo@v1" in again["skipped"]

        changed = json.loads(json.dumps(manifest))
        changed["assets"][0]["desc_en"] = "different"
        with pytest.raises(SeriesLibraryError, match="draft"):
            import_series_manifest(root, changed)

        locked_edit = {
            "schema_version": 1,
            "assets": [{
                "id": "CHR_xiaolongnv",
                "version": 1,
                "kind": "character",
                "status": "locked",
                "desc_en": "rewritten",
                "approved_by": "张景",
                "files": {"front": {"path": "CHR_xiaolongnv/v1/front.png"}},
            }],
        }
        before = get_series_asset(root, "CHR_xiaolongnv@v1")["desc_en"]
        with pytest.raises(SeriesLibraryError, match="locked"):
            import_series_manifest(root, locked_edit)
        assert get_series_asset(root, "CHR_xiaolongnv@v1")["desc_en"] == before

    def test_import_creates_library_and_history_on_later_write(self, tmp_path):
        from ComfyTV.storage.series_library import import_series_manifest, list_series_assets

        root = tmp_path / "fresh"
        root.mkdir()
        _write_stub(root, "CHR_yangguo/v1/front.png", b"yangguo-front")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        import_series_manifest(root, manifest)
        listed = list_series_assets(root)
        assert listed["series"]["id"] == "gumu-sx"
        assert [row["id"] for row in listed["assets"]] == ["CHR_yangguo"]
        text = (root / "library.yaml").read_text(encoding="utf-8")
        assert text.startswith("# 系列资产库索引")
        assert "schema_version: 1" in text


class TestShotResolve:
    def test_e01_shot_codes_resolve_to_image_files(self, tmp_path, monkeypatch):
        from ComfyTV.nodes.stages.common.shot_inputs import shot_reference_images
        from ComfyTV.storage.series_library import bind_series_library, resolve_series_shot

        root = materialize(tmp_path / "gumu-sx")
        _user_dir(tmp_path, monkeypatch)
        bind_series_library(root)
        shot = {
            "shot_id": "E01S01_01",
            "characters": [
                {"id": "CHR_yangguo@v1", "outfit": "OUT_yangguo_heiyi@v1", "look": None},
                {
                    "id": "CHR_xiaolongnv@v1",
                    "outfit": "OUT_xiaolongnv_baiyi@v1",
                    "look": None,
                },
            ],
            "location": {"id": "NEW:LOC_linzhaoying_shishi", "camera_card": None},
            "props": [{"id": "NEW:PRP_zhutai_linzhaoying", "state": "lit"}],
            "refs": [{"role": "first_frame", "src": "episode:E01S01_01/first.png"}],
            "prompts": {
                "krea2": "In the chamber {LOC}, {CHR_xiaolongnv} stands beside {CHR_yangguo}.",
            },
        }
        resolved = resolve_series_shot(root, shot)
        images = [row for row in resolved["results"] if row.get("resolved") and row.get("media_type") == "image"]
        assert any(row["id"] == "CHR_xiaolongnv" and row["view"] == "front" for row in images)
        assert all(row["kind"] != "episode" or row["resolved"] is False for row in resolved["results"])
        assert any(row["kind"] == "unresolved" for row in resolved["results"])

        urls = shot_reference_images(shot, ["shared.png"])
        assert any("CHR_xiaolongnv/v1" in url and "front.png" in url for url in urls)
        assert "shared.png" not in urls
        assert all("NEW:" not in url for url in urls)

    def test_existing_image_urls_stay_in_front(self, tmp_path, monkeypatch):
        from ComfyTV.nodes.stages.common.shot_inputs import shot_reference_images
        from ComfyTV.storage.series_library import bind_series_library

        root = materialize(tmp_path / "gumu-sx")
        _user_dir(tmp_path, monkeypatch)
        bind_series_library(root)
        shot = {
            "reference_images": ["/view?filename=already.png&type=input"],
            "characters": [{"id": "CHR_xiaolongnv@v1"}],
        }
        assert shot_reference_images(shot, ["shared.png"]) == [
            "/view?filename=already.png&type=input",
        ]


class TestMcpAndHttp:
    def test_mcp_actions(self, tmp_path, monkeypatch):
        import asyncio

        from ComfyTV.api.mcp_tools import TOOLS
        from ComfyTV.bot.guide import CORE_MCP_TOOLS

        assert "series_library" in TOOLS
        assert "series_library" in CORE_MCP_TOOLS
        _user_dir(tmp_path, monkeypatch)
        root = materialize(tmp_path / "gumu-sx")
        _write_stub(root, "CHR_yangguo/v1/front.png", b"yangguo-front")
        handler = TOOLS["series_library"]["handler"]

        async def run():
            bound = await handler({"action": "bind", "root": str(root)})
            assert bound["series_id"] == "gumu-sx"
            listed = await handler({"action": "list", "kind": "character"})
            assert [row["id"] for row in listed["assets"]] == ["CHR_xiaolongnv"]
            got = await handler({"action": "get", "code": "LOC_gumu_shishi@v1"})
            assert got["asset"]["lighting_states"] == ["night_candle", "dark_candle_out"]
            resolved = await handler({"action": "resolve", "ref": "STY_01@v1#style"})
            assert resolved["resolved"]["absolute_path"].endswith("style.png")
            imported = await handler({
                "action": "import_manifest",
                "manifest_path": str(MANIFEST),
            })
            assert imported["added"] == ["CHR_yangguo@v1"]
            locked = await handler({
                "action": "lock",
                "code": "CHR_yangguo@v1",
                "approved_by": "张景",
            })
            assert locked["asset"]["status"] == "locked"
            nxt = await handler({"action": "new_version", "code": "CHR_yangguo"})
            assert nxt["asset"]["version"] == 2
            assert nxt["asset"]["status"] == "draft"
            shot = await handler({
                "action": "resolve",
                "shot": {
                    "characters": [{"id": "CHR_xiaolongnv@v1"}],
                    "props": [{"id": "PRP_zhutai@v1", "state": "out"}],
                },
            })
            views = {(row["id"], row.get("view")) for row in shot["results"] if row.get("resolved")}
            assert ("CHR_xiaolongnv", "front") in views
            assert ("PRP_zhutai", "out") in views

        asyncio.run(run())  # may already be in a loop; the http test uses pytest-asyncio

    @pytest.fixture()
    async def client(self, tmp_path, monkeypatch):
        _user_dir(tmp_path, monkeypatch)
        from ComfyTV import api  # noqa: F401
        import server
        app = web.Application()
        app.router.add_routes(server.PromptServer.instance.routes)
        test_server = TestServer(app)
        test_client = TestClient(test_server)
        await test_client.start_server()
        yield test_client
        await test_client.close()

    async def test_http_resolve_and_lock(self, client, tmp_path):
        root = materialize(tmp_path / "gumu-sx")
        listed = await client.get("/comfytv/series_library", params={"root": str(root)})
        assert listed.status == 200
        body = await listed.json()
        assert any(row["id"] == "VOX_xiaolongnv" for row in body["assets"])

        resolved = await client.post("/comfytv/series_library/resolve", json={
            "root": str(root),
            "ref": "LOOK_xiaolongnv_baiyi_opened@v1",
        })
        assert resolved.status == 200
        payload = await resolved.json()
        assert payload["resolved"]["view"] == "front"
        assert payload["resolved"]["absolute_path"].endswith(
            "LOOK_xiaolongnv_baiyi_opened/v1/front.png")

        _write_stub(root, "BIB_series_bible/v1/series_bible.md", b"# bible")
        registered = await client.post("/comfytv/series_library/register", json={
            "root": str(root),
            "asset": {
                "id": "BIB_series_bible",
                "version": 1,
                "kind": "bible",
                "status": "draft",
                "name_zh": "系列圣经",
                "files": {
                    "series_bible": {
                        "path": "BIB_series_bible/v1/series_bible.md",
                        "media_type": "text",
                        "usage": ["text"],
                    },
                },
            },
        })
        assert registered.status == 200
        conflict = await client.post("/comfytv/series_library/lock", json={
            "root": str(root),
            "code": "CHR_xiaolongnv@v1",
            "approved_by": "张景",
        })
        assert conflict.status == 409
