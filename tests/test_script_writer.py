"""Script-writer sessions: versions, selection rewrite, import receipts.

The LLM is a fake. Nothing here reads a real project tree.
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


SECTIONS = ("改编说明", "人物表", "节拍与爽点", "分集/分场梗概", "叙事资产清单")

SCRIPT = """## E01S03 石室 夜 (预算 40s)
- E01S03_b01 [动作] 她站在门边。
- E01S03_b02 [动作] 烛火跳了一下。
- E01S03_b03 [台词 小龙女] 你来了。 {语气: 冷}
- E01S03_b04 [动作] 她没有回头。
- E01S03_b05 [独白 杨过] 石室比记忆里更冷。
- E01S03_b06 [动作] 门外有风。
"""


class FakeLlm:
    model = "fake"

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def complete(self, *, system, user, model):
        self.calls.append({"system": system, "user": user, "model": model})
        return self.handler(system, user)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, text: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes(data)
    return {"bytes": len(data), "sha256": _sha(data)}


def _directions(n=4):
    rows = []
    for i in range(n):
        rows.append({
            "id": f"d{i + 1}",
            "logline": f"方向{i + 1}的一句话梗概",
            "conflict": f"冲突{i + 1}",
            "characters": ["小龙女", "杨过"],
            "selling_points": f"看点{i + 1}",
            "risks": "可生成性未测",
            "genre": f"题材{i + 1}",
            "ending": f"结局{i + 1}",
        })
    return json.dumps(rows, ensure_ascii=False)


def _outline(*, quoted=False):
    dialogue = "她说：“你来了。”" if quoted else "她只点了点头。"
    parts = []
    for name in SECTIONS:
        body = dialogue if "梗概" in name else "没有补写设定。"
        parts.append(f"## {name}\n\n{body}\n")
    outline = "\n".join(parts)
    characters = "# 人物\n\n小龙女。\n"
    hooks = "hooks:\n  - id: h1\n    text: 门开了\n"
    return f"<<<OUTLINE>>>\n{outline}\n<<<CHARACTERS>>>\n{characters}\n<<<HOOKS>>>\n{hooks}\n"


def _task(user: str) -> str:
    for line in user.splitlines():
        if line.startswith("task:"):
            return line.split(":", 1)[1].strip()
    return ""


def _router(system, user):
    task = _task(user)
    if task == "directions":
        n = 4
        for line in user.splitlines():
            if line.startswith("n:"):
                n = int(line.split(":", 1)[1].strip())
        return _directions(n)
    if task == "choose":
        return _outline(quoted=False)
    if task == "revise":
        return user
    if task == "episode":
        return f"<<<SCRIPT>>>\n{SCRIPT}\n<<<LINES>>>\n- L1 小龙女 你来了。\n"
    if task == "review":
        return "# review\n\n8 分。\n"
    raise AssertionError(f"unexpected task {task}")


def _manifest(files, *, project="gumu-shuangxiu", status="active",
              lineage="E01", episode="E01", session_id=None):
    body = {
        "schema": "comfytv/script-import.v1",
        "project": project,
        "series": "gumu-sx",
        "lineage": lineage,
        "status": status,
        "episode": episode,
        "file_count": len(files),
        "files": files,
    }
    if session_id:
        body["session_id"] = session_id
    return body


def _file_entry(rel: str, digest: dict, *, doc=None):
    entry = {"source_path": rel, "sha256": digest["sha256"], "bytes": digest["bytes"]}
    if doc:
        entry["doc"] = doc
    return entry


def _import_pair(tmp_path: Path):
    from ComfyTV.storage.script_writer import import_scripts

    source = tmp_path / "box"
    script = (
        "# E01\n\nP01_xiaolongnv 站在门边。\n\n"
        "## E01S03 石室 夜 (预算 40s)\n"
        "- E01S03_b04 [动作] 她没有回头。\n"
    )
    shots = {
        "episode": "E01",
        "shots": [
            {"shot_id": f"E01S{(i // 4) + 1:02d}_{(i % 4) + 1:02d}", "who": "P01_xiaolongnv"}
            for i in range(22)
        ],
    }
    older = "# v1\n\n这是定稿前一版，不是真相源。\n"
    script_meta = _write(source / "episodes/E01/script_v2.md", script)
    shots_meta = _write(
        source / "episodes/E01/shots.json",
        json.dumps(shots, ensure_ascii=False),
    )
    older_meta = _write(source / "episodes/E01/script_v1.md", older)
    manifest = _manifest([
        _file_entry("episodes/E01/script_v2.md", script_meta, doc="script"),
        _file_entry("episodes/E01/shots.json", shots_meta, doc="shots"),
        _file_entry("episodes/E01/script_v1.md", older_meta, doc="script_v1"),
    ])
    root = tmp_path / "spark-project"
    root.mkdir(parents=True, exist_ok=True)
    (root / "library.yaml").write_text("schema_version: 1\nassets: []\n", encoding="utf-8")
    library_before = (root / "library.yaml").read_bytes()
    report = import_scripts(root, manifest, source_dir=source)
    return root, source, report, library_before, script, shots


class TestImport:
    def test_e01_import_keeps_bytes_ids_and_library(self, tmp_path):
        from ComfyTV.storage.script_writer import current_storyboard, verify_imports

        root, _source, report, library_before, script, shots = _import_pair(tmp_path)
        assert report["file_count"] == 3
        assert report["added"] == 3
        assert report["status"] == "active"
        assert (root / "library.yaml").read_bytes() == library_before

        session_id = report["session_id"]
        imported = root / "script" / "sessions" / session_id / "imported"
        stored_script = (imported / "episodes/E01/script_v2.md").read_bytes()
        stored_shots = (imported / "episodes/E01/shots.json").read_bytes()
        assert stored_script == script.encode("utf-8")
        assert stored_shots == json.dumps(shots, ensure_ascii=False).encode("utf-8")
        assert b"P01_xiaolongnv" in stored_script
        assert b"CHR_" not in stored_script
        assert report["shot_count"] == 22
        assert report["shot_ids"][0] == "E01S01_01"
        assert report["shot_ids"][-1] == "E01S06_02"
        board = current_storyboard(root, project="gumu-shuangxiu")
        assert board["doc"] == "shots"
        assert board["shot_ids"] == report["shot_ids"]
        verified = verify_imports(root, session_id)
        assert verified["ok"] is True
        assert verified["file_count"] == 3
        assert verified["receipt_sha256"] == report["receipt_sha256"]

    def test_rejects_bad_sha_and_file_count_without_writing(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, import_scripts

        source = tmp_path / "box"
        meta = _write(source / "episodes/E01/script_v2.md", "hello\n")
        root = tmp_path / "spark"
        bad_hash = _manifest([
            _file_entry("episodes/E01/script_v2.md", {**meta, "sha256": "0" * 64}, doc="script"),
        ])
        with pytest.raises(ScriptError, match="sha256"):
            import_scripts(root, bad_hash, source_dir=source)
        assert not (root / "script").exists()

        bad_count = _manifest([
            _file_entry("episodes/E01/script_v2.md", meta, doc="script"),
        ])
        bad_count["file_count"] = 9
        with pytest.raises(ScriptError, match="file_count"):
            import_scripts(root, bad_count, source_dir=source)
        assert not (root / "script").exists()

    def test_archived_x01_does_not_replace_e01(self, tmp_path):
        from ComfyTV.storage.script_writer import (
            ScriptError,
            current_storyboard,
            get_session,
            import_scripts,
        )

        root, source, report, _library, script, _shots = _import_pair(tmp_path)
        e01_id = report["session_id"]
        e01_dir = root / "script" / "sessions" / e01_id
        before = {
            path.relative_to(e01_dir).as_posix(): _sha(path.read_bytes())
            for path in e01_dir.rglob("*") if path.is_file()
        }
        x_script = _write(
            source / "script" / "script.md",
            "# X01\n\nE01S01_01 仍是旧编号，不要改成 X01。\nP01_xiaolongnv\n",
        )
        story = _write(
            source / "storyboard.yaml",
            "shots:\n  - id: X01S01_01\n    note: archived\n",
        )
        state = _write(
            source / "state.json",
            json.dumps({"superseded": True, "lineage": "X01", "shots": []}),
        )
        archived = _manifest(
            [
                _file_entry("script/script.md", x_script, doc="script"),
                _file_entry("storyboard.yaml", story, doc="storyboard"),
                _file_entry("state.json", state, doc="state"),
            ],
            status="archived",
            lineage="X01",
            episode=None,
            session_id="gumu-shuangxiu-X01",
        )
        saved = import_scripts(root, archived, source_dir=source)
        assert saved["status"] == "archived"
        assert saved["session_id"] == "gumu-shuangxiu-X01"
        after = {
            path.relative_to(e01_dir).as_posix(): _sha(path.read_bytes())
            for path in e01_dir.rglob("*") if path.is_file()
        }
        assert after == before
        stored = (
            root / "script" / "sessions" / "gumu-shuangxiu-X01"
            / "imported" / "script" / "script.md"
        ).read_text(encoding="utf-8")
        assert "E01S01_01" in stored
        assert "X01S01_01" not in stored
        board = current_storyboard(root, project="gumu-shuangxiu")
        assert board["session_id"] == e01_id
        archived_session = get_session(root, "gumu-shuangxiu-X01")
        assert archived_session["storyboard_current"] is False
        assert archived_session["status"] == "archived"

        clash = _manifest(
            [_file_entry("episodes/E01/script_v2.md", {
                "sha256": _sha(script.encode("utf-8")),
                "bytes": len(script.encode("utf-8")),
            }, doc="script")],
            status="archived",
            lineage="X01",
            episode=None,
            session_id="gumu-shuangxiu-X01b",
        )
        with pytest.raises(ScriptError, match="episodes/"):
            import_scripts(root, clash, source_dir=source)

    def test_same_manifest_skips_and_different_bytes_do_not_overwrite(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, import_scripts

        root, source, report, _library, script, _shots = _import_pair(tmp_path)
        again = import_scripts(
            root,
            json.loads((root / "script" / "sessions" / report["session_id"] / "import_manifest.json").read_text(encoding="utf-8")),
            source_dir=source,
        )
        assert again["added"] == 0
        assert again["skipped"] == 3
        v1 = (
            root / "script" / "sessions" / report["session_id"]
            / ".versions" / "script" / "v1.md"
        ).read_bytes()
        changed = script + "多了一句。\n"
        meta = _write(source / "episodes/E01/script_v2.md", changed)
        mutated = _manifest([
            _file_entry("episodes/E01/script_v2.md", meta, doc="script"),
        ], session_id=report["session_id"])
        with pytest.raises(ScriptError, match="cannot be changed"):
            import_scripts(root, mutated, source_dir=source)
        assert v1 == (
            root / "script" / "sessions" / report["session_id"]
            / ".versions" / "script" / "v1.md"
        ).read_bytes()

    def test_gumu_bixie_is_excluded_until_config_allows_it(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, import_scripts

        source = tmp_path / "box"
        meta = _write(source / "script" / "script.md", "独立项目\n")
        root = tmp_path / "bixie"
        manifest = _manifest(
            [_file_entry("script/script.md", meta, doc="script")],
            project="gumu-bixie",
            lineage="E01",
            episode="E01",
        )
        with pytest.raises(ScriptError, match="gumu-bixie"):
            import_scripts(root, manifest, source_dir=source)
        assert not (root / "script" / "sessions").exists()

        root.mkdir(parents=True, exist_ok=True)
        (root / "script").mkdir()
        (root / "script" / "config.json").write_text(
            json.dumps({"excluded_projects": []}),
            encoding="utf-8",
        )
        allowed = import_scripts(root, manifest, source_dir=source)
        assert allowed["project"] == "gumu-bixie"


class TestDialogue:
    def _ready(self, tmp_path, llm):
        from ComfyTV.storage.script_writer import (
            confirm_session,
            run_choose,
            run_directions,
            run_episode,
            start_session,
        )

        root = tmp_path / "proj"
        started = start_session(
            root,
            project="gumu-shuangxiu",
            episode="E01",
            idea="一句想法",
            source_text="",
            reference_works="",
        )
        assert "未指定" in started["docs"]["brief"]["text"]
        session_id = started["session_id"]
        directions = run_directions(root, session_id, llm, n=4)
        assert len(directions["directions"]) == 4
        chosen = run_choose(root, session_id, llm, direction_id="d2")
        outline = chosen["docs"]["outline"]
        for name in SECTIONS:
            assert f"## {name}" in outline
        confirmed = confirm_session(root, session_id, approved_by="张景")
        assert confirmed["stage"] == "confirmed"
        assert confirmed["confirmed"]["by"] == "张景"
        episode = run_episode(root, session_id, llm, episode="E01")
        assert "E01S03_b04" in episode["docs"]["script"]
        assert "L1" in episode["docs"]["lines"]
        return root, session_id

    def test_directions_choose_confirm_and_episode(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, confirm_session, start_session

        llm = FakeLlm(_router)
        root, session_id = self._ready(tmp_path, llm)
        assert llm.calls[0]["model"] == "deepseek-v4-pro"
        other = tmp_path / "other"
        started = start_session(other, project="demo", idea="想法")
        with pytest.raises(ScriptError, match="张景"):
            confirm_session(other, started["session_id"], approved_by="视频制作助手")

    def test_quoted_synopsis_is_rejected(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, run_choose, run_directions, start_session

        root = tmp_path / "proj"
        started = start_session(root, project="demo", idea="想法")
        session_id = started["session_id"]

        def handler(system, user):
            if _task(user) == "directions":
                return _directions(4)
            return _outline(quoted=True)

        llm = FakeLlm(handler)
        run_directions(root, session_id, llm, n=4)
        with pytest.raises(ScriptError, match="引号"):
            run_choose(root, session_id, llm, direction_id="d1")
        versions = root / "script" / "sessions" / session_id / ".versions" / "outline"
        assert not versions.exists()

    def test_identical_directions_are_rejected(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, run_directions, start_session

        root = tmp_path / "proj"
        started = start_session(root, project="demo", idea="想法")
        row = json.loads(_directions(1))[0]

        def handler(system, user):
            return json.dumps([row, dict(row, id="d2")], ensure_ascii=False)

        with pytest.raises(ScriptError, match="distinct"):
            run_directions(root, started["session_id"], FakeLlm(handler), n=2)

    def test_confirm_required_before_episode(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, run_episode, start_session

        root = tmp_path / "proj"
        started = start_session(root, project="demo", idea="想法")
        with pytest.raises(ScriptError, match="confirmed"):
            run_episode(root, started["session_id"], FakeLlm(_router), episode="E01")


class TestRewrite:
    def _imported(self, tmp_path):
        from ComfyTV.storage.script_writer import import_scripts

        source = tmp_path / "box"
        meta = _write(source / "episodes/E01/script_v2.md", SCRIPT)
        root = tmp_path / "spark"
        report = import_scripts(
            root,
            _manifest([_file_entry("episodes/E01/script_v2.md", meta, doc="script")]),
            source_dir=source,
        )
        v1 = (
            root / "script" / "sessions" / report["session_id"] / ".versions" / "script" / "v1.md"
        ).read_bytes()
        return root, report["session_id"], v1

    def test_out_of_bounds_rewrite_is_rejected(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, run_revise

        root, session_id, v1 = self._imported(tmp_path)
        leaked = SCRIPT.replace("她站在门边。", "她已经离开。")

        def handler(system, user):
            return leaked

        with pytest.raises(ScriptError, match="outside"):
            run_revise(
                root, session_id, FakeLlm(handler),
                doc="script",
                instruction="更口语一点",
                selection={"beat_ids": ["E01S03_b03", "E01S03_b04", "E01S03_b05"]},
            )
        assert (
            root / "script" / "sessions" / session_id / ".versions" / "script" / "v1.md"
        ).read_bytes() == v1
        assert not (root / "script" / "sessions" / session_id / ".versions" / "script" / "v2.md").exists()

    def test_in_bounds_rewrite_keeps_outside_hash(self, tmp_path):
        from ComfyTV.storage.script_writer import accept_candidate, get_session, run_revise

        root, session_id, v1 = self._imported(tmp_path)
        replacement = (
            "- E01S03_b03 [台词 小龙女] 你来啦。 {语气: 冷}\n"
            "- E01S03_b04 [动作] 她没有回头。\n"
            "- E01S03_b05 [独白 杨过] 石室比记忆里更冷。\n"
        )

        def handler(system, user):
            return replacement

        llm = FakeLlm(handler)
        candidate = run_revise(
            root, session_id, llm,
            doc="script",
            instruction="更口语一点",
            selection={"beat_ids": ["E01S03_b03", "E01S03_b04", "E01S03_b05"]},
        )
        assert "E01S03_b03" in llm.calls[0]["user"]
        assert candidate["sha256_outside_changed"] is False
        assert get_session(root, session_id)["docs"]["script"]["current"] == 1
        accepted = accept_candidate(root, session_id, candidate["candidate_id"])
        assert accepted["version"] == 2
        text = get_session(root, session_id)["docs"]["script"]["text"]
        assert "你来啦" in text
        assert "她站在门边。" in text
        assert (
            root / "script" / "sessions" / session_id / ".versions" / "script" / "v1.md"
        ).read_bytes() == v1
        frozen = (
            root / "script" / "sessions" / session_id / "imported" / "episodes/E01/script_v2.md"
        ).read_bytes()
        assert frozen == v1

    def test_changed_beat_id_is_rejected_and_insert_gets_a_new_id(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, get_session, run_revise

        root, session_id, v1 = self._imported(tmp_path)

        def renamed(system, user):
            return "- E01S03_b09 [动作] 她没有回头。\n"

        with pytest.raises(ScriptError, match="E01S03_b04"):
            run_revise(
                root, session_id, FakeLlm(renamed),
                doc="script",
                instruction="改一下",
                selection={"beat_ids": ["E01S03_b04"]},
            )
        assert (
            root / "script" / "sessions" / session_id / ".versions" / "script" / "v1.md"
        ).read_bytes() == v1

        def inserted(system, user):
            return (
                "- E01S03_b04 [动作] 她没有回头。\n"
                "- [动作] 她抬了一下眼。\n"
            )

        candidate = run_revise(
            root, session_id, FakeLlm(inserted),
            doc="script",
            instruction="加一个节拍",
            selection={"beat_ids": ["E01S03_b04"]},
        )
        text = candidate["text"]
        for old in ("E01S03_b01", "E01S03_b02", "E01S03_b03", "E01S03_b04", "E01S03_b05", "E01S03_b06"):
            assert old in text
        assert "E01S03_b07" in text
        assert "她抬了一下眼" in text
        session = get_session(root, session_id)
        assert session["docs"]["script"]["current"] == 1

    def test_accept_manual_and_revert_keep_history(self, tmp_path):
        from ComfyTV.storage.script_writer import (
            diff_doc,
            get_session,
            revert_doc,
            save_manual,
        )

        root, session_id, v1 = self._imported(tmp_path)
        edited = SCRIPT.replace("她站在门边。", "她靠在门边。")
        saved = save_manual(root, session_id, doc="script", content=edited)
        assert saved["version"] == 2
        assert saved["by"] == "user"
        feedback = get_session(root, session_id)["docs"]["feedback"]["text"]
        assert "她站在门边。" in feedback
        assert "她靠在门边。" in feedback
        assert "→" in feedback
        reverted = revert_doc(root, session_id, doc="script", to=1)
        assert reverted["version"] == 3
        assert get_session(root, session_id)["docs"]["script"]["text"] == SCRIPT
        assert (
            root / "script" / "sessions" / session_id / ".versions" / "script" / "v1.md"
        ).read_bytes() == v1
        diff = diff_doc(root, session_id, doc="script", a=1, b=2)
        replaced = [row for row in diff["hunks"] if row["op"] == "replace"]
        assert any(row.get("beat_id") == "E01S03_b01" for row in replaced)
        assert all(row.get("beat_id") != "E01S03_b06" for row in replaced)

    def test_manual_edit_cannot_rename_a_beat(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, save_manual

        root, session_id, v1 = self._imported(tmp_path)
        renamed = SCRIPT.replace("E01S03_b04", "E01S03_b99")
        with pytest.raises(ScriptError, match="E01S03_b04"):
            save_manual(root, session_id, doc="script", content=renamed)
        assert (
            root / "script" / "sessions" / session_id / ".versions" / "script" / "v1.md"
        ).read_bytes() == v1

    def test_archived_session_is_frozen(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, import_scripts, save_manual

        source = tmp_path / "box"
        meta = _write(source / "script" / "script.md", SCRIPT)
        root = tmp_path / "spark"
        report = import_scripts(
            root,
            _manifest(
                [_file_entry("script/script.md", meta, doc="script")],
                status="archived",
                lineage="X01",
                episode=None,
                session_id="gumu-shuangxiu-X01",
            ),
            source_dir=source,
        )
        with pytest.raises(ScriptError, match="archived"):
            save_manual(root, report["session_id"], doc="script", content=SCRIPT + "\n")


class TestReviewAndJobs:
    def test_review_can_be_turned_off(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, run_review, start_session

        root = tmp_path / "proj"
        started = start_session(root, project="demo", idea="想法")
        (root / "script" / "config.json").write_text(
            json.dumps({"review_enabled": False}),
            encoding="utf-8",
        )
        with pytest.raises(ScriptError, match="disabled"):
            run_review(root, started["session_id"], FakeLlm(_router), doc="brief")

    def test_review_writes_a_version_when_enabled(self, tmp_path):
        from ComfyTV.storage.script_writer import get_session, run_review, start_session

        root = tmp_path / "proj"
        started = start_session(root, project="demo", idea="想法")
        result = run_review(root, started["session_id"], FakeLlm(_router), doc="brief")
        assert "8 分" in result["text"]
        assert get_session(root, started["session_id"])["docs"]["review"]["current"] == 1

    def test_prompt_file_is_replaceable(self, tmp_path):
        from ComfyTV.storage.script_writer import run_directions, start_session

        root = tmp_path / "proj"
        started = start_session(root, project="demo", idea="想法")
        prompt = root / "script" / "prompts" / "directions.md"
        prompt.parent.mkdir(parents=True)
        prompt.write_text("UNIQUE_PROMPT_MARKER\n", encoding="utf-8")
        llm = FakeLlm(_router)
        run_directions(root, started["session_id"], llm, n=4)
        assert "UNIQUE_PROMPT_MARKER" in llm.calls[0]["system"]

    def test_job_returns_before_the_fake_llm_runs(self, tmp_path):
        from ComfyTV.storage.script_writer import dispatch, get_session, pump

        root = tmp_path / "proj"
        started = dispatch({
            "action": "start",
            "root": str(root),
            "project": "demo",
            "idea": "一句想法",
        })
        queued = dispatch({
            "action": "directions",
            "root": str(root),
            "session_id": started["session_id"],
            "n": 4,
        })
        assert queued["status"] == "queued"
        assert "directions" not in queued
        session = get_session(root, started["session_id"])
        assert session["directions"] == []
        done = pump(root, FakeLlm(_router), session_id=started["session_id"])
        assert done[0]["status"] == "done"
        assert len(done[0]["result"]["directions"]) == 4
        status = dispatch({
            "action": "status",
            "root": str(root),
            "session_id": started["session_id"],
            "job_id": queued["job_id"],
        })
        assert status["status"] == "done"

    def test_to_shots_is_not_available(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, dispatch

        with pytest.raises(ScriptError, match="to_shots"):
            dispatch({"action": "to_shots", "root": str(tmp_path), "session_id": "x"})

    def test_unavailable_does_not_save_a_softer_draft(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, run_directions, start_session

        root = tmp_path / "proj"
        started = start_session(root, project="demo", idea="想法")

        def handler(system, user):
            return "UNAVAILABLE"

        with pytest.raises(ScriptError, match="unavailable"):
            run_directions(root, started["session_id"], FakeLlm(handler), n=4)
        assert get_directions(root, started["session_id"]) == []


def get_directions(root, session_id):
    from ComfyTV.storage.script_writer import get_session

    return get_session(root, session_id)["directions"]


class TestReviewFixes:
    def test_noncontiguous_beats_are_rejected(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, run_revise

        root, session_id, _v1 = TestRewrite()._imported(tmp_path)
        changed = SCRIPT.replace("烛火跳了一下。", "烛火灭了。")
        with pytest.raises(ScriptError, match="contiguous"):
            run_revise(
                root, session_id, FakeLlm(lambda system, user: changed),
                doc="script",
                instruction="改中间",
                selection={"beat_ids": ["E01S03_b01", "E01S03_b03"]},
            )

    def test_outline_edit_clears_confirmation(self, tmp_path):
        from ComfyTV.storage.script_writer import (
            ScriptError,
            get_session,
            run_episode,
            save_manual,
        )

        root, session_id = TestDialogue()._ready(tmp_path, FakeLlm(_router))
        outline = get_session(root, session_id)["docs"]["outline"]["text"]
        save_manual(root, session_id, doc="outline", content=outline.replace("没有补写设定。", "仍然没有补写设定。", 1))
        session = get_session(root, session_id)
        assert session["stage"] == "outline"
        assert session["confirmed"] is None
        with pytest.raises(ScriptError, match="confirmed"):
            run_episode(root, session_id, FakeLlm(_router), episode="E01")

    def test_duplicate_source_path_writes_nothing(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, import_scripts

        source = tmp_path / "box"
        meta = _write(source / "episodes/E01/script_v2.md", "hello\n")
        root = tmp_path / "spark"
        manifest = _manifest([
            _file_entry("episodes/E01/script_v2.md", meta, doc="script"),
            _file_entry("episodes/E01/script_v2.md", meta, doc="script_copy"),
        ])
        with pytest.raises(ScriptError, match="more than once"):
            import_scripts(root, manifest, source_dir=source)
        assert not (root / "script" / "sessions").exists()

    def test_archived_manifest_cannot_reuse_the_active_session(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, get_session, import_scripts

        root, source, report, _library, _script, _shots = _import_pair(tmp_path)
        x_script = _write(source / "script" / "script.md", "archived\n")
        archived = _manifest(
            [_file_entry("script/script.md", x_script, doc="script_archive")],
            status="archived",
            lineage="X01",
            episode=None,
            session_id=report["session_id"],
        )
        with pytest.raises(ScriptError, match="status"):
            import_scripts(root, archived, source_dir=source)
        session = get_session(root, report["session_id"])
        assert session["status"] == "active"
        assert session["storyboard_current"] is True

    def test_continue_cannot_cross_documents(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, run_revise

        root, session_id = TestDialogue()._ready(tmp_path, FakeLlm(_router))
        outline = _outline(quoted=False).split("<<<CHARACTERS>>>")[0].replace("<<<OUTLINE>>>", "").strip() + "\n"
        candidate = run_revise(
            root, session_id, FakeLlm(lambda system, user: outline),
            doc="outline", instruction="改标题以外的一句",
        )
        with pytest.raises(ScriptError, match="different document"):
            run_revise(
                root, session_id, FakeLlm(lambda system, user: "x"),
                doc="characters", instruction="继续", candidate_id=candidate["candidate_id"],
            )

    def test_duplicate_direction_ids_are_rejected(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, run_directions, start_session

        root = tmp_path / "proj"
        started = start_session(root, project="demo", idea="想法")
        rows = json.loads(_directions(2))
        rows[1]["id"] = rows[0]["id"]

        def handler(system, user):
            return json.dumps(rows, ensure_ascii=False)

        with pytest.raises(ScriptError, match="duplicate direction id"):
            run_directions(root, started["session_id"], FakeLlm(handler), n=2)

    def test_duplicate_beat_ids_are_rejected(self, tmp_path):
        from ComfyTV.storage.script_writer import (
            ScriptError,
            confirm_session,
            run_choose,
            run_directions,
            run_episode,
            start_session,
        )

        root = tmp_path / "proj"
        started = start_session(root, project="demo", idea="想法")
        session_id = started["session_id"]
        llm = FakeLlm(_router)
        run_directions(root, session_id, llm, n=4)
        run_choose(root, session_id, llm, direction_id="d1")
        confirm_session(root, session_id, approved_by="张景")
        duplicated = SCRIPT + "- E01S03_b01 [动作] 她又站了一次。\n"

        def handler(system, user):
            return f"<<<SCRIPT>>>\n{duplicated}\n<<<LINES>>>\n- L1\n"

        with pytest.raises(ScriptError, match="duplicate beat id"):
            run_episode(root, session_id, FakeLlm(handler), episode="E01")

    def test_outline_rewrite_rejects_quoted_synopsis(self, tmp_path):
        from ComfyTV.storage.script_writer import ScriptError, get_session, run_revise

        root, session_id = TestDialogue()._ready(tmp_path, FakeLlm(_router))
        bad = _outline(quoted=True).split("<<<CHARACTERS>>>")[0].replace("<<<OUTLINE>>>", "")
        with pytest.raises(ScriptError, match="引号"):
            run_revise(
                root, session_id, FakeLlm(lambda system, user: bad),
                doc="outline", instruction="加一句对白",
            )
        assert get_session(root, session_id)["docs"]["outline"]["current"] == 1

    def test_project_llm_config_is_kept_without_the_key(self, tmp_path, monkeypatch):
        from ComfyTV.storage.script_writer import start_session

        monkeypatch.setenv("DEEPSEEK_API_KEY", "file-secret-key")
        root = tmp_path / "proj"
        (root / "script").mkdir(parents=True)
        (root / "script" / "config.json").write_text(json.dumps({
            "llm": {
                "provider": "local",
                "local_model": "qwen",
                "local_base_url": "http://127.0.0.1:9/v1",
            },
        }), encoding="utf-8")
        started = start_session(root, project="demo", idea="想法")
        assert started["llm"]["provider"] == "local"
        assert started["llm"]["model"] == "qwen"
        assert started["llm"]["base_url"] == "http://127.0.0.1:9/v1"
        stored = (root / "script" / "sessions" / started["session_id"] / "session.json").read_text(encoding="utf-8")
        assert "file-secret-key" not in stored
        assert "api_key" not in started["llm"]

    def test_job_usage_and_unexpected_failure(self, tmp_path):
        from ComfyTV.storage.script_writer import dispatch, pump

        root = tmp_path / "proj"
        started = dispatch({
            "action": "start", "root": str(root), "project": "demo", "idea": "想法",
        })

        class UsageLlm(FakeLlm):
            def complete(self, *, system, user, model):
                self.last_usage = {"total_tokens": 12}
                return super().complete(system=system, user=user, model=model)

        done = pump(root, UsageLlm(_router), session_id=started["session_id"])
        assert done == []
        queued = dispatch({
            "action": "review", "root": str(root),
            "session_id": started["session_id"], "doc": "brief",
        })
        done = pump(root, UsageLlm(_router), session_id=started["session_id"])
        assert done[0]["result"]["usage"]["total_tokens"] == 12

        class Boom:
            def complete(self, *, system, user, model):
                raise RuntimeError("secret-boom")

        queued = dispatch({
            "action": "review", "root": str(root),
            "session_id": started["session_id"], "doc": "brief",
        })
        failed = pump(root, Boom(), session_id=started["session_id"])
        assert failed[0]["status"] == "error"
        assert failed[0]["error"] == "script writer job failed"
        assert "secret-boom" not in failed[0]["error"]
        assert queued["job_id"] == failed[0]["job_id"]

    def test_two_workers_claim_one_job(self, tmp_path):
        from ComfyTV.storage.script_writer import dispatch, pump

        root = tmp_path / "proj"
        started = dispatch({
            "action": "start", "root": str(root), "project": "demo", "idea": "想法",
        })
        dispatch({
            "action": "directions", "root": str(root),
            "session_id": started["session_id"], "n": 4,
        })
        calls = {"n": 0}

        class Slow(FakeLlm):
            def complete(self, *, system, user, model):
                calls["n"] += 1
                return super().complete(system=system, user=user, model=model)

        llm = Slow(_router)
        results = []

        def run():
            results.append(pump(root, llm, session_id=started["session_id"]))

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        finished = [job for batch in results for job in batch]
        assert calls["n"] == 1
        assert len(finished) == 1
        assert finished[0]["status"] == "done"


class TestLlmConfig:
    def test_key_is_not_stored_or_logged(self, tmp_path, monkeypatch, caplog):
        from ComfyTV.storage.script_llm import OpenAICompatChat, ScriptError

        secret = "secret-key-do-not-log"
        monkeypatch.setenv("DEEPSEEK_API_KEY", secret)

        class Handler(BaseHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                body = f"nope {secret}".encode()
                self.send_response(500)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = make_server(Handler)
        try:
            host, port = server.server_address
            client = OpenAICompatChat(
                base_url=f"http://{host}:{port}/v1",
                model="deepseek-v4-pro",
                api_key=secret,
            )
            with caplog.at_level("ERROR"):
                with pytest.raises(ScriptError) as caught:
                    client.complete(system="s", user="u", model="deepseek-v4-pro")
            assert secret not in str(caught.value)
            assert secret not in caplog.text

            class SplitHandler(BaseHandler):
                def do_POST(self):
                    length = int(self.headers.get("Content-Length", "0"))
                    self.rfile.read(length)
                    body = (b"x" * 290 + secret.encode())
                    self.send_response(500)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

        finally:
            server.shutdown()
        server = make_server(SplitHandler)
        try:
            host, port = server.server_address
            client = OpenAICompatChat(
                base_url=f"http://{host}:{port}/v1",
                model="deepseek-v4-pro",
                api_key=secret,
            )
            with pytest.raises(ScriptError) as caught:
                client.complete(system="s", user="u", model="deepseek-v4-pro")
            assert secret not in str(caught.value)
            assert secret[:8] not in str(caught.value)
        finally:
            server.shutdown()

    def test_local_endpoint_omits_auth_when_no_key(self, tmp_path, monkeypatch):
        from ComfyTV.storage.script_llm import OpenAICompatChat

        seen = {}

        class Handler(BaseHandler):
            def do_POST(self):
                seen["auth"] = self.headers.get("Authorization")
                seen["path"] = self.path
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                seen["body"] = json.loads(raw.decode())
                payload = json.dumps({
                    "choices": [{"message": {"content": "本地回答"}}],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        monkeypatch.delenv("COMFYTV_LOCAL_LLM_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        server = make_server(Handler)
        try:
            host, port = server.server_address
            client = OpenAICompatChat(
                base_url=f"http://{host}:{port}/v1",
                model="local-model",
                api_key="",
            )
            text = client.complete(system="s", user="u", model="local-model")
            assert text == "本地回答"
            assert seen["auth"] is None
            assert seen["path"] == "/v1/chat/completions"
            assert seen["body"]["model"] == "local-model"
        finally:
            server.shutdown()

    def test_key_file_is_read_and_not_returned(self, tmp_path, monkeypatch):
        from ComfyTV.storage.script_llm import resolve_llm

        key_file = tmp_path / "deepseek.key"
        key_file.write_text("file-secret-key\n", encoding="utf-8")
        monkeypatch.setenv("COMFYTV_SCRIPT_LLM_KEY_FILE", str(key_file))
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("COMFYTV_SCRIPT_LLM_API_KEY", raising=False)
        resolved = resolve_llm(tmp_path, {"provider": "deepseek"})
        assert resolved["model"] == "deepseek-v4-pro"
        assert resolved["api_key"] == "file-secret-key"
        assert "api_key" not in resolved["public"]
        assert "file-secret-key" not in json.dumps(resolved["public"])


class BaseHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return


def make_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class TestHttp:
    @pytest.fixture()
    async def client(self):
        from ComfyTV import api  # noqa: F401
        import server

        app = web.Application()
        app.router.add_routes(server.PromptServer.instance.routes)
        test_server = TestServer(app)
        test_client = TestClient(test_server)
        await test_client.start_server()
        yield test_client
        await test_client.close()

    async def test_http_import(self, client, tmp_path):
        source = tmp_path / "box"
        meta = _write(source / "episodes/E01/script_v2.md", "P01_yangguo\n")
        root = tmp_path / "spark"
        manifest = _manifest([
            _file_entry("episodes/E01/script_v2.md", meta, doc="script"),
        ])
        response = await client.post("/comfytv/script_writer", json={
            "action": "import",
            "root": str(root),
            "source_dir": str(source),
            "manifest": manifest,
        })
        assert response.status == 200
        body = await response.json()
        assert body["added"] == 1
        assert body["file_count"] == 1
        stored = (
            root / "script" / "sessions" / body["session_id"]
            / "imported" / "episodes/E01/script_v2.md"
        ).read_text(encoding="utf-8")
        assert stored == "P01_yangguo\n"
