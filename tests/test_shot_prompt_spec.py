"""Optional shot-prompt fields, reference slots, and validate extensions.

Older shots.json documents stay on the same validation result. New fields are
checked only when they are present, unless strict mode is requested.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ComfyTV.nodes.stages.common.shot_document import (
    dump_shots_document,
    frames_for,
    load_shots_document,
    validate_shots_document,
)
from ComfyTV.nodes.stages.common.shot_prompt_spec import (
    ShotPromptConfig,
    assign_ref_slots,
    build_alias_index,
    canonicalize_shot,
    default_align_s,
    effective_constraints,
    h3_inputs,
    inferred_summary_tasks,
    merge_storyboard_fields,
    non_diegetic_music_text,
    normalize_config,
    reference_cap_for_workflow,
    soundscape_text,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "e01_shots.json"
E01_ERROR = "M episode: total 126.0s vs target 90s (±15%)"


def _doc(**over) -> dict:
    shot = {
        "shot_id": "E01S01_01",
        "scene_id": "E01S01",
        "order": 1,
        "duration_s": 6,
        "mode": "Ref2VA",
        "variation_type": "small",
        "characters": [{"id": "CHR_xiaolongnv@v1", "visible": "face"}],
        "dialogue": [],
        "refs": [{"role": "first_frame", "src": "episode:E01S01_01/first.png"}],
        "continuity": {"prev": None, "cut_motive": "open"},
    }
    shot.update(over)
    return {"episode": "E01", "shots": [shot]}


def _strict_doc() -> dict:
    doc = _doc(
        frames=frames_for(6),
        first_frame_desc="她站在石室中央。",
        last_frame_desc="她停下，看着门口。",
        end_state="她停下。",
        end_hold={"type": "line", "ref": "L1"},
        opening={"positions": "中央", "facing": "向左"},
        spatial={"anchor": "烛台", "distances": "一步"},
        performance="平静",
        optics="50mm, eye level, focus on the eyes",
        physics="袖口随抬手滞后",
        positive_locks=["白衣交领扣着"],
        visual_style={"style_ref": "STY_01@v1#style", "note": "暖烛"},
        constraints=["不出现现代物品"],
        aspect_ratio="16:9",
        dialogue=[{
            "line_id": "L1",
            "speaker": "CHR_xiaolongnv",
            "kind": "onscreen",
            "text": "好",
            "lips_visible": True,
            "start_s": 0.2,
            "language": "Chinese",
            "cutoff": False,
            "cross_cut": False,
        }],
        refs=[
            {
                "slot": 1,
                "role": "first_frame",
                "src": "episode:E01S01_01/first.png",
                "align_s": 0,
            },
            {
                "slot": 2,
                "role": "character",
                "src": "CHR_xiaolongnv@v1#front",
                "label": "小龙女",
                "retention": "fully_preserved",
            },
            {
                "slot": 1,
                "role": "voice",
                "media": "audio",
                "src": "VOX_xiaolongnv@v1#voice_ref",
                "retention": "reference",
            },
        ],
        subjects=[{
            "n": 1,
            "kind": "identity",
            "label": "小龙女",
            "text": "The woman in <Picture 2>.",
            "pictures": [2],
            "retention": "fully_preserved",
            "src": "CHR_xiaolongnv@v1#front",
        }],
        video_plan={
            "summary_en": "She speaks once and holds.",
            "summary_tasks": ["reference generation", "audio reference"],
            "soundscape_en": "A quiet stone room.",
            "retention_analysis": "<Subject 1> (appears in [Shot 1]): fully_preserved - face and robe.",
            "detailed_description": "The target video uses the stone room.",
            "integrated_description": "[Shot 1] She stands and speaks.",
            "non_diegetic_music": "N/A",
            "segments": [{
                "t0": 0,
                "t1": 6,
                "cut_type": "opening",
                "shot_size": "MS",
                "focus": "eyes",
                "line_ids": ["L1"],
                "lines": ["L1"],
                "ending": "She holds the look.",
            }],
        },
    )
    doc["aspect_ratio"] = "16:9"
    return doc


class TestLegacyDocuments:
    def test_e01_validate_and_round_trip_are_unchanged(self):
        original = json.loads(FIXTURE.read_text(encoding="utf-8"))
        loaded = load_shots_document(FIXTURE.read_text(encoding="utf-8"))
        assert loaded == original
        assert json.loads(dump_shots_document(loaded)) == original
        errors, warns = validate_shots_document(loaded)
        assert errors == [E01_ERROR]
        assert warns == []
        assert loaded == original

    def test_legacy_columns_do_not_gain_spec_fields(self):
        raw = {
            "title": "keep",
            "shots": [{"shot_no": "1", "duration": "4", "prompt": "door"}],
        }
        loaded = load_shots_document(raw)
        assert "aspect_ratio" not in loaded
        assert "slot" not in loaded["shots"][0]
        assert "end_hold" not in loaded["shots"][0]
        assert json.loads(dump_shots_document(loaded)) == raw

    def test_minimal_document_still_has_no_errors(self):
        errors, warns = validate_shots_document(_doc())
        assert errors == []
        assert warns == []


class TestSlotOrder:
    def _mixed(self) -> dict:
        return {
            "characters": [
                {"id": "CHR_yangguo@v1"},
                {"id": "CHR_xiaolongnv@v1"},
            ],
            "dialogue": [{"speaker": "CHR_xiaolongnv", "line_id": "L1"}],
            "refs": [
                {"role": "prev_last_frame", "src": "episode:prev/last.png"},
                {"role": "character", "src": "CHR_yangguo@v1#front", "label": "杨过"},
                {"role": "character", "src": "CHR_xiaolongnv@v1#front", "label": "小龙女"},
                {"role": "location", "src": "LOC_room@v1#empty"},
                {"role": "prop", "src": "PRP_zhutai@v1#lit"},
                {"role": "style", "src": "STY_01@v1#style"},
                {"role": "first_frame", "src": "episode:this/first.png"},
                {"role": "voice", "media": "audio", "src": "VOX_xiaolongnv@v1#voice_ref"},
            ],
        }

    def test_default_order_puts_the_start_frame_first(self):
        shot = self._mixed()
        before = copy.deepcopy(shot["refs"])
        refs = assign_ref_slots(shot)
        assert shot["refs"] == before
        pictures = [ref for ref in refs if ref.get("media") != "audio"]
        assert [ref["role"] for ref in pictures] == [
            "first_frame",
            "prev_last_frame",
            "character",
            "character",
            "location",
            "prop",
            "style",
        ]
        assert pictures[2]["label"] == "小龙女"
        assert pictures[3]["label"] == "杨过"
        assert [ref["slot"] for ref in pictures] == list(range(1, 8))
        audio = [ref for ref in refs if ref.get("media") == "audio"]
        assert audio[0]["slot"] == 1

    def test_libtv_order_puts_the_tail_frame_at_picture_4(self):
        shot = {
            "dialogue": [
                {"speaker": "CHR_a"},
                {"speaker": "CHR_b"},
            ],
            "refs": [
                {"role": "prev_last_frame", "src": "episode:prev/last.png"},
                {"role": "location", "src": "LOC_room@v1#empty"},
                {"role": "character", "src": "CHR_b@v1#front"},
                {"role": "character", "src": "CHR_a@v1#front"},
            ],
        }
        refs = assign_ref_slots(shot, config=ShotPromptConfig(slot_order="libtv"))
        assert [(ref["src"], ref["slot"]) for ref in refs] == [
            ("CHR_a@v1#front", 1),
            ("CHR_b@v1#front", 2),
            ("LOC_room@v1#empty", 3),
            ("episode:prev/last.png", 4),
        ]

    def test_apply_writes_the_numbered_refs(self):
        shot = _doc()["shots"][0]
        assign_ref_slots(shot, apply=True)
        assert shot["refs"][0]["slot"] == 1


class TestValidateExtensions:
    def test_strict_h3_document_passes(self):
        errors, warns = validate_shots_document(
            _strict_doc(),
            config=ShotPromptConfig(strict=True),
        )
        assert errors == []
        assert warns == []

    def test_strict_reports_missing_structured_fields(self):
        errors, _warns = validate_shots_document(
            _doc(),
            config=normalize_config(strict=True),
        )
        text = "\n".join(errors)
        assert "aspect_ratio is required" in text
        assert "opening picture is required" in text
        assert "ending picture is required" in text
        assert "end_hold is required" in text
        assert "at least one segment" in text

    def test_segment_time_errors(self):
        gap = _doc(video_plan={"segments": [
            {"t0": 0, "t1": 2},
            {"t0": 3, "t1": 6},
        ]})
        errors, _warns = validate_shots_document(gap)
        assert any("does not meet" in error for error in errors)

        late = _doc(video_plan={"segments": [{"t0": 1, "t1": 6}]})
        errors, _warns = validate_shots_document(late)
        assert any("start at 0" in error for error in errors)

        short = _doc(video_plan={"segments": [{"t0": 0, "t1": 5}]})
        errors, _warns = validate_shots_document(short)
        assert any("not duration_s" in error for error in errors)

        backwards = _doc(video_plan={"segments": [{"t0": 4, "t1": 2}]})
        errors, _warns = validate_shots_document(backwards)
        assert any("not before" in error for error in errors)

    def test_bad_cut_type_and_unknown_line_id(self):
        doc = _doc(video_plan={"segments": [{
            "t0": 0,
            "t1": 6,
            "cut_type": "smash",
            "line_ids": ["L404"],
        }]})
        errors, _warns = validate_shots_document(doc)
        assert any("cut_type" in error for error in errors)
        assert any("L404" in error and "dialogue" in error for error in errors)

    def test_frames_stay_a_warning_until_asked(self):
        doc = _doc(frames=1)
        errors, warns = validate_shots_document(doc)
        assert errors == []
        assert any("17n+5" in warning for warning in warns)
        errors, warns = validate_shots_document(
            doc, config=ShotPromptConfig(frames_mismatch="error"),
        )
        assert any("17n+5" in error for error in errors)
        assert not any("17n+5" in warning for warning in warns)

    def test_duration_must_be_whole_seconds(self):
        errors, _warns = validate_shots_document(_doc(duration_s=6.5))
        assert any("integer" in error for error in errors)

    def test_picture_slots_must_be_dense(self):
        doc = _doc(refs=[
            {"slot": 1, "role": "first_frame", "src": "episode:a/first.png"},
            {"slot": 3, "role": "character", "src": "CHR_xiaolongnv@v1#front"},
        ])
        errors, _warns = validate_shots_document(doc)
        assert any("no gaps" in error for error in errors)

        swapped = _doc(refs=[
            {"slot": 2, "role": "character", "src": "CHR_xiaolongnv@v1#front"},
            {"slot": 1, "role": "first_frame", "src": "episode:a/first.png"},
        ])
        errors, _warns = validate_shots_document(swapped)
        assert not any("slot" in error for error in errors)

    def test_picture_cap_is_separate_from_audio(self):
        pictures = [
            {"slot": index, "role": "prop", "src": f"PRP_item{index}@v1#lit"}
            for index in range(1, 11)
        ]
        errors, _warns = validate_shots_document(_doc(refs=pictures))
        assert any("above the limit 9" in error for error in errors)

        nine = pictures[:9]
        nine.append({
            "slot": 1,
            "role": "voice",
            "media": "audio",
            "src": "VOX_xiaolongnv@v1#voice_ref",
        })
        errors, _warns = validate_shots_document(_doc(refs=nine))
        assert not any("above the limit" in error for error in errors)

        capped = _doc(refs=[
            {"role": "first_frame", "src": "episode:a/first.png"},
            {"role": "character", "src": "CHR_xiaolongnv@v1#front"},
            {"role": "prop", "src": "PRP_zhutai@v1#lit"},
        ])
        errors, _warns = validate_shots_document(capped, max_refs=2, ref_limit_reason="FL2VA")
        assert any("above the limit 2" in error and "FL2VA" in error for error in errors)

    def test_sheet_view_is_rejected_until_allowed(self):
        doc = _doc(refs=[{"role": "character", "src": "CHR_xiaolongnv@v1#sheet"}])
        errors, _warns = validate_shots_document(doc)
        assert any("sheet" in error for error in errors)
        errors, _warns = validate_shots_document(
            doc, config=ShotPromptConfig(allow_sheet_in_video_refs=True),
        )
        assert not any("sheet" in error for error in errors)

    def test_chain_tail_is_off_until_configured(self):
        doc = _doc(continuity={"prev": "E01S00_09", "cut_motive": "cut"})
        errors, _warns = validate_shots_document(doc)
        assert not any("prev_last_frame ref" in error for error in errors)
        errors, _warns = validate_shots_document(
            doc, config=ShotPromptConfig(require_chain_tail_frame=True),
        )
        assert any("without a prev_last_frame ref" in error for error in errors)

    def test_names_new_refs_and_external_paths(self):
        named = _doc(refs=[{"role": "character", "src": "小龙女"}])
        errors, _warns = validate_shots_document(named)
        assert any("unresolved" in error for error in errors)

        index = build_alias_index({"小龙女": "CHR_xiaolongnv@v1#front"})
        errors, warns = validate_shots_document(named, alias_index=index)
        assert errors == []
        assert any("canonical id" in warning for warning in warns)

        fresh = _doc(refs=[{"role": "location", "src": "NEW:LOC_huacong"}])
        errors, warns = validate_shots_document(fresh)
        assert errors == []
        assert any("unresolved" in warning for warning in warns)
        errors, _warns = validate_shots_document(
            fresh, config=ShotPromptConfig(unresolved_new_refs="error"),
        )
        assert any("unresolved" in error for error in errors)

        external = _doc(refs=[{"role": "character", "src": "/workspace/vp/face.png"}])
        errors, warns = validate_shots_document(external)
        assert errors == []
        assert any("ComfyTV asset code" in warning for warning in warns)

    def test_h3_field_shapes(self):
        doc = _doc(
            end_hold={"type": "line", "ref": "missing"},
            optics=3,
            video_plan={
                "summary_tasks": ["not-a-task"],
                "segments": [{
                    "t0": 0,
                    "t1": 6,
                    "line_ids": [],
                }],
            },
            subjects=[{
                "n": 1,
                "kind": "identity",
                "pictures": [4],
                "retention": "fully_copy",
            }],
            refs=[
                {
                    "slot": 1,
                    "role": "first_frame",
                    "src": "episode:a/first.png",
                    "align_s": 20,
                    "retention": "fully_copy",
                },
                {"slot": 1, "role": "location", "src": "LOC_a@v1#empty"},
                {"slot": 1, "role": "location", "src": "LOC_b@v1#empty"},
            ],
            dialogue=[{
                "line_id": "L1",
                "speaker": "CHR_xiaolongnv",
                "kind": "onscreen",
                "text": "好",
                "cutoff": "yes",
            }],
        )
        errors, _warns = validate_shots_document(doc)
        text = "\n".join(errors)
        assert "end_hold.ref missing" in text
        assert "optics must be a string" in text
        assert "not an H3 summary task" in text
        assert "picture slot 4" in text
        assert "retention" in text
        assert "align_s" in text
        assert "one location reference" in text
        assert "cutoff must be a boolean" in text

    def test_constraints_can_turn_a_default_off(self):
        shot = {"constraints": ["-无额外字幕、水印、配乐", "不出现现代物品"]}
        assert effective_constraints(shot) == [
            "人物与道具保持一致",
            "不重复已完成动作",
            "不出现现代物品",
        ]
        errors, _warns = validate_shots_document(_doc(constraints=["-不存在的限制"]))
        assert any("cannot be turned off" in error for error in errors)

    def test_shot_ids_skip_other_shots(self):
        doc = _doc()
        second = dict(doc["shots"][0])
        second.update({
            "shot_id": "E01S01_02",
            "order": 2,
            "refs": [
                {"slot": 1, "role": "first_frame", "src": "episode:a/first.png"},
                {"slot": 3, "role": "prop", "src": "PRP_zhutai@v1#lit"},
            ],
            "continuity": {"prev": "E01S01_01", "cut_motive": "next"},
        })
        doc["shots"].append(second)
        errors, _warns = validate_shots_document(doc, shot_ids=["E01S01_01"])
        assert not any("no gaps" in error for error in errors)


class TestAliasesAndStoryboardYaml:
    def test_library_names_resolve_to_front_not_sheet(self):
        index = build_alias_index(assets=[{
            "id": "CHR_xiaolongnv",
            "version": 1,
            "kind": "character",
            "status": "locked",
            "name_zh": "小龙女",
            "aliases": ["龙儿"],
            "files": {"front": {}, "sheet": {}},
        }])
        assert index.lookup("小龙女") == ("CHR_xiaolongnv@v1#front", None)
        assert index.lookup("龙儿") == ("CHR_xiaolongnv@v1#front", None)
        shot = {"refs": [{"role": "character", "src": "小龙女"}]}
        result = canonicalize_shot(shot, index)
        assert shot["refs"][0]["src"] == "小龙女"
        assert result["refs"][0]["src"] == "CHR_xiaolongnv@v1#front"
        assert result["issues"] == []

    def test_duplicate_names_conflict(self):
        index = build_alias_index(assets=[
            {
                "id": "CHR_a", "version": 1, "kind": "character", "status": "locked",
                "name_zh": "小龙女", "files": {"front": {}},
            },
            {
                "id": "CHR_b", "version": 1, "kind": "character", "status": "locked",
                "name_zh": "小龙女", "files": {"front": {}},
            },
        ])
        shot = {"refs": [{"role": "character", "src": "小龙女"}]}
        result = canonicalize_shot(shot, index, apply=True)
        assert shot["refs"][0]["src"] == "小龙女"
        assert any(issue.startswith("conflict:") for issue in result["issues"])

    def test_storyboard_yaml_merge_is_off_by_default(self):
        shot = {"continuity": {"prev": None, "cut_motive": "open"}}
        source = {
            "chain_from": "E01S01_00",
            "chain_method": "cut",
            "motion_source": "tail",
            "eyeline": "向左",
        }
        kept = merge_storyboard_fields(shot, source)
        assert kept["continuity"]["prev"] is None
        assert "opening" not in kept
        assert "chain_method" not in kept
        merged = merge_storyboard_fields(
            shot, source, config=ShotPromptConfig(merge_storyboard_yaml=True),
        )
        assert merged["continuity"]["prev"] == "E01S01_00"
        assert merged["opening"]["eyeline"] == "向左"
        assert merged["chain_method"] == "cut"
        assert merged["motion_source"] == "tail"
        assert merged["chain_from"] == "E01S01_00"


class TestH3Inputs:
    def test_full_shot_exposes_ref2va_and_base_sections(self):
        doc = _strict_doc()
        shot = doc["shots"][0]
        before = copy.deepcopy(shot)
        inputs = h3_inputs(shot, doc)
        assert shot == before
        assert inputs["sections"]["ref2va"] == (
            "subject_definitions",
            "summary",
            "retention_analysis",
            "detailed_description",
            "overall_soundscape",
            "non_diegetic_music",
        )
        assert inputs["sections"]["base"][0] == "alignment"
        assert inputs["summary"] == "She speaks once and holds."
        assert inputs["summary_tasks"] == ["reference generation", "audio reference"]
        assert inputs["retention_analysis"].startswith("<Subject 1>")
        assert inputs["soundscape"] == "A quiet stone room."
        assert inputs["non_diegetic_music"] == "N/A"
        assert inputs["pictures"][0]["align_s"] == 0
        assert inputs["pictures"][1]["slot"] == 2
        assert inputs["audios"][0]["src"].startswith("VOX_")
        assert inputs["subjects"][0]["pictures"] == [2]
        assert "不出现现代物品" in inputs["constraints"]
        assert inputs["optics"].startswith("50mm")
        assert inputs["end_hold"]["ref"] == "L1"
        assert inputs["dialogue"][0]["language"] == "Chinese"

    def test_missing_authored_text_falls_back_without_writing(self):
        shot = _doc(
            audio={"ambience": "石室回声", "h3_music": "none"},
            refs=[
                {"role": "first_frame", "src": "episode:a/first.png"},
                {"role": "last_frame", "src": "episode:a/last.png"},
                {"role": "prev_last_frame", "src": "episode:prev/last.png"},
            ],
        )["shots"][0]
        assert soundscape_text(shot) == "石室回声"
        assert non_diegetic_music_text(shot) is None
        assert default_align_s(shot["refs"][0], 6) == 0
        assert default_align_s(shot["refs"][1], 6) == 6
        tasks = inferred_summary_tasks(shot)
        assert tasks[0] == "reference generation"
        assert "video continuation" in tasks
        inputs = h3_inputs(shot)
        assert inputs["soundscape"] == "石室回声"
        assert inputs["non_diegetic_music"] is None
        assert inputs["detailed_description"] is None
        assert "video_plan" not in shot

    def test_workflow_caps_follow_the_h3_nodes(self):
        assert reference_cap_for_workflow("Local MiniMax H3 FLF2V")[0] == 2
        assert reference_cap_for_workflow("MiniMaxH3ImageToVideo")[0] == 2
        assert reference_cap_for_workflow("Local MiniMax H3 R2V")[0] == 9
        assert reference_cap_for_workflow("") == (None, None)

    def test_strict_keyword_promotes_checks_unless_overridden(self):
        promoted = normalize_config(strict=True)
        assert promoted.require_chain_tail_frame is True
        assert promoted.frames_mismatch == "error"
        held = normalize_config(strict=True, require_chain_tail_frame=False)
        assert held.strict is True
        assert held.require_chain_tail_frame is False


class TestStoryboardValidateTool:
    async def test_validate_accepts_strict_workflow_and_aliases(self):
        from ComfyTV.api.mcp_tools.storyboard_tool import _storyboard

        passed = await _storyboard({"action": "validate", "document": _strict_doc()})
        assert passed["ok"] is True

        missing = await _storyboard({
            "action": "validate",
            "document": _doc(),
            "strict": True,
        })
        assert missing["ok"] is False
        assert any("end_hold" in error for error in missing["errors"])

        capped = await _storyboard({
            "action": "validate",
            "workflow": "Local MiniMax H3 FLF2V",
            "document": _doc(refs=[
                {"role": "first_frame", "src": "episode:a/first.png"},
                {"role": "last_frame", "src": "episode:a/last.png"},
                {"role": "character", "src": "CHR_xiaolongnv@v1#front"},
            ]),
        })
        assert capped["ok"] is False
        assert any("limit 2" in error for error in capped["errors"])

        named = await _storyboard({
            "action": "validate",
            "aliases": {"小龙女": "CHR_xiaolongnv@v1#front"},
            "document": _doc(refs=[{"role": "character", "src": "小龙女"}]),
        })
        assert named["ok"] is True
        assert any("canonical id" in warning for warning in named["warnings"])


def test_unknown_slot_order_is_rejected():
    with pytest.raises(ValueError, match="slot_order"):
        ShotPromptConfig(slot_order="tutorial")
