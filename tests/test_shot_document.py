"""Storyboard documents keep production shots.json fields and custom keys."""

from __future__ import annotations

import json
from pathlib import Path

from ComfyTV.nodes.stages import _common as c
from ComfyTV.nodes.stages.common.shot_document import (
    dump_shots_document,
    load_shots_document,
    parse_asset_ref,
    shot_image_prompt,
    validate_shots_document,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "e01_shots.json"


def _valid_doc(**shot_over) -> dict:
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
        "extras": 0,
        "prompts": {"h3_image": "chamber", "krea2": "chamber"},
    }
    shot.update(shot_over)
    return {"episode": "E01", "shots": [shot]}


class TestRoundTrip:
    def test_e01_fixture_is_unchanged_by_the_shaper(self):
        original = json.loads(FIXTURE.read_text(encoding="utf-8"))
        shaped = c._shape_storyboard_from_llm(FIXTURE.read_text(encoding="utf-8"))
        assert json.loads(shaped) == original
        assert json.loads(c._shape_storyboard_from_llm(shaped)) == original

    def test_load_dump_matches_the_fixture(self):
        original = json.loads(FIXTURE.read_text(encoding="utf-8"))
        loaded = load_shots_document(FIXTURE.read_text(encoding="utf-8"))
        assert json.loads(dump_shots_document(loaded)) == original

    def test_custom_keys_on_the_document_and_the_shot_survive(self):
        original = json.loads(FIXTURE.read_text(encoding="utf-8"))
        original["studio_tag"] = "night"
        original["shots"][0]["pipeline_note"] = {"keep": True}
        original["shots"][0]["camera_notes"] = "hold on the candle"
        text = json.dumps(original, ensure_ascii=False)
        assert json.loads(c._shape_storyboard_from_llm(text)) == original

    def test_legacy_json_keeps_unknown_keys_and_sibling_envelope_keys(self):
        raw = json.dumps({
            "title": "keep",
            "shots": [{
                "shot_no": "1",
                "duration": "4",
                "prompt": "door",
                "pipeline_note": "keep",
            }],
        })
        data = json.loads(c._shape_storyboard_from_llm(raw))
        assert data["title"] == "keep"
        shot = data["shots"][0]
        assert shot["pipeline_note"] == "keep"
        assert shot["prompt"] == "door"
        assert shot["duration"] == "4s"

    def test_legacy_sixteen_field_text_still_shapes(self):
        text = "镜号: 1\n时长: 4\n画面描述: 文本镜\n分镜提示词: from text\n"
        data = json.loads(c._shape_storyboard_from_llm(text))
        assert data["shots"][0]["shot_no"] == "1"
        assert data["shots"][0]["prompt"] == "from text"
        assert "shot_id" not in data["shots"][0]


class TestPromptAndRefs:
    def test_image_prompt_prefers_h3_then_krea_then_legacy(self):
        assert shot_image_prompt({
            "prompts": {"h3_image": " h3 ", "krea2": "krea"},
            "image_prompt": "legacy",
        }) == "h3"
        assert shot_image_prompt({"prompts": {"krea2": "krea"}}) == "krea"
        assert shot_image_prompt({"prompt": "door"}) == "door"
        assert shot_image_prompt({"scene_purpose": "not a prompt"}) == ""

    def test_parse_asset_ref_strips_version_and_view(self):
        assert parse_asset_ref("CHR_xiaolongnv@v1#front") == {
            "type": "series", "id": "CHR_xiaolongnv", "version": 1, "view": "front",
        }
        assert parse_asset_ref("NEW:LOC_huacong")["type"] == "new"
        assert parse_asset_ref("episode:E01S01_01/first.png")["type"] == "episode"


class TestValidate:
    def test_minimal_document_has_no_errors(self):
        errors, _warns = validate_shots_document(_valid_doc())
        assert errors == []

    def test_duplicate_id_short_duration_and_unknown_speaker(self):
        doc = _valid_doc(
            duration_s=3,
            dialogue=[{
                "line_id": "L1", "speaker": "CHR_missing", "kind": "onscreen",
                "text": "你好", "lips_visible": True,
            }],
        )
        doc["shots"].append(dict(doc["shots"][0], order=2))
        errors, _warns = validate_shots_document(doc)
        assert any("outside 5–15s" in error for error in errors)
        assert any("duplicate shot_id" in error for error in errors)
        assert any("not in characters" in error for error in errors)

    def test_dialogue_that_does_not_fit_is_an_error(self):
        doc = _valid_doc(dialogue=[{
            "line_id": "L1",
            "speaker": "CHR_xiaolongnv",
            "kind": "onscreen",
            "text": "一二三四五六七八九十一二三四五六七八九十甲",
            "lips_visible": True,
            "start_s": 0.2,
        }])
        errors, _warns = validate_shots_document(doc)
        assert any("needs" in error and "duration" in error for error in errors)

    def test_e01_fixture_reports_lists_not_exceptions(self):
        errors, warns = validate_shots_document(json.loads(FIXTURE.read_text(encoding="utf-8")))
        assert isinstance(errors, list)
        assert isinstance(warns, list)
        assert errors == ["M episode: total 126.0s vs target 90s (±15%)"]

    def test_episode_total_sums_duration_s_not_cut_to(self):
        doc = _valid_doc(duration_s=6, cut_to=5)
        second = dict(doc["shots"][0])
        second.update({
            "shot_id": "E01S01_02",
            "order": 2,
            "duration_s": 6,
            "cut_to": 5,
            "continuity": {"prev": "E01S01_01", "cut_motive": "next"},
        })
        doc["shots"].append(second)
        doc["target_duration_s"] = 10
        errors, _warns = validate_shots_document(doc)
        assert any("total 12.0s vs target 10s" in error for error in errors)
        assert not any("total 10.0s" in error for error in errors)

    def test_single_shot_skips_episode_target(self):
        doc = _valid_doc(duration_s=6, cut_to=5)
        doc["target_duration_s"] = 90
        errors, warns = validate_shots_document(doc)
        assert not any("episode:" in item for item in errors + warns)

    def test_partial_board_skips_episode_target(self):
        doc = _valid_doc(duration_s=6, cut_to=5)
        second = dict(doc["shots"][0])
        second.update({
            "shot_id": "E01S01_02",
            "order": 2,
            "continuity": {"prev": "E01S01_01", "cut_motive": "next"},
        })
        doc["shots"].append(second)
        doc["target_duration_s"] = 90
        errors, warns = validate_shots_document(doc)
        assert not any("episode:" in item for item in errors + warns)

    def test_full_episode_total_uses_duration_s(self):
        errors, _warns = validate_shots_document(json.loads(FIXTURE.read_text(encoding="utf-8")))
        assert any("total 126.0s vs target 90s" in error for error in errors)
