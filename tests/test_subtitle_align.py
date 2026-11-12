"""Script-aligned subtitles: Whisper times, original dialogue text."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ComfyTV.runners.text_overlay import _NO_LINE_START, wrap_text


def _cells(text: str) -> int:
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


def _words(text: str, start: float, end: float) -> dict:
    chars = list(text)
    if not chars:
        return {"start": start, "end": end, "text": text, "words": []}
    span = end - start
    words = []
    for index, ch in enumerate(chars):
        words.append({
            "start": start + span * index / len(chars),
            "end": start + span * (index + 1) / len(chars),
            "text": ch,
        })
    return {"start": start, "end": end, "text": text, "words": words}


def _cue_texts(srt: str) -> list[str]:
    blocks = [block for block in srt.strip().split("\n\n") if block.strip()]
    texts = []
    for block in blocks:
        lines = block.splitlines()
        texts.append("\n".join(lines[2:]))
    return texts


LIAN_SCRIPT = "过儿专心些，心神不定是练不成玉女心经的。"
LIAN_HEARD = "过儿专心些,心神不定是炼不成玉女心经的。"


class TestLianHomophone:
    def test_word_times_keep_the_script_character(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        result = align_subtitles(
            [_words(LIAN_HEARD, 0.0, 6.1)],
            LIAN_SCRIPT,
            max_line_cells=80,
        )
        cue = _cue_texts(result["srt"])[0]
        assert cue == LIAN_SCRIPT
        assert "炼" not in cue
        assert "00:00:00,000" in result["srt"]
        assert "00:00:06,100" in result["srt"]

        sentence = result["report"]["sentences"][0]
        assert sentence["text"] == LIAN_SCRIPT
        assert sentence["heard"] == "过儿专心些心神不定是炼不成玉女心经的"
        assert sentence["status"] == "aligned"
        assert sentence["content_chars"] == 18
        assert sentence["exact"] == 17
        assert sentence["confidence"] == pytest.approx(17 / 18, abs=1e-4)
        assert sentence["start"] == pytest.approx(0.0, abs=1e-3)
        assert sentence["end"] == pytest.approx(6.1, abs=1e-3)
        assert sentence["substitutions"] == [{"script": "练", "whisper": "炼", "at": 11}]
        assert result["report"]["low_confidence"] == []
        assert result["report"]["interpolated"] == []

    def test_segment_without_words_still_replaces_the_homophone(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        result = align_subtitles(
            [{"start": 1.0, "end": 5.0, "text": "这心经要两人同炼"}],
            "这心经要两人同练",
            max_line_cells=80,
        )
        assert _cue_texts(result["srt"]) == ["这心经要两人同练"]
        sentence = result["report"]["sentences"][0]
        assert sentence["start"] == pytest.approx(1.0, abs=1e-3)
        assert sentence["end"] == pytest.approx(5.0, abs=1e-3)
        assert sentence["substitutions"] == [{"script": "练", "whisper": "炼", "at": 7}]
        assert sentence["confidence"] == pytest.approx(7 / 8, abs=1e-4)


class TestEdits:
    def test_missing_and_extra_characters_do_not_shift_the_line(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        # 上 is missing, 啊 is inserted between 能 and 炼, 哦 trails the line.
        whisper = [{
            "start": 0.0,
            "end": 8.0,
            "text": "寒玉床不能啊炼哦",
            "words": [
                {"start": 0, "end": 1, "text": "寒"},
                {"start": 1, "end": 2, "text": "玉"},
                {"start": 2, "end": 3, "text": "床"},
                {"start": 3, "end": 4, "text": "不"},
                {"start": 4, "end": 5, "text": "能"},
                {"start": 5, "end": 6, "text": "啊"},
                {"start": 6, "end": 7, "text": "炼"},
                {"start": 7, "end": 8, "text": "哦"},
            ],
        }]
        result = align_subtitles(whisper, "寒玉床上不能练", max_line_cells=80)
        sentence = result["report"]["sentences"][0]
        assert sentence["text"] == "寒玉床上不能练"
        assert sentence["missing"] == [{"script": "上", "at": 3}]
        assert sentence["extra"] == [{"whisper": "啊"}]
        assert {"script": "练", "whisper": "炼", "at": 6} in sentence["substitutions"]
        assert sentence["start"] == pytest.approx(0.0, abs=1e-3)
        assert sentence["end"] == pytest.approx(7.0, abs=1e-3)
        assert sentence["exact"] == 5
        assert sentence["content_chars"] == 7

        leading = align_subtitles(
            [_words("呃不能炼", 0.0, 4.0)],
            "不能练",
            max_line_cells=80,
        )["report"]["sentences"][0]
        assert leading["text"] == "不能练"
        assert leading["start"] == pytest.approx(1.0, abs=1e-3)
        assert leading["end"] == pytest.approx(4.0, abs=1e-3)

    def test_punctuation_width_and_case_do_not_change_the_match(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        result = align_subtitles(
            [_words("你好,世界!", 1.0, 2.0)],
            "你好，世界！",
            max_line_cells=80,
        )
        sentence = result["report"]["sentences"][0]
        assert sentence["text"] == "你好，世界！"
        assert sentence["confidence"] == 1.0
        assert sentence["substitutions"] == []
        assert sentence["start"] == pytest.approx(1.0, abs=1e-3)
        assert sentence["end"] == pytest.approx(2.0, abs=1e-3)
        assert "," not in _cue_texts(result["srt"])[0]

        folded = align_subtitles(
            [{"start": 0.0, "end": 1.0, "text": "第３集OK"}],
            "第3集Ok",
            max_line_cells=40,
        )["report"]["sentences"][0]
        assert folded["text"] == "第3集Ok"
        assert folded["confidence"] == 1.0
        assert folded["substitutions"] == []

    def test_srt_timeline_is_segment_level(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        srt = "1\n00:00:00,000 --> 00:00:01,000\n同炼\n"
        result = align_subtitles(srt, "同练", max_line_cells=40)
        assert _cue_texts(result["srt"]) == ["同练"]
        assert result["report"]["sentences"][0]["substitutions"][0]["whisper"] == "炼"

    def test_numeric_srt_body_keeps_its_whisper_time(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        srt = "1\n00:00:05,000 --> 00:00:06,000\n2026\n"
        result = align_subtitles(srt, "2026", max_line_cells=40)
        sentence = result["report"]["sentences"][0]
        assert sentence["status"] == "aligned"
        assert sentence["text"] == "2026"
        assert sentence["heard"] == "2026"
        assert sentence["start"] == pytest.approx(5.0, abs=1e-3)
        assert sentence["end"] == pytest.approx(6.0, abs=1e-3)

        numbered = align_subtitles(
            "1\n00:00:01,000 --> 00:00:02,000\n1\n",
            "1",
            max_line_cells=40,
        )["report"]["sentences"][0]
        assert numbered["status"] == "aligned"
        assert numbered["start"] == pytest.approx(1.0, abs=1e-3)
        assert numbered["end"] == pytest.approx(2.0, abs=1e-3)


class TestUnmatchedAndConfidence:
    def test_unrecognized_line_is_interpolated_and_marked(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        script = "第一句。\n这句完全没读。\n第三句到此。"
        whisper = [
            _words("第一句。", 0.0, 1.0),
            _words("第三句到此。", 3.0, 4.5),
        ]
        result = align_subtitles(whisper, script, max_line_cells=80)
        sentences = result["report"]["sentences"]
        assert [row["status"] for row in sentences] == [
            "aligned", "interpolated", "aligned",
        ]
        missed = sentences[1]
        assert missed["text"] == "这句完全没读。"
        assert missed["mark"] == "interpolated"
        assert missed["start"] == pytest.approx(1.0, abs=1e-3)
        assert missed["end"] == pytest.approx(3.0, abs=1e-3)
        assert missed["note"] == "剧本有此句，Whisper 未识别，时间由相邻句插值"
        assert result["report"]["interpolated"] == [missed]
        assert result["report"]["low_confidence"] == []
        assert "未识别并插值 1 句：L002" in result["report"]["summary"]
        assert "这句完全没读。" in result["srt"]
        assert "未识别" not in result["srt"]

    def test_edges_borrow_the_neighbor_clock(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        script = "开头没读呀\n中间这句\n结尾也没读"
        result = align_subtitles(
            [_words("中间这句", 2.0, 3.0)],
            script,
            max_line_cells=80,
        )
        rows = result["report"]["sentences"]
        assert rows[0]["status"] == "interpolated"
        assert rows[0]["start"] == pytest.approx(0.75, abs=1e-3)
        assert rows[0]["end"] == pytest.approx(2.0, abs=1e-3)
        assert rows[1]["status"] == "aligned"
        assert rows[1]["start"] == pytest.approx(2.0, abs=1e-3)
        assert rows[1]["end"] == pytest.approx(3.0, abs=1e-3)
        assert rows[2]["status"] == "interpolated"
        assert rows[2]["start"] == pytest.approx(3.0, abs=1e-3)
        assert rows[2]["end"] == pytest.approx(4.25, abs=1e-3)
        assert [row["line_id"] for row in result["report"]["interpolated"]] == ["L001", "L003"]

    def test_low_confidence_lines_are_listed_apart_from_misses(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        script = "甲乙丙丁戊己庚辛"
        heard = "甲乙丙丁壬癸子丑"
        result = align_subtitles([_words(heard, 0.0, 2.0)], script, max_line_cells=80)
        sentence = result["report"]["sentences"][0]
        assert sentence["status"] == "aligned"
        assert sentence["text"] == script
        assert sentence["heard"] == heard
        assert sentence["exact"] == 4
        assert sentence["confidence"] == pytest.approx(0.5, abs=1e-4)
        assert result["report"]["low_confidence"] == [sentence]
        assert result["report"]["interpolated"] == []
        assert "低置信度 1 句：L001" in result["report"]["summary"]
        assert sentence["note"] == "对齐置信度低于阈值"

        loose = align_subtitles(
            [_words(heard, 0.0, 2.0)],
            script,
            max_line_cells=80,
            low_confidence=0.4,
        )
        assert loose["report"]["low_confidence"] == []

    def test_nothing_recognized_is_paced_and_marked(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        result = align_subtitles([], "一二三四。", max_line_cells=40)
        sentence = result["report"]["sentences"][0]
        assert sentence["status"] == "interpolated"
        assert sentence["start"] == pytest.approx(0.0, abs=1e-3)
        assert sentence["end"] == pytest.approx(1.0, abs=1e-3)
        assert result["report"]["low_confidence"] == []
        assert _cue_texts(result["srt"]) == ["一二三四。"]


class TestCuesAndShots:
    def test_wrap_reuses_subtitle_line_breaks(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        samples = ["一二三四，五六七八", "一二三四《五六七》"]
        for text in samples:
            result = align_subtitles([_words(text, 0.0, 1.0)], text, max_line_cells=10)
            cue = _cue_texts(result["srt"])[0]
            assert cue == wrap_text(text, _cells, 10)
            assert result["srt"].count("-->") == 1
            for line in cue.split("\n"):
                assert _cells(line) <= 10
                assert not line.startswith(tuple(_NO_LINE_START))

        long = "这是一句比较长的中文字幕，用来测试它在视频画面里会不会超出边界。"
        wrapped = align_subtitles([_words(long, 0.0, 4.0)], long)
        cue = _cue_texts(wrapped["srt"])[0]
        assert "\n" in cue
        assert cue == wrap_text(long, _cells, 32)
        for line in cue.split("\n"):
            assert _cells(line) <= 32
            assert not line.startswith(tuple(_NO_LINE_START))

    def test_shots_keep_order_and_ignore_the_planned_start(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        script = {
            "shots": [
                {
                    "shot_id": "E01S01_02",
                    "dialogue": [{
                        "line_id": "L001",
                        "speaker": "CHR_xiaolongnv",
                        "text": "第一句。",
                        "start_s": 9.0,
                    }],
                },
                {
                    "shot_id": "E01S01_03",
                    "dialogue": [{
                        "line_id": "L002",
                        "speaker": "CHR_yangguo",
                        "text": "第二句。",
                    }],
                },
            ],
        }
        whisper = [
            _words("第一句。", 0.2, 1.0),
            _words("第二句。", 2.0, 3.5),
        ]
        result = align_subtitles(whisper, script, max_line_cells=40)
        rows = result["report"]["sentences"]
        assert [(row["shot_id"], row["line_id"], row["speaker"]) for row in rows] == [
            ("E01S01_02", "L001", "CHR_xiaolongnv"),
            ("E01S01_03", "L002", "CHR_yangguo"),
        ]
        assert rows[0]["start"] == pytest.approx(0.2, abs=1e-3)
        assert rows[1]["start"] == pytest.approx(2.0, abs=1e-3)
        assert _cue_texts(result["srt"]) == ["第一句。", "第二句。"]

    def test_per_shot_clocks_are_placed_in_order(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        script = {
            "shots": [
                {"shot_id": "S1", "dialogue": [{"line_id": "L1", "text": "同练"}]},
                {"shot_id": "S2", "dialogue": [{"line_id": "L2", "text": "你好"}]},
            ],
        }
        local = align_subtitles(
            [
                {"shot_id": "S1", "start": 0.0, "end": 1.0, "text": "同炼"},
                {"shot_id": "S2", "start": 0.0, "end": 1.2, "text": "你好"},
            ],
            script,
            max_line_cells=40,
        )
        rows = local["report"]["sentences"]
        assert rows[0]["text"] == "同练"
        assert rows[0]["end"] == pytest.approx(1.0, abs=1e-3)
        assert rows[1]["text"] == "你好"
        assert rows[1]["start"] == pytest.approx(1.0, abs=1e-3)
        assert rows[1]["end"] == pytest.approx(2.2, abs=1e-3)

        absolute = align_subtitles(
            [
                {"shot_id": "S1", "start": 0.0, "end": 1.0, "text": "同炼"},
                {"shot_id": "S2", "start": 5.0, "end": 6.2, "text": "你好"},
            ],
            script,
            max_line_cells=40,
        )
        assert absolute["report"]["sentences"][1]["start"] == pytest.approx(5.0, abs=1e-3)
        assert absolute["report"]["sentences"][1]["end"] == pytest.approx(6.2, abs=1e-3)

    def test_untagged_script_uses_the_combined_shot_clock(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        single = align_subtitles(
            {"S1": [{"start": 10.0, "end": 11.0, "text": "你好"}]},
            "你好",
            max_line_cells=40,
        )["report"]["sentences"][0]
        assert single["status"] == "aligned"
        assert single["start"] == pytest.approx(10.0, abs=1e-3)
        assert single["end"] == pytest.approx(11.0, abs=1e-3)

        combined = align_subtitles(
            {
                "S1": [{"start": 0.0, "end": 1.0, "text": "你好"}],
                "S2": [{"start": 0.0, "end": 1.2, "text": "世界"}],
            },
            "你好\n世界",
            max_line_cells=40,
        )["report"]["sentences"]
        assert combined[0]["status"] == "aligned"
        assert combined[0]["end"] == pytest.approx(1.0, abs=1e-3)
        assert combined[1]["text"] == "世界"
        assert combined[1]["start"] == pytest.approx(1.0, abs=1e-3)
        assert combined[1]["end"] == pytest.approx(2.2, abs=1e-3)

    def test_long_alignment_still_keeps_the_homophone(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        prefix = "甲" * 400
        script = prefix + "练" + prefix
        heard = prefix + "炼" + prefix
        result = align_subtitles(
            [{"start": 0.0, "end": 10.0, "text": heard}],
            script,
            max_line_cells=80,
        )
        sentence = result["report"]["sentences"][0]
        assert sentence["text"] == script
        assert sentence["substitutions"] == [{"script": "练", "whisper": "炼", "at": 400}]
        assert sentence["exact"] == 800
        assert "炼" not in result["srt"]

    def test_by_shot_map_matches_each_shot_on_its_own(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        script = {
            "shots": [
                {"shot_id": "S1", "dialogue": [{"line_id": "L1", "text": "同练"}]},
                {"shot_id": "S2", "dialogue": [{"line_id": "L2", "text": "寒玉床上不能练。"}]},
            ],
        }
        result = align_subtitles(
            {
                "S2": [_words("寒玉床上不能炼。", 0.0, 2.0)],
                "S1": [_words("同炼", 0.0, 1.0)],
            },
            script,
            max_line_cells=80,
        )
        rows = result["report"]["sentences"]
        assert rows[0]["text"] == "同练"
        assert rows[0]["end"] == pytest.approx(1.0, abs=1e-3)
        assert rows[1]["text"] == "寒玉床上不能练。"
        assert rows[1]["start"] == pytest.approx(1.0, abs=1e-3)
        assert rows[1]["substitutions"][0]["whisper"] == "炼"

    def test_empty_script_is_an_empty_srt(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        result = align_subtitles([_words("你好", 0.0, 1.0)], "  \n")
        assert result["srt"] == ""
        assert result["report"]["sentences"] == []
        assert result["report"]["summary"] == "低置信度 0 句。未识别并插值 0 句。"

    def test_invalid_json_is_rejected(self):
        from ComfyTV.runners.subtitle_align import align_subtitles

        with pytest.raises(ValueError, match="script"):
            align_subtitles([], "{")
        with pytest.raises(ValueError, match="whisper"):
            align_subtitles("你好你好", "你好")


class TestNode:
    def test_schema_is_registered_beside_whisper(self):
        from ComfyTV.nodes.stages import stage_classes
        from ComfyTV.nodes.stages.common.meta import NON_STAGE_NODES
        from ComfyTV.nodes.subtitle_align import SubtitleAlignStage

        schema = SubtitleAlignStage.define_schema()
        assert schema.kw["node_id"] == "ComfyTV.SubtitleAlignStage"
        names = [item.args[0] for item in schema.kw["inputs"] if item.args]
        assert names[:5] == [
            "segments", "script", "max_line_cells", "low_confidence", "media",
        ]
        assert [item.args[0] for item in schema.kw["outputs"]] == ["srt", "report", "text"]
        assert "SubtitleAlignStage" in NON_STAGE_NODES
        assert "SubtitleAlignStage" in {cls.__name__ for cls in stage_classes()}

    def test_execute_writes_script_srt_and_report(self, tmp_path, monkeypatch):
        import folder_paths
        from ComfyTV.nodes.subtitle_align import SubtitleAlignStage

        monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
        result = SubtitleAlignStage.execute(
            segments=json.dumps([_words(LIAN_HEARD, 0.0, 6.1)], ensure_ascii=False),
            script=LIAN_SCRIPT,
            max_line_cells=80,
            low_confidence=0.6,
            media="e01.wav",
        )
        srt = result.values[0]
        assert "练" in srt and "炼" not in srt
        report = json.loads(result.values[1])
        assert report["sentences"][0]["substitutions"][0] == {
            "script": "练", "whisper": "炼", "at": 11,
        }
        assert result.values[2] == LIAN_SCRIPT
        written = Path(result.ui["srt_file"][0])
        assert written.name == "e01.srt"
        assert written.read_text(encoding="utf-8") == srt
        report_path = Path(result.ui["report_file"][0])
        assert report_path.name == "e01.align.json"
        assert "练" in report_path.read_text(encoding="utf-8")

        unnamed = SubtitleAlignStage.execute(
            segments="[]",
            script="一二三四。",
            max_line_cells=40,
        )
        assert Path(unnamed.ui["srt_file"][0]).name == "subtitle-align.srt"
        body = json.loads(unnamed.values[1])
        assert body["interpolated"][0]["mark"] == "interpolated"

        with pytest.raises(RuntimeError, match="script"):
            SubtitleAlignStage.execute(segments="[]", script="  ")
        with pytest.raises(RuntimeError, match="script"):
            SubtitleAlignStage.execute(segments="[]", script="{")
