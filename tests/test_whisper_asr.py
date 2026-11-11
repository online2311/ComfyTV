"""Local Whisper ASR helpers that do not load the model."""
from __future__ import annotations

import json
import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")


def _write_pcm_wav(path: Path, sample_rate: int, seconds: float, freq: float = 440.0) -> None:
    n = int(sample_rate * seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n):
            sample = int(12000 * math.sin(2 * math.pi * freq * i / sample_rate))
            frames += struct.pack("<h", sample)
        handle.writeframes(bytes(frames))


def _snapshot(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


class TestSrtAndTimestamps:
    def test_segments_become_srt_cues(self):
        from ComfyTV.runners.whisper_asr import segments_to_srt

        srt = segments_to_srt([
            {"start": 0.0, "end": 1.2, "text": " 你好 "},
            {"start": 1.2, "end": 1.0, "text": ""},
            {"start": 1.25, "end": 3.25, "text": "世界"},
        ])
        assert srt == (
            "1\n00:00:00,000 --> 00:00:01,200\n你好\n"
            "\n"
            "2\n00:00:01,250 --> 00:00:03,250\n世界\n"
        )

    def test_empty_segments_are_empty_srt_and_json(self):
        from ComfyTV.runners.whisper_asr import pack_transcript

        srt, text, raw = pack_transcript([])
        assert srt == ""
        assert text == ""
        assert json.loads(raw) == []

    def test_json_keeps_chinese_and_optional_words(self):
        from ComfyTV.runners.whisper_asr import segments_to_json

        raw = segments_to_json([{
            "start": 0.0,
            "end": 0.4,
            "text": "你好",
            "words": [
                {"start": 0.0, "end": 0.2, "text": "你"},
                {"start": 0.2, "end": 0.4, "text": "好"},
            ],
        }])
        data = json.loads(raw)
        assert data[0]["text"] == "你好"
        assert "你好" in raw
        assert data[0]["words"][1]["text"] == "好"

    def test_timestamp_tokens_become_segments(self):
        from ComfyTV.runners.whisper_asr import parse_whisper_timestamp_text

        parsed = parse_whisper_timestamp_text(
            "<|0.00|>你好<|1.20|><|1.20|>世界<|2.50|>"
        )
        assert parsed == [
            {"start": 0.0, "end": 1.2, "text": "你好"},
            {"start": 1.2, "end": 2.5, "text": "世界"},
        ]

    def test_transcript_joins_chinese_without_spaces_and_english_with_them(self):
        from ComfyTV.runners.whisper_asr import join_transcript

        assert join_transcript([
            {"text": " 你好"}, {"text": "世界"},
        ]) == "你好世界"
        assert join_transcript([
            {"text": "Hello"}, {"text": "world"},
        ]) == "Hello world"

    def test_token_pieces_group_into_words(self):
        from ComfyTV.runners.whisper_asr import group_token_words

        words = group_token_words([
            ("你", 0.0, 0.2),
            ("好", 0.2, 0.4),
            (" hello", 0.5, 0.8),
            (" world", 0.8, 1.2),
        ])
        assert [w["text"] for w in words] == ["你", "好", "hello", "world"]
        assert words[2]["start"] == 0.5
        assert words[2]["end"] == 0.8

    def test_generate_segments_decode_to_text_and_words(self):
        from ComfyTV.runners.whisper_asr import segments_from_generate_output

        class _Tok:
            def decode(self, ids, skip_special_tokens=True, decode_with_timestamps=False):
                ids = list(ids)
                if decode_with_timestamps:
                    return "<|0.00|>fallback<|1.00|>"
                if ids == [7]:
                    return "你"
                if ids == [8]:
                    return "好"
                if ids == [7, 8]:
                    return "你好"
                return ""

        outputs = {
            "sequences": [7, 8],
            "segments": [[{
                "start": 0.0,
                "end": 0.4,
                "tokens": [7, 8],
                "token_timestamps": [0.0, 0.2],
            }]],
        }
        segs = segments_from_generate_output(
            outputs, _Tok(), timestamps="word", duration_s=0.4,
        )
        assert segs[0]["text"] == "你好"
        assert [w["text"] for w in segs[0]["words"]] == ["你", "好"]

    def test_plain_ids_fall_back_to_timestamp_text(self):
        from ComfyTV.runners.whisper_asr import segments_from_generate_output

        class _Tok:
            def decode(self, ids, skip_special_tokens=True, decode_with_timestamps=False):
                if decode_with_timestamps:
                    return "<|0.00|>你好<|1.00|>"
                return "你好"

        segs = segments_from_generate_output(
            [1, 2, 3], _Tok(), timestamps="segment", duration_s=1.0,
        )
        assert segs == [{"start": 0.0, "end": 1.0, "text": "你好"}]

    def test_none_mode_collapses_to_one_cue(self):
        from ComfyTV.runners.whisper_asr import finalize_segments

        out = finalize_segments(
            [{"start": 0.0, "end": 1.0, "text": "你好"},
             {"start": 1.0, "end": 2.0, "text": "世界"}],
            timestamps="none",
            duration_s=2.0,
        )
        assert out == [{"start": 0.0, "end": 2.0, "text": "你好世界"}]

    def test_byte_level_chinese_tokens_merge_into_characters(self):
        # System test: 专 / 练 / 玉 / 杨 / 课 were emitted as U+FFFD words.
        from ComfyTV.runners.whisper_asr import segments_from_generate_output

        chars = "专练玉杨课"
        token_bytes: dict[int, bytes] = {}
        ids: list[int] = []
        nid = 1
        for ch in chars:
            for byte in ch.encode("utf-8"):
                token_bytes[nid] = bytes([byte])
                ids.append(nid)
                nid += 1
        token_bytes[nid] = "心".encode("utf-8")
        all_ids = ids + [nid]
        times = [i * 0.1 for i in range(len(all_ids))]
        segs = segments_from_generate_output(
            {"segments": [[{
                "start": 0.0,
                "end": 2.0,
                "tokens": all_ids,
                "token_timestamps": times,
                "text": chars + "心",
            }]]},
            _Utf8ByteTok(token_bytes),
            timestamps="word",
            duration_s=2.0,
        )
        words = segs[0]["words"]
        assert segs[0]["text"] == chars + "心"
        assert [w["text"] for w in words] == list(chars + "心")
        assert all("\ufffd" not in w["text"] for w in words)
        # 专 is bytes at 0.0, 0.1, 0.2. Its end is the next token at 0.3.
        assert words[0]["start"] == 0.0
        assert words[0]["end"] == 0.3
        # 心 is one whole token at 1.5; nothing follows, so it ends at the segment.
        assert words[-1]["start"] == 1.5
        assert words[-1]["end"] == 2.0

    def test_uneven_byte_tokens_merge_until_the_character_decodes(self):
        from ComfyTV.runners.whisper_asr import segments_from_generate_output

        raw = "专".encode("utf-8")
        token_bytes = {1: raw[:2], 2: raw[2:], 3: "心".encode("utf-8")}
        segs = segments_from_generate_output(
            {"segments": [[{
                "start": 0.0,
                "end": 1.0,
                "tokens": [1, 2, 3],
                "token_timestamps": [0.0, 0.2, 0.5],
                "text": "专心",
            }]]},
            _Utf8ByteTok(token_bytes),
            timestamps="word",
            duration_s=1.0,
        )
        words = segs[0]["words"]
        assert [w["text"] for w in words] == ["专", "心"]
        assert words[0]["start"] == 0.0
        assert words[0]["end"] == 0.5
        assert words[1]["start"] == 0.5
        assert words[1]["end"] == 1.0

    def test_incomplete_byte_tail_is_not_a_replacement_word(self):
        from ComfyTV.runners.whisper_asr import segments_from_generate_output

        raw = "专".encode("utf-8")
        token_bytes = {1: raw[:1], 2: raw[1:2]}
        segs = segments_from_generate_output(
            {"segments": [[{
                "start": 0.0,
                "end": 0.4,
                "tokens": [1, 2],
                "token_timestamps": [0.0, 0.2],
                "text": "专",
            }]]},
            _Utf8ByteTok(token_bytes),
            timestamps="word",
            duration_s=0.4,
        )
        words = segs[0].get("words") or []
        assert words == []
        assert all("\ufffd" not in w["text"] for w in words)

    def test_tokens_crossing_character_boundaries_keep_complete_characters(self):
        # One token can end a character and start the next: b'\xe4\xb8' + b'\x93\xe7'
        # decodes to 专 plus an unfinished byte. The finished character stays.
        from ComfyTV.runners.whisper_asr import segments_from_generate_output

        token_bytes = {1: b"\xe4\xb8", 2: b"\x93\xe7"}
        segs = segments_from_generate_output(
            {"segments": [[{
                "start": 0.0,
                "end": 1.0,
                "tokens": [1, 2],
                "token_timestamps": [0.0, 0.4],
                "text": "专",
            }]]},
            _Utf8ByteTok(token_bytes),
            timestamps="word",
            duration_s=1.0,
        )
        words = segs[0].get("words") or []
        assert [w["text"] for w in words] == ["专"]
        assert words[0]["start"] == 0.0
        assert words[0]["end"] == 1.0

        # The leftover byte of 练 is finished by the following token.
        token_bytes[3] = b"\xbb\x83"
        segs = segments_from_generate_output(
            {"segments": [[{
                "start": 0.0,
                "end": 1.0,
                "tokens": [1, 2, 3],
                "token_timestamps": [0.0, 0.4, 0.7],
                "text": "专练",
            }]]},
            _Utf8ByteTok(token_bytes),
            timestamps="word",
            duration_s=1.0,
        )
        words = segs[0]["words"]
        assert [w["text"] for w in words] == ["专", "练"]
        assert words[0]["start"] == 0.0
        assert words[0]["end"] == 0.7
        assert words[1]["start"] == 0.7
        assert words[1]["end"] == 1.0

    def test_word_end_is_clamped_inside_the_segment(self):
        # System test: segment end 6.10 s, last word end 6.20 s.
        from ComfyTV.runners.whisper_asr import segments_from_generate_output

        class _Tok:
            def decode(self, ids, skip_special_tokens=True, decode_with_timestamps=False):
                mapping = {1: "的", 2: "。"}
                return "".join(mapping.get(int(i), "") for i in list(ids))

        segs = segments_from_generate_output(
            {"segments": [[{
                "start": 0.0,
                "end": 6.10,
                "tokens": [1, 2],
                "token_timestamps": [6.00, 6.20],
                "text": "的。",
            }]]},
            _Tok(),
            timestamps="word",
            duration_s=6.2,
        )
        words = segs[0]["words"]
        assert segs[0]["end"] == 6.1
        assert [w["text"] for w in words] == ["的", "。"]
        for word in words:
            assert segs[0]["start"] <= word["start"] <= word["end"] <= segs[0]["end"]
        assert words[-1]["end"] == 6.1

    def test_zh_punctuation_becomes_fullwidth_without_touching_srt_times(self):
        from ComfyTV.runners.whisper_asr import finish_transcript

        sample = "过儿专心些,心神不定是炼不成玉女心经的。"
        srt, text, raw = finish_transcript(
            [{
                "start": 0.0,
                "end": 6.10,
                "text": sample,
                "words": [
                    {"start": 1.0, "end": 1.2, "text": "些,"},
                    {"start": 5.9, "end": 6.1, "text": "。"},
                ],
            }],
            timestamps="word",
            duration_s=6.10,
            normalize_punctuation=True,
        )
        assert text == "过儿专心些，心神不定是炼不成玉女心经的。"
        assert "些，心神" in srt
        assert "00:00:06,100" in srt
        assert "00:00:00,000" in srt
        data = json.loads(raw)
        assert data[0]["text"] == "过儿专心些，心神不定是炼不成玉女心经的。"
        assert data[0]["words"][0]["text"] == "些，"
        assert data[0]["end"] == 6.1

        kept, kept_text, _kept_raw = finish_transcript(
            [{"start": 0.0, "end": 1.0, "text": "Hello, world. 3.5!"}],
            timestamps="segment",
            duration_s=1.0,
            normalize_punctuation=False,
        )
        assert kept_text == "Hello, world. 3.5!"
        assert "Hello, world." in kept

        forced, forced_text, _forced_raw = finish_transcript(
            [{"start": 0.0, "end": 1.0, "text": "真的?好!注意:停止;好(注)第3.5集..."}],
            timestamps="segment",
            duration_s=1.0,
            normalize_punctuation=True,
        )
        assert forced_text == "真的？好！注意：停止；好（注）第3.5集……"
        assert "00:00:01,000" in forced


class TestAudioFormat:
    def test_resample_keeps_a_constant_and_the_new_length(self):
        from ComfyTV.runners.whisper_asr import resample_audio

        src = np.full(48000, 0.25, dtype=np.float32)
        out = resample_audio(src, 48000, 16000)
        assert out.shape == (16000,)
        assert out.dtype == np.float32
        # Polyphase resampling rings at the very edges; the body stays put.
        assert np.allclose(out[64:-64], 0.25, atol=1e-3)

    def test_numpy_resample_matches_when_scipy_is_absent(self, monkeypatch):
        from ComfyTV.runners import whisper_asr

        monkeypatch.setattr(whisper_asr, "_resample_poly", None)
        monkeypatch.setattr(whisper_asr, "_resample_poly_checked", True)
        src = np.full(48000, 0.25, dtype=np.float32)
        out = whisper_asr.resample_audio(src, 48000, 16000)
        assert out.shape == (16000,)
        assert np.allclose(out, 0.25, atol=1e-3)

    def test_same_rate_is_unchanged(self):
        from ComfyTV.runners.whisper_asr import resample_audio

        src = np.linspace(-1, 1, 160, dtype=np.float32)
        out = resample_audio(src, 16000, 16000)
        assert np.allclose(out, src)

    def test_stereo_audio_dict_becomes_mono_16k(self):
        from ComfyTV.runners.whisper_asr import audio_dict_to_mono16k

        class _Tensor:
            def __init__(self, arr):
                self._arr = arr

            def detach(self):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._arr

        wave = np.ones((1, 2, 4800), dtype=np.float32) * 0.5
        out = audio_dict_to_mono16k({
            "waveform": _Tensor(wave),
            "sample_rate": 48000,
        })
        assert out.shape == (1600,)
        assert np.allclose(out[32:-32], 0.5, atol=1e-3)

    def test_wav_roundtrip_is_mono_16k(self, tmp_path):
        from ComfyTV.runners.whisper_asr import load_media_mono16k

        path = tmp_path / "tone.wav"
        _write_pcm_wav(path, 48000, 0.5)
        out = load_media_mono16k(str(path))
        assert abs(out.shape[0] - 8000) <= 8
        assert float(np.max(np.abs(out))) > 0.05

    def test_av_decode_used_when_soundfile_cannot_read(self, tmp_path, monkeypatch):
        from ComfyTV.runners import whisper_asr

        path = tmp_path / "tone.wav"
        _write_pcm_wav(path, 16000, 0.25)

        def _boom(*_a, **_k):
            raise RuntimeError("soundfile declined")

        monkeypatch.setattr(whisper_asr, "sf", types_module_sf(_boom))
        out = whisper_asr.load_media_mono16k(str(path))
        assert abs(out.shape[0] - 4000) <= 8

    def test_video_container_audio_is_decoded(self, tmp_path):
        from ComfyTV.runners.whisper_asr import load_media_mono16k

        path = tmp_path / "clip.mkv"
        _write_tone_video(path)
        if not path.exists() or path.stat().st_size == 0:
            pytest.skip("av could not mux a video+audio container")
        out = load_media_mono16k(str(path))
        assert out.shape[0] > 100
        assert float(np.max(np.abs(out))) > 0.05

    def test_view_url_and_annotated_path_resolve(self, tmp_path, monkeypatch):
        import folder_paths
        from ComfyTV.runners.whisper_asr import resolve_media_path

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        target = out_dir / "a.wav"
        target.write_bytes(b"not-read")
        monkeypatch.setattr(
            folder_paths, "get_directory_by_type",
            lambda kind: str(out_dir if kind == "output" else tmp_path / kind),
        )
        if hasattr(folder_paths, "get_annotated_filepath"):
            monkeypatch.delattr(folder_paths, "get_annotated_filepath", raising=False)

        view = resolve_media_path("/view?filename=a.wav&type=output&subfolder=")
        assert Path(view) == target.resolve()
        annotated = resolve_media_path("a.wav [output]")
        assert Path(annotated) == target.resolve()
        assert resolve_media_path(str(target)) == str(target.resolve())

    def test_missing_media_raises(self, tmp_path):
        from ComfyTV.runners.whisper_asr import resolve_media_path

        with pytest.raises(RuntimeError, match="not found"):
            resolve_media_path(str(tmp_path / "missing.wav"))


class _Utf8ByteTok:
    """Tokenizer stand-in: each id is raw UTF-8 bytes, like Whisper byte-level BPE."""

    def __init__(self, token_bytes: dict[int, bytes]):
        self.token_bytes = token_bytes

    def decode(self, ids, skip_special_tokens=True, decode_with_timestamps=False):
        blob = b"".join(self.token_bytes[int(i)] for i in list(ids))
        return blob.decode("utf-8", errors="replace")


def types_module_sf(read):
    class _SF:
        @staticmethod
        def read(*args, **kwargs):
            return read(*args, **kwargs)

    return _SF


def _write_tone_video(path: Path) -> None:
    """Mux a short silent video with a sine audio track. Skip-friendly."""
    sr = 16000
    n = sr // 2
    samples = (0.2 * np.sin(2 * np.pi * 440 * np.arange(n) / sr)).astype(np.float32)
    try:
        container = av.open(str(path), "w")
    except Exception:
        return
    ok = False
    try:
        audio = container.add_stream("aac", rate=sr)
        video = container.add_stream("libx264", rate=12)
        video.width = 32
        video.height = 32
        video.pix_fmt = "yuv420p"
        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1), format="fltp", layout="mono",
        )
        frame.sample_rate = sr
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
        blank = np.zeros((32, 32, 3), dtype=np.uint8)
        vframe = av.VideoFrame.from_ndarray(blank, format="rgb24")
        for packet in video.encode(vframe):
            container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        ok = True
    except Exception:
        ok = False
    finally:
        container.close()
    if not ok:
        path.unlink(missing_ok=True)


class TestModelDir:
    def test_explicit_then_env_then_folder_then_fallback(self, tmp_path, monkeypatch):
        from ComfyTV.runners import whisper_asr as asr

        explicit = _snapshot(tmp_path / "explicit")
        env = _snapshot(tmp_path / "from-env")
        tiny = _snapshot(tmp_path / "models" / "whisper" / "whisper-tiny")
        large = _snapshot(tmp_path / "models" / "whisper" / "whisper-large-v3")
        fallback = _snapshot(tmp_path / "fallback" / "whisper-large-v3")
        monkeypatch.setattr(asr, "FALLBACK_MODEL_DIR", str(fallback))
        monkeypatch.delenv(asr.ENV_MODEL_DIR, raising=False)

        assert asr.resolve_model_dir(str(explicit)) == str(explicit.resolve())

        monkeypatch.setenv(asr.ENV_MODEL_DIR, str(env))
        assert asr.resolve_model_dir("") == str(env.resolve())
        monkeypatch.setenv(asr.ENV_MODEL_DIR, str(tmp_path / "nope"))
        with pytest.raises(RuntimeError, match="config.json"):
            asr.resolve_model_dir(None)

        monkeypatch.delenv(asr.ENV_MODEL_DIR, raising=False)

        def _folders(kind):
            if kind == "whisper":
                return [str(tmp_path / "models" / "whisper")]
            return []

        monkeypatch.setattr(asr, "_folder_paths", _folders)
        assert asr.resolve_model_dir("") == str(large.resolve())
        assert tiny.exists()

        monkeypatch.setattr(asr, "_folder_paths", lambda _kind: [])
        assert asr.resolve_model_dir("") == str(fallback.resolve())

        monkeypatch.setattr(asr, "FALLBACK_MODEL_DIR", str(tmp_path / "absent"))
        with pytest.raises(RuntimeError, match="never downloads"):
            asr.resolve_model_dir("")

    def test_register_whisper_folders_once(self, tmp_path, monkeypatch):
        import folder_paths
        from ComfyTV.runners.whisper_asr import register_whisper_model_folders

        store: dict[str, list[str]] = {}

        def _add(kind, path):
            store.setdefault(kind, []).append(path)

        monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path), raising=False)
        monkeypatch.setattr(folder_paths, "add_model_folder_path", _add, raising=False)
        monkeypatch.setattr(
            folder_paths, "get_folder_paths",
            lambda kind: store.get(kind, []),
            raising=False,
        )
        register_whisper_model_folders()
        register_whisper_model_folders()
        assert store["whisper"] == [str(tmp_path / "whisper")]
        assert store["stt"] == [str(tmp_path / "stt")]
        assert (tmp_path / "whisper").is_dir()


class TestWorkflowRegistration:
    def test_speech_to_text_workflow_seeds_and_binds_source(self, reset_db, tmp_path, monkeypatch):
        from ComfyTV.runners import refresh_registry, workflow_db as wdb
        from ComfyTV.nodes.stages.common.workflow_lists import default_for, labels_for
        from ComfyTV.nodes.stages import video_text

        src = Path(__file__).resolve().parent.parent / "workflows" / "speech-to-text"
        assert (src / "local-whisper-large-v3.json").is_file()
        wdir = tmp_path / "workflows"
        dest = wdir / "speech-to-text"
        dest.mkdir(parents=True)
        for item in src.iterdir():
            if item.is_file():
                shutil.copy(item, dest / item.name)
        monkeypatch.setattr(wdb.seed, "_WORKFLOWS_DIR", wdir)
        monkeypatch.setattr(wdb.seed, "_LEGACY_WORKFLOWS_DIR", tmp_path / "legacy-unused")
        seeded = wdb.seed_workflows_from_disk(("speech-to-text",))
        assert seeded["total"] == 1

        cfg = wdb.get_workflow_config("speech-to-text", "Local Whisper large-v3")
        assert cfg["result_type"] == "graph_output_first"
        assert cfg["result_node"] == "1"
        assert cfg["bindings"] == [{
            "node_id": "1",
            "input_name": "media",
            "from": "option:source",
            "default": None,
            "prefix": None,
            "suffix": None,
            "required": True,
            "error_msg": "Speech-to-text needs an upstream audio or video.",
            "cast": None,
        }]
        gui = json.loads((dest / "local-whisper-large-v3.json").read_text(encoding="utf-8"))
        node = gui["nodes"][0]
        assert node["type"] == "ComfyTV.WhisperTranscribe"
        assert node["widgets_values"][1] == "zh"
        assert node["widgets_values"][2] == "transcribe"

        refresh_registry()
        assert labels_for("speech-to-text") == ["Local Whisper large-v3"]
        assert default_for("speech-to-text") == "Local Whisper large-v3"

        schema = video_text.SubtitleGenStage.define_schema()
        workflow = next(item for item in schema.kw["inputs"] if item.args[0] == "workflow")
        assert "Local Whisper large-v3" in workflow.kw["options"]

    def test_stage_passes_audio_or_video_as_source(self, monkeypatch):
        import asyncio
        from ComfyTV.nodes.stages import video_text

        captured = {}

        async def fake_run(cls, **kw):
            captured.update(kw)
            return None

        monkeypatch.setattr(video_text, "run_stage_workflow", fake_run)
        asyncio.run(video_text.SubtitleGenStage.execute(
            project_id="p", audio="/view?filename=a.wav&type=output",
        ))
        assert captured["upstream"]["audio"] == ["/view?filename=a.wav&type=output"]
        assert captured["options"]["source"] == "/view?filename=a.wav&type=output"

        captured.clear()
        asyncio.run(video_text.SubtitleGenStage.execute(
            project_id="p", video="/view?filename=b.mp4&type=output",
        ))
        assert captured["upstream"]["videos"] == ["/view?filename=b.mp4&type=output"]
        assert "audio" not in captured["upstream"]
        assert captured["options"]["source"] == "/view?filename=b.mp4&type=output"


class TestNodeExecute:
    def test_schema_and_media_execute_without_loading_weights(self, tmp_path, monkeypatch):
        from ComfyTV.nodes import whisper_transcribe as node
        from ComfyTV.nodes.stages.common.meta import NON_STAGE_NODES

        schema = node.WhisperTranscribe.define_schema()
        assert schema.kw["node_id"] == "ComfyTV.WhisperTranscribe"
        assert schema.kw["is_output_node"] is True
        names = [item.args[0] for item in schema.kw["inputs"] if item.args]
        assert names[:3] == ["audio", "media", "language"]
        assert "WhisperTranscribe" in NON_STAGE_NODES

        wav = tmp_path / "in.wav"
        _write_pcm_wav(wav, 16000, 0.2)
        seen = {}

        def _fake(samples, **kwargs):
            seen["n"] = int(samples.shape[0])
            seen.update(kwargs)
            return ("1\n00:00:00,000 --> 00:00:00,200\n你好\n", "你好", "[]")

        monkeypatch.setattr(node, "run_whisper", _fake)
        result = node.WhisperTranscribe.execute(
            media=str(wav), language="zh", task="transcribe",
            timestamps="segment", unload=False, model_dir="",
        )
        assert result.values[0].startswith("1\n")
        assert result.values[1] == "你好"
        assert seen["language"] == "zh"
        assert seen["task"] == "transcribe"
        assert seen["n"] > 0

        wave = np.ones((1, 1, 1600), dtype=np.float32)
        node.WhisperTranscribe.execute(
            audio={"waveform": wave, "sample_rate": 16000}, media=str(wav),
        )
        assert seen["n"] == 1600

        with pytest.raises(RuntimeError, match="AUDIO"):
            node.WhisperTranscribe.execute(media="", audio=None)

    def test_solo_run_returns_ui_and_writes_srt(self, tmp_path, monkeypatch):
        import folder_paths
        from ComfyTV.nodes import whisper_transcribe as node

        monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
        wav = tmp_path / "s2pro_00001.flac"
        _write_pcm_wav(wav, 16000, 0.2)
        srt = "1\n00:00:00,000 --> 00:00:00,200\n你好\n"
        segments = "[{\"text\":\"你好\"}]"

        def _fake(samples, **kwargs):
            return (srt, "你好", segments)

        monkeypatch.setattr(node, "run_whisper", _fake)
        result = node.WhisperTranscribe.execute(
            media=str(wav), language="zh", timestamps="segment",
        )
        assert result.ui["text"] == [srt]
        assert result.ui["segments"] == [segments]
        written = Path(result.ui["srt_file"][0])
        assert written.parent == (tmp_path / "comfytv").resolve()
        assert written.name == "s2pro_00001.srt"
        assert written.read_text(encoding="utf-8") == srt
        assert result.values[0] == srt

        view = node.WhisperTranscribe.execute(
            audio={"waveform": np.ones((1, 1, 1600), dtype=np.float32), "sample_rate": 16000},
            media="/view?filename=s2pro_00001.flac&subfolder=comfytv&type=output",
            language="zh",
        )
        assert Path(view.ui["srt_file"][0]).name == "s2pro_00001.srt"

        audio_only = node.WhisperTranscribe.execute(
            audio={"waveform": np.ones((1, 1, 1600), dtype=np.float32), "sample_rate": 16000},
            media="",
            language="en",
        )
        assert Path(audio_only.ui["srt_file"][0]).name == "whisper.srt"

    def test_normalize_punct_defaults_on_for_zh(self, tmp_path, monkeypatch):
        from ComfyTV.nodes import whisper_transcribe as node

        schema = node.WhisperTranscribe.define_schema()
        punct = next(
            item for item in schema.kw["inputs"]
            if item.args and item.args[0] == "normalize_punct"
        )
        assert punct.kw["default"] == "auto"
        assert list(punct.kw["options"]) == ["auto", "on", "off"]

        import folder_paths

        monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
        wav = tmp_path / "in.wav"
        _write_pcm_wav(wav, 16000, 0.1)
        seen = {}

        def _fake(samples, **kwargs):
            seen.update(kwargs)
            return ("", "", "[]")

        monkeypatch.setattr(node, "run_whisper", _fake)
        node.WhisperTranscribe.execute(media=str(wav), language="zh")
        assert seen["normalize_punctuation"] is True
        node.WhisperTranscribe.execute(media=str(wav), language="zh", task="translate")
        assert seen["normalize_punctuation"] is False
        node.WhisperTranscribe.execute(
            media=str(wav), language="zh", task="translate", normalize_punct="on",
        )
        assert seen["normalize_punctuation"] is True
        node.WhisperTranscribe.execute(media=str(wav), language="en")
        assert seen["normalize_punctuation"] is False
        node.WhisperTranscribe.execute(media=str(wav), language="zh", normalize_punct="off")
        assert seen["normalize_punctuation"] is False
        node.WhisperTranscribe.execute(media=str(wav), language="en", normalize_punct="on")
        assert seen["normalize_punctuation"] is True
