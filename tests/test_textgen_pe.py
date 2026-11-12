"""MiniMax-H3-Image-PE reasoning effort: prompt text, detection, empty answers."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from ComfyTV.nodes import textgen_pe as pe
from ComfyTV.nodes.textgen_pe import (
    LOW_INSTRUCTIONS,
    XHIGH_INSTRUCTIONS,
    PeDetection,
    TextGeneratePE,
    call_text_generate_v2,
    compose_system_prompt,
    detect_pe,
    ensure_answer_present,
    system_prompt_for,
)


def _stage(clip_name: str, model_type: str | None = None):
    class Qwen35TEModel_:
        pass

    stage = Qwen35TEModel_()
    stage.clip_name = clip_name
    stage.clip = clip_name
    inner = SimpleNamespace(transformer=SimpleNamespace(model_type=model_type or clip_name))
    setattr(stage, clip_name, inner)
    return stage


def _clip(*, clip_name: str, paths=None, parent_paths=None, model_type: str | None = None, clip_type="MINIMAX"):
    parent = None
    if parent_paths is not None:
        parent = SimpleNamespace(
            cached_patcher_init=(object(), (list(parent_paths), None, clip_type, {})),
            parent=None,
        )
    if paths is None:
        cached = None
    else:
        cached = (object(), (list(paths), None, clip_type, {}))
    return SimpleNamespace(
        cond_stage_model=_stage(clip_name, model_type),
        tokenizer=SimpleNamespace(clip_name=clip_name),
        patcher=SimpleNamespace(cached_patcher_init=cached, parent=parent),
    )


PE_FILE = "/models/text_encoders/minimax_h3_image_pe_nvfp4.safetensors"
NINE_B_FILE = "/models/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors"
STOCK_27B_FILE = "/models/text_encoders/qwen3.5_27b.safetensors"


def test_xhigh_and_low_instructions_match_the_official_template():
    assert system_prompt_for("", "xhigh", thinking=True, is_pe=False) == XHIGH_INSTRUCTIONS
    assert system_prompt_for("", "low", thinking=True, is_pe=False) == LOW_INSTRUCTIONS
    assert XHIGH_INSTRUCTIONS == (
        "Reasoning effort is set to xhigh. Please think carefully through the task, "
        "validate key assumptions, consider plausible alternatives, and prioritize "
        "correctness, consistency, and clarity in the final answer."
    )
    assert LOW_INSTRUCTIONS == (
        "Reasoning effort is set to low. Keep your thinking brief and focused, "
        "moving directly to the conclusion without unnecessary elaboration."
    )


def test_medium_and_off_insert_nothing():
    assert system_prompt_for("", "medium", thinking=True, is_pe=True) == ""
    assert system_prompt_for("", "off", thinking=True, is_pe=True) == ""
    assert system_prompt_for("keep me", "medium", thinking=True, is_pe=True) == "keep me"
    assert system_prompt_for("keep me", "off", thinking=True, is_pe=True) == "keep me"


def test_user_system_is_joined_with_a_blank_line_and_trimmed():
    assert compose_system_prompt("  hello  ", XHIGH_INSTRUCTIONS) == (
        XHIGH_INSTRUCTIONS + "\n\n" + "hello"
    )
    assert compose_system_prompt("  hello  ", LOW_INSTRUCTIONS) == (
        LOW_INSTRUCTIONS + "\n\n" + "hello"
    )
    assert compose_system_prompt("   ", XHIGH_INSTRUCTIONS) == XHIGH_INSTRUCTIONS
    assert compose_system_prompt(None, "") == ""


def test_thinking_off_never_inserts_the_effort_line():
    for effort in ("auto", "xhigh", "medium", "low", "off"):
        assert system_prompt_for("keep me", effort, thinking=False, is_pe=True) == "keep me"
        assert system_prompt_for("", effort, thinking=False, is_pe=True) == ""


def test_auto_uses_xhigh_only_for_pe():
    assert system_prompt_for("", "auto", thinking=True, is_pe=True) == XHIGH_INSTRUCTIONS
    assert system_prompt_for("keep me", "auto", thinking=True, is_pe=False) == "keep me"
    assert system_prompt_for("", "auto", thinking=True, is_pe=False) == ""


def test_unknown_effort_inserts_nothing():
    assert system_prompt_for("keep me", "high", thinking=True, is_pe=True) == "keep me"


def test_pe_filename_matches_and_class_name_does_not():
    clip = _clip(clip_name="qwen35_27b", paths=[PE_FILE])
    found = detect_pe(clip)
    assert type(clip.cond_stage_model).__name__ == "Qwen35TEModel_"
    assert found == PeDetection(True, PE_FILE, False, True)

    hyphenated = _clip(
        clip_name="qwen35_27b",
        paths=["/models/text_encoders/MiniMax-H3-Image-PE.safetensors"],
    )
    assert detect_pe(hyphenated).is_pe is True


def test_qwen35_9b_is_not_pe_even_with_a_pe_looking_name():
    by_file = _clip(clip_name="qwen35_9b", paths=[NINE_B_FILE])
    assert type(by_file.cond_stage_model).__name__ == "Qwen35TEModel_"
    assert detect_pe(by_file, model_file=PE_FILE) == PeDetection(False, "", True, True)

    misleading = _clip(clip_name="qwen35_9b", paths=[PE_FILE], model_type="qwen35_9b")
    assert detect_pe(misleading).rejected_as_9b is True
    assert detect_pe(misleading).is_pe is False

    fallback_only = _clip(clip_name="qwen35_9b", paths=None)
    assert detect_pe(fallback_only, model_file="minimax_h3_image_pe_nvfp4.safetensors").is_pe is False


def test_stock_qwen35_27b_is_not_pe_and_ignores_the_fallback():
    clip = _clip(clip_name="qwen35_27b", paths=[STOCK_27B_FILE], clip_type="MINIMAX")
    assert type(clip.cond_stage_model).__name__ == "Qwen35TEModel_"
    found = detect_pe(clip, model_file="minimax_h3_image_pe_nvfp4.safetensors")
    assert found.is_pe is False
    assert found.rejected_as_9b is False
    assert found.saw_checkpoint is True

    folder_only = _clip(
        clip_name="qwen35_27b",
        paths=["/models/minimax_h3_image_pe/qwen3.5_27b.safetensors"],
    )
    assert detect_pe(folder_only).is_pe is False


def test_missing_filename_is_not_pe_and_does_not_raise():
    unsure = _clip(clip_name="qwen35_27b", paths=None)
    assert detect_pe(unsure) == PeDetection(False, "", False, False)
    assert detect_pe(unsure, model_file="   ") == PeDetection(False, "", False, False)

    class _Boom:
        @property
        def patcher(self):
            raise RuntimeError("secret /home/user/keys")

    assert detect_pe(_Boom()).is_pe is False
    assert detect_pe(None).is_pe is False
    assert detect_pe(object()).is_pe is False


def test_model_file_fallback_is_used_only_without_a_checkpoint_path():
    missing = _clip(clip_name="qwen35_27b", paths=None)
    found = detect_pe(missing, model_file="/opt/models/minimax_h3_image_pe_nvfp4.safetensors")
    assert found.is_pe is True
    assert found.saw_checkpoint is False

    from_parent = _clip(clip_name="qwen35_27b", paths=None, parent_paths=[PE_FILE])
    assert detect_pe(from_parent).is_pe is True

    single = _clip(clip_name="qwen35_27b", paths=None)
    single.patcher.cached_patcher_init = (object(), (PE_FILE, None, "MINIMAX", {}))
    assert detect_pe(single).is_pe is True


def test_schema_matches_text_generate_v2_and_adds_effort():
    schema = TextGeneratePE.define_schema()
    assert schema.kw["node_id"] == "ComfyTV.TextGeneratePE"
    ids = [item.args[0] for item in schema.kw["inputs"]]
    assert ids == [
        "clip",
        "prompt",
        "images",
        "videos",
        "audios",
        "max_length",
        "sampling_mode",
        "thinking",
        "use_default_template",
        "mtp",
        "system_prompt",
        "reasoning_effort",
        "model_file",
    ]
    effort = schema.kw["inputs"][-2]
    assert effort.kw["options"] == ["auto", "xhigh", "medium", "low", "off"]
    assert effort.kw["default"] == "auto"
    length = schema.kw["inputs"][5]
    assert length.kw["default"] == 4096
    assert length.kw["max"] >= 8192
    assert schema.kw["inputs"][7].kw["default"] is True


class _Core:
    signature_used = None

    @classmethod
    def execute(
        cls,
        clip,
        prompt,
        max_length,
        sampling_mode,
        image=None,
        thinking=False,
        use_default_template=True,
        video=None,
        audio=None,
        mtp="auto",
        system_prompt="",
        images=None,
        videos=None,
        audios=None,
    ):
        cls.signature_used = {
            "clip": clip,
            "prompt": prompt,
            "max_length": max_length,
            "sampling_mode": sampling_mode,
            "image": image,
            "thinking": thinking,
            "use_default_template": use_default_template,
            "video": video,
            "audio": audio,
            "mtp": mtp,
            "system_prompt": system_prompt,
            "images": images,
            "videos": videos,
            "audios": audios,
        }
        return cls.result


def _install_core(monkeypatch, result):
    _Core.result = result
    _Core.signature_used = None
    monkeypatch.setattr(pe, "_load_text_generate_v2", lambda: _Core)


def test_execute_calls_text_generate_v2_with_the_composed_system_prompt(monkeypatch):
    from comfy_api.latest import io

    out = io.NodeOutput("expanded caption", "some reasoning")
    _install_core(monkeypatch, out)
    clip = _clip(clip_name="qwen35_27b", paths=[PE_FILE])
    sampling = {"sampling_mode": "on", "temperature": 1.0, "seed": 7}
    result = TextGeneratePE.execute(
        clip,
        "a rainy alley",
        4096,
        sampling,
        images={"image1": object()},
        videos={"video1": object()},
        audios={"audio1": object()},
        thinking=True,
        use_default_template=True,
        mtp="auto",
        system_prompt="Be concrete.",
        reasoning_effort="auto",
        image=None,
        video=None,
        audio=None,
    )
    assert result is out
    called = _Core.signature_used
    assert called["system_prompt"] == XHIGH_INSTRUCTIONS + "\n\n" + "Be concrete."
    assert called["thinking"] is True
    assert called["max_length"] == 4096
    assert called["prompt"] == "a rainy alley"
    assert called["sampling_mode"] is sampling
    assert "reasoning_effort" not in called
    assert "model_file" not in called


def test_auto_on_9b_and_stock_27b_does_not_insert_instructions(monkeypatch):
    from comfy_api.latest import io

    _install_core(monkeypatch, io.NodeOutput("ok", ""))
    TextGeneratePE.execute(
        _clip(clip_name="qwen35_9b", paths=[NINE_B_FILE]),
        "prompt",
        128,
        {"sampling_mode": "off"},
        system_prompt="plain",
        reasoning_effort="auto",
        thinking=True,
    )
    assert _Core.signature_used["system_prompt"] == "plain"

    TextGeneratePE.execute(
        _clip(clip_name="qwen35_27b", paths=[STOCK_27B_FILE]),
        "prompt",
        128,
        {"sampling_mode": "off"},
        system_prompt="plain",
        reasoning_effort="auto",
        thinking=True,
        model_file="minimax_h3_image_pe_nvfp4.safetensors",
    )
    assert _Core.signature_used["system_prompt"] == "plain"


def test_manual_xhigh_still_inserts_on_a_stock_27b(monkeypatch):
    from comfy_api.latest import io

    _install_core(monkeypatch, io.NodeOutput("ok", ""))
    TextGeneratePE.execute(
        _clip(clip_name="qwen35_27b", paths=[STOCK_27B_FILE]),
        "prompt",
        32,
        {"sampling_mode": "off"},
        reasoning_effort="xhigh",
        thinking=True,
    )
    assert _Core.signature_used["system_prompt"] == XHIGH_INSTRUCTIONS


def test_empty_answer_is_an_error_for_blank_and_thinking_only_output(monkeypatch):
    from comfy_api.latest import io

    _install_core(monkeypatch, io.NodeOutput("", ""))
    with pytest.raises(RuntimeError, match="thinking is off") as blank:
        TextGeneratePE.execute(
            _clip(clip_name="qwen35_27b", paths=[PE_FILE]),
            "prompt",
            512,
            {"sampling_mode": "off"},
            thinking=False,
            reasoning_effort="off",
        )
    assert "max_length" in str(blank.value)
    assert "512" in str(blank.value)
    assert "4096" in str(blank.value)

    _install_core(monkeypatch, io.NodeOutput("   ", "only the reasoning"))
    with pytest.raises(RuntimeError, match="reasoning but the answer text is empty"):
        TextGeneratePE.execute(
            _clip(clip_name="qwen35_27b", paths=[PE_FILE]),
            "prompt",
            512,
            {"sampling_mode": "off"},
            thinking=True,
            reasoning_effort="xhigh",
        )


def test_empty_check_message_covers_a_fully_empty_generation():
    with pytest.raises(RuntimeError, match="generation is empty") as caught:
        ensure_answer_present("", "", thinking=True, max_length=4096)
    text = str(caught.value)
    assert "thinking is off" in text
    assert "max_length" in text
    ensure_answer_present("answer", "", thinking=False, max_length=8)


def test_call_filters_kwargs_to_the_v045_signature(monkeypatch):
    class _Narrow:
        seen = None

        @classmethod
        def execute(cls, clip, prompt, max_length, sampling_mode, system_prompt=""):
            cls.seen = {
                "clip": clip,
                "prompt": prompt,
                "max_length": max_length,
                "sampling_mode": sampling_mode,
                "system_prompt": system_prompt,
            }
            return "narrow"

    monkeypatch.setattr(pe, "_load_text_generate_v2", lambda: _Narrow)
    result = call_text_generate_v2(
        clip="c",
        prompt="p",
        max_length=4,
        sampling_mode={"sampling_mode": "off"},
        system_prompt="sys",
        images={"image1": 1},
        videos=None,
        audios=None,
        reasoning_effort="xhigh",
    )
    assert result == "narrow"
    assert _Narrow.seen["system_prompt"] == "sys"
    assert "images" not in _Narrow.seen


def test_node_is_registered_outside_the_stage_catalog():
    from ComfyTV.nodes.stages import stage_classes
    from ComfyTV.nodes.stages.common.meta import NON_STAGE_NODES

    assert TextGeneratePE in stage_classes()
    assert "TextGeneratePE" in NON_STAGE_NODES


_LOG = "ComfyTV.nodes.textgen_pe"
_SECRET_PROMPT = "SECRET_PROMPT_DO_NOT_LOG"
_SECRET_PATH = "/home/user/secrets/api_key_SHOULD_NOT_APPEAR.safetensors"


def _records(caplog, level):
    return [record for record in caplog.records if record.levelno == level and record.name == _LOG]


def _run(monkeypatch, clip, *, result, max_length=128, thinking=True, reasoning_effort="auto",
         system_prompt="", prompt=_SECRET_PROMPT, model_file=""):
    from comfy_api.latest import io

    _install_core(monkeypatch, result if result is not None else io.NodeOutput("ok", ""))
    return TextGeneratePE.execute(
        clip,
        prompt,
        max_length,
        {"sampling_mode": "off"},
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        system_prompt=system_prompt,
        model_file=model_file,
    )


def test_pe_auto_logs_that_the_xhigh_sentence_was_inserted(monkeypatch, caplog):
    from comfy_api.latest import io

    with caplog.at_level(logging.INFO, logger=_LOG):
        _run(
            monkeypatch,
            _clip(clip_name="qwen35_27b", paths=[PE_FILE]),
            result=io.NodeOutput("answer", "thought"),
            reasoning_effort="auto",
            system_prompt="Be concrete.",
        )
    inserted = _records(caplog, logging.INFO)
    assert len(inserted) == 1
    message = inserted[0].message
    assert "inserted xhigh reasoning sentence" in message
    assert "model=pe" in message
    assert "requested=auto" in message
    assert "decided=xhigh" in message
    assert _SECRET_PROMPT not in caplog.text
    assert XHIGH_INSTRUCTIONS not in caplog.text
    assert PE_FILE not in caplog.text


def test_skip_logs_model_kind_and_decided_effort(monkeypatch, caplog):
    from comfy_api.latest import io

    cases = [
        (_clip(clip_name="qwen35_9b", paths=[NINE_B_FILE]), "auto", True, "9b", "off"),
        (_clip(clip_name="qwen35_27b", paths=[STOCK_27B_FILE]), "auto", True, "other", "off"),
        (_clip(clip_name="qwen35_27b", paths=[PE_FILE]), "xhigh", False, "pe", "off"),
        (_clip(clip_name="qwen35_27b", paths=[PE_FILE]), "low", True, "pe", "low"),
        (_clip(clip_name="qwen35_27b", paths=None), "auto", True, "unknown", "off"),
    ]
    for clip, effort, thinking, kind, decided in cases:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=_LOG):
            _run(
                monkeypatch,
                clip,
                result=io.NodeOutput("answer", ""),
                reasoning_effort=effort,
                thinking=thinking,
            )
        skipped = _records(caplog, logging.INFO)
        assert len(skipped) == 1, caplog.text
        message = skipped[0].message
        assert "skipped xhigh reasoning sentence" in message
        assert f"model={kind}" in message
        assert f"requested={effort}" in message
        assert f"decided={decided}" in message
        assert _SECRET_PROMPT not in caplog.text
        assert NINE_B_FILE not in caplog.text


def test_manual_xhigh_on_a_stock_model_logs_the_insert(monkeypatch, caplog):
    from comfy_api.latest import io

    with caplog.at_level(logging.INFO, logger=_LOG):
        _run(
            monkeypatch,
            _clip(clip_name="qwen35_27b", paths=[STOCK_27B_FILE]),
            result=io.NodeOutput("answer", ""),
            reasoning_effort="xhigh",
        )
    message = _records(caplog, logging.INFO)[0].message
    assert "inserted xhigh reasoning sentence" in message
    assert "model=other" in message
    assert "decided=xhigh" in message


def test_empty_answer_after_thinking_warns_to_raise_max_length(monkeypatch, caplog):
    from comfy_api.latest import io

    with caplog.at_level(logging.INFO, logger=_LOG):
        with pytest.raises(RuntimeError, match="reasoning but the answer text is empty"):
            _run(
                monkeypatch,
                _clip(clip_name="qwen35_9b", paths=[NINE_B_FILE]),
                result=io.NodeOutput("", "thinking ate the budget"),
                max_length=4096,
                thinking=True,
                reasoning_effort="auto",
            )
    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 1
    message = warnings[0].message
    assert "hit max_length=4096" in message
    assert "answer=empty" in message
    assert "8192" in message
    assert _SECRET_PROMPT not in caplog.text
    assert "thinking ate the budget" not in caplog.text


def test_thinking_off_empty_answer_does_not_claim_the_token_limit(monkeypatch, caplog):
    from comfy_api.latest import io

    with caplog.at_level(logging.INFO, logger=_LOG):
        with pytest.raises(RuntimeError, match="generation is empty"):
            _run(
                monkeypatch,
                _clip(clip_name="qwen35_27b", paths=[PE_FILE]),
                result=io.NodeOutput("  ", ""),
                thinking=False,
                reasoning_effort="off",
            )
    assert _records(caplog, logging.WARNING) == []


def test_truncated_json_after_thinking_warns_without_failing(monkeypatch, caplog):
    from comfy_api.latest import io

    with caplog.at_level(logging.INFO, logger=_LOG):
        result = _run(
            monkeypatch,
            _clip(clip_name="qwen35_9b", paths=[NINE_B_FILE]),
            result=io.NodeOutput('{"shots": [{"id": 1}', "done thinking"),
            max_length=4096,
            thinking=True,
            reasoning_effort="auto",
        )
    assert result.values[0].startswith('{"shots"')
    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert "hit max_length=4096" in warnings[0].message
    assert "answer=truncated" in warnings[0].message
    assert "Raise max_length" in warnings[0].message
    assert '{"shots"' not in caplog.text


def test_finish_length_warns_even_when_the_answer_looks_complete(monkeypatch, caplog):
    from comfy_api.latest import io

    class _CountingCore:
        @classmethod
        def execute(cls, clip, prompt, max_length, sampling_mode, **kwargs):
            clip.generate(prompt, False, max_length)
            return io.NodeOutput('{"ok": true}', "")

    monkeypatch.setattr(pe, "_load_text_generate_v2", lambda: _CountingCore)
    clip = _clip(clip_name="qwen35_9b", paths=[NINE_B_FILE])

    def original(*args, **kwargs):
        return list(range(int(args[2])))

    clip.generate = original

    with caplog.at_level(logging.INFO, logger=_LOG):
        TextGeneratePE.execute(
            clip,
            _SECRET_PROMPT,
            4096,
            {"sampling_mode": "off"},
            thinking=True,
            reasoning_effort="auto",
        )
    assert clip.generate is original
    warnings = _records(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert "hit max_length=4096" in warnings[0].message
    assert "answer=complete" in warnings[0].message
    assert _SECRET_PROMPT not in caplog.text
    assert NINE_B_FILE not in caplog.text


def test_closed_json_and_prose_after_thinking_do_not_warn(monkeypatch, caplog):
    from comfy_api.latest import io

    for answer in ('{"a": "}"}', "use {braces} in the caption", "plain answer"):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=_LOG):
            _run(
                monkeypatch,
                _clip(clip_name="qwen35_9b", paths=[NINE_B_FILE]),
                result=io.NodeOutput(answer, "thought"),
                thinking=True,
                reasoning_effort="auto",
                max_length=4096,
            )
        assert _records(caplog, logging.WARNING) == [], answer


def test_class_generate_is_not_left_on_the_instance(monkeypatch):
    from comfy_api.latest import io

    class _CountingCore:
        @classmethod
        def execute(cls, clip, prompt, max_length, sampling_mode, **kwargs):
            assert len(clip.generate(max_length=4)) == 4
            return io.NodeOutput("answer", "")

    class _Clip:
        def __init__(self):
            inner = _clip(clip_name="qwen35_27b", paths=[PE_FILE])
            self.cond_stage_model = inner.cond_stage_model
            self.tokenizer = inner.tokenizer
            self.patcher = inner.patcher

        def generate(self, *args, **kwargs):
            return [0, 1, 2, 3]

    monkeypatch.setattr(pe, "_load_text_generate_v2", lambda: _CountingCore)
    clip = _Clip()
    assert "generate" not in clip.__dict__
    TextGeneratePE.execute(
        clip,
        "prompt",
        32,
        {"sampling_mode": "off"},
        thinking=True,
        reasoning_effort="xhigh",
    )
    assert "generate" not in clip.__dict__
    assert len(clip.generate()) == 4


def test_short_completion_does_not_warn(monkeypatch, caplog):
    from comfy_api.latest import io

    class _CountingCore:
        @classmethod
        def execute(cls, clip, prompt, max_length, sampling_mode, **kwargs):
            clip.generate(max_length=8)
            return io.NodeOutput('{"ok": true}', "brief")

    monkeypatch.setattr(pe, "_load_text_generate_v2", lambda: _CountingCore)
    clip = _clip(clip_name="qwen35_9b", paths=[NINE_B_FILE])

    def original(*args, **kwargs):
        return [1, 2, 3]

    clip.generate = original

    with caplog.at_level(logging.INFO, logger=_LOG):
        TextGeneratePE.execute(
            clip,
            _SECRET_PROMPT,
            4096,
            {"sampling_mode": "off"},
            thinking=True,
            reasoning_effort="auto",
            model_file=_SECRET_PATH,
        )
    assert clip.generate is original
    assert _records(caplog, logging.WARNING) == []
    assert _SECRET_PATH not in caplog.text
    assert _SECRET_PROMPT not in caplog.text


def test_generate_observer_is_removed_when_the_answer_is_empty(monkeypatch):
    from comfy_api.latest import io

    class _CountingCore:
        @classmethod
        def execute(cls, clip, prompt, max_length, sampling_mode, **kwargs):
            clip.generate(max_length=max_length)
            return io.NodeOutput("", "reasoning")

    monkeypatch.setattr(pe, "_load_text_generate_v2", lambda: _CountingCore)
    clip = _clip(clip_name="qwen35_9b", paths=[NINE_B_FILE])

    def original(*args, **kwargs):
        return list(range(int(kwargs["max_length"])))

    clip.generate = original
    with pytest.raises(RuntimeError):
        TextGeneratePE.execute(
            clip,
            "prompt",
            32,
            {"sampling_mode": "off"},
            thinking=True,
            reasoning_effort="auto",
        )
    assert clip.generate is original


def test_unobservable_generate_still_runs(monkeypatch, caplog):
    from comfy_api.latest import io

    class _Frozen:
        def __init__(self):
            object.__setattr__(self, "clip", _clip(clip_name="qwen35_27b", paths=[PE_FILE]))

        def __getattr__(self, name):
            return getattr(self.clip, name)

        def __setattr__(self, name, value):
            if name == "generate":
                raise AttributeError("frozen")
            setattr(self.clip, name, value)

    clip = _Frozen()

    def original(*args, **kwargs):
        return [1]

    clip.clip.generate = original
    with caplog.at_level(logging.DEBUG, logger=_LOG):
        result = _run(
            monkeypatch,
            clip,
            result=io.NodeOutput("answer", ""),
            reasoning_effort="xhigh",
        )
    assert result.values[0] == "answer"
    assert clip.clip.generate is original
    assert any("could not observe generate" in record.message for record in caplog.records)
