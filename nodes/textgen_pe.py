"""Text generation node for MiniMax-H3-Image-PE reasoning effort.

ComfyUI v0.45.0 loads the converted PE file
``text_encoders/minimax_h3_image_pe_nvfp4.safetensors`` as ``Qwen35TEModel_``
with ``clip_name == "qwen35_27b"``. The same class and the same 27B shape are
what a stock Qwen3.5-27B text encoder gets. ``qwen3.5_9b_w4a8_gptq`` is also a
``Qwen35TEModel_``, distinguished only by ``clip_name == "qwen35_9b"`` (hidden
size 4096, 32 layers). Layer count alone is not a 9B test: Qwen3.5-4B has 32
layers too. ``CLIPLoader`` type ``minimax`` is not a PE test either. The
Qwen3.5 weight branch in ``load_text_encoder_state_dicts`` does not look at
that type, so a stock 27B loaded with type ``minimax`` is still
``Qwen35TEModel_``.

The checkpoint path is the signal that separates PE from those models.
``comfy.sd.load_clip`` stores the files it opened on
``clip.patcher.cached_patcher_init`` as
``(loader, (ckpt_paths, embedding_directory, clip_type, model_options))``, and
``ModelPatcher.clone`` copies that tuple. A name is PE only when it contains
``minimax_h3_image_pe``. Anything else, including a missing path, is not PE
and does not raise. The optional ``model_file`` input is used only when the
CLIP object has no checkpoint path. A Qwen3.5-9B identity (``qwen35_9b``, or a
9B filename such as ``qwen3.5_9b``) is never PE, even if another string
contains the PE marker.

``auto`` follows that result: PE gets the official default ``xhigh``
instruction, and every other model gets no extra system text. ``xhigh``,
``medium``, ``low``, and ``off`` are manual and do not depend on detection.
``medium`` and ``off`` insert nothing. Thinking off inserts nothing, matching
the official template (the effort block is omitted when ``enable_thinking``
is false).

The composed system text is handed to core ``TextGenerateV2.execute``. This
module does not copy that sampler. An empty answer is a node error. The bot
LLM path is left alone: ``generate_chat`` treats an empty ``content`` plus
``tool_calls`` as a normal turn, and ``prompt_family`` already refuses names
containing ``minimax`` or ``h3``. Storyboard and Text stage workflows call
core ``TextGenerate`` inside a graph; an empty string there is parsed as "no
shot list" or returned as workflow output, which is a different contract.
"""
from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal, Never

from comfy_api.latest import io

try:
    from comfy_extras.nodes_textgen import TextGenerateV2 as _TextGenerateV2
except ImportError:  # unit tests and CI do not install ComfyUI core
    _TextGenerateV2 = None

_log = logging.getLogger(__name__)

PE_FILENAME_MARK = "minimax_h3_image_pe"
QWEN35_9B = "qwen35_9b"

XHIGH_INSTRUCTIONS = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer."
)
LOW_INSTRUCTIONS = (
    "Reasoning effort is set to low. Keep your thinking brief and focused, "
    "moving directly to the conclusion without unnecessary elaboration."
)

OFFICIAL_MAX_TOKENS = 4096

_Effort = Literal["auto", "xhigh", "medium", "low", "off"]
_Resolved = Literal["xhigh", "medium", "low", "off"]
_EFFORTS: tuple[_Effort, ...] = ("auto", "xhigh", "medium", "low", "off")


@dataclass(frozen=True)
class PeDetection:
    """Result of a PE filename check. ``is_pe`` is false when the check is unsure."""

    is_pe: bool
    matched_name: str
    rejected_as_9b: bool
    saw_checkpoint: bool


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_effort(value: Any) -> _Effort:
    key = str(value or "auto").strip().lower()
    for item in _EFFORTS:
        if key == item:
            return item
    return "off"


def _basename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _looks_like_9b_filename(value: str) -> bool:
    folded = _basename(value).casefold()
    compact = folded.replace(".", "_").replace("-", "_")
    return (
        "qwen3.5_9b" in folded
        or "qwen3.5-9b" in folded
        or QWEN35_9B in folded
        or "qwen3_5_9b" in compact
    )


def _marked_pe_filename(value: str) -> bool:
    base = _basename(value).casefold().replace("-", "_")
    return PE_FILENAME_MARK in base


def _identity_names(clip: Any) -> list[str]:
    names: list[str] = []
    stage = getattr(clip, "cond_stage_model", None)
    if stage is not None:
        clip_name = getattr(stage, "clip_name", None)
        if isinstance(clip_name, str) and clip_name:
            names.append(clip_name)
        attr = getattr(stage, "clip", None)
        inner = getattr(stage, attr, None) if isinstance(attr, str) else None
        transformer = getattr(inner, "transformer", None) if inner is not None else None
        model_type = getattr(transformer, "model_type", None) if transformer is not None else None
        if isinstance(model_type, str) and model_type:
            names.append(model_type)
    tokenizer = getattr(clip, "tokenizer", None)
    tokenizer_name = getattr(tokenizer, "clip_name", None) if tokenizer is not None else None
    if isinstance(tokenizer_name, str) and tokenizer_name:
        names.append(tokenizer_name)
    return names


def _is_qwen35_9b(clip: Any, filenames: list[str]) -> bool:
    for name in _identity_names(clip):
        if name.casefold() == QWEN35_9B:
            return True
    return any(_looks_like_9b_filename(name) for name in filenames)


def _names_from_cached(cached: Any) -> list[str]:
    if not isinstance(cached, tuple) or len(cached) < 2:
        return []
    args = cached[1]
    if not isinstance(args, tuple) or not args:
        return []
    first = args[0]
    raw: list[Any]
    if isinstance(first, (str, os.PathLike)):
        raw = [first]
    elif isinstance(first, (list, tuple)):
        raw = list(first)
    else:
        return []
    names: list[str] = []
    for item in raw:
        if isinstance(item, (str, os.PathLike)):
            text = os.fspath(item)
            if text:
                names.append(text)
    return names


def _checkpoint_names(clip: Any) -> list[str]:
    names: list[str] = []
    patcher = getattr(clip, "patcher", None)
    seen: set[int] = set()
    while patcher is not None and id(patcher) not in seen:
        seen.add(id(patcher))
        names.extend(_names_from_cached(getattr(patcher, "cached_patcher_init", None)))
        patcher = getattr(patcher, "parent", None)
    return names


def detect_pe(clip: Any, model_file: str = "") -> PeDetection:
    """Return whether this CLIP is MiniMax-H3-Image-PE.

    A missing path, an unexpected CLIP shape, or any other failure is "not
    PE". Callers must not treat that as an error.
    """
    try:
        return _detect_pe(clip, model_file)
    except Exception as exc:
        _log.debug("PE detection failed (%s); treating the model as not PE", type(exc).__name__)
        return PeDetection(False, "", False, False)


def _detect_pe(clip: Any, model_file: str) -> PeDetection:
    checkpoints = _checkpoint_names(clip)
    fallback = (model_file or "").strip()
    if checkpoints:
        candidates = checkpoints
        saw_checkpoint = True
    elif fallback:
        candidates = [fallback]
        saw_checkpoint = False
    else:
        candidates = []
        saw_checkpoint = False

    if _is_qwen35_9b(clip, candidates):
        return PeDetection(False, "", True, saw_checkpoint)

    for name in candidates:
        if _marked_pe_filename(name):
            return PeDetection(True, name, False, saw_checkpoint)
    return PeDetection(False, "", False, saw_checkpoint)


def resolved_effort(effort: Any, *, thinking: bool, is_pe: bool) -> _Resolved:
    """Official effort after ``auto`` and the thinking switch.

    Thinking off matches the template: the effort block is not inserted.
    ``auto`` on a non-PE or unidentified model inserts nothing.
    """
    if not thinking:
        return "off"
    choice = _normalize_effort(effort)
    if choice == "auto":
        return "xhigh" if is_pe else "off"
    if choice == "xhigh":
        return "xhigh"
    if choice == "medium":
        return "medium"
    if choice == "low":
        return "low"
    if choice == "off":
        return "off"
    unexpected: Never = choice
    raise AssertionError(unexpected)


def reasoning_instructions(effort: _Resolved) -> str:
    if effort == "xhigh":
        return XHIGH_INSTRUCTIONS
    if effort == "low":
        return LOW_INSTRUCTIONS
    if effort == "medium" or effort == "off":
        return ""
    unexpected: Never = effort
    raise AssertionError(unexpected)


def compose_system_prompt(user_system: Any, instructions: str) -> str:
    """Join the official effort line with the user system text.

    With both, the template emits ``instructions + '\\n\\n' + trimmed content``.
    With only the effort line, the system block is that line. With neither, the
    user string is returned unchanged so a non-PE run matches TextGenerateV2.
    """
    if not instructions:
        return "" if user_system is None else str(user_system)
    content = ("" if user_system is None else str(user_system)).strip()
    if content:
        return instructions + "\n\n" + content
    return instructions


def system_prompt_for(
    user_system: Any,
    effort: Any,
    *,
    thinking: bool,
    is_pe: bool,
) -> str:
    instructions = reasoning_instructions(
        resolved_effort(effort, thinking=thinking, is_pe=is_pe)
    )
    return compose_system_prompt(user_system, instructions)


def empty_answer_message(
    *,
    thinking: bool,
    max_length: int,
    reasoning_chars: int,
) -> str:
    if reasoning_chars:
        observed = (
            "the model returned reasoning but the answer text is empty "
            f"({reasoning_chars} reasoning characters)"
        )
    else:
        observed = "the generation is empty"
    return (
        "ComfyTV.TextGeneratePE produced no answer text "
        f"({observed}). thinking={thinking}, max_length={max_length}. "
        "Possible causes: thinking is off, so MiniMax-H3-Image-PE emits the "
        "end token immediately and the text is empty; or max_length is too "
        "small to finish thinking and the answer. The official recommendation "
        f"is {OFFICIAL_MAX_TOKENS} tokens, including thinking."
    )


def ensure_answer_present(
    text: Any,
    thinking_text: Any,
    *,
    thinking: bool,
    max_length: int,
) -> None:
    answer = ("" if text is None else str(text)).strip()
    if answer:
        return
    reasoning = ("" if thinking_text is None else str(thinking_text)).strip()
    raise RuntimeError(
        empty_answer_message(
            thinking=thinking,
            max_length=max_length,
            reasoning_chars=len(reasoning),
        )
    )


def _result_texts(result: Any) -> tuple[str, str]:
    values = getattr(result, "args", None)
    if values is None:
        values = getattr(result, "values", None)
    if values is None and isinstance(result, (tuple, list)):
        values = result
    if not values:
        return "", ""
    text = values[0] if len(values) > 0 else ""
    reasoning = values[1] if len(values) > 1 else ""
    return "" if text is None else str(text), "" if reasoning is None else str(reasoning)


def _load_text_generate_v2():
    if _TextGenerateV2 is None:
        raise RuntimeError(
            "ComfyTV.TextGeneratePE needs ComfyUI's TextGenerateV2 "
            "(comfy_extras.nodes_textgen)."
        )
    return _TextGenerateV2


def call_text_generate_v2(**kwargs: Any) -> Any:
    """Call core TextGenerateV2 with the v0.45.0 argument names it accepts."""
    node = _load_text_generate_v2()
    signature = inspect.signature(node.execute)
    parameters = signature.parameters
    if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return node.execute(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    return node.execute(**accepted)


def _sampling_options():
    return [
        io.DynamicCombo.Option(
            key="on",
            inputs=[
                io.Float.Input("temperature", default=0.7, min=0.01, max=2.0, step=0.000001),
                io.Int.Input("top_k", default=64, min=0, max=1000),
                io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01),
                io.Float.Input("repetition_penalty", default=1.05, min=0.0, max=5.0, step=0.01),
                io.Float.Input("presence_penalty", default=0.0, min=0.0, max=5.0, step=0.01),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
            ],
        ),
        io.DynamicCombo.Option(key="off", inputs=[]),
    ]


class TextGeneratePE(io.ComfyNode):
    """Generate Text with MiniMax-H3-Image-PE reasoning effort.

    Inputs match TextGenerateV2. ``reasoning_effort`` selects the official
    system line, and ``model_file`` is the filename fallback. Defaults follow
    the PE card: thinking on, 4096 tokens.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyTV.TextGeneratePE",
            display_name="Generate Text (PE)",
            category="ComfyTV/Generate",
            description=(
                "TextGenerateV2 plus MiniMax-H3-Image-PE reasoning effort. "
                "auto inserts the official xhigh system line only when the "
                "checkpoint name contains minimax_h3_image_pe. Qwen3.5-9B and "
                "a stock Qwen3.5-27B stay on the normal template. An empty "
                "answer is an error."
            ),
            search_aliases=["LLM", "minimax", "PE", "reasoning", "TextGenerate"],
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image{i}" for i in range(1, 51)],
                        min=0,
                    ),
                ),
                io.Autogrow.Input(
                    "videos",
                    optional=True,
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input(
                            "video",
                            tooltip="Video frames as image batch. Assumed to be 24 FPS; subsampled to the model's frame rate internally.",
                        ),
                        names=[f"video{i}" for i in range(1, 101)],
                        min=0,
                    ),
                ),
                io.Autogrow.Input(
                    "audios",
                    optional=True,
                    template=io.Autogrow.TemplateNames(
                        io.Audio.Input("audio"),
                        names=[f"audio{i}" for i in range(1, 101)],
                        min=0,
                    ),
                ),
                io.Int.Input(
                    "max_length",
                    default=OFFICIAL_MAX_TOKENS,
                    min=1,
                    max=32768,
                    tooltip=(
                        "Maximum tokens to generate, including thinking. "
                        "4096 is the official MiniMax-H3-Image-PE recommendation."
                    ),
                ),
                io.DynamicCombo.Input(
                    "sampling_mode",
                    options=_sampling_options(),
                    display_name="Sampling Mode",
                ),
                io.Boolean.Input(
                    "thinking",
                    optional=True,
                    default=True,
                    tooltip=(
                        "Operate in thinking mode if the model supports it. "
                        "PE defaults to on. Off skips the reasoning-effort line "
                        "and often makes PE emit the end token with an empty answer."
                    ),
                ),
                io.Boolean.Input(
                    "use_default_template",
                    optional=True,
                    default=True,
                    tooltip="Use the built in system prompt/template if the model has one. The reasoning-effort line is part of that system prompt.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "mtp",
                    options=["auto", "off", "2", "3", "4", "5", "tree2s1", "tree2s2", "tree3s1", "tree4s1m8"],
                    default="auto",
                    optional=True,
                    tooltip="Speculative decoding with the checkpoint's multi-token-prediction head. Passed through to TextGenerateV2.",
                ),
                io.String.Input(
                    "system_prompt",
                    force_input=True,
                    optional=True,
                    tooltip="User system text. When a reasoning-effort line is active it is placed first, then a blank line, then this text.",
                ),
                io.Combo.Input(
                    "reasoning_effort",
                    options=list(_EFFORTS),
                    default="auto",
                    tooltip=(
                        "auto inserts the official xhigh line only for a PE checkpoint "
                        "(filename contains minimax_h3_image_pe) and inserts nothing otherwise. "
                        "xhigh and low insert that official line. medium and off insert nothing. "
                        "Ignored when thinking is off."
                    ),
                ),
                io.String.Input(
                    "model_file",
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "Fallback checkpoint name, used only when the CLIP object has no loaded path. "
                        "A name containing minimax_h3_image_pe is treated as PE. "
                        "Qwen3.5-9B is never treated as PE."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="generated_text"),
                io.String.Output(display_name="thinking"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        prompt,
        max_length,
        sampling_mode,
        images=None,
        videos=None,
        audios=None,
        thinking=True,
        use_default_template=True,
        mtp="auto",
        system_prompt="",
        reasoning_effort="auto",
        model_file="",
        image=None,
        video=None,
        audio=None,
    ):
        thinking_on = _as_bool(thinking, default=True)
        length = int(max_length)
        detection = detect_pe(clip, model_file or "")
        composed = system_prompt_for(
            system_prompt,
            reasoning_effort,
            thinking=thinking_on,
            is_pe=detection.is_pe,
        )
        result = call_text_generate_v2(
            clip=clip,
            prompt=prompt,
            max_length=length,
            sampling_mode=sampling_mode,
            image=image,
            thinking=thinking_on,
            use_default_template=use_default_template,
            video=video,
            audio=audio,
            mtp=mtp,
            system_prompt=composed,
            images=images,
            videos=videos,
            audios=audios,
        )
        text, reasoning = _result_texts(result)
        ensure_answer_present(
            text,
            reasoning,
            thinking=thinking_on,
            max_length=length,
        )
        return result
