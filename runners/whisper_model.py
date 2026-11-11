"""Load openai/whisper-large-v3 locally and run long-form generate().

Imported only when torch and transformers are installed. The load call matches
the Spark environment: transformers 5.x ``dtype`` (not ``torch_dtype``), no
``device_map`` (accelerate is not installed), ``local_files_only=True``.
``low_cpu_mem_usage`` is still passed; transformers 5.22 pops and ignores it.
"""
from __future__ import annotations

import gc
import logging
import threading

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .whisper_asr import (
    LANGUAGES,
    TARGET_SR,
    finalize_segments,
    pack_transcript,
    resolve_model_dir,
    segments_from_generate_output,
)

_log = logging.getLogger(__name__)
_LOCK = threading.Lock()
_CACHE: dict = {"path": None, "model": None, "processor": None}

_TASKS = ("transcribe", "translate")
_TIMESTAMP_MODES = ("segment", "word", "none")


def unload_whisper() -> None:
    """Drop the cached model and release CUDA cache. Does not touch ComfyUI /free."""
    with _LOCK:
        _CACHE["model"] = None
        _CACHE["processor"] = None
        _CACHE["path"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load(path: str):
    with _LOCK:
        if _CACHE["path"] == path and _CACHE["model"] is not None:
            return _CACHE["model"], _CACHE["processor"]
        _CACHE["model"] = None
        _CACHE["processor"] = None
        _CACHE["path"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log.info("[ComfyTV] loading local Whisper snapshot %s", path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            path,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to("cuda")
        model.eval()
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        _CACHE["path"] = path
        _CACHE["model"] = model
        _CACHE["processor"] = processor
        return model, processor


def _check_options(language: str, task: str, timestamps: str) -> None:
    if task not in _TASKS:
        raise RuntimeError(f"Whisper task must be one of {_TASKS}, got {task!r}")
    if timestamps not in _TIMESTAMP_MODES:
        raise RuntimeError(
            f"Whisper timestamps must be one of {_TIMESTAMP_MODES}, got {timestamps!r}"
        )
    if language not in LANGUAGES:
        raise RuntimeError(
            f"Whisper language must be one of {LANGUAGES}, got {language!r}"
        )


def run_whisper(
    samples,
    *,
    language: str = "zh",
    task: str = "transcribe",
    timestamps: str = "segment",
    model_dir: str = "",
    unload: bool = False,
) -> tuple[str, str, str]:
    """Transcribe mono 16 kHz audio. Returns ``(srt, text, segments_json)``."""
    _check_options(language, task, timestamps)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Whisper Transcribe loads the model with .to('cuda') and needs a CUDA device."
        )
    audio = torch.as_tensor(samples, dtype=torch.float32).reshape(-1).cpu().numpy()
    if audio.size == 0:
        return "", "", "[]"
    path = resolve_model_dir(model_dir)
    model, processor = _load(path)
    previous_attn = getattr(model.model.config, "_attn_implementation", None)
    try:
        inputs = processor(
            audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
        features = inputs.input_features.to(device=model.device, dtype=model.dtype)
        long_form = int(features.shape[-1]) > 3000
        want_timestamps = timestamps != "none" or long_form
        gen_kwargs = {
            "task": task,
            "return_timestamps": bool(want_timestamps),
        }
        mask = getattr(inputs, "attention_mask", None)
        if mask is None and isinstance(inputs, dict):
            mask = inputs.get("attention_mask")
        if mask is not None:
            gen_kwargs["attention_mask"] = mask.to(device=model.device)
        if language != "auto":
            gen_kwargs["language"] = language
        if timestamps == "word":
            gen_kwargs["return_token_timestamps"] = True
        if want_timestamps:
            gen_kwargs["return_segments"] = True
        if long_form:
            # The model's own long-form loop, not pipeline chunk_length_s.
            gen_kwargs["condition_on_prev_tokens"] = True
            gen_kwargs["temperature"] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            gen_kwargs["compression_ratio_threshold"] = 1.35
            gen_kwargs["logprob_threshold"] = -1.0
            gen_kwargs["no_speech_threshold"] = 0.6
        with torch.inference_mode():
            try:
                outputs = model.generate(features, **gen_kwargs)
            except ValueError as exc:
                if "return_segments" not in str(exc):
                    raise
                gen_kwargs.pop("return_segments", None)
                outputs = model.generate(features, **gen_kwargs)
        duration = float(audio.shape[0]) / float(TARGET_SR)
        segments = segments_from_generate_output(
            outputs,
            processor.tokenizer,
            timestamps=timestamps,
            duration_s=duration,
        )
        segments = finalize_segments(
            segments, timestamps=timestamps, duration_s=duration,
        )
        return pack_transcript(segments)
    finally:
        if timestamps == "word" and previous_attn is not None:
            model.model.config._attn_implementation = previous_attn
        if unload:
            unload_whisper()
