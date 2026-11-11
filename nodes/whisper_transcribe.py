"""ComfyUI node: local Whisper large-v3 transcription."""
from __future__ import annotations

from comfy_api.latest import io

from ..runners.whisper_asr import (
    LANGUAGES,
    TARGET_SR,
    audio_dict_to_mono16k,
    load_media_mono16k,
    register_whisper_model_folders,
    resolve_media_path,
    write_output_srt,
)

try:
    from ..runners.whisper_model import run_whisper
except ImportError:
    # torch / transformers live in the ComfyUI venv. CI imports this module
    # through the stage registry and does not install them.
    run_whisper = None

register_whisper_model_folders()


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _punctuation_enabled(option, language: str, task: str = "transcribe") -> bool:
    """``auto`` follows the text Whisper will emit.

    zh transcription uses full-width punctuation. ``translate`` always writes
    English, so auto leaves that text alone. ``on`` and ``off`` override this.
    """
    if isinstance(option, bool):
        return option
    mode = str(option or "auto").strip().lower()
    if mode in {"on", "true", "1", "yes"}:
        return True
    if mode in {"off", "false", "0", "no"}:
        return False
    if (task or "transcribe").strip().lower() == "translate":
        return False
    return (language or "zh").strip().lower() == "zh"


def _has_audio(audio) -> bool:
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return False
    wave = audio["waveform"]
    if hasattr(wave, "numel"):
        return int(wave.numel()) > 0
    size = getattr(wave, "size", None)
    if isinstance(size, int):
        return size > 0
    return True


class WhisperTranscribe(io.ComfyNode):
    """Offline speech-to-text. Output slot 0 is SRT for SubtitleGenStage.

    The node is an output node, so a run with nothing wired downstream still
    has to fill ``/history``. ``ui["text"]`` is the SRT, ``ui["segments"]`` is
    the segments JSON, and ``ui["srt_file"]`` is the absolute path of
    ``output/comfytv/<name>.srt``.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyTV.WhisperTranscribe",
            display_name="Whisper Transcribe",
            category="ComfyTV/Audio",
            description=(
                "Local Whisper large-v3. Defaults suit a Chinese short drama "
                f"(language zh, {TARGET_SR} Hz internally). Weights stay on disk. "
                "A solo run returns the SRT and segments on the node UI and writes "
                "output/comfytv/<name>.srt."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="ComfyUI AUDIO dict. Used when wired; otherwise media is read from disk.",
                ),
                io.String.Input(
                    "media",
                    default="",
                    tooltip=(
                        "Audio or video file. Absolute path, /view? URL, or "
                        "'name.wav [output]'. Video audio is decoded with PyAV."
                    ),
                ),
                io.Combo.Input(
                    "language",
                    options=list(LANGUAGES),
                    default="zh",
                    tooltip="zh is the default for Chinese short drama. auto lets Whisper detect the language.",
                ),
                io.Combo.Input(
                    "task",
                    options=["transcribe", "translate"],
                    default="transcribe",
                    tooltip="transcribe keeps the spoken language. translate renders English.",
                ),
                io.Combo.Input(
                    "timestamps",
                    options=["segment", "word", "none"],
                    default="segment",
                    tooltip="segment fills SRT cues. word also adds per-word times in the JSON output.",
                ),
                io.Boolean.Input(
                    "unload",
                    default=False,
                    tooltip="Drop the cached model and call torch.cuda.empty_cache() after this run.",
                ),
                io.String.Input(
                    "model_dir",
                    default="",
                    tooltip=(
                        "Local Hugging Face snapshot. Empty uses COMFYTV_WHISPER_MODEL, "
                        "then models/whisper or models/stt, then the documented fallback "
                        "/home/nodecloud/whisper-v3/models/whisper-large-v3. Never downloads."
                    ),
                ),
                io.Combo.Input(
                    "normalize_punct",
                    options=["auto", "on", "off"],
                    default="auto",
                    tooltip=(
                        "auto rewrites half-width sentence punctuation to full-width "
                        "for Chinese transcription (language zh). translate writes English, "
                        "so auto leaves it unchanged. on forces full-width; off leaves the text unchanged. "
                        "A decimal point between digits stays '.'."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("srt", display_name="SRT"),
                io.String.Output("text", display_name="text"),
                io.String.Output("segments", display_name="segments"),
            ],
        )

    @classmethod
    def execute(
        cls,
        audio=None,
        media="",
        language="zh",
        task="transcribe",
        timestamps="segment",
        unload=False,
        model_dir="",
        normalize_punct="auto",
    ):
        language = language or "zh"
        task = task or "transcribe"
        timestamps = timestamps or "segment"
        if _has_audio(audio):
            samples = audio_dict_to_mono16k(audio)
        elif (media or "").strip():
            samples = load_media_mono16k(resolve_media_path(media))
        else:
            raise RuntimeError(
                "Whisper Transcribe needs an AUDIO input or a media file path."
            )
        if run_whisper is None:
            raise RuntimeError(
                "Whisper Transcribe needs torch and transformers in this ComfyUI process."
            )
        srt, text, segments = run_whisper(
            samples,
            language=language,
            task=task,
            timestamps=timestamps,
            model_dir=model_dir or "",
            unload=_as_bool(unload),
            normalize_punctuation=_punctuation_enabled(normalize_punct, language, task),
        )
        path = write_output_srt(media, srt)
        return io.NodeOutput(
            srt,
            text,
            segments,
            ui={
                "text": [srt],
                "segments": [segments],
                "srt_file": [path],
            },
        )
