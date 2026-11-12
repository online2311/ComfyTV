"""ComfyUI node: replace Whisper text with the script, keeping Whisper times."""
from __future__ import annotations

import json
from pathlib import Path

from comfy_api.latest import io

from ..runners.subtitle_align import (
    DEFAULT_LOW_CONFIDENCE,
    DEFAULT_MAX_LINE_CELLS,
    align_subtitles,
)
from ..runners.whisper_asr import write_output_srt


class SubtitleAlignStage(io.ComfyNode):
    """SRT whose words are the script and whose times are Whisper's.

    Wire Whisper Transcribe's segments output, or paste that JSON (word or
    segment times) or an SRT. The script is a shots document, a list of
    lines, or one sentence per line. A solo run writes the SRT and a JSON
    report under ``output/comfytv``.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyTV.SubtitleAlignStage",
            display_name="Subtitle Script Align",
            category="ComfyTV/Audio",
            description=(
                "Character-level alignment of script dialogue onto a Whisper "
                "timeline. The SRT uses the script's characters and Whisper's "
                "times, so a homophone such as 练/炼 is written as the script."
            ),
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "segments",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Whisper segments JSON (word times when present, otherwise "
                        "one span per segment), an SRT, or a per-shot map. "
                        "Empty means nothing was recognized."
                    ),
                ),
                io.String.Input(
                    "script",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Shots JSON (dialogue text by shot), a list of "
                        "{shot_id, line_id, text}, or one sentence per line."
                    ),
                ),
                io.Int.Input(
                    "max_line_cells",
                    default=DEFAULT_MAX_LINE_CELLS,
                    min=2,
                    max=80,
                    step=1,
                    tooltip=(
                        "Subtitle line budget in the same cells the burner uses: "
                        "a CJK character is 2. Closing punctuation stays on the "
                        "previous line and opening punctuation stays with the next."
                    ),
                ),
                io.Float.Input(
                    "low_confidence",
                    default=DEFAULT_LOW_CONFIDENCE,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip=(
                        "Sentences whose exact-character ratio is below this "
                        "are listed in the report. Their cue text is still the script."
                    ),
                ),
                io.String.Input(
                    "media",
                    default="",
                    tooltip=(
                        "Filename stem for output/comfytv/<stem>.srt and "
                        "<stem>.align.json. Empty writes subtitle-align."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("srt", display_name="SRT"),
                io.String.Output("report", display_name="report"),
                io.String.Output("text", display_name="text"),
            ],
        )

    @classmethod
    def execute(
        cls,
        segments="",
        script="",
        max_line_cells=DEFAULT_MAX_LINE_CELLS,
        low_confidence=DEFAULT_LOW_CONFIDENCE,
        media="",
    ):
        if not str(script or "").strip():
            raise RuntimeError(
                "Subtitle Align needs the script dialogue "
                "(shots JSON or one sentence per line)."
            )
        try:
            cells = int(max_line_cells)
        except (TypeError, ValueError):
            cells = DEFAULT_MAX_LINE_CELLS
        cells = min(80, max(2, cells))
        try:
            result = align_subtitles(
                segments or "",
                script,
                max_line_cells=cells,
                low_confidence=low_confidence,
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        stem = (media or "").strip() or "subtitle-align.wav"
        srt_path = Path(write_output_srt(stem, result["srt"]))
        report_path = srt_path.with_name(srt_path.stem + ".align.json")
        report_json = json.dumps(result["report"], ensure_ascii=False, indent=2)
        report_path.write_text(report_json, encoding="utf-8")
        return io.NodeOutput(
            result["srt"],
            report_json,
            result["text"],
            ui={
                "text": [result["srt"]],
                "report": [report_json],
                "srt_file": [str(srt_path)],
                "report_file": [str(report_path)],
            },
        )
