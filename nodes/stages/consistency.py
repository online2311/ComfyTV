"""Score a shot video against locked character refs and reshoot below threshold.

The run keeps every take. Sorted scores are the handoff for stop ③ (pick a
take). A repeat execution with the same inputs and the same force_run_token
reuses that report, so changing the pick does not submit another video.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from ._common import *  # noqa: F401, F403
from ...runners.consistency_rescore import (
    DEFAULT_FRAME_COUNT,
    DEFAULT_MAX_RESHOOTS,
    DEFAULT_THRESHOLD,
    ConsistencyRescoreError,
    apply_selection,
    cache_key,
    coerce_video_url,
    run_consistency_rescore,
)


class ConsistencyRescoreStage(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyTV.ConsistencyRescoreStage",
            display_name="Consistency Rescore",
            category="ComfyTV/Video",
            inputs=[
                *_standard_stage_inputs(),
                io.Combo.Input(
                    "workflow",
                    options=labels_for("video") or [""],
                    default=default_for("video"),
                    tooltip="Video workflow used when a take is below threshold and must be reshot.",
                ),
                io.Combo.Input("resolution", options=RESOLUTIONS, default="720P"),
                io.Combo.Input("aspect_ratio", options=ASPECT_RATIOS, default="16:9"),
                io.Int.Input(
                    "duration_s",
                    default=VIDEO_DURATION_DEFAULT_S,
                    min=VIDEO_DURATION_MIN_S,
                    max=VIDEO_DURATION_MAX_S,
                    step=1,
                ),
                io.Float.Input(
                    "threshold",
                    default=DEFAULT_THRESHOLD,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Minimum similarity to a locked character reference. Below this, reshoot.",
                ),
                io.Int.Input(
                    "max_reshoots",
                    default=DEFAULT_MAX_RESHOOTS,
                    min=0,
                    max=8,
                    step=1,
                    tooltip="How many new seeds may be submitted after the incoming take. Default 2.",
                ),
                io.Int.Input(
                    "frame_count",
                    default=DEFAULT_FRAME_COUNT,
                    min=1,
                    max=12,
                    step=1,
                    tooltip="Frames sampled from each take and placed on the contact sheet.",
                ),
                io.Int.Input(
                    "seed",
                    default=-1,
                    min=-1,
                    max=2_147_483_647,
                    step=1,
                    tooltip="Seed of the incoming take. Each reshoot adds 1.",
                ),
                io.Combo.Input(
                    "backend",
                    options=["auto", "clip_vision", "face"],
                    default="auto",
                    tooltip="Similarity backend. auto uses CLIP vision when that node and a clip_vision model are loaded.",
                ),
                io.String.Input(
                    "series_root",
                    default="",
                    tooltip="Series folder that holds library.yaml. Empty uses the bound series library.",
                ),
                io.String.Input(
                    "comfy_base_url",
                    default="",
                    tooltip="ComfyUI origin used for GET /queue before a reshoot is submitted.",
                ),
                _main_prompt_input(
                    tooltip="Motion prompt for a reshoot. Empty uses the shot's motion_desc.",
                ),
                io.Autogrow.Input("images", template=_image_template(9)),
                _selected_index_input(),
                _custom_params_input(),
                COMFYTV_STORYBOARD.Input("storyboard", optional=True),
                io.Int.Input("shot_index", default=1, min=1, max=999, step=1),
                COMFYTV_VIDEO.Input("video", optional=True),
            ],
            outputs=[
                COMFYTV_IMAGES.Output("takes"),
                COMFYTV_VIDEO.Output("video"),
                COMFYTV_IMAGE.Output("contact_sheet"),
            ],
            is_output_node=True,
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    async def execute(
        cls,
        force_run_token=0,
        project_id="",
        parent_output_id=0,
        workflow="",
        resolution="720P",
        aspect_ratio="16:9",
        duration_s=VIDEO_DURATION_DEFAULT_S,
        threshold=DEFAULT_THRESHOLD,
        max_reshoots=DEFAULT_MAX_RESHOOTS,
        frame_count=DEFAULT_FRAME_COUNT,
        seed=-1,
        backend="auto",
        series_root="",
        comfy_base_url="",
        main_prompt="",
        images=None,
        selected_index=1,
        custom_params="{}",
        storyboard=None,
        shot_index=1,
        video="",
        cache_path=None,
    ):
        key = cache_key(
            video_url=video or "",
            shot_index=int(shot_index or 1),
            threshold=float(threshold),
            max_reshoots=int(max_reshoots),
            frame_count=int(frame_count),
            seed=int(seed),
            backend=backend or "auto",
            workflow=workflow or "",
            series_root=series_root or "",
            force_run_token=int(force_run_token or 0),
        )
        cache_file = _cache_file(cache_path, project_id, _node_id(cls))
        stored = _read_cache(cache_file, key)
        if stored is None:
            report = run_consistency_rescore(
                video_url=video,
                storyboard=storyboard,
                shot_index=int(shot_index or 1),
                series_root=series_root or None,
                backend=backend or "auto",
                base_url=comfy_base_url or None,
                threshold=float(threshold),
                max_reshoots=int(max_reshoots),
                frame_count=int(frame_count),
                seed=int(seed),
                workflow=workflow or "",
                prompt=main_prompt or "",
                images=_autogrow_values(images),
                options={
                    "resolution": resolution,
                    "aspect_ratio": aspect_ratio,
                    "duration_s": duration_s,
                },
                selected_index=int(selected_index or 1),
                allow_running=True,
                submitter=_NestedReshoot(),
                custom_params=custom_params or "{}",
            )
            if inspect.isawaitable(report):
                report = await report
            report["cache_key"] = key
            _write_cache(cache_file, report)
        else:
            report = stored
        report = apply_selection(report, int(selected_index or 1))
        return _stage_emit_auto(
            cls,
            project_id=project_id,
            payload_str=json.dumps(report),
            parent_output_id=parent_output_id,
            picked_payload=report.get("selected_video") or "",
            picked_index=report.get("selected_index"),
            extra_outputs=(report.get("contact_sheet") or "",),
        )


class _NestedReshoot:
    """Generate the next take inside this prompt after GET /queue has no pending work.

    A second POST /prompt cannot start while this stage is the running prompt.
    """

    async def submit(self, job):
        if not str(job.workflow or "").strip():
            raise ConsistencyRescoreError("reshoot needs a video workflow label")
        options = dict(job.options or {})
        options["seed"] = int(job.seed)
        payload = await invoke_runner(
            kind="video",
            label=job.workflow,
            main_prompt=job.prompt or "",
            upstream={
                "texts": [],
                "images": list(job.images or []),
                "videos": [],
                "audio": [],
            },
            options=options,
            custom_params=job.custom_params or "{}",
        )
        return coerce_video_url(payload)


def _node_id(cls):
    hidden = getattr(cls, "hidden", None)
    return getattr(hidden, "unique_id", None)


def _cache_file(cache_path, project_id: str, node_id) -> Path:
    if cache_path:
        return Path(cache_path)
    base = Path(folder_paths.get_output_directory()) / "comfytv" / "consistency"
    label = f"{project_id or 'default'}-{node_id if node_id is not None else 'stage'}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    return base / f"{safe}.json"


def _read_cache(cache_path, key: str) -> dict | None:
    if not cache_path:
        return None
    path = Path(cache_path)
    if not path.is_file():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(stored, dict) or stored.get("cache_key") != key:
        return None
    return stored


def _write_cache(cache_path, report: dict) -> None:
    if not cache_path:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
