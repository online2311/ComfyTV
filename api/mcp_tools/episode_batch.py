"""MCP tool: plan or serially submit one VideoStage per storyboard shot."""

from __future__ import annotations

from ..episode_batch import run_episode_batch
from ..headless_queue import QueueBusy
from ...nodes.stages.common.episode_video import EpisodeBatchError


async def _episode_batch(args: dict) -> dict:
    try:
        return await run_episode_batch(args or {})
    except QueueBusy as exc:
        raise ValueError(str(exc)) from exc
    except EpisodeBatchError as exc:
        raise ValueError(str(exc)) from exc


_SHOT_ITEM = {"anyOf": [{"type": "string"}, {"type": "integer"}]}

TOOLS: dict[str, dict] = {
    "episode_batch": {
        "description": (
            "Plan or submit a whole episode as one VideoStage per shot from a "
            "shots.json document or a StoryboardStage. action 'plan' is a dry "
            "run: it resolves each shot's reference images through the series "
            "library (asset codes and episode: frames), picks an independent "
            "reproducible seed, builds the video prompt, chooses Ref2VA or "
            "FL2VA, and attaches dialogue audio. It does not call a model and "
            "does not submit. A shot with both a first frame and a last frame "
            "uses FL2VA; otherwise Ref2VA. video_mode, then mode, overrides "
            "that (auto keeps the frame rule). If the chosen mode is FL2VA, "
            "the shot has mapped dialogue, and that FL2VA workflow does not "
            "bind upstream audio, plan and run use Ref2VA instead "
            "(reason dialogue_downgrade): the first frame stays picture 1, "
            "the last frame is another reference image (still an error past "
            "9, and nothing is dropped), and the audio is wired to audio0 so "
            "the Ref2VA workflow receives it. video_mode FL2VA is explicit "
            "and is not downgraded; the audio stays on the slot and the plan "
            "warns that the workflow will not receive it. VideoStage duration_s is the "
            "shot's duration_s. cut_to is the edit point inside that length "
            "and must not be longer; it is not substituted for duration_s. "
            "Dialogue audio is mapped only by audio.by_line, audio.by_shot, "
            "or audio.dir plus audio.pattern ({shot_id}, {line_id}, {order}, "
            "{episode}, {scene_id}). Lines in one shot are concatenated in "
            "dialogue order onto audio.audio0. A line with no mapping, or a "
            "mapped file that is missing, is an error naming the shot and "
            "line. Audio longer than duration_s or cut_to is kept whole and "
            "reported; it is not trimmed. Shots with no dialogue get no audio "
            "slot. action 'run' materializes those VideoStage nodes on the "
            "headless graph and submits one at a time. Before each submit it "
            "reads ComfyUI's queue (the same list as GET /queue) and waits "
            "until nothing is running or pending. It waits for that prompt to "
            "finish before the next shot. It does not interrupt or free the "
            "queue. Close the ComfyTV page first. Completed shots are skipped "
            "on a later run (state_path). shots limits the run to those "
            "shot_id values or 1-based orders. max_shots submits at most that "
            "many new shots and leaves the rest pending so a short client "
            "timeout can continue by calling run again. rerun true, or a list "
            "of shot ids, submits those shots even when a previous run "
            "finished them."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["plan", "run"]},
                "document": {"type": "object"},
                "document_path": {"type": "string"},
                "node": {"type": "string"},
                "project_id": {"type": "string"},
                "library_root": {"type": "string"},
                "frames_root": {"type": "string"},
                "base_seed": {"type": "integer"},
                "shots": {"type": "array", "items": _SHOT_ITEM},
                "workflows": {
                    "type": "object",
                    "properties": {
                        "Ref2VA": {"type": "string"},
                        "FL2VA": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                "audio": {
                    "type": "object",
                    "properties": {
                        "dir": {"type": "string"},
                        "pattern": {"type": "string"},
                        "by_line": {"type": "object"},
                        "by_shot": {"type": "object"},
                    },
                    "additionalProperties": False,
                },
                "resolution": {"type": "string"},
                "aspect_ratio": {"type": "string"},
                "work_dir": {"type": "string"},
                "state_path": {"type": "string"},
                "require_locked": {"type": "boolean"},
                "max_shots": {"type": "integer"},
                "rerun": {
                    "anyOf": [
                        {"type": "boolean"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        "handler": _episode_batch,
    },
}
