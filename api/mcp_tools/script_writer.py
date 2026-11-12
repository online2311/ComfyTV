"""MCP tool for script sessions, versions, and manifest import."""

from ...storage.script_writer import dispatch


async def _script_writer(args: dict) -> dict:
    return dispatch(args if isinstance(args, dict) else {})


TOOLS: dict[str, dict] = {
    "script_writer": {
        "description": (
            "Script sessions stored under a project directory's script/ folder. "
            "The default root is <ComfyUI input>/comfytv/projects/<project> "
            "(COMFYTV_ASSET_ROOT replaces the comfytv directory). An explicit "
            "root must stay inside that projects directory. Give root or "
            "project; there is no user-directory fallback. This does not read or write "
            "library.yaml; character IDs stay on series_library. "
            "Versions only grow: .versions/<doc>/v<n>.<ext> is written once and "
            "kept. An import copies bytes into imported/<source_path> and never "
            "overwrites them. "
            "action 'import' takes manifest (schema comfytv/script-import.v1) "
            "and source_dir. The manifest lists file_count and files of "
            "{source_path, sha256, bytes, doc}. status 'active' is the official "
            "episode (episodes/E01/script_v2.md as doc script, shots.json as doc "
            "shots). status 'archived' keeps an older lineage such as root "
            "script/, storyboard.yaml, and state.json, and cannot use an "
            "episodes/ path or replace the active session. Shot ids and legacy "
            "prefixes inside the files are not rewritten. "
            "gumu-bixie is refused unless script/config.json sets "
            "excluded_projects to an empty list. "
            "actions that call the model (directions, choose, revise, "
            "write_episode, review) return job_id. Poll status. "
            "confirm requires approved_by 张景. review is skipped when "
            "script/config.json sets review_enabled to false. "
            "to_shots is not this tool. The model defaults to DeepSeek V4 Pro "
            "over an OpenAI-compatible chat completions endpoint; set llm "
            "{provider:'local', model} for a local endpoint. API keys are read "
            "from the environment or llm-keys files and are not stored."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "start", "directions", "choose", "revise", "accept",
                        "save_manual", "confirm", "write_episode", "diff",
                        "revert", "review", "get", "import", "status",
                        "verify", "current_storyboard",
                    ],
                },
                "root": {"type": "string"},
                "project": {"type": "string"},
                "episode": {"type": "string"},
                "idea": {"type": "string"},
                "source_text": {"type": "string"},
                "reference_works": {"type": "string"},
                "series": {"type": "string"},
                "llm": {"type": "object"},
                "session_id": {"type": "string"},
                "n": {"type": "integer"},
                "extra": {"type": "string"},
                "direction_id": {"type": "string"},
                "note": {"type": "string"},
                "doc": {"type": "string"},
                "instruction": {"type": "string"},
                "selection": {"type": "object"},
                "candidate_id": {"type": "string"},
                "content": {"type": "string"},
                "approved_by": {"type": "string"},
                "a": {"type": "integer"},
                "b": {"type": "integer"},
                "to": {"type": "integer"},
                "job_id": {"type": "string"},
                "manifest": {"type": "object"},
                "source_dir": {"type": "string"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "handler": _script_writer,
    },
}
