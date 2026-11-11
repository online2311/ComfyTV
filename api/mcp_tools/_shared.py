from ..headless_dispatch import execute_headless, page_is_live  # noqa: F401
from ..mcp_commands import submit_command  # noqa: F401 — patched in tests


def _no_args_schema() -> dict:
    return {"type": "object", "properties": {}, "additionalProperties": False}

def _command_payload(args: dict, keys: tuple[str, ...]) -> dict:
    return {k: args[k] for k in keys if args.get(k) is not None}


async def submit_or_headless(action: str, payload: dict, timeout: float = 15.0) -> dict:
    """Use the open page when it is live; otherwise run the command server-side."""
    project_id = payload.get("project_id")
    pid = project_id if isinstance(project_id, str) else None
    if page_is_live(pid):
        return await submit_command(action, payload, timeout=timeout)
    return await execute_headless(action, payload)
