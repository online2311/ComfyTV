import logging
import uuid
from datetime import datetime

from .base import Runner, RunnerContext
from ._workflow_resolve import (  # noqa: F401
    _Resolver,
    _SHORT_SIDE_BY_TIER,
    _aspect_ratio_value,
    _cast,
    _composite_masked_image,
    _resolve_default,
    _resolve_length,
    _resolve_wh,
    _view_url_to_annotated,
)
from ._workflow_mutate import (  # noqa: F401
    _apply_overrides,
    _apply_prunes,
    _apply_text_replacements,
    _auto_detect_result,
    _auto_prune_unbound,
    _output_node_ids,
    _split_runner_id,
)
from ._nested_exec import (  # noqa: F401
    _extract_result,
    _filter_subprompt_preview,
    _run_subprompt,
    _save_files_from,
    _translate_subprompt_event,
    _view_url,
)

_log = logging.getLogger(__name__)


def prepare_workflow(runner_id: str, kinds, ctx: RunnerContext) -> tuple[dict, dict]:
    from . import workflow_db
    kind, label = _split_runner_id(runner_id)

    if ctx.kind not in kinds:
        raise NotImplementedError(
            f"{runner_id} doesn't handle kind={ctx.kind!r} "
            f"(declared: {sorted(kinds)})"
        )

    config = workflow_db.get_workflow_for_invoke(kind, label)
    if config is None:
        raise RuntimeError(
            f"workflow {kind!r}/{label!r} not in DB — startup seed missed it?"
        )

    import copy
    workflow = copy.deepcopy(config["api_json"])

    pruned_nodes = _apply_prunes(workflow, config, ctx)
    pruned_nodes |= _auto_prune_unbound(workflow, config, ctx)
    resolver = _Resolver(config, ctx)
    _apply_overrides(workflow, config, resolver, pruned_nodes)
    _apply_text_replacements(workflow, datetime.now())

    result_meta = _custom_multi_result(config) or config.get("result") or {}
    if not result_meta.get("node"):
        result_meta = _auto_detect_result(workflow, ctx.kind)
    result_node = result_meta.get("node")
    _stamp_force_run_token(workflow, getattr(ctx, "force_run_token", None))
    if not result_node:
        raise RuntimeError(
            f"{runner_id}: no result node — select the stage on the canvas, "
            f"open the ComfyTV sidebar, and pick a node under 'Result' "
            f"(or ship a `_preset.json` declaring `result`)."
        )
    return workflow, result_meta


def _custom_multi_result(config: dict) -> dict:
    from .custom_io import normalize_custom_io, result_meta_for
    custom_io = (config.get("meta") or {}).get("custom_io")
    if not custom_io:
        return {}
    return result_meta_for(normalize_custom_io(custom_io))


def _is_cached_loader(class_type: str) -> bool:
    """Weight and media loaders keep their output cache across a force re-run.

    ``CheckpointLoaderSimple``, ``UNETLoader``, ``CLIPLoader``, ``VAELoader``,
    ``LoadImage``, and pack names such as ``VHS_LoadVideo`` do not depend on
    the run token. A downstream sampler's cache key includes its own inputs,
    so stamping the sampler (or any custom node that is not a loader) misses
    that cache without reloading weights.
    """
    name = str(class_type or "")
    if "loader" in name.lower():
        return True
    short = name.rsplit("|", 1)[-1].rsplit(".", 1)[-1]
    return short.startswith("Load") or short.startswith("load") or "_Load" in short


def _stamp_force_run_token(workflow: dict, token) -> None:
    """Put the token on generation nodes as a literal input.

    ComfyUI v0.45 hashes every non-link input in ``CacheKeySetInputSignature``.
    ``fingerprint_inputs`` on ImageStage would only change the outer node, and
    the nested ``PromptExecutor`` keeps its own classic cache. A new token
    therefore has to be inside the sub-prompt. Seed and the other declared
    widgets are left as they are. Loaders are not stamped, so their cached
    weights stay loaded. Nodes that do not declare this input ignore it.
    """
    if token is None:
        return
    try:
        token = int(token)
    except (TypeError, ValueError):
        return
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if _is_cached_loader(str(node.get("class_type") or "")):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
            node["inputs"] = inputs
        inputs["_comfytv_force_run_token"] = token


def _execute_node_ids(workflow: dict, result_meta: dict) -> list[str]:
    if result_meta.get("type") == "multi":
        ids = [str(o.get("node") or "") for o in result_meta.get("outputs") or []]
        return [nid for nid in ids if nid and nid in workflow]
    return _output_node_ids(workflow, result_meta.get("node"))


class LocalComfyUIRunner(Runner):

    async def invoke(self, ctx: RunnerContext):
        workflow, result_meta = prepare_workflow(self.id, self.kinds, ctx)
        result_node = result_meta.get("node")

        sub_prompt_id = str(uuid.uuid4())
        _log.info("[ComfyTV/%s] comfytv-%s  nodes=%d",
                  self.id, sub_prompt_id[:8], len(workflow))

        execute_outputs = _execute_node_ids(workflow, result_meta)
        executor = await _run_subprompt(workflow, sub_prompt_id,
                                        execute_outputs=execute_outputs)
        return await _extract_result(executor, result_meta)
