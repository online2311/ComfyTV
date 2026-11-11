"""force_run_token is a cache key for ImageStage's nested ComfyUI prompt."""

from __future__ import annotations

import json

import pytest

from ComfyTV.runners.base import RunnerContext


def _config():
    return {
        "api_json": {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5.safetensors"}},
            "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen.safetensors"}},
            "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen.safetensors"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen.safetensors"}},
            "5": {"class_type": "LoadImage", "inputs": {"image": "ref.png"}},
            "6": {"class_type": "VHS_LoadVideo", "inputs": {"video": "clip.mp4"}},
            "7": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 4, "model": ["1", 0]}},
            "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0]}},
            "9": {"class_type": "MyCustomSampler", "inputs": {"model": ["2", 0]}},
        },
        "result": {"type": "ui_save_url", "node": "8"},
        "inputs": {},
        "sizing": {},
        "prune_when_missing": [],
    }


def test_prepare_workflow_stamps_a_new_token_without_touching_seed(monkeypatch):
    from ComfyTV.runners import local_comfy as lc
    from ComfyTV.runners import workflow_db

    monkeypatch.setattr(workflow_db, "get_workflow_for_invoke", lambda _k, _l: _config())
    ctx = RunnerContext(kind="image", main_prompt="cat")
    ctx.force_run_token = 5
    first, _meta = lc.prepare_workflow("image/demo", {"image"}, ctx)
    assert first["7"]["inputs"]["seed"] == 1
    assert first["7"]["inputs"]["steps"] == 4
    assert first["7"]["inputs"]["_comfytv_force_run_token"] == 5
    assert first["8"]["inputs"]["_comfytv_force_run_token"] == 5
    assert first["9"]["inputs"]["_comfytv_force_run_token"] == 5
    assert first["8"]["inputs"]["images"] == ["7", 0]
    assert first["7"]["inputs"]["model"] == ["1", 0]
    for loader_id in ("1", "2", "3", "4", "5", "6"):
        assert "_comfytv_force_run_token" not in first[loader_id]["inputs"]

    ctx.force_run_token = 6
    second, _meta = lc.prepare_workflow("image/demo", {"image"}, ctx)
    assert second["7"]["inputs"]["_comfytv_force_run_token"] == 6
    assert first["7"]["inputs"]["_comfytv_force_run_token"] == 5


def test_prepare_workflow_leaves_the_prompt_alone_without_a_token(monkeypatch):
    from ComfyTV.runners import local_comfy as lc
    from ComfyTV.runners import workflow_db

    monkeypatch.setattr(workflow_db, "get_workflow_for_invoke", lambda _k, _l: _config())
    workflow, _meta = lc.prepare_workflow(
        "image/demo", {"image"}, RunnerContext(kind="image", main_prompt="cat"),
    )
    assert "_comfytv_force_run_token" not in workflow["7"]["inputs"]


@pytest.mark.asyncio
async def test_image_stage_forwards_force_run_token(monkeypatch):
    from ComfyTV.nodes.stages import generators as g

    captured: dict = {}

    async def fake_invoke(**kwargs):
        captured.update(kwargs)
        return json.dumps({"images": [{"image_url": "/view?filename=a.png&type=output"}]})

    monkeypatch.setattr(g, "invoke_runner", fake_invoke)
    monkeypatch.setattr(g, "_stage_emit_auto", lambda *args, **kwargs: {"ok": True})
    await g.ImageStage.execute(
        force_run_token=99,
        workflow="demo",
        resolution="1K",
        aspect_ratio="1:1",
        batch_size=1,
        main_prompt="cat",
    )
    assert captured["force_run_token"] == 99
