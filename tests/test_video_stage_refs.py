"""VideoStage passes up to 9 reference images, numbered <Picture 1>..N."""

from __future__ import annotations

import json

import pytest

from ComfyTV.nodes.stages.common.ref_slots import ImageCap


def _urls(n: int) -> list[str]:
    return [f"/view?filename=ref{i}.png&type=input" for i in range(1, n + 1)]


@pytest.fixture
def video_stage(monkeypatch):
    from ComfyTV.nodes.stages import generators as g

    calls: list[dict] = []

    async def fake_run(_cls, **kwargs):
        calls.append(kwargs)
        return "ran"

    monkeypatch.setattr(g, "run_stage_workflow", fake_run)
    monkeypatch.setattr(g, "load_image_cap", lambda kind, label: None)
    return g, calls


@pytest.mark.parametrize("count", [1, 3, 9])
async def test_picture_numbers_follow_the_images_passed_to_the_model(video_stage, count):
    g, calls = video_stage
    ordered = _urls(count)
    images: dict[str, str] = {}
    for i in reversed(range(count)):
        images[f"image{i}"] = ordered[i]
    prompt = " ".join(f"@image_{i}" for i in range(1, count + 1))
    await g.VideoStage.execute(
        workflow="Local MiniMax H3 R2V",
        main_prompt=prompt,
        images=images,
        seed=11,
        custom_params=json.dumps({"items": [
            {"key": "seed", "value": 7},
            {"key": "negative_prompt", "value": "blur"},
        ]}),
    )
    assert calls[0]["upstream"]["images"] == ordered
    assert calls[0]["main_prompt"] == " ".join(
        f"<Picture {i}>" for i in range(1, count + 1)
    )
    assert calls[0]["options"]["seed"] == 11
    assert json.loads(calls[0]["custom_params"])["items"] == [
        {"key": "negative_prompt", "value": "blur"},
    ]


async def test_legacy_custom_params_seed_is_used_when_the_field_is_unset(video_stage):
    g, calls = video_stage
    await g.VideoStage.execute(
        workflow="GXS",
        main_prompt="hold",
        images=_urls(3),
        seed=-1,
        custom_params=json.dumps({"items": [{"key": "seed", "value": 20261113}]}),
    )
    assert calls[0]["options"]["seed"] == 20261113
    assert calls[0]["upstream"]["images"] == _urls(3)
    assert json.loads(calls[0]["custom_params"])["items"] == []


async def test_legacy_three_picture_params_fill_empty_slots(video_stage):
    g, calls = video_stage
    await g.VideoStage.execute(
        workflow="GXS",
        main_prompt="@image_1 @image_2 @image_3",
        images=None,
        custom_params=json.dumps({"items": [
            {"key": "picture_2", "value": "/view?filename=b.png&type=input"},
            {"key": "picture_1", "value": "/view?filename=a.png&type=input"},
            {"key": "picture_3", "value": "/view?filename=c.png&type=input"},
        ]}),
    )
    assert calls[0]["upstream"]["images"] == [
        "/view?filename=a.png&type=input",
        "/view?filename=b.png&type=input",
        "/view?filename=c.png&type=input",
    ]
    assert calls[0]["main_prompt"] == "<Picture 1> <Picture 2> <Picture 3>"
    assert json.loads(calls[0]["custom_params"])["items"] == []


async def test_a_three_slot_model_rejects_a_fourth_image(video_stage, monkeypatch):
    g, calls = video_stage
    monkeypatch.setattr(
        g, "load_image_cap",
        lambda kind, label: ImageCap(3, "this workflow binds 3 reference-image inputs"),
    )
    await g.VideoStage.execute(workflow="GXS", main_prompt="ok", images=_urls(3))
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="3") as raised:
        await g.VideoStage.execute(workflow="GXS", main_prompt="too many", images=_urls(4))
    assert "not dropped" in str(raised.value)
    assert len(calls) == 1


async def test_nine_images_fit_a_minimax_reference_workflow(video_stage, monkeypatch):
    g, calls = video_stage
    monkeypatch.setattr(
        g, "load_image_cap",
        lambda kind, label: ImageCap(
            9, "ComfyUI v0.45.0 MiniMaxH3ReferenceToVideo accepts 9 reference images",
        ),
    )
    await g.VideoStage.execute(
        workflow="Local MiniMax H3 R2V",
        main_prompt="@image_9",
        images=_urls(9),
        seed=0,
    )
    assert len(calls[0]["upstream"]["images"]) == 9
    assert calls[0]["main_prompt"] == "<Picture 9>"
    assert calls[0]["options"]["seed"] == 0
