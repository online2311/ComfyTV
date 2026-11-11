"""ShotImagesStage iterates a storyboard once per shot."""

from __future__ import annotations

import json

import pytest


def _storyboard(*shots: dict) -> str:
    return json.dumps({"shots": list(shots)})


@pytest.fixture
def shot_stage(monkeypatch, reset_db):
    from ComfyTV.nodes.stages import generators as g

    calls: list[dict] = []

    async def fake_invoke(**kwargs):
        calls.append(kwargs)
        prompt = kwargs["main_prompt"]
        if prompt == "boom":
            raise RuntimeError("sampler failed")
        return json.dumps({
            "images": [{
                "index": "1",
                "label": "#1",
                "image_url": f"/view?filename={prompt}.png&type=output",
            }],
        })

    monkeypatch.setattr(g, "invoke_runner", fake_invoke)
    monkeypatch.setattr(g.RUNNER_REGISTRY, "by_label", lambda *_a, **_k: object())
    return g, calls


async def test_one_image_per_shot_shares_refs_and_stage_seed(shot_stage):
    g, calls = shot_stage
    custom = json.dumps({"items": [{"key": "seed", "value": 7}]})
    refs = ["/view?filename=shared.png&type=input"]
    out = await g.ShotImagesStage.execute(
        workflow="Flux Schnell (per-shot)",
        resolution="1K",
        aspect_ratio="16:9",
        storyboard=_storyboard(
            {"shot_no": "1", "prompt": "wide"},
            {"shot_no": "2", "image_prompt": "close"},
        ),
        images=refs,
        custom_params=custom,
        selected_index=2,
    )
    assert [c["main_prompt"] for c in calls] == ["wide", "close"]
    assert all(c["upstream"]["images"] == refs for c in calls)
    assert all(c["custom_params"] == custom for c in calls)
    assert all("seed" not in c["options"] for c in calls)
    assert all(c["options"]["batch_size"] == 1 for c in calls)
    payload = json.loads(out.values[0])
    assert [img["index"] for img in payload["images"]] == ["1", "2"]
    assert out.values[1].endswith("close.png&type=output")


async def test_production_shot_uses_h3_image_and_leaves_asset_codes_alone(shot_stage):
    g, calls = shot_stage
    storyboard = json.dumps({
        "episode": "E01",
        "shots": [{
            "shot_id": "E01S01_01",
            "duration_s": 6,
            "prompts": {"h3_image": "the chamber", "krea2": "krea text"},
            "refs": [{"role": "character", "src": "CHR_yangguo@v1#front"}],
            "extras": 0,
            "custom_note": "keep",
        }],
    })
    out = await g.ShotImagesStage.execute(
        workflow="Flux",
        storyboard=storyboard,
        images=["/view?filename=shared.png&type=input"],
    )
    assert calls[0]["main_prompt"] == "the chamber"
    assert calls[0]["upstream"]["images"] == ["/view?filename=shared.png&type=input"]
    assert "CHR_yangguo" not in json.dumps(calls[0]["upstream"])
    assert json.loads(out.values[0])["images"][0]["index"] == "E01S01_01"


async def test_krea2_prompt_is_used_when_h3_image_is_absent(shot_stage):
    g, calls = shot_stage
    await g.ShotImagesStage.execute(
        workflow="Flux",
        storyboard=_storyboard({
            "shot_id": "E01S01_02",
            "prompts": {"krea2": "still of the door"},
        }),
    )
    assert calls[0]["main_prompt"] == "still of the door"


async def test_empty_storyboard_names_the_missing_shots(shot_stage):
    g, calls = shot_stage
    with pytest.raises(RuntimeError, match="no shots"):
        await g.ShotImagesStage.execute(workflow="Flux", storyboard="{}")
    assert calls == []


async def test_shots_without_prompts_are_reported(shot_stage):
    g, _calls = shot_stage
    with pytest.raises(RuntimeError, match="none of the 1 shots"):
        await g.ShotImagesStage.execute(
            workflow="Flux",
            storyboard=_storyboard({"shot_no": "1", "scene_purpose": "no prompt field"}),
        )


async def test_runner_error_is_included_when_nothing_is_produced(shot_stage):
    g, _calls = shot_stage
    with pytest.raises(RuntimeError, match="sampler failed"):
        await g.ShotImagesStage.execute(
            workflow="Flux",
            storyboard=_storyboard({"shot_no": "1", "prompt": "boom"}),
        )


async def test_a_failed_shot_does_not_drop_the_ones_that_succeeded(shot_stage):
    g, _calls = shot_stage
    out = await g.ShotImagesStage.execute(
        workflow="Flux",
        storyboard=_storyboard(
            {"shot_no": "1", "prompt": "boom"},
            {"shot_no": "2", "prompt": "ok"},
        ),
    )
    payload = json.loads(out.values[0])
    assert [img["prompt"] for img in payload["images"]] == ["ok"]


async def test_cancel_recorded_by_the_nested_executor_stops_later_shots(
    shot_stage, monkeypatch,
):
    """The nested run clears the interrupt flag and reports execution_interrupted."""
    g, calls = shot_stage

    class InterruptProcessingException(Exception):
        pass

    import comfy.model_management as mm
    monkeypatch.setattr(
        mm, "InterruptProcessingException", InterruptProcessingException, raising=False,
    )
    from ComfyTV.runners._nested_exec import raise_if_nested_failed

    class _Stopped:
        success = False
        status_messages = [
            ("execution_interrupted", {"prompt_id": "sub", "node_id": "31"}),
        ]

    async def fake_invoke(**kwargs):
        calls.append(kwargs)
        if kwargs["main_prompt"] == "first":
            raise_if_nested_failed(_Stopped(), "sub")
        return json.dumps({"images": [{
            "index": "1", "label": "#1", "image_url": "/view?filename=ok.png",
        }]})

    monkeypatch.setattr(g, "invoke_runner", fake_invoke)
    with pytest.raises(InterruptProcessingException):
        await g.ShotImagesStage.execute(
            workflow="Flux",
            storyboard=_storyboard(
                {"shot_no": "1", "prompt": "first"},
                {"shot_no": "2", "prompt": "second"},
            ),
        )
    assert [c["main_prompt"] for c in calls] == ["first"]


async def test_an_empty_image_list_is_not_blamed_on_missing_prompts(shot_stage, monkeypatch):
    g, _calls = shot_stage

    async def empty_images(**_kwargs):
        return json.dumps({"images": []})

    monkeypatch.setattr(g, "invoke_runner", empty_images)
    with pytest.raises(RuntimeError, match="returned no shots") as raised:
        await g.ShotImagesStage.execute(
            workflow="Flux",
            storyboard=_storyboard(
                {"shot_no": "1", "scene_purpose": "no prompt"},
                {"shot_no": "2", "prompt": "ran"},
            ),
        )
    assert "none of the" not in str(raised.value)


async def test_interrupt_stops_the_remaining_shots(shot_stage, monkeypatch):
    g, calls = shot_stage

    class InterruptProcessingException(Exception):
        pass

    async def exploding(**kwargs):
        calls.append(kwargs)
        raise InterruptProcessingException("cancelled")

    monkeypatch.setattr(g, "invoke_runner", exploding)
    with pytest.raises(InterruptProcessingException):
        await g.ShotImagesStage.execute(
            workflow="Flux",
            storyboard=_storyboard(
                {"shot_no": "1", "prompt": "first"},
                {"shot_no": "2", "prompt": "second"},
            ),
        )
    assert len(calls) == 1


async def test_per_shot_refs_and_seed_override_the_shared_values(shot_stage):
    g, calls = shot_stage
    shared = ["/view?filename=shared.png&type=input"]
    own = [
        "/view?filename=face.png&type=input",
        "not-a-url",
        {"src": "https://example.com/room.png"},
    ]
    await g.ShotImagesStage.execute(
        workflow="Flux",
        storyboard=_storyboard(
            {"shot_no": "1", "prompt": "own", "reference_images": own, "seed": 11},
            {"shot_no": "2", "prompt": "url", "image_url": "/view?filename=only.png&type=input"},
            {"shot_no": "3", "prompt": "text", "reference_img": "特写"},
            {"shot_no": "4", "prompt": "shared", "seed": ""},
        ),
        images=shared,
        custom_params='{"items":[{"key":"seed","value":7}]}',
    )
    assert calls[0]["upstream"]["images"] == [
        "/view?filename=face.png&type=input",
        "https://example.com/room.png",
    ]
    assert calls[0]["options"]["seed"] == 11
    assert calls[1]["upstream"]["images"] == ["/view?filename=only.png&type=input"]
    assert "seed" not in calls[1]["options"]
    assert calls[2]["upstream"]["images"] == shared
    assert calls[3]["upstream"]["images"] == shared
    assert "seed" not in calls[3]["options"]
    assert calls[3]["custom_params"] == calls[0]["custom_params"]


async def test_refs_objects_are_capped_at_eight(shot_stage):
    g, calls = shot_stage
    refs = [{"image_url": f"/view?filename={i}.png&type=input"} for i in range(10)]
    await g.ShotImagesStage.execute(
        workflow="Flux",
        storyboard=_storyboard({"shot_no": "1", "prompt": "many", "refs": refs}),
    )
    assert len(calls[0]["upstream"]["images"]) == 8
    assert calls[0]["upstream"]["images"][0].endswith("0.png&type=input")


async def test_missing_workflow_is_a_runner_error(monkeypatch, reset_db):
    from ComfyTV.nodes.stages import generators as g

    monkeypatch.setattr(g.RUNNER_REGISTRY, "by_label", lambda *_a, **_k: None)
    with pytest.raises(g.StageRunnerMissing, match="shot-images"):
        await g.ShotImagesStage.execute(
            workflow="missing",
            storyboard=_storyboard({"shot_no": "1", "prompt": "wide"}),
        )
