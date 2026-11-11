"""B1, B4, and F1: injected storyboard data, stable image indexes, per-shot slots.

The t_sb_chain case supplies a storyboard document and wires it into Shot
Images, then into a video stage. The LLM must not run when that document is
present.
"""

from __future__ import annotations

import json

import pytest
import yaml
from comfy_api.latest import io

import ComfyTV.runners.media as media
from ComfyTV.nodes.stages import director as dstage


def _doc(*shots: dict, **envelope) -> str:
    return json.dumps({"shots": list(shots), **envelope}, ensure_ascii=False)


def _bind_library(tmp_path, monkeypatch) -> None:
    from ComfyTV.storage.series_library import bind_series_library

    root = tmp_path / "gumu-sx"
    rel = "CHR_xiaolongnv/v1/front.png"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_bytes(b"front")
    doc = {
        "schema_version": 1,
        "series": {"id": "gumu-sx"},
        "assets": [{
            "id": "CHR_xiaolongnv",
            "version": 1,
            "kind": "character",
            "status": "locked",
            "approved_by": "tester",
            "files": {"front": {"path": rel}},
        }],
    }
    (root / "library.yaml").write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    user = tmp_path / "user" / "default"
    user.mkdir(parents=True, exist_ok=True)
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(user))
    bind_series_library(root)


@pytest.fixture
def chain(monkeypatch, reset_db, tmp_path):
    from ComfyTV.nodes.stages import generators as g

    llm_calls: list[dict] = []
    image_calls: list[dict] = []
    video_calls: list[dict] = []

    async def fake_run(cls, **kwargs):
        name = getattr(cls, "__name__", "")
        if name == "StoryboardStage":
            llm_calls.append(kwargs)
            raise AssertionError("StoryboardStage called the LLM")
        video_calls.append(kwargs)
        return "video-ran"

    async def fake_invoke(**kwargs):
        image_calls.append(kwargs)
        n = len(image_calls)
        return json.dumps({
            "images": [{
                "image_url": f"/view?filename=shot{n}.png&type=output",
            }],
        })

    monkeypatch.setattr(g, "run_stage_workflow", fake_run)
    monkeypatch.setattr(g, "invoke_runner", fake_invoke)
    monkeypatch.setattr(g.RUNNER_REGISTRY, "by_label", lambda *_a, **_k: object())
    monkeypatch.setattr(g, "load_image_cap", lambda *_a, **_k: None)
    _bind_library(tmp_path, monkeypatch)
    return g, llm_calls, image_calls, video_calls


INJECTED = _doc(
    {
        "shot_no": 2,
        "image_prompt": "INJECTED stone lantern",
        "seed": 4242,
        "reference_images": [
            "/view?filename=lantern.png&type=input",
            "CHR_xiaolongnv@v1#front",
        ],
        "pipeline_note": "keep-me",
    },
    {
        "shot_no": "99",
        "image_prompt": "INJECTED door",
        "seed": "7",
        "characters": [{"id": "CHR_xiaolongnv@v1"}],
    },
)


async def test_t_sb_chain_uses_injected_data_and_skips_the_llm(chain):
    """Report t_sb_chain: storyboard_data is the shot list; the LLM stays idle."""
    g, llm_calls, image_calls, video_calls = chain
    out = await g.StoryboardStage.execute(
        workflow="DeepSeek V4 Pro Storyboard (API)",
        main_prompt="a different premise that must not be sent",
        shot_count=3,
        total_duration_s=19,
        storyboard_data=INJECTED,
        force_run_token=1_700_000_000,
    )
    assert llm_calls == []
    doc = json.loads(out.values[0])
    assert doc["shots"][0]["image_prompt"] == "INJECTED stone lantern"
    assert doc["shots"][0]["seed"] == 4242
    assert doc["shots"][0]["reference_images"] == [
        "/view?filename=lantern.png&type=input",
        "CHR_xiaolongnv@v1#front",
    ]
    assert doc["shots"][0]["pipeline_note"] == "keep-me"
    assert doc["shots"][1]["characters"] == [{"id": "CHR_xiaolongnv@v1"}]

    images = await g.ShotImagesStage.execute(
        workflow="Krea2 Turbo Shot Draft",
        storyboard=out.values[0],
        seed=222,
        selected_index=2,
    )
    assert [call["main_prompt"] for call in image_calls] == [
        "INJECTED stone lantern",
        "INJECTED door",
    ]
    assert image_calls[0]["options"]["seed"] == 4242
    assert image_calls[0]["upstream"]["images"] == [
        "/view?filename=lantern.png&type=input",
    ]
    assert image_calls[1]["options"]["seed"] == 7
    assert any("CHR_xiaolongnv/v1" in url and "front.png" in url
               for url in image_calls[1]["upstream"]["images"])

    payload = json.loads(images.values[0])
    assert [img["index"] for img in payload["images"]] == [1, 2]
    assert all(isinstance(img["index"], int) for img in payload["images"])
    assert [img["shot_no"] for img in payload["images"]] == ["2", "99"]
    assert payload["images"][0]["seed"] == 4242
    assert payload["images"][0]["reference_images"] == image_calls[0]["upstream"]["images"]
    assert images.values[1] == "/view?filename=shot2.png&type=output"

    await g.VideoStage.execute(
        workflow="Local MiniMax H3 R2V",
        main_prompt="hold",
        storyboard=out.values[0],
        shot_index=1,
        seed=222,
    )
    assert video_calls[0]["options"]["seed"] == 4242
    assert video_calls[0]["upstream"]["images"] == image_calls[0]["upstream"]["images"]

    await g.VideoStage.execute(
        workflow="Local MiniMax H3 R2V",
        storyboard=out.values[0],
        shot_index=2,
        seed=-1,
    )
    assert video_calls[1]["options"]["seed"] == 7
    assert video_calls[1]["upstream"]["images"] == image_calls[1]["upstream"]["images"]

    await g.VideoStage.execute(
        workflow="Local MiniMax H3 R2V",
        main_prompt="hold",
        images=["/view?filename=frame.png&type=output"],
        storyboard=out.values[0],
        shot_index=1,
        seed=-1,
    )
    assert video_calls[2]["upstream"]["images"][0] == "/view?filename=frame.png&type=output"
    assert "/view?filename=lantern.png&type=input" in video_calls[2]["upstream"]["images"]
    assert video_calls[2]["options"]["seed"] == 4242


async def test_force_run_does_not_turn_supplied_data_into_an_llm_rewrite(chain):
    g, llm_calls, _images, _video = chain
    await g.StoryboardStage.execute(
        workflow="DeepSeek",
        storyboard_data=INJECTED,
        force_run_token=99,
        source="auto",
    )
    await g.StoryboardStage.execute(
        workflow="DeepSeek",
        storyboard_data=INJECTED,
        force_run_token=100,
        source="data",
    )
    assert llm_calls == []


def _node_output(payload: str):
    return io.NodeOutput(payload)


async def test_source_llm_still_rewrites_and_keeps_character_slots(chain, monkeypatch):
    g, llm_calls, _images, _video = chain

    async def fake_run(cls, **kwargs):
        llm_calls.append(kwargs)
        raw = "镜号: 1\n时长: 4\n角色: 林岳_赛车手\n分镜提示词: the car\n"
        return _node_output(kwargs["transform"](raw))

    monkeypatch.setattr(g, "run_stage_workflow", fake_run)
    out = await g.StoryboardStage.execute(
        workflow="DeepSeek",
        main_prompt="crash",
        storyboard_data=INJECTED,
        source="llm",
        force_run_token=5,
        characters=(
            "- 林岳_赛车手: 32岁男性 "
            "@/view?filename=lin.png&type=input "
            "CHR_xiaolongnv@v1#front seed=11"
        ),
    )
    assert llm_calls
    assert "crash" in llm_calls[0]["main_prompt"]
    shot = json.loads(out.values[0])["shots"][0]
    assert shot["image_prompt"] == "the car"
    assert shot["seed"] == 11
    assert "/view?filename=lin.png&type=input" in shot["reference_images"]
    assert "CHR_xiaolongnv@v1#front" in shot["reference_images"]


async def test_empty_data_follows_the_source_switch(chain, monkeypatch):
    g, llm_calls, _images, _video = chain

    async def fake_run(cls, **kwargs):
        llm_calls.append(kwargs)
        return _node_output(kwargs["transform"]("镜号: 1\n时长: 4\n分镜提示词: from llm\n"))

    monkeypatch.setattr(g, "run_stage_workflow", fake_run)
    out = await g.StoryboardStage.execute(
        workflow="DeepSeek",
        main_prompt="premise",
        storyboard_data="   ",
        source="auto",
    )
    assert len(llm_calls) == 1
    assert json.loads(out.values[0])["shots"][0]["prompt"] == "from llm"

    with pytest.raises(RuntimeError, match="storyboard_data"):
        await g.StoryboardStage.execute(
            workflow="DeepSeek",
            storyboard_data="",
            source="data",
        )
    assert len(llm_calls) == 1


async def test_invalid_storyboard_data_is_not_sent_to_the_llm(chain):
    g, llm_calls, _images, _video = chain
    with pytest.raises(RuntimeError, match="JSON"):
        await g.StoryboardStage.execute(
            workflow="DeepSeek",
            storyboard_data="{not json",
            source="auto",
        )
    assert llm_calls == []


async def test_production_document_is_passed_through_unchanged(chain):
    g, llm_calls, image_calls, _video = chain
    original = {
        "episode": "E01",
        "studio_tag": "night",
        "shots": [{
            "shot_id": "E01S01_01",
            "duration_s": 6,
            "seed": 3,
            "reference_images": ["CHR_xiaolongnv@v1#front"],
            "custom_note": {"keep": True},
            "prompts": {"h3_image": "the chamber"},
        }],
    }
    out = await g.StoryboardStage.execute(
        workflow="DeepSeek",
        storyboard_data=json.dumps(original, ensure_ascii=False),
    )
    assert llm_calls == []
    assert json.loads(out.values[0]) == original
    images = await g.ShotImagesStage.execute(
        workflow="Flux",
        storyboard=out.values[0],
    )
    assert image_calls[0]["main_prompt"] == "the chamber"
    assert image_calls[0]["options"]["seed"] == 3
    assert any("front.png" in url for url in image_calls[0]["upstream"]["images"])
    row = json.loads(images.values[0])["images"][0]
    assert row["index"] == 1
    assert isinstance(row["index"], int)
    assert row["shot_no"] == "E01S01_01"
    assert row["label"] == "Shot E01S01_01"


async def test_shot_index_types_stay_stable_for_int_and_str_shot_numbers(chain):
    g, _llm, _images, _video = chain
    out = await g.ShotImagesStage.execute(
        workflow="Flux",
        storyboard=_doc(
            {"shot_no": 12, "prompt": "wide"},
            {"shot_no": "99", "image_prompt": "close"},
            {"shot_id": "E01S01_03", "prompt": "third"},
        ),
        selected_index=2,
    )
    rows = json.loads(out.values[0])["images"]
    assert [(row["index"], row["shot_no"]) for row in rows] == [
        (1, "12"),
        (2, "99"),
        (3, "E01S01_03"),
    ]
    assert out.values[1] == "/view?filename=shot2.png&type=output"


async def test_director_clips_take_per_shot_slots_when_the_clip_has_none(
    chain, monkeypatch,
):
    g, _llm, _images, _video = chain
    calls: list[dict] = []

    async def fake_invoke(**kwargs):
        calls.append(kwargs)
        return f"/view?filename=clip{len(calls)}.mp4"

    monkeypatch.setattr(dstage, "invoke_runner", fake_invoke)
    monkeypatch.setattr(dstage, "load_image_cap", lambda *_a, **_k: None)
    monkeypatch.setattr(dstage, "_mention_style_for", lambda _label: "")

    monkeypatch.setattr(media, "concat_videos", lambda urls, progress=None: "/view?filename=final.mp4")
    monkeypatch.setattr(media, "extract_frame", lambda url, position="last": "/view?filename=frame.png")

    storyboard = await g.StoryboardStage.execute(
        workflow="DeepSeek",
        storyboard_data=INJECTED,
    )
    timeline = json.dumps({
        "version": 1,
        "settings": {},
        "clips": [
            {"id": "c1", "prompt": "lantern", "duration_s": 4},
            {"id": "c2", "prompt": "door", "duration_s": 4, "seed": 99,
             "images": ["/view?filename=own.png&type=input"]},
        ],
    })
    await dstage.DirectorStage.execute(
        project_id="default",
        workflow="Local MiniMax H3 R2V",
        timeline_data=timeline,
        storyboard=storyboard.values[0],
    )
    assert calls[0]["options"]["seed"] == 4242
    assert calls[0]["upstream"]["images"] == ["/view?filename=lantern.png&type=input"]
    assert calls[1]["options"]["seed"] == 99
    assert calls[1]["upstream"]["images"] == ["/view?filename=own.png&type=input"]
