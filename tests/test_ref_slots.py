"""Reference-image slots and first-class seeds for video stages.

Slots are 1-based and match the order passed to the model:
<Picture 1> is the first image, <Picture 9> the ninth. A stage accepts
at most 9. A workflow may accept fewer; extras are an error.
"""

from __future__ import annotations

import json

import pytest

from ComfyTV.nodes.stages.common.ref_slots import (
    MAX_REFERENCE_IMAGES,
    ReferenceSlotError,
    align_picture_prompt,
    ordered_reference_images,
    picture_slots,
    prepare_references,
    reference_image_cap,
)


def _urls(n: int) -> list[str]:
    return [f"/view?filename=ref{i}.png&type=input" for i in range(1, n + 1)]


def _params(*items: dict) -> str:
    return json.dumps({"items": list(items)})


@pytest.mark.parametrize("count", [1, 3, 9])
def test_slots_are_numbered_in_pass_order(count):
    images = _urls(count)
    slots = picture_slots(images)
    assert [s["n"] for s in slots] == list(range(1, count + 1))
    assert [s["tag"] for s in slots] == [f"<Picture {n}>" for n in range(1, count + 1)]
    assert [s["url"] for s in slots] == images
    prompt = " ".join(f"@image_{n}" for n in range(1, count + 1))
    assert align_picture_prompt(prompt, count) == " ".join(
        f"<Picture {n}>" for n in range(1, count + 1)
    )


def test_dict_order_follows_slot_index_not_insertion_order():
    images = ordered_reference_images({
        "image2": "/view?filename=c.png&type=input",
        "image0": "/view?filename=a.png&type=input",
        "image1": "/view?filename=b.png&type=input",
    })
    assert [s["tag"] for s in picture_slots(images)] == [
        "<Picture 1>", "<Picture 2>", "<Picture 3>",
    ]
    assert images[0].endswith("a.png&type=input")
    assert images[2].endswith("c.png&type=input")


def test_empty_holes_do_not_shift_later_pictures():
    images = ordered_reference_images({
        "image0": "/view?filename=a.png&type=input",
        "image1": None,
        "image2": "/view?filename=c.png&type=input",
    })
    assert [s["url"] for s in picture_slots(images)] == [
        "/view?filename=a.png&type=input",
        "/view?filename=c.png&type=input",
    ]


def test_tenth_image_is_an_error_not_a_silent_drop():
    with pytest.raises(ReferenceSlotError, match="9") as raised:
        prepare_references(
            _urls(10), seed_field=-1, custom_params="{}",
            cap=None, where="VideoStage",
        )
    assert "not dropped" in str(raised.value)
    assert "<Picture 9>" in str(raised.value)


def test_prompt_picture_past_the_passed_list_is_an_error():
    with pytest.raises(ReferenceSlotError, match="Picture 4"):
        align_picture_prompt("hold <Picture 4>", 3)
    with pytest.raises(ReferenceSlotError, match="Picture 2"):
        align_picture_prompt("use @image_2", 1)


def test_seed_field_wins_and_is_written_out_of_custom_params():
    images, seed, written = prepare_references(
        _urls(1),
        seed_field=11,
        custom_params=_params(
            {"key": "seed", "value": 7},
            {"key": "negative_prompt", "value": "blur"},
        ),
        cap=None,
        where="VideoStage",
    )
    assert images == _urls(1)
    assert seed == 11
    data = json.loads(written)
    assert data["items"] == [{"key": "negative_prompt", "value": "blur"}]


def test_unset_seed_reads_legacy_custom_params_and_writes_it_out():
    _images, seed, written = prepare_references(
        [],
        seed_field=-1,
        custom_params=_params(
            {"key": "seed", "value": "42"},
            {"key": "negative_prompt", "value": "blur"},
        ),
        cap=None,
        where="VideoStage",
    )
    assert seed == 42
    assert json.loads(written)["items"] == [
        {"key": "negative_prompt", "value": "blur"},
    ]


def test_legacy_three_picture_slots_are_read_in_order_when_unwired():
    images, seed, written = prepare_references(
        [],
        seed_field=-1,
        custom_params=_params(
            {"key": "picture_3", "value": "/view?filename=c.png&type=input"},
            {"key": "picture_1", "value": "/view?filename=a.png&type=input"},
            {"key": "seed", "value": 5},
            {"key": "picture_2", "value": "/view?filename=b.png&type=input"},
        ),
        cap=3,
        where="GXS",
    )
    assert [s["tag"] for s in picture_slots(images)] == [
        "<Picture 1>", "<Picture 2>", "<Picture 3>",
    ]
    assert images[0].endswith("a.png&type=input")
    assert images[2].endswith("c.png&type=input")
    assert seed == 5
    assert json.loads(written)["items"] == []


def test_wired_images_are_the_new_form_and_legacy_picture_keys_are_dropped():
    images, _seed, written = prepare_references(
        ["/view?filename=wired.png&type=input"],
        seed_field=-1,
        custom_params=_params(
            {"key": "picture_1", "value": "/view?filename=old.png&type=input"},
            {"key": "seed", "value": 3},
        ),
        cap=3,
        where="VideoStage",
    )
    assert images == ["/view?filename=wired.png&type=input"]
    assert json.loads(written)["items"] == []


def test_three_slot_workflow_accepts_one_and_three_and_rejects_four():
    cfg = {
        "inputs": {
            "10": {"image": {"from": "upstream_image:annotated[0]"}},
            "11": {"image": {"from": "upstream_image:annotated[1]"}},
            "12": {"image": {"from": "upstream_image:annotated[2]"}},
        },
        "api_json": {"99": {"class_type": "GXSRef2VA"}},
    }
    cap = reference_image_cap(cfg)
    assert cap is not None and cap.limit == 3
    prepare_references(_urls(1), seed_field=-1, custom_params="{}", cap=cap, where="GXS")
    prepare_references(_urls(3), seed_field=-1, custom_params="{}", cap=cap, where="GXS")
    with pytest.raises(ReferenceSlotError, match="3") as raised:
        prepare_references(_urls(4), seed_field=-1, custom_params="{}", cap=cap, where="GXS")
    assert "not dropped" in str(raised.value)


def test_wan_video_edit_cap_is_four():
    cfg = {
        "inputs": {
            str(i): {"image": {"from": f"upstream_image:annotated[{i}]"}}
            for i in range(4)
        },
        "api_json": {"7": {"class_type": "Wan2VideoEditApi"}},
    }
    cap = reference_image_cap(cfg)
    assert cap is not None and cap.limit == 4
    assert "Wan2VideoEditApi" in cap.reason
    with pytest.raises(ReferenceSlotError, match="4"):
        prepare_references(
            _urls(5), seed_field=-1, custom_params="{}", cap=cap, where="Wan",
        )


def test_minimax_reference_node_documents_a_cap_of_nine():
    cfg = {
        "inputs": {
            str(i): {"image": {"from": f"upstream_image:annotated[{i}]"}}
            for i in range(9)
        },
        "api_json": {"136": {"class_type": "MiniMaxH3ReferenceToVideo"}},
    }
    cap = reference_image_cap(cfg)
    assert cap is not None and cap.limit == 9
    images, _seed, _written = prepare_references(
        _urls(9), seed_field=0, custom_params="{}", cap=cap, where="H3",
    )
    assert len(images) == 9
    assert picture_slots(images)[-1]["tag"] == "<Picture 9>"


def test_minimax_first_last_frame_rejects_a_third_image():
    cfg = {
        "inputs": {
            "114": {"image": {"from": "upstream_image:annotated[0]"}},
            "200": {"image": {"from": "upstream_image:annotated[1]"}},
        },
        "api_json": {"104": {"class_type": "MiniMaxH3ImageToVideo"}},
    }
    cap = reference_image_cap(cfg)
    assert cap is not None and cap.limit == 2
    assert "MiniMaxH3ImageToVideo" in cap.reason
    prepare_references(_urls(2), seed_field=-1, custom_params="{}", cap=cap, where="FLF")
    with pytest.raises(ReferenceSlotError, match="2"):
        prepare_references(_urls(3), seed_field=-1, custom_params="{}", cap=cap, where="FLF")


def test_documented_meta_cap_overrides_a_wider_graph():
    cfg = {
        "meta": {"max_reference_images": 3},
        "inputs": {
            str(i): {"image": {"from": f"upstream_image:annotated[{i}]"}}
            for i in range(9)
        },
        "api_json": {"136": {"class_type": "MiniMaxH3ReferenceToVideo"}},
    }
    cap = reference_image_cap(cfg)
    assert cap is not None and cap.limit == 3
    assert "max_reference_images" in cap.reason


def test_zero_is_a_real_seed_and_bool_is_not():
    _images, seed, _written = prepare_references(
        [], seed_field=0, custom_params=_params({"key": "seed", "value": 9}),
        cap=None, where="VideoStage",
    )
    assert seed == 0
    _images, seed, _written = prepare_references(
        [], seed_field=True, custom_params=_params({"key": "seed", "value": 9}),
        cap=None, where="VideoStage",
    )
    assert seed == 9


def test_stage_maximum_constant_is_nine():
    assert MAX_REFERENCE_IMAGES == 9


def test_a_model_that_could_take_images_still_errors_when_none_are_bound():
    cfg = {
        "inputs": {},
        "api_json": {"104": {"class_type": "MiniMaxH3ImageToVideo"}},
    }
    cap = reference_image_cap(cfg)
    assert cap is not None and cap.limit == 0
    with pytest.raises(ReferenceSlotError, match="not dropped"):
        prepare_references(
            _urls(1), seed_field=-1, custom_params="{}", cap=cap, where="T2V",
        )
    prepare_references([], seed_field=-1, custom_params="{}", cap=cap, where="T2V")
