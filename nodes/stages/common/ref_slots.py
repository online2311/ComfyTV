"""Reference-image slots and first-class seeds for video stages.

Slots are 1-based and follow the order passed to the model. ``<Picture 1>``
is the first image and ``<Picture 9>`` is the ninth. ComfyUI v0.45.0
(commit 989d93d0) numbers MiniMax H3 references the same way:
``MiniMaxH3ReferenceToVideo`` walks ``ref_images`` in order and skips empty
slots, so a hole would renumber every later picture. Callers therefore pass
a dense list.

A stage accepts at most 9 images. A workflow may accept fewer. The cap is
``meta.max_reference_images`` when that is set, otherwise the tighter of the
known ComfyUI node cap and the number of bound ``upstream_image`` inputs.
Supplying more than the cap raises ``ReferenceSlotError``. Fewer images than
the cap are kept, so a workflow that still binds 3 slots keeps running.

Seed is a stage or shot field. ``custom_params`` items with key ``seed``,
and the older ``picture_1``..``picture_9`` (or ``reference_images``) items,
are still read. The value returned for ``custom_params`` is the new form:
those keys are removed, and the seed / image list are returned on their own.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from ....runners.workflow_db.config import get_workflow_for_invoke

_log = logging.getLogger(__name__)

MAX_REFERENCE_IMAGES = 9
UNSET_SEED = -1

# Caps taken from ComfyUI v0.45.0 node schemas (989d93d0), not from a
# workflow that happens to wire fewer LoadImage nodes.
VIDEO_NODE_IMAGE_CAPS: dict[str, int] = {
    "MiniMaxH3ReferenceToVideo": 9,
    "MiniMaxH3ImageToVideo": 2,
    "Wan2VideoEditApi": 4,
}

_AT_IMAGE_RE = re.compile(r"@image_(\d+)\b", re.IGNORECASE)
_PICTURE_RE = re.compile(r"<Picture\s+(\d+)\s*>", re.IGNORECASE)
_LEGACY_PICTURE_RE = re.compile(r"^picture_(\d+)$")
_SLOT_INDEX_RE = re.compile(r"(\d+)$")
_UPSTREAM_IMAGE_RE = re.compile(
    r"^upstream_image:(?:annotated|value|masked)(?:\[(\d+)\])?$"
)
_LIST_KEYS = ("reference_images", "pictures", "refs")


class ReferenceSlotError(RuntimeError):
    """More reference images were supplied than a stage or model accepts."""


@dataclass(frozen=True)
class ImageCap:
    limit: int
    reason: str


def picture_tag(n: int) -> str:
    return f"<Picture {n}>"


def picture_slots(images: list) -> list[dict]:
    return [
        {"n": i, "tag": picture_tag(i), "url": url}
        for i, url in enumerate(images, 1)
    ]


def ordered_reference_images(group: Any) -> list:
    """Non-empty images in slot order. Dict keys ending in a number sort by it."""
    if group is None:
        return []
    if isinstance(group, dict):
        keyed: list[tuple[int, Any]] = []
        rest: list[Any] = []
        for key, value in group.items():
            if value is None or value == "":
                continue
            match = _SLOT_INDEX_RE.search(str(key))
            if match:
                keyed.append((int(match.group(1)), value))
            else:
                rest.append(value)
        keyed.sort(key=lambda item: item[0])
        return [value for _index, value in keyed] + rest
    if isinstance(group, (list, tuple)):
        return [value for value in group if value not in (None, "")]
    return [group]


def _range_label(count: int) -> str:
    if count <= 0:
        return "no <Picture> slots"
    if count == 1:
        return "<Picture 1>"
    return f"<Picture 1>..<Picture {count}>"


def _cap_error(where: str, got: int, limit: int, reason: str) -> ReferenceSlotError:
    if limit <= 0:
        return ReferenceSlotError(
            f"{where} accepts no reference images ({reason}); "
            f"{got} were provided and were not dropped."
        )
    return ReferenceSlotError(
        f"{where} accepts at most {limit} reference images "
        f"({_range_label(limit)}; {reason}); "
        f"{got} were provided and were not dropped."
    )


def align_picture_prompt(prompt: str, image_count: int) -> str:
    """Rewrite ``@image_N`` to ``<Picture N>`` for slots that were passed.

    A tag past the images actually passed is an error, so the prompt cannot
    name a picture the model will never see.
    """
    def replace_at(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= image_count:
            return picture_tag(n)
        if n >= 1:
            raise ReferenceSlotError(
                f"<Picture {n}> does not match a passed reference image; "
                f"{image_count} image(s) were passed ({_range_label(image_count)})."
            )
        return match.group(0)

    text = _AT_IMAGE_RE.sub(replace_at, prompt or "")
    for match in _PICTURE_RE.finditer(text):
        n = int(match.group(1))
        if n < 1 or n > image_count:
            raise ReferenceSlotError(
                f"<Picture {n}> does not match a passed reference image; "
                f"{image_count} image(s) were passed ({_range_label(image_count)})."
            )
    return text


def _parse_items(custom_params: Any) -> list[dict] | None:
    if custom_params in (None, ""):
        return []
    try:
        data = json.loads(custom_params) if isinstance(custom_params, str) else custom_params
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    items = data.get("items")
    if items is None:
        return []
    if not isinstance(items, list):
        return None
    return [item for item in items if isinstance(item, dict)]


def _coerce_seed(raw: Any) -> int | None:
    if raw is None or isinstance(raw, bool) or raw == "":
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            return int(text, 10)
        except ValueError:
            return None
    return None


def _is_url(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("image_url", "url", "payload_url", "src"):
            found = _is_url(value.get(key))
            if found:
                return found
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    lower = text.lower()
    if (
        lower.startswith(("http://", "https://", "/view", "data:image/"))
        or "filename=" in lower
    ):
        return text
    return None


def _legacy_urls(raw: Any) -> list[str]:
    if isinstance(raw, str):
        one = _is_url(raw)
        return [one] if one else []
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for item in raw:
        url = _is_url(item)
        if url and url not in out:
            out.append(url)
    return out


def _legacy_pictures(items: list[dict]) -> list[str]:
    numbered: list[tuple[int, str]] = []
    listed: list[str] = []
    for item in items:
        key = str(item.get("key") or "")
        match = _LEGACY_PICTURE_RE.match(key)
        if match:
            url = _is_url(item.get("value"))
            if url:
                numbered.append((int(match.group(1)), url))
            continue
        if key in _LIST_KEYS:
            for url in _legacy_urls(item.get("value")):
                if url not in listed:
                    listed.append(url)
    numbered.sort(key=lambda item: item[0])
    out: list[str] = []
    for _index, url in numbered:
        if url not in out:
            out.append(url)
    for url in listed:
        if url not in out:
            out.append(url)
    return out


def _legacy_seed(items: list[dict]) -> int | None:
    for item in items:
        if item.get("key") == "seed":
            return _coerce_seed(item.get("value"))
    return None


def _is_legacy_key(key: str) -> bool:
    return key == "seed" or key in _LIST_KEYS or _LEGACY_PICTURE_RE.match(key) is not None


def _write_items(custom_params: Any, items: list[dict] | None) -> str:
    if items is None:
        if isinstance(custom_params, str):
            return custom_params
        return "{}"
    kept = [
        {"key": item.get("key"), "value": item.get("value")}
        for item in items
        if not _is_legacy_key(str(item.get("key") or ""))
    ]
    return json.dumps({"items": kept}, ensure_ascii=False)


def resolve_stage_seed(field: Any, items: list[dict] | None) -> int | None:
    """Widget value wins. ``-1`` (and a missing widget) means unset."""
    if not isinstance(field, bool):
        seed = _coerce_seed(field)
        if seed is not None and seed != UNSET_SEED:
            return seed
    if not items:
        return None
    return _legacy_seed(items)


def binding_span(config: dict) -> int | None:
    indices: list[int] = []
    for fields in (config.get("inputs") or {}).values():
        if not isinstance(fields, dict):
            continue
        for spec in fields.values():
            if not isinstance(spec, dict):
                continue
            match = _UPSTREAM_IMAGE_RE.match(str(spec.get("from") or ""))
            if match:
                indices.append(int(match.group(1) or 0))
    if not indices:
        return None
    return max(indices) + 1


def node_image_cap(api_json: Any) -> tuple[int, str] | None:
    if not isinstance(api_json, dict):
        return None
    found: list[tuple[int, str]] = []
    for node in api_json.values():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        cap = VIDEO_NODE_IMAGE_CAPS.get(class_type)
        if cap is not None:
            found.append((cap, class_type))
    if not found:
        return None
    limit, class_type = min(found, key=lambda item: item[0])
    return (
        limit,
        f"ComfyUI v0.45.0 {class_type} accepts {limit} reference images",
    )


def reference_image_cap(config: dict | None) -> ImageCap | None:
    """Cap for a workflow config. ``None`` means the workflow is unknown."""
    if not config:
        return None
    meta = config.get("meta") if isinstance(config.get("meta"), dict) else {}
    documented = meta.get("max_reference_images")
    if isinstance(documented, int) and not isinstance(documented, bool):
        limit = max(0, min(MAX_REFERENCE_IMAGES, documented))
        return ImageCap(limit, f"workflow meta.max_reference_images is {documented}")

    span = binding_span(config)
    node = node_image_cap(config.get("api_json"))
    if span is None:
        reason = "this workflow binds no reference-image inputs"
        if node is not None:
            reason = f"{node[1]}, but this workflow binds no reference-image inputs"
        return ImageCap(0, reason)
    if node is None or span < node[0]:
        return ImageCap(span, f"this workflow binds {span} reference-image inputs")
    return ImageCap(node[0], node[1])


def load_image_cap(kind: str, label: str) -> ImageCap | None:
    """Look up a saved workflow. Missing or unreadable workflows are uncapped
    beyond the stage maximum of 9.
    """
    if not label:
        return None
    try:
        cfg = get_workflow_for_invoke(kind, label)
    except Exception as exc:
        _log.warning(
            "[ComfyTV/ref-slots] could not read %s/%s image cap: %s",
            kind, label, exc,
        )
        return None
    return reference_image_cap(cfg)


def prepare_references(
    wired: Any,
    *,
    seed_field: Any,
    custom_params: Any,
    cap: ImageCap | int | None,
    where: str,
) -> tuple[list, int | None, str]:
    """Return ``(images, seed, custom_params)`` in the new form.

    Wired images win over legacy ``picture_N`` / ``reference_images`` items.
    When nothing is wired, those items are the image list. Either way the
    returned ``custom_params`` no longer carries seed or picture items.
    """
    items = _parse_items(custom_params)
    images = ordered_reference_images(wired)
    if not images and items:
        images = _legacy_pictures(items)

    limit = MAX_REFERENCE_IMAGES
    reason = f"a stage accepts at most {MAX_REFERENCE_IMAGES} reference images"
    if isinstance(cap, ImageCap):
        limit = min(limit, cap.limit)
        if cap.limit < MAX_REFERENCE_IMAGES:
            reason = cap.reason
    elif isinstance(cap, int) and not isinstance(cap, bool):
        limit = min(limit, cap)
        reason = f"cap {cap}"

    if len(images) > limit:
        raise _cap_error(where, len(images), limit, reason)

    seed = resolve_stage_seed(seed_field, items)
    return images, seed, _write_items(custom_params, items)
