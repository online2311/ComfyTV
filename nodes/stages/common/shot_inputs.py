"""Optional per-shot reference images and seeds for ShotImagesStage.

A shot that carries none of these falls back to the stage-level shared
reference list and the stage seed field. A seed still stored in
custom_params is read when the stage seed is unset (-1). Existing
storyboards that only have prompt / image_prompt keep that behavior.
At most 9 reference images; a longer list is an error.
"""

from __future__ import annotations

from typing import Any

from ....storage.series_library import image_urls_for_shot
from .ref_slots import MAX_REFERENCE_IMAGES, ReferenceSlotError

MAX_SHOT_REFS = MAX_REFERENCE_IMAGES


def image_ref_url(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("image_url", "url", "payload_url", "src"):
            found = image_ref_url(value.get(key))
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


def image_ref_urls(raw: Any) -> list[str]:
    if isinstance(raw, str):
        one = image_ref_url(raw)
        return [one] if one else []
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for item in raw:
        url = image_ref_url(item)
        if url and url not in out:
            out.append(url)
    return out


def _bounded_refs(urls: list[str], where: str) -> list[str]:
    if len(urls) > MAX_SHOT_REFS:
        raise ReferenceSlotError(
            f"{where} accepts at most {MAX_SHOT_REFS} reference images "
            f"(<Picture 1>..<Picture {MAX_SHOT_REFS}>); "
            f"{len(urls)} were provided and were not dropped."
        )
    return urls


def shot_reference_images(shot: Any, shared: list | None) -> list:
    """Per-shot refs when the shot has any image URL, otherwise the shared list.

    When the shot has no image URL, asset codes such as ``CHR_xiaolongnv@v1#front``
    (or character / location / prop ids in the shot) resolve through the bound
    series library.yaml. The shot dict itself is left unchanged.
    """
    if isinstance(shot, dict):
        for key in ("reference_images", "refs"):
            urls = image_ref_urls(shot.get(key))
            if urls:
                return _bounded_refs(urls, "a shot")
        for key in ("image_url", "reference_img"):
            url = image_ref_url(shot.get(key))
            if url:
                return [url]
        resolved = image_urls_for_shot(shot)
        if resolved:
            return _bounded_refs(resolved, "a shot")
    return _bounded_refs(list(shared or []), "ShotImagesStage")


def shot_seed(shot: Any) -> int | None:
    """The shot's own seed, or None when the stage seed should be used."""
    if not isinstance(shot, dict) or "seed" not in shot:
        return None
    raw = shot.get("seed")
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
