"""Replaceable similarity backends for shot consistency scoring.

Scoring uses a ComfyUI node that is already loaded (CLIP vision, or a face
node that has a registered adapter). Nothing is installed into the venv.
"""

from __future__ import annotations

from typing import Callable

import folder_paths
import nodes

CLIP_LOADER = "CLIPVisionLoader"
CLIP_ENCODE = "CLIPVisionEncode"

FACE_NODE_NAMES = (
    "InsightFaceLoader",
    "IPAdapterInsightFaceLoader",
    "InstantIDFaceAnalysis",
    "PulidInsightFaceLoader",
    "ReActorFaceSwap",
)

_FACTORIES: dict[str, Callable[[], "SimilarityBackend"]] = {}


class ConsistencyBackendUnavailable(RuntimeError):
    """CLIP vision and face nodes cannot score this take."""


class SimilarityBackend:
    """Object with ``name`` and ``score_pair(frame_path, reference_path) -> float``."""

    name = "similarity"

    def score_pair(self, frame_path: str, reference_path: str) -> float:
        raise ConsistencyBackendUnavailable(
            f"{self.name} has no score_pair implementation"
        )


def register_similarity_backend(name: str, factory: Callable[[], SimilarityBackend]) -> None:
    _FACTORIES[name] = factory


def unregister_similarity_backend(name: str) -> None:
    _FACTORIES.pop(name, None)


def node_mappings() -> dict:
    return getattr(nodes, "NODE_CLASS_MAPPINGS", None) or {}


def clip_vision_models() -> list[str]:
    try:
        names = folder_paths.get_filename_list("clip_vision") or []
    except Exception:
        return []
    return [str(name) for name in names if name]


def present_face_nodes() -> list[str]:
    mappings = node_mappings()
    return [name for name in FACE_NODE_NAMES if name in mappings]


def clip_vision_available() -> tuple[bool, str]:
    mappings = node_mappings()
    missing = [name for name in (CLIP_LOADER, CLIP_ENCODE) if name not in mappings]
    if missing:
        return False, "missing " + ", ".join(missing)
    models = clip_vision_models()
    if not models:
        return False, "clip_vision folder has no model"
    return True, models[0]


def _unavailable(reason: str) -> ConsistencyBackendUnavailable:
    faces = ", ".join(FACE_NODE_NAMES)
    present = present_face_nodes()
    return ConsistencyBackendUnavailable(
        "No similarity backend is available. "
        f"CLIPVisionLoader / CLIPVisionEncode: {reason}. "
        f"Face nodes looked up: {faces}. "
        f"Present: {', '.join(present) if present else 'none'}. "
        "Refusing to score. Do not pip install packages into the ComfyUI venv."
    )


def build_backend(name: str) -> SimilarityBackend:
    choice = (name or "auto").strip() or "auto"
    if choice in _FACTORIES:
        return _FACTORIES[choice]()
    if choice == "clip_vision":
        ok, detail = clip_vision_available()
        if not ok:
            raise _unavailable(detail)
        return ClipVisionBackend(detail)
    if choice == "face":
        present = present_face_nodes()
        if not present:
            raise _unavailable("face backend was requested and no face node is loaded")
        return FaceNodeBackend(present[0])
    if choice == "auto":
        ok, detail = clip_vision_available()
        if ok:
            return ClipVisionBackend(detail)
        present = present_face_nodes()
        if present:
            return FaceNodeBackend(present[0])
        raise _unavailable("neither backend is loaded")
    raise ConsistencyBackendUnavailable(f"unknown similarity backend {choice!r}")


class FaceNodeBackend(SimilarityBackend):
    name = "face"

    def __init__(self, node_name: str):
        self.node_name = node_name

    def score_pair(self, frame_path: str, reference_path: str) -> float:
        raise ConsistencyBackendUnavailable(
            f"Face node {self.node_name} is loaded but this build has no scoring "
            "adapter for it. Use clip_vision or pass a SimilarityBackend. "
            "Do not pip install packages into the ComfyUI venv."
        )


class ClipVisionBackend(SimilarityBackend):
    name = "clip_vision"

    def __init__(self, model_name: str, mappings: dict | None = None):
        self.model_name = model_name
        self._mappings = mappings if mappings is not None else node_mappings()
        self._clip = None

    def score_pair(self, frame_path: str, reference_path: str) -> float:
        left = self.embed(frame_path)
        right = self.embed(reference_path)
        return cosine(left, right)

    def embed(self, path: str) -> list[float]:
        image = load_image_value(path)
        encoded = _call_encode(self._encoder(), self._vision(), image)
        return _embeds(encoded)

    def _vision(self):
        if self._clip is None:
            loader_cls = self._mappings[CLIP_LOADER]
            loader = loader_cls()
            loaded = loader.load_clip(self.model_name)
            self._clip = loaded[0]
        return self._clip

    def _encoder(self):
        return self._mappings[CLIP_ENCODE]()


def load_image_value(path: str):
    try:
        import torch
    except ImportError as exc:
        raise ConsistencyBackendUnavailable(
            "torch is not importable, so CLIP vision embeddings cannot be built. "
            "Do not pip install packages into the ComfyUI venv."
        ) from exc
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    pixels = list(image.getdata())
    tensor = torch.tensor(pixels, dtype=torch.float32).view(1, height, width, 3)
    return tensor / 255.0


def _call_encode(encoder, clip, image):
    if hasattr(encoder, "encode"):
        return encoder.encode(clip, image)
    return encoder.execute(clip_vision=clip, image=image)


def _embeds(result) -> list[float]:
    output = result[0] if isinstance(result, (list, tuple)) else result
    embeds = getattr(output, "image_embeds", None)
    if embeds is None and isinstance(output, dict):
        embeds = output.get("image_embeds")
    if embeds is None:
        embeds = getattr(output, "penultimate_hidden_states", None)
    if embeds is None:
        raise ConsistencyBackendUnavailable(
            "CLIP vision encode returned no image embeds"
        )
    return _flat(embeds)


def _flat(value) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "reshape"):
        try:
            value = value.reshape(-1)
        except TypeError:
            pass
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    flat: list[float] = []

    def walk(item) -> None:
        if isinstance(item, (list, tuple)):
            for part in item:
                walk(part)
            return
        flat.append(float(item))

    walk(value)
    if not flat:
        raise ConsistencyBackendUnavailable("CLIP vision embedding was empty")
    return flat


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ConsistencyBackendUnavailable(
            "CLIP vision embeddings differ in length"
        )
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / ((left_norm ** 0.5) * (right_norm ** 0.5))
