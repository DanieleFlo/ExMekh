"""Ranking and embedding adapters exercised without downloading model weights."""

import io
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from exmekh_lib.core.web_navigation import ranking
from exmekh_lib.core.web_navigation.image_discovery import DiscoveredImage

RESULTS = [
    {
        "title": "Python guide",
        "url": "https://docs.example.org/guide",
        "snippet": "language reference",
    },
    {
        "title": "Other guide",
        "url": "https://example.org/other",
        "snippet": "other subject",
    },
]


class _TextBackend:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.asarray(
            [[1.0, 0.0]] + [[1.0, 0.0]] * (len(texts) - 1), dtype=np.float32
        )


class _ImageBackend:
    def __init__(self) -> None:
        self.text_calls = 0
        self.image_calls = 0

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.text_calls += 1
        return np.asarray([[1.0, 0.0]] * len(texts), dtype=np.float32)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        self.image_calls += 1
        return np.asarray([[1.0, 0.0]] * len(images), dtype=np.float32)


def test_lru_cache_evicts_oldest_and_refreshes_read_entry() -> None:
    with pytest.raises(ValueError, match="positive"):
        ranking.EmbeddingLRUCache(0)
    cache = ranking.EmbeddingLRUCache(capacity=2)
    first = np.asarray([1.0], dtype=np.float32)
    assert cache.get("a") is None
    cache.put("a", first)
    cache.put("b", first)
    assert cache.get("a") is first
    cache.put("c", first)
    assert cache.get("b") is None
    cache.clear()
    assert cache.get("a") is None


def test_text_embedding_cache_batches_misses_then_reuses_documents() -> None:
    backend = _TextBackend()
    cache = ranking.EmbeddingLRUCache()
    first = ranking.compute_semantic_similarity(
        "python", RESULTS, backend=backend, cache=cache
    )
    second = ranking.compute_semantic_similarity(
        "python", RESULTS, backend=backend, cache=cache
    )
    assert first == second == [1.0, 1.0]
    assert [len(call) for call in backend.calls] == [3, 1]
    assert "Domain: docs.example.org" in ranking.build_embedding_text(RESULTS[0])
    assert ranking.compute_semantic_similarity("topic", [], backend=backend) == []


def test_invalid_embedding_batch_is_rejected() -> None:
    backend = SimpleNamespace(encode=lambda texts: np.asarray([1.0, 0.0]))
    with pytest.raises(ValueError, match="invalid batch shape"):
        ranking.compute_semantic_similarity("topic", RESULTS, backend=backend)


def test_image_embedding_cache_reuses_local_text_and_remote_vectors() -> None:
    backend = _ImageBackend()
    cache = ranking._ImageEmbeddingCache()
    image = Image.new("RGB", (2, 2), "blue")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    remote = [DiscoveredImage(0, "https://example.org/image.png", stream.getvalue())]
    for _ in range(2):
        assert cache.local_image(image, backend).shape == (2,)
        assert cache.candidate_texts(RESULTS, backend).shape == (2, 2)
        assert cache.remote_images(remote, backend).shape == (1, 2)
    assert backend.image_calls == 2
    assert backend.text_calls == 1


def test_image_similarity_and_three_way_siglip_ranking() -> None:
    backend = _ImageBackend()
    image = Image.new("RGB", (2, 2), "blue")
    assert ranking.compute_image_similarity("topic", RESULTS, backend=backend) is None
    assert (
        ranking.compute_image_similarity(
            "topic", [], local_image=image, backend=backend
        )
        is None
    )
    assert ranking.compute_image_similarity(
        "topic", RESULTS, local_image=image, backend=backend
    ) == [1.0, 1.0]
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    remote = [DiscoveredImage(1, "https://example.org/image.png", stream.getvalue())]
    ranked, urls = ranking.compute_siglip_ranking(
        "guide", RESULTS, local_image=image, discovered_images=remote, backend=backend
    )
    assert ranked[0] == 1
    assert urls == {1: "https://example.org/image.png"}


def test_bm25_phrase_and_rank_fusion_are_deterministic() -> None:
    assert ranking.bm25_scores("topic", []) == []
    scores = ranking.bm25_scores('"Python guide" reference', RESULTS)
    assert scores[0] > scores[1]
    with pytest.raises(ValueError, match="positive"):
        ranking.reciprocal_rank_fusion([([0], 1.0)], k=0)
    fused = ranking.reciprocal_rank_fusion([([0, 1], 1.0), ([1], 0.0)])
    assert fused[0] > fused[1]


def test_rank_results_uses_semantic_signal_and_recovers_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ranking, "compute_semantic_similarity", lambda *args: [0.0, 1.0]
    )
    assert len(ranking.rank_results("guide", RESULTS, image_ranking=[1, 0])) == 2

    def unavailable(*args: object) -> None:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(ranking, "compute_semantic_similarity", unavailable)
    assert len(ranking.rank_results("guide", RESULTS)) == 2


def test_backend_singletons_load_once(monkeypatch: pytest.MonkeyPatch) -> None:
    text_instances: list[object] = []
    image_instances: list[object] = []
    monkeypatch.setattr(ranking, "_BACKEND", None)
    monkeypatch.setattr(ranking, "_IMAGE_BACKEND", None)
    monkeypatch.setattr(
        ranking,
        "OpenVINOEmbeddingBackend",
        lambda: text_instances.append(object()) or text_instances[-1],
    )
    monkeypatch.setattr(
        ranking,
        "SigLIPEmbeddingBackend",
        lambda: image_instances.append(object()) or image_instances[-1],
    )
    assert ranking.get_embedding_backend() is ranking.get_embedding_backend()
    assert (
        ranking.get_image_embedding_backend() is ranking.get_image_embedding_backend()
    )
    assert len(text_instances) == len(image_instances) == 1


def test_backend_double_check_reuses_an_instance_loaded_by_another_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_backend = object()
    image_backend = object()

    class LoadOnLock:
        def __init__(self, attribute: str, value: object) -> None:
            self.attribute = attribute
            self.value = value

        def __enter__(self) -> None:
            monkeypatch.setattr(ranking, self.attribute, self.value)

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(ranking, "_BACKEND", None)
    monkeypatch.setattr(ranking, "_BACKEND_LOCK", LoadOnLock("_BACKEND", text_backend))
    monkeypatch.setattr(ranking, "_IMAGE_BACKEND", None)
    monkeypatch.setattr(
        ranking, "_IMAGE_BACKEND_LOCK", LoadOnLock("_IMAGE_BACKEND", image_backend)
    )
    assert ranking.get_embedding_backend() is text_backend
    assert ranking.get_image_embedding_backend() is image_backend


def test_default_semantic_backend_uses_process_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _TextBackend()
    monkeypatch.setattr(ranking, "_BACKEND", backend)
    monkeypatch.setattr(ranking, "_EMBEDDING_CACHE", ranking.EmbeddingLRUCache())
    assert ranking.compute_semantic_similarity("topic", RESULTS) == [1.0, 1.0]
    assert ranking.compute_semantic_similarity("topic", RESULTS) == [1.0, 1.0]
    assert [len(batch) for batch in backend.calls] == [3, 1]


def test_openvino_adapter_normalizes_cls_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_key = object()

    class CompiledModel:
        inputs = [SimpleNamespace(get_any_name=lambda: "input_ids")]

        def output(self, index: int) -> object:
            return output_key

        def __call__(self, inputs: object) -> dict[object, np.ndarray]:
            return {
                output_key: np.asarray([[[3.0, 4.0], [0.0, 0.0]]], dtype=np.float32)
            }

    class Core:
        def read_model(self, path: str) -> str:
            return path

        def compile_model(self, model: str, device: str) -> CompiledModel:
            assert device == "CPU"
            return CompiledModel()

    def tokenizer(texts: list[str], **kwargs: object) -> dict[str, np.ndarray]:
        return {"input_ids": np.asarray([[1]]), "ignored": np.asarray([[2]])}

    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=Core))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda **kwargs: "model.xml"),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda name: tokenizer)
        ),
    )
    backend = ranking.OpenVINOEmbeddingBackend()
    assert np.allclose(backend.encode(["text"]), [[0.6, 0.8]])


def test_siglip_adapter_uses_processor_and_feature_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tensor:
        def to(self, device: object) -> "Tensor":
            return self

        def detach(self) -> "Tensor":
            return self

        def float(self) -> "Tensor":
            return self

        def cpu(self) -> "Tensor":
            return self

        def numpy(self) -> np.ndarray:
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    class Inputs(dict[str, object]):
        def to(self, device: object) -> "Inputs":
            return self

    class Model:
        def to(self, device: object) -> None:
            pass

        def eval(self) -> None:
            pass

        def get_text_features(self, **kwargs: object) -> object:
            return SimpleNamespace(pooler_output=Tensor())

        def get_image_features(self, **kwargs: object) -> Tensor:
            return Tensor()

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        device=lambda name: name,
        inference_mode=nullcontext,
        nn=SimpleNamespace(
            functional=SimpleNamespace(normalize=lambda value, dim: value)
        ),
    )
    processor = SimpleNamespace(from_pretrained=lambda name: lambda **kwargs: Inputs())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Model()),
            AutoProcessor=processor,
        ),
    )
    backend = ranking.SigLIPEmbeddingBackend()
    assert backend.encode_texts(["topic"]).tolist() == [[1.0, 0.0]]
    assert backend.encode_images([Image.new("RGB", (1, 1))]).tolist() == [[1.0, 0.0]]
