import pytest


def test_bm25_prefers_relevant_title() -> None:
    pytest.importorskip("numpy")
    from exmekh_lib.core.web_navigation.ranking import bm25_scores  # noqa: PLC0415

    results = [
        {
            "title": "Other topic",
            "url": "https://example.org/other",
            "snippet": "unrelated",
        },
        {
            "title": "Python documentation",
            "url": "https://docs.python.org",
            "snippet": "Python reference",
        },
    ]
    scores = bm25_scores("Python documentation", results)
    assert scores[1] > scores[0]


def test_visual_ranking_uses_injected_embeddings() -> None:
    numpy = pytest.importorskip("numpy")
    from PIL import Image  # noqa: PLC0415

    from exmekh_lib.core.web_navigation.ranking import (  # noqa: PLC0415
        compute_siglip_ranking,
    )

    class FakeBackend:
        def encode_images(self, images: object) -> object:
            return numpy.asarray([[1.0, 0.0]], dtype=numpy.float32)

        def encode_texts(self, texts: object) -> object:
            return numpy.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=numpy.float32)

    results = [
        {"title": "first", "url": "https://example.org/first", "snippet": ""},
        {"title": "second", "url": "https://example.org/second", "snippet": ""},
    ]
    ranking, image_urls = compute_siglip_ranking(
        "topic",
        results,
        local_image=Image.new("RGB", (2, 2)),
        discovered_images=[],
        backend=FakeBackend(),
    )
    assert ranking == [1, 0]
    assert image_urls == {}


def test_injected_embedding_backends_do_not_share_cached_vectors() -> None:
    numpy = pytest.importorskip("numpy")
    from PIL import Image  # noqa: PLC0415

    from exmekh_lib.core.web_navigation.ranking import (  # noqa: PLC0415
        compute_siglip_ranking,
    )

    class Backend:
        def __init__(self, text_vectors: list[list[float]]) -> None:
            self.text_vectors = text_vectors

        def encode_images(self, images: object) -> object:
            return numpy.asarray([[1.0, 0.0]], dtype=numpy.float32)

        def encode_texts(self, texts: object) -> object:
            return numpy.asarray(self.text_vectors, dtype=numpy.float32)

    results = [
        {"title": "first", "url": "https://cache-test.example/first", "snippet": ""},
        {"title": "second", "url": "https://cache-test.example/second", "snippet": ""},
    ]
    image = Image.new("RGB", (3, 3), "magenta")
    first, _ = compute_siglip_ranking(
        "topic",
        results,
        local_image=image,
        discovered_images=[],
        backend=Backend([[0.0, 1.0], [1.0, 0.0]]),
    )
    second, _ = compute_siglip_ranking(
        "topic",
        results,
        local_image=image,
        discovered_images=[],
        backend=Backend([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert first == [1, 0]
    assert second == [0, 1]


def test_injected_semantic_backends_do_not_share_cached_vectors() -> None:
    numpy = pytest.importorskip("numpy")
    from exmekh_lib.core.web_navigation.ranking import (  # noqa: PLC0415
        compute_semantic_similarity,
    )

    class Backend:
        def __init__(self, document_vector: list[float]) -> None:
            self.document_vector = document_vector

        def encode(self, texts: object) -> object:
            return numpy.asarray(
                [[1.0, 0.0], self.document_vector], dtype=numpy.float32
            )

    results = [
        {"title": "page", "url": "https://semantic-cache.example/page", "snippet": ""}
    ]
    first = compute_semantic_similarity("topic", results, backend=Backend([1.0, 0.0]))
    second = compute_semantic_similarity("topic", results, backend=Backend([0.0, 1.0]))
    assert first == [1.0]
    assert second == [0.0]
