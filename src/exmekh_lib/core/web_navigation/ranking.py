"""Extensible hybrid ranking for public-web search candidates."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from typing import Protocol
from urllib.parse import urlparse

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from exmekh_lib.config.web import WEB_RANKING
from exmekh_lib.config.web_text import WEB_TEXT
from exmekh_lib.core.models.native_suite import EmbeddingItem, native_suite_client

from .image_discovery import DiscoveredImage, decode_image

EMBEDDING_CACHE_SIZE = WEB_RANKING.embedding_cache_size
RRF_K = WEB_RANKING.rrf_k
BM25_RRF_WEIGHT = WEB_RANKING.bm25_rrf_weight
SEMANTIC_RRF_WEIGHT = WEB_RANKING.semantic_rrf_weight
PROVIDER_RRF_WEIGHT = WEB_RANKING.provider_rrf_weight
IMAGE_RRF_WEIGHT = WEB_RANKING.image_rrf_weight

Embedding = NDArray[np.float32]
_LOGGER = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    """Encode text batches for semantic ranking."""

    def encode(self, texts: Sequence[str]) -> Embedding:
        """Return one normalized embedding for every supplied text."""
        ...


class ImageEmbeddingBackend(Protocol):
    """Encode text and images into one normalized multimodal embedding space."""

    def encode_texts(self, texts: Sequence[str]) -> Embedding:
        """Return one normalized embedding for every supplied text."""
        ...

    def encode_images(self, images: Sequence[Image.Image]) -> Embedding:
        """Return one normalized embedding for every supplied image."""
        ...


class NativeEmbeddingBackend:
    """Text and image vectors come from one versioned native model space."""

    def encode(self, texts: Sequence[str]) -> Embedding:
        return self.encode_texts(texts)

    def encode_texts(self, texts: Sequence[str]) -> Embedding:
        with native_suite_client() as suite:
            result = suite.embeddings(texts, task="similarity")
        return np.asarray(
            [row["embedding"] for row in result["data"]], dtype=np.float32
        )

    def encode_images(self, images: Sequence[Image.Image]) -> Embedding:
        with native_suite_client() as suite:
            items = [
                EmbeddingItem(content=[suite.upload_image(image)]) for image in images
            ]
            result = suite.embeddings(items, task="similarity")
        return np.asarray(
            [row["embedding"] for row in result["data"]], dtype=np.float32
        )


class EmbeddingLRUCache:
    """Store bounded document embeddings by canonical URL."""

    def __init__(self, capacity: int = EMBEDDING_CACHE_SIZE) -> None:
        """Create a cache with a positive maximum entry count."""
        if capacity < 1:
            raise ValueError(WEB_TEXT.embedding_cache_capacity_must_be_positive)
        self._capacity = capacity
        self._entries: OrderedDict[str, Embedding] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, url: str) -> Embedding | None:
        """Return and refresh one cached embedding when present."""
        with self._lock:
            embedding = self._entries.get(url)
            if embedding is not None:
                self._entries.move_to_end(url)
            return embedding

    def put(self, url: str, embedding: Embedding) -> None:
        """Store one embedding and evict the least recently used entry."""
        with self._lock:
            self._entries[url] = embedding
            self._entries.move_to_end(url)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Remove every cached embedding."""
        with self._lock:
            self._entries.clear()


class _ImageEmbeddingCache:
    """Cache vectors for one image embedding space."""

    def __init__(self) -> None:
        self._local_images = EmbeddingLRUCache()
        self._candidate_texts = EmbeddingLRUCache()
        self._remote_images = EmbeddingLRUCache()

    def local_image(
        self, image: Image.Image, backend: ImageEmbeddingBackend
    ) -> Embedding:
        digest = hashlib.sha256(
            f"{image.mode}:{image.width}x{image.height}:".encode("ascii")
            + image.tobytes()
        ).hexdigest()
        cached = self._local_images.get(digest)
        if cached is not None:
            return cached
        embedding = backend.encode_images([image])[0]
        self._local_images.put(digest, embedding)
        return embedding

    def candidate_texts(
        self, results: Sequence[Mapping[str, object]], backend: ImageEmbeddingBackend
    ) -> Embedding:
        embeddings: list[Embedding | None] = []
        missing: list[int] = []
        for index, result in enumerate(results):
            cached = self._candidate_texts.get(str(result["url"]))
            embeddings.append(cached)
            if cached is None:
                missing.append(index)
        if missing:
            encoded = backend.encode_texts(
                [build_embedding_text(results[index]) for index in missing]
            )
            for offset, index in enumerate(missing):
                embeddings[index] = encoded[offset]
                self._candidate_texts.put(str(results[index]["url"]), encoded[offset])
        return np.asarray(embeddings, dtype=np.float32)

    def remote_images(
        self, images: Sequence[DiscoveredImage], backend: ImageEmbeddingBackend
    ) -> Embedding:
        embeddings: list[Embedding | None] = []
        missing: list[int] = []
        for index, item in enumerate(images):
            cached = self._remote_images.get(item.url)
            embeddings.append(cached)
            if cached is None:
                missing.append(index)
        if missing:
            encoded = backend.encode_images(
                [decode_image(images[index].content) for index in missing]
            )
            for offset, index in enumerate(missing):
                embeddings[index] = encoded[offset]
                self._remote_images.put(images[index].url, encoded[offset])
        return np.asarray(embeddings, dtype=np.float32)


_NATIVE_BACKEND = NativeEmbeddingBackend()
_SEMANTIC_INFERENCE_LOCK = threading.Lock()
_IMAGE_INFERENCE_LOCK = threading.Lock()


def get_embedding_backend() -> EmbeddingBackend:
    """Return the native HTTP backend; the server owns model residency."""
    return _NATIVE_BACKEND


def get_image_embedding_backend() -> ImageEmbeddingBackend:
    """Return the native image/text backend."""
    return _NATIVE_BACKEND


def build_embedding_text(result: Mapping[str, object]) -> str:
    """Build a natural field-labelled representation of one search result."""
    hostname = urlparse(str(result.get("url", ""))).hostname or ""
    domain = hostname.removeprefix("www.")
    return WEB_TEXT.title_domain_snippet.format(
        result_get_strip=str(result.get("title", "")).strip(),
        domain=domain,
        result_get_strip_2=str(result.get("snippet", "")).strip(),
    )


def bm25_scores(query: str, results: Sequence[Mapping[str, object]]) -> list[int]:
    """Score candidates with corpus-derived BM25 weights and no language rules."""
    if not results:
        return []
    query_terms = Counter(_tokenize(query))
    documents = [_weighted_document_tokens(result) for result in results]
    document_sets = [set(tokens) for tokens in documents]
    average_length = sum(len(tokens) for tokens in documents) / len(documents)
    phrases = [phrase.strip().lower() for phrase in re.findall(r'"([^"]+)"', query)]
    scores: list[int] = []
    for result, tokens in zip(results, documents):
        frequencies = Counter(tokens)
        length = max(1, len(tokens))
        score = 0.0
        for term, query_frequency in query_terms.items():
            frequency = frequencies[term]
            if frequency == 0:
                continue
            document_frequency = sum(term in terms for terms in document_sets)
            inverse_frequency = math.log(
                1
                + (len(documents) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            normalization = frequency + WEB_RANKING.bm25_k1 * (
                1
                - WEB_RANKING.bm25_b
                + WEB_RANKING.bm25_b * length / max(1.0, average_length)
            )
            score += (
                inverse_frequency
                * frequency
                * (WEB_RANKING.bm25_k1 + 1)
                / normalization
                * min(query_frequency, WEB_RANKING.max_query_term_frequency)
            )
        searchable = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
        score += sum(
            WEB_RANKING.phrase_bonus
            for phrase in phrases
            if phrase and phrase in searchable
        )
        scores.append(round(score * WEB_RANKING.score_scale))
    return scores


def compute_semantic_similarity(
    query: str,
    results: Sequence[Mapping[str, object]],
    *,
    backend: EmbeddingBackend | None = None,
    cache: EmbeddingLRUCache | None = None,
) -> list[float]:
    """Compute cosine similarities while batching the query and cache misses."""
    if not results:
        return []
    # A process cache can mix endpoints or stale page contents. Reuse is explicit.
    resolved_cache = cache if cache is not None else EmbeddingLRUCache()
    with _SEMANTIC_INFERENCE_LOCK:
        cached: dict[int, Embedding] = {}
        missing_indices: list[int] = []
        for index, result in enumerate(results):
            embedding = resolved_cache.get(str(result["url"]))
            if embedding is None:
                missing_indices.append(index)
            else:
                cached[index] = embedding

        texts = [query]
        texts.extend(build_embedding_text(results[index]) for index in missing_indices)
        vectors = (backend or get_embedding_backend()).encode(texts)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError(WEB_TEXT.embedding_backend_returned_an_invalid_batch_shape)
        query_embedding = vectors[0]
        for offset, result_index in enumerate(missing_indices, start=1):
            document_embedding = vectors[offset]
            cached[result_index] = document_embedding
            resolved_cache.put(str(results[result_index]["url"]), document_embedding)

        return [
            float(np.dot(query_embedding, cached[index]))
            for index in range(len(results))
        ]


def compute_image_similarity(
    query: str,
    results: Sequence[Mapping[str, object]],
    *,
    local_image: Image.Image | None = None,
    backend: ImageEmbeddingBackend | None = None,
) -> list[float] | None:
    """Compare a local query image with every candidate's natural result text."""
    del query
    if local_image is None or not results:
        return None
    resolved_backend = backend or get_image_embedding_backend()
    embedding_cache = _ImageEmbeddingCache()
    with _IMAGE_INFERENCE_LOCK:
        image_embedding = embedding_cache.local_image(local_image, resolved_backend)
        text_embeddings = embedding_cache.candidate_texts(results, resolved_backend)
        return [float(np.dot(image_embedding, item)) for item in text_embeddings]


def compute_multimodal_ranking(
    query: str,
    results: Sequence[Mapping[str, object]],
    *,
    local_image: Image.Image,
    discovered_images: Sequence[DiscoveredImage],
    backend: ImageEmbeddingBackend | None = None,
) -> tuple[list[int], dict[int, str]]:
    """Fuse image-to-text, text-to-image, and image-to-image native rankings."""
    resolved_backend = backend or get_image_embedding_backend()
    embedding_cache = _ImageEmbeddingCache()
    with _IMAGE_INFERENCE_LOCK:
        local_embedding = embedding_cache.local_image(local_image, resolved_backend)
        candidate_texts = embedding_cache.candidate_texts(results, resolved_backend)
        image_to_text = [
            float(np.dot(local_embedding, embedding)) for embedding in candidate_texts
        ]
        rankings: list[tuple[Sequence[int], float]] = [
            (_ranking_from_scores(image_to_text), 1.0)
        ]
        image_urls: dict[int, str] = {}
        if discovered_images:
            remote_embeddings = embedding_cache.remote_images(
                discovered_images, resolved_backend
            )
            query_embedding = resolved_backend.encode_texts([query])[0]
            text_to_image = [
                (item.candidate_index, float(np.dot(query_embedding, embedding)))
                for item, embedding in zip(discovered_images, remote_embeddings)
            ]
            image_to_image = [
                (item.candidate_index, float(np.dot(local_embedding, embedding)))
                for item, embedding in zip(discovered_images, remote_embeddings)
            ]
            rankings.extend(
                [
                    (_ranking_from_indexed_scores(text_to_image), 1.0),
                    (_ranking_from_indexed_scores(image_to_image), 1.0),
                ]
            )
            image_urls = {item.candidate_index: item.url for item in discovered_images}
        fused = reciprocal_rank_fusion(rankings)
        ranking = sorted(fused, key=lambda index: (-fused[index], index))
        return ranking, image_urls


def reciprocal_rank_fusion(
    rankings: Sequence[tuple[Sequence[int], float]], *, k: int = RRF_K
) -> dict[int, float]:
    """Fuse ranked candidate indices using weighted reciprocal rank fusion."""
    if k < 1:
        raise ValueError(WEB_TEXT.rrf_k_must_be_positive)
    scores: dict[int, float] = {}
    for ranking, weight in rankings:
        if weight <= 0:
            continue
        for rank, candidate_index in enumerate(ranking, start=1):
            scores[candidate_index] = scores.get(candidate_index, 0.0) + weight / (
                k + rank
            )
    return scores


def rank_results(
    query: str,
    results: Sequence[Mapping[str, object]],
    *,
    image_ranking: Sequence[int] | None = None,
) -> list[tuple[int, float]]:
    """Rank candidates by BM25, native similarity, images and provider order."""
    bm25_ranking = _ranking_from_scores(bm25_scores(query, results))
    rankings: list[tuple[Sequence[int], float]] = [
        (bm25_ranking, BM25_RRF_WEIGHT),
        (list(range(len(results))), PROVIDER_RRF_WEIGHT),
    ]
    try:
        semantic_scores = compute_semantic_similarity(query, results)
    except Exception as error:  # noqa: BLE001 - semantic ranking is an optional signal.
        _LOGGER.warning(WEB_TEXT.native_semantic_ranking_unavailable, error)
    else:
        rankings.append((_ranking_from_scores(semantic_scores), SEMANTIC_RRF_WEIGHT))
    if image_ranking is not None:
        rankings.append((image_ranking, IMAGE_RRF_WEIGHT))

    fused_scores = reciprocal_rank_fusion(rankings)
    return sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))


def _ranking_from_scores(scores: Sequence[int | float]) -> list[int]:
    """Convert scores to a stable descending ranking of candidate indices."""
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def _ranking_from_indexed_scores(scores: Sequence[tuple[int, float]]) -> list[int]:
    """Convert candidate-indexed scores into a stable partial ranking."""
    return [index for index, _ in sorted(scores, key=lambda item: (-item[1], item[0]))]


def _tokenize(value: object) -> list[str]:
    """Tokenize Unicode words without assumptions about language or subject."""
    return [
        token
        for token in re.findall(r"[^\W_]+", str(value).lower(), flags=re.UNICODE)
        if len(token) > 1
    ]


def _weighted_document_tokens(result: Mapping[str, object]) -> list[str]:
    """Build one field-weighted token stream for deterministic BM25 ranking."""
    title = _tokenize(result.get("title", ""))
    url = _tokenize(result.get("url", ""))
    snippet = _tokenize(result.get("snippet", ""))
    return title * WEB_RANKING.title_weight + url * WEB_RANKING.url_weight + snippet
