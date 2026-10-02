"""Image discovery, safe fetches, and local image input boundaries."""

import asyncio
import io
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from exmekh_lib.core.web_navigation import image_discovery as images
from exmekh_lib.core.web_navigation.errors import WebNavigationError


def _image_bytes(format: str = "PNG") -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(stream, format=format)
    return stream.getvalue()


def test_local_query_image_validates_root_path_size_and_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "photo.png"
    image.write_bytes(_image_bytes())
    assert images.load_local_query_image(
        image_path="photo.png", workroot=tmp_path
    ).size == (2, 2)
    for path, root, kind in [
        ("photo.png", None, "invalid_image_path"),
        ("", tmp_path, "invalid_image_path"),
        ("missing.png", tmp_path, "invalid_image_path"),
    ]:
        with pytest.raises(WebNavigationError) as failure:
            images.load_local_query_image(image_path=path, workroot=root)
        assert failure.value.kind == kind

    monkeypatch.setattr(images, "MAX_LOCAL_IMAGE_BYTES", 1)
    with pytest.raises(WebNavigationError) as failure:
        images.load_local_query_image(image_path="photo.png", workroot=tmp_path)
    assert failure.value.kind == "invalid_image"


def test_local_query_image_rechecks_read_size_and_reports_io_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "photo.png"
    image.write_bytes(_image_bytes())
    monkeypatch.setattr(
        Path, "read_bytes", lambda path: b"x" * (images.MAX_LOCAL_IMAGE_BYTES + 1)
    )
    with pytest.raises(WebNavigationError, match="no larger than 10 MB"):
        images.load_local_query_image(image_path="photo.png", workroot=tmp_path)

    def unreadable(path: Path) -> bytes:
        raise OSError("file changed")

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    with pytest.raises(WebNavigationError, match="could not be read"):
        images.load_local_query_image(image_path="photo.png", workroot=tmp_path)

    monkeypatch.setattr(images, "MAX_LOCAL_IMAGE_BYTES", 10 * 1024 * 1024)
    (tmp_path / "plain.txt").write_text("not an image")
    with pytest.raises(WebNavigationError) as failure:
        images.load_local_query_image(image_path="plain.txt", workroot=tmp_path)
    assert failure.value.kind == "invalid_image"


def test_image_decoder_rejects_unsupported_or_oversized_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert images.decode_image(_image_bytes()).mode == "RGB"
    with pytest.raises(WebNavigationError, match="Only JPEG"):
        images.decode_image(_image_bytes("GIF"))
    with pytest.raises(WebNavigationError, match="valid supported image"):
        images.decode_image(b"not an image")
    monkeypatch.setattr(images, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(WebNavigationError, match="dimensions"):
        images.decode_image(_image_bytes())


def test_image_decoder_rejects_animated_supported_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Animated:
        format = "PNG"
        is_animated = True

        def __enter__(self) -> "Animated":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(images.Image, "open", lambda stream: Animated())
    with pytest.raises(WebNavigationError, match="Animated images"):
        images.decode_image(b"image")


def test_public_resolver_rejects_mixed_dns_and_dns_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def resolve(addresses: list[str]) -> list[dict[str, object]]:
        loop = asyncio.get_running_loop()

        async def fake_getaddrinfo(
            *args: object, **kwargs: object
        ) -> list[tuple[object, ...]]:
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))
                for address in addresses
            ]

        monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
        resolver = images.PublicResolver()
        result = await resolver.resolve("public.example", 443)
        await resolver.close()
        return result

    records = asyncio.run(resolve(["93.184.215.14"]))
    assert records[0]["host"] == "93.184.215.14"
    with pytest.raises(WebNavigationError, match="not publicly routable"):
        asyncio.run(resolve(["93.184.215.14", "127.0.0.1"]))
    with pytest.raises(WebNavigationError, match="not publicly routable"):
        asyncio.run(resolve([]))

    async def failing() -> None:
        loop = asyncio.get_running_loop()

        async def error(*args: object, **kwargs: object) -> None:
            raise OSError("unavailable")

        monkeypatch.setattr(loop, "getaddrinfo", error)
        await images.PublicResolver().resolve("missing.example", 443)

    with pytest.raises(WebNavigationError) as failure:
        asyncio.run(failing())
    assert failure.value.kind == "dns_error"


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        (b'<meta property="og:image" content="/og.png">', "https://example.org/og.png"),
        (
            b'<meta name="twitter:image" content="/social.png">',
            "https://example.org/social.png",
        ),
        (
            b'<script type="application/ld+json">{"image":{"url":"/json.png"}}</script>',
            "https://example.org/json.png",
        ),
        (
            b'<main><img src="/small.png" width="10" height="10"><img src="/large.png" width="30" height="30"></main>',
            "https://example.org/large.png",
        ),
    ],
)
def test_image_selection_uses_metadata_then_significant_page_images(
    html: bytes, expected: str
) -> None:
    assert (
        images.extract_significant_image_url(html, base_url="https://example.org/page")
        == expected
    )


def test_image_selection_ignores_invalid_metadata_and_unsafe_urls() -> None:
    html = b'<meta property="og:image" content="data:image/png,x"><main><img src="/ok.png"></main>'
    assert (
        images.extract_significant_image_url(html, base_url="https://example.org/")
        == "https://example.org/ok.png"
    )
    assert (
        images.extract_significant_image_url(42, base_url="https://example.org/")
        is None
    )  # type: ignore[arg-type]
    assert (
        images.extract_significant_image_url(
            b"<p>No images</p>", base_url="https://example.org/"
        )
        is None
    )
    assert (
        images._declared_image_area(
            SimpleNamespace(get=lambda name, default: "invalid")
        )
        == 0
    )


def test_image_selection_tries_later_json_ld_and_inline_images() -> None:
    document = (
        b'<script type="application/ld+json">{"image":["data:image/png,x","/valid.png"]}</script>'
        b'<main><img src="data:image/png,x"><img src="/fallback.png"></main>'
    )
    assert (
        images.extract_significant_image_url(document, base_url="https://example.org/")
        == "https://example.org/valid.png"
    )
    without_json = b'<main><img src="data:image/png,x"><img src="/fallback.png"></main>'
    assert (
        images.extract_significant_image_url(
            without_json, base_url="https://example.org/"
        )
        == "https://example.org/fallback.png"
    )
    multiple_scripts = (
        b'<script type="application/ld+json">{"image":["data:image/png,x"]}</script>'
        b'<script type="application/ld+json">{"image":"/second.png"}</script>'
    )
    assert (
        images.extract_significant_image_url(
            multiple_scripts, base_url="https://example.org/"
        )
        == "https://example.org/second.png"
    )


def test_json_ld_nested_image_forms_are_discovered() -> None:
    payload = {"@graph": [{"image": ["/a.png", 4]}, {"image": {"url": "/b.png"}}]}
    assert images._json_ld_images(payload) == ["/a.png", "/b.png"]
    assert images._json_ld_images({"image": "/single.png"}) == ["/single.png"]
    assert images._json_ld_images([{"image": "/list.png"}]) == ["/list.png"]
    assert images._json_ld_images([]) == []
    assert images._json_ld_images(42) == []
    assert (
        images.extract_significant_image_url(
            b'<script type="application/ld+json">{invalid</script>',
            base_url="https://example.org/",
        )
        is None
    )


class _Response:
    def __init__(
        self,
        status: int,
        *,
        location: str | None = None,
        mime: str = "image/png",
        body: bytes = b"image",
        declared_size: int | None = None,
        url: str = "https://example.org/image.png",
    ) -> None:
        self.status = status
        self.headers = {"Content-Type": mime}
        if location is not None:
            self.headers["Location"] = location
        self.content_length = declared_size
        self.url = url
        self.content = self
        self.body = body

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    async def iter_chunked(self, size: int) -> Any:
        yield self.body


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses

    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    def get(self, url: str, *, allow_redirects: bool) -> _Response:
        assert not allow_redirects
        return self.responses.pop(0)


def _fake_http(monkeypatch: pytest.MonkeyPatch, responses: list[_Response]) -> None:
    monkeypatch.setattr(images.aiohttp, "TCPConnector", lambda **kwargs: object())
    monkeypatch.setattr(
        images.aiohttp, "ClientSession", lambda **kwargs: _Session(responses)
    )


def test_remote_image_fetch_checks_redirect_mime_status_and_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_http(monkeypatch, [_Response(302, location="/image.png"), _Response(200)])
    body, url = asyncio.run(
        images._fetch_bytes(
            "https://example.org/start", allowed_mime_types={"image/png"}, max_bytes=10
        )
    )
    assert (body, url) == (b"image", "https://example.org/image.png")

    for response, message in [
        (_Response(404), "HTTP 404"),
        (_Response(200, mime="text/html"), "content type"),
        (_Response(200, declared_size=100), "size limit"),
        (_Response(200, body=b"too-long-content"), "size limit"),
        (_Response(302), "omitted its destination"),
    ]:
        _fake_http(monkeypatch, [response])
        with pytest.raises(WebNavigationError, match=message):
            asyncio.run(
                images._fetch_bytes(
                    "https://example.org/start",
                    allowed_mime_types={"image/png"},
                    max_bytes=10,
                )
            )

    _fake_http(monkeypatch, [_Response(302, location="/next")] * 4)
    with pytest.raises(WebNavigationError, match="redirect limit"):
        asyncio.run(
            images._fetch_bytes(
                "https://example.org/start",
                allowed_mime_types={"image/png"},
                max_bytes=10,
            )
        )


def test_discovery_obeys_robots_and_omits_individual_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def robots(url: str) -> bool:
        return not url.endswith("blocked")

    async def fetch(url: str, **kwargs: object) -> tuple[bytes, str]:
        if url.endswith("broken"):
            raise WebNavigationError("page failed", kind="image_discovery_failed")
        if url.endswith("image.png"):
            return _image_bytes(), url
        return b'<meta property="og:image" content="/image.png">', url

    monkeypatch.setattr(images, "_robots_allows", robots)
    monkeypatch.setattr(images, "_fetch_bytes", fetch)
    candidates = [
        {"url": "https://example.org/page"},
        {"url": "https://example.org/blocked"},
        {"url": "https://example.org/broken"},
    ]
    result = images.discover_result_images(candidates, [0, 1, 2], timeout_seconds=1)
    assert len(result) == 1
    assert result[0].candidate_index == 0
    assert result[0].url == "https://example.org/image.png"


def test_discovery_returns_empty_when_pages_have_no_image_or_budget_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def allowed(url: str) -> bool:
        return True

    async def no_image(url: str, **kwargs: object) -> tuple[bytes, str]:
        return b"<main><p>Text only</p></main>", url

    monkeypatch.setattr(images, "_robots_allows", allowed)
    monkeypatch.setattr(images, "_fetch_bytes", no_image)
    candidates = [{"url": "https://example.org/page"}]
    assert images.discover_result_images(candidates, [0], timeout_seconds=1) == []

    async def slow(url: str) -> bool:
        await asyncio.sleep(0.05)
        return True

    monkeypatch.setattr(images, "_robots_allows", slow)
    assert (
        asyncio.run(
            images._discover_result_images(candidates, [0], timeout_seconds=0.001)
        )
        == []
    )


def test_robots_failure_defaults_to_allow_and_disallow_is_respected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failure(*args: object, **kwargs: object) -> None:
        raise WebNavigationError("missing robots", kind="image_discovery_failed")

    monkeypatch.setattr(images, "_fetch_bytes", failure)
    assert asyncio.run(images._robots_allows("https://example.org/page"))

    async def denied(*args: object, **kwargs: object) -> tuple[bytes, str]:
        return b"User-agent: *\nDisallow: /page", "https://example.org/robots.txt"

    monkeypatch.setattr(images, "_fetch_bytes", denied)
    assert not asyncio.run(images._robots_allows("https://example.org/page"))
