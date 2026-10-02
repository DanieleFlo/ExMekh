"""Safe local-image validation and remote result-image discovery."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import aiohttp
from aiohttp.abc import AbstractResolver, ResolveResult
from lxml import html
from PIL import Image, UnidentifiedImageError

from exmekh_lib.config.web import WEB_FORMAT, WEB_NAVIGATION

from .errors import WebNavigationError

MAX_LOCAL_IMAGE_BYTES = WEB_NAVIGATION.max_local_image_bytes
MAX_REMOTE_IMAGE_BYTES = WEB_NAVIGATION.max_remote_image_bytes
MAX_HTML_BYTES = WEB_NAVIGATION.max_html_bytes
MAX_IMAGE_PIXELS = WEB_NAVIGATION.max_image_pixels
MAX_REDIRECTS = WEB_NAVIGATION.max_redirects
IMAGE_DISCOVERY_CONCURRENCY = WEB_NAVIGATION.image_discovery_concurrency
IMAGE_CANDIDATE_LIMIT = WEB_NAVIGATION.image_candidate_limit
ALLOWED_IMAGE_FORMATS = WEB_FORMAT.allowed_image_formats
ALLOWED_IMAGE_MIME_TYPES = WEB_FORMAT.allowed_image_mime_types
HTML_MIME_TYPES = WEB_FORMAT.html_mime_types
USER_AGENT = WEB_FORMAT.user_agent


@dataclass(frozen=True, slots=True)
class DiscoveredImage:
    """Hold one validated image selected for a search candidate."""

    candidate_index: int
    url: str
    content: bytes


class PublicResolver(AbstractResolver):
    """Resolve hosts only when every returned address is globally routable."""

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_UNSPEC
    ) -> list[ResolveResult]:
        """Resolve one host and reject mixed public/private DNS responses."""
        loop = asyncio.get_running_loop()
        try:
            records = await loop.getaddrinfo(
                host, port, family=family, type=socket.SOCK_STREAM
            )
        except OSError as error:
            raise WebNavigationError(
                f"Could not resolve URL hostname: {host}.", kind="dns_error"
            ) from error
        addresses = {ipaddress.ip_address(record[4][0]) for record in records}
        if not addresses or any(not address.is_global for address in addresses):
            raise WebNavigationError(
                f"URL hostname is not publicly routable: {host}.",
                kind="blocked_url",
            )
        return [
            {
                "hostname": host,
                "host": str(record[4][0]),
                "port": port,
                "family": record[0],
                "proto": record[2],
                "flags": socket.AI_NUMERICHOST,
            }
            for record in records
        ]

    async def close(self) -> None:
        """Release no resources because resolution is delegated to the event loop."""


def load_local_query_image(*, image_path: str, workroot: Path | None) -> Image.Image:
    """Resolve and validate one local query image strictly inside workroot."""
    if workroot is None:
        raise WebNavigationError(
            "image_path requires an injected workroot.", kind="invalid_image_path"
        )
    root = Path(workroot).resolve()
    raw_path = Path(str(image_path).strip())
    if not str(image_path).strip():
        raise WebNavigationError(
            "image_path must be a non-empty path.", kind="invalid_image_path"
        )
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise WebNavigationError(
            "image_path must resolve to a file inside workroot.",
            kind="invalid_image_path",
        ) from error
    if not resolved.is_file() or resolved.stat().st_size > MAX_LOCAL_IMAGE_BYTES:
        raise WebNavigationError(
            f"image_path must be an image file no larger than {MAX_LOCAL_IMAGE_BYTES / (1024 * 1024):g} MB.",
            kind="invalid_image",
        )
    try:
        content = resolved.read_bytes()
        if len(content) > MAX_LOCAL_IMAGE_BYTES:
            raise WebNavigationError(
                f"image_path must be an image file no larger than {MAX_LOCAL_IMAGE_BYTES / (1024 * 1024):g} MB.",
                kind="invalid_image",
            )
        return decode_image(content)
    except OSError as error:
        raise WebNavigationError(
            "The local image could not be read.", kind="invalid_image"
        ) from error


def decode_image(content: bytes) -> Image.Image:
    """Decode one bounded, non-animated JPEG, PNG, or WebP image as RGB."""
    try:
        with Image.open(BytesIO(content)) as image:
            if image.format not in ALLOWED_IMAGE_FORMATS:
                raise WebNavigationError(
                    "Only JPEG, PNG, and WebP images are supported.",
                    kind="invalid_image",
                )
            if bool(getattr(image, "is_animated", False)):
                raise WebNavigationError(
                    "Animated images are not supported.", kind="invalid_image"
                )
            width, height = image.size
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                raise WebNavigationError(
                    f"Image dimensions exceed the {MAX_IMAGE_PIXELS / 1_000_000:g} megapixel limit.",
                    kind="invalid_image",
                )
            image.verify()
        with Image.open(BytesIO(content)) as image:
            return image.convert("RGB")
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise WebNavigationError(
            "The file does not contain a valid supported image.",
            kind="invalid_image",
        ) from error


def discover_result_images(
    candidates: Sequence[Mapping[str, object]],
    candidate_indices: Sequence[int],
    *,
    timeout_seconds: int,
) -> list[DiscoveredImage]:
    """Discover one safe image for each selected result within one time budget."""
    return asyncio.run(
        _discover_result_images(
            candidates,
            candidate_indices,
            timeout_seconds=timeout_seconds,
        )
    )


async def _discover_result_images(
    candidates: Sequence[Mapping[str, object]],
    candidate_indices: Sequence[int],
    *,
    timeout_seconds: int,
) -> list[DiscoveredImage]:
    """Run bounded concurrent static image discovery for selected candidates."""
    semaphore = asyncio.Semaphore(IMAGE_DISCOVERY_CONCURRENCY)

    async def discover(index: int) -> DiscoveredImage | None:
        """Return one validated image or omit a candidate-specific failure."""
        async with semaphore:
            try:
                page_url = str(candidates[index]["url"])
                if not await _robots_allows(page_url):
                    return None
                page_content, final_url = await _fetch_bytes(
                    page_url,
                    allowed_mime_types=HTML_MIME_TYPES,
                    max_bytes=MAX_HTML_BYTES,
                )
                image_url = extract_significant_image_url(
                    page_content, base_url=final_url
                )
                if image_url is None:
                    return None
                image_content, final_image_url = await _fetch_bytes(
                    image_url,
                    allowed_mime_types=ALLOWED_IMAGE_MIME_TYPES,
                    max_bytes=MAX_REMOTE_IMAGE_BYTES,
                )
                decode_image(image_content)
                return DiscoveredImage(index, final_image_url, image_content)
            except (WebNavigationError, aiohttp.ClientError, asyncio.TimeoutError):
                return None

    tasks = [asyncio.create_task(discover(index)) for index in candidate_indices]
    try:
        resolved = await asyncio.wait_for(
            asyncio.gather(*tasks), timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return []
    return [item for item in resolved if item is not None]


async def _robots_allows(url: str) -> bool:
    """Return whether robots.txt allows the discovery user agent to read a page."""
    parsed = urlparse(url)
    robots_url = urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))
    try:
        content, _ = await _fetch_bytes(
            robots_url,
            allowed_mime_types={"text/plain", "text/html"},
            max_bytes=WEB_NAVIGATION.max_robots_bytes,
        )
    except (WebNavigationError, aiohttp.ClientError, asyncio.TimeoutError):
        return True
    parser = RobotFileParser()
    parser.parse(content.decode("utf-8", errors="replace").splitlines())
    return parser.can_fetch(USER_AGENT, url)


async def _fetch_bytes(
    url: str, *, allowed_mime_types: Set[str], max_bytes: int
) -> tuple[bytes, str]:
    """Fetch public content with pinned DNS, manual redirects, and byte limits."""
    current_url = _canonicalize_remote_url(url)
    redirect_count = 0
    while True:
        connector = aiohttp.TCPConnector(
            resolver=PublicResolver(), use_dns_cache=False, ttl_dns_cache=0
        )
        timeout = aiohttp.ClientTimeout(
            total=WEB_NAVIGATION.image_fetch_timeout_seconds
        )
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            trust_env=False,
            headers={"User-Agent": USER_AGENT},
        ) as session:
            async with session.get(current_url, allow_redirects=False) as response:
                if response.status in {301, 302, 303, 307, 308}:
                    if redirect_count >= MAX_REDIRECTS:
                        raise WebNavigationError(
                            "Remote image discovery exceeded the redirect limit.",
                            kind="redirect_limit_exceeded",
                        )
                    location = response.headers.get("Location")
                    if not location:
                        raise WebNavigationError(
                            "Remote redirect omitted its destination.",
                            kind="image_discovery_failed",
                        )
                    current_url = _canonicalize_remote_url(
                        urljoin(current_url, location)
                    )
                    redirect_count += 1
                    continue
                if response.status < 200 or response.status >= 300:
                    raise WebNavigationError(
                        f"Remote resource returned HTTP {response.status}.",
                        kind="image_discovery_failed",
                    )
                content_type = (
                    response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                )
                if content_type not in allowed_mime_types:
                    raise WebNavigationError(
                        f"Unsupported remote content type: {content_type or 'missing'}.",
                        kind="image_discovery_failed",
                    )
                declared_length = response.content_length
                if declared_length is not None and declared_length > max_bytes:
                    raise WebNavigationError(
                        "Remote resource exceeds its size limit.",
                        kind="image_discovery_failed",
                    )
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.content.iter_chunked(
                    WEB_NAVIGATION.download_chunk_bytes
                ):
                    size += len(chunk)
                    if size > max_bytes:
                        raise WebNavigationError(
                            "Remote resource exceeds its size limit.",
                            kind="image_discovery_failed",
                        )
                    chunks.append(chunk)
                return b"".join(chunks), str(response.url)


def extract_significant_image_url(content: bytes, *, base_url: str) -> str | None:
    """Select one significant image URL using deterministic metadata precedence."""
    try:
        document = html.fromstring(content, base_url=base_url)
    except (ValueError, TypeError):
        return None
    selectors = (
        "//meta[translate(@property, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='og:image']/@content",
        "//meta[translate(@name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='twitter:image']/@content",
        "//meta[translate(@property, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='twitter:image']/@content",
    )
    for selector in selectors:
        values = document.xpath(selector)
        if values and (resolved := _resolve_image_url(str(values[0]), base_url)):
            return resolved
    for script_text in document.xpath("//script[@type='application/ld+json']/text()"):
        try:
            payload = json.loads(str(script_text))
        except (TypeError, ValueError):
            continue
        for value in _json_ld_images(payload):
            if resolved := _resolve_image_url(value, base_url):
                return resolved
    image_nodes = document.xpath("//main//img[@src] | //article//img[@src]")
    ranked_nodes = sorted(
        enumerate(image_nodes),
        key=lambda item: (-_declared_image_area(item[1]), item[0]),
    )
    for _, node in ranked_nodes:
        if resolved := _resolve_image_url(str(node.get("src", "")), base_url):
            return resolved
    return None


def _json_ld_images(value: object) -> list[str]:
    """Collect image URLs from nested JSON-LD objects in document order."""
    found: list[str] = []
    if isinstance(value, dict):
        image = value.get("image")
        if isinstance(image, str):
            found.append(image)
        elif isinstance(image, list):
            found.extend(item for item in image if isinstance(item, str))
        elif isinstance(image, dict) and isinstance(image.get("url"), str):
            found.append(str(image["url"]))
        for child in value.values():
            if child is not image:
                found.extend(_json_ld_images(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_json_ld_images(child))
    return found


def _declared_image_area(node: object) -> int:
    """Return the declared pixel area of one HTML image node."""
    try:
        width = int(getattr(node, "get")("width", 0))
        height = int(getattr(node, "get")("height", 0))
    except (TypeError, ValueError):
        return 0
    return max(0, width) * max(0, height)


def _resolve_image_url(value: str, base_url: str) -> str | None:
    """Resolve and canonicalize one absolute or document-relative image URL."""
    try:
        return _canonicalize_remote_url(urljoin(base_url, value.strip()))
    except WebNavigationError:
        return None


def _canonicalize_remote_url(url: str) -> str:
    """Validate the URL shape before its hostname is resolved by PublicResolver."""
    parsed = urlparse(str(url).strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise WebNavigationError(
            "Only credential-free absolute HTTP(S) URLs are supported.",
            kind="invalid_url",
        )
    return urlunparse(parsed._replace(fragment=""))
