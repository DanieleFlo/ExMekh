"""Public-network boundary and redirect-policy regression tests."""

import asyncio
import socket
from types import SimpleNamespace

import pytest

from exmekh_lib.core.web_navigation import security
from exmekh_lib.core.web_navigation.errors import WebNavigationError


@pytest.mark.parametrize(
    ("url", "kind"),
    [
        ("file:///etc/passwd", "invalid_url"),
        ("example.org/path", "invalid_url"),
        ("https://user:pass@example.org/", "invalid_url"),
        ("https://127.0.0.1/", "blocked_url"),
        ("http://[::1]/", "blocked_url"),
    ],
)
def test_url_validation_rejects_unsafe_destinations(url: str, kind: str) -> None:
    with pytest.raises(WebNavigationError) as failure:
        security.validate_public_url(url)
    assert failure.value.kind == kind


def test_normalized_origin_rejects_relative_urls() -> None:
    with pytest.raises(WebNavigationError) as failure:
        security.normalized_origin("/relative")
    assert failure.value.kind == "invalid_url"


def test_url_validation_checks_every_resolved_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (None, None, None, None, ("93.184.215.14", 443)),
            (None, None, None, None, ("10.0.0.1", 443)),
        ],
    )
    with pytest.raises(WebNavigationError) as failure:
        security.validate_public_url("https://mixed.example/")
    assert failure.value.kind == "blocked_url"

    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [])
    with pytest.raises(WebNavigationError) as failure:
        security.validate_public_url("https://empty.example/")
    assert failure.value.kind == "blocked_url"

    def dns_error(*args: object, **kwargs: object) -> None:
        raise OSError("DNS unavailable")

    monkeypatch.setattr(socket, "getaddrinfo", dns_error)
    with pytest.raises(WebNavigationError) as failure:
        security.validate_public_url("https://missing.example/")
    assert failure.value.kind == "dns_error"


def test_navigation_policy_counts_hops_and_validates_action_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "validate_public_url", lambda url: url)
    with pytest.raises(ValueError, match="must not be negative"):
        security.NavigationPolicy(
            initial_url="https://source.example/", max_navigations=-1
        )
    policy = security.NavigationPolicy(
        initial_url="https://source.example/", max_navigations=1
    )
    assert policy.initial_origin == ("https", "source.example", 443)
    assert policy.observe_main_frame_navigation("https://source.example/start")
    assert policy.observe_main_frame_navigation("https://source.example/next")
    assert policy.validate_action_url("https://source.example/last") == (
        "https://source.example/last"
    )
    assert not policy.observe_main_frame_navigation("https://source.example/again")
    assert policy.violation is not None
    assert policy.violation.kind == "redirect_limit_exceeded"
    with pytest.raises(WebNavigationError, match="left the initial origin"):
        policy.validate_action_url("https://other.example/")


class _Route:
    def __init__(self, url: str, responses: list[tuple[int, str | None]]) -> None:
        self.request = SimpleNamespace(
            url=url,
            method="GET",
            is_navigation_request=lambda: True,
            frame=None,
        )
        self.responses = responses
        self.aborted = False
        self.continued = False
        self.fulfilled = False
        self.fetched: list[str] = []

    async def fetch(
        self, *, max_redirects: int, url: str | None = None
    ) -> SimpleNamespace:
        assert max_redirects == 0
        self.fetched.append(url or self.request.url)
        status, location = self.responses.pop(0)
        return SimpleNamespace(
            status=status, headers={"location": location} if location else {}
        )

    async def abort(self) -> None:
        self.aborted = True

    async def continue_(self) -> None:
        self.continued = True

    async def fulfill(self, *, response: object) -> None:
        self.fulfilled = True


def test_route_guard_fulfills_safe_document_and_continues_subresources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "validate_public_url", lambda url: url)
    route = _Route("https://source.example/page", [(200, None)])
    asyncio.run(security.build_route_guard()(route))
    assert route.fulfilled and not route.aborted

    route = _Route("https://source.example/image.png", [])
    route.request.is_navigation_request = lambda: False
    asyncio.run(security.build_route_guard()(route))
    assert route.continued and not route.fetched


def test_route_guard_rejects_non_http_and_private_requests() -> None:
    route = _Route("data:text/plain,hello", [])
    asyncio.run(security.build_route_guard()(route))
    assert route.aborted
    route = _Route("http://127.0.0.1/private", [])
    asyncio.run(security.build_route_guard()(route))
    assert route.aborted


def test_route_guard_aborts_main_navigation_rejected_by_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "validate_public_url", lambda url: url)
    route = _Route("https://other.example/page", [])
    policy = security.NavigationPolicy(initial_url="https://source.example/")
    asyncio.run(security.build_route_guard(policy=policy)(route))
    assert route.aborted and not route.fetched


def test_redirect_without_location_is_fulfilled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "validate_public_url", lambda url: url)
    route = _Route("https://source.example/start", [(302, None)])
    asyncio.run(security.build_route_guard()(route))
    assert route.fulfilled


def test_redirect_limit_without_policy_still_blocks_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "validate_public_url", lambda url: url)
    route = _Route("https://source.example/start", [(302, "/next")] * 7)
    asyncio.run(security.build_route_guard()(route))
    assert route.aborted


def test_route_guard_follows_relative_redirects_only_within_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "validate_public_url", lambda url: url)
    route = _Route("https://source.example/start", [(302, "/middle"), (200, None)])
    policy = security.NavigationPolicy(initial_url=route.request.url)
    asyncio.run(security.build_route_guard(policy=policy)(route))
    assert route.fetched == [
        "https://source.example/start",
        "https://source.example/middle",
    ]
    assert route.fulfilled and policy.violation is None


def test_route_guard_blocks_unsafe_redirect_without_policy() -> None:
    route = _Route("https://93.184.215.14/start", [(302, "http://127.0.0.1/")])
    asyncio.run(security.build_route_guard()(route))
    assert route.aborted and not route.fulfilled


def test_redirect_preflight_limits_hops_and_respects_post_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "validate_public_url", lambda url: url)
    route = _Route("https://source.example/start", [(302, "/next")] * 7)
    policy = security.NavigationPolicy(
        initial_url=route.request.url, max_navigations=20
    )
    asyncio.run(security.build_route_guard(policy=policy)(route))
    assert route.aborted
    assert policy.violation is not None
    assert policy.violation.kind == "redirect_limit_exceeded"

    post = _Route("https://source.example/start", [(302, "/next")])
    post.request.method = "POST"
    asyncio.run(security.build_route_guard()(post))
    assert post.fulfilled
    assert len(post.fetched) == 1


def test_main_frame_detection_ignores_other_frames() -> None:
    request = SimpleNamespace(is_navigation_request=lambda: True, frame=object())
    page = SimpleNamespace(main_frame=object())
    assert not security._is_main_frame_navigation(request=request, page=page)
    assert security._is_main_frame_navigation(request=request, page=None)
