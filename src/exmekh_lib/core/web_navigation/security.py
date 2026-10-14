"""Network-boundary validation for browser-backed capabilities."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from urllib.parse import urljoin, urlparse

from exmekh_lib.config.web import WEB_NAVIGATION
from exmekh_lib.config.web_text import WEB_TEXT

from .errors import WebNavigationError

MAX_MAIN_FRAME_NAVIGATIONS = WEB_NAVIGATION.max_main_frame_navigations


def normalized_origin(url: str) -> tuple[str, str, int]:
    """Return the normalized scheme, hostname, and effective port for one URL."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise WebNavigationError(
            WEB_TEXT.only_absolute_http_and_https_urls_are_supported, kind="invalid_url"
        )
    default_port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme.lower(), parsed.hostname.lower(), parsed.port or default_port


class NavigationPolicy:
    """Track same-origin main-frame navigation within a finite hop budget."""

    def __init__(
        self, *, initial_url: str, max_navigations: int = MAX_MAIN_FRAME_NAVIGATIONS
    ) -> None:
        if max_navigations < 0:
            raise ValueError(WEB_TEXT.max_navigations_must_not_be_negative)
        self._initial_origin = normalized_origin(validate_public_url(initial_url))
        self._max_navigations = max_navigations
        self._navigation_count = 0
        self._initial_request_seen = False
        self._violation: WebNavigationError | None = None

    @property
    def initial_origin(self) -> tuple[str, str, int]:
        """Return the exact origin allowed for the browser lifecycle."""
        return self._initial_origin

    @property
    def violation(self) -> WebNavigationError | None:
        return self._violation

    def record_violation(self, error: WebNavigationError) -> None:
        """Keep the first blocked navigation as the cause of failure."""
        if self._violation is None:
            self._violation = error

    def observe_main_frame_navigation(self, url: str) -> bool:
        """Record one document request and return whether it may continue."""
        try:
            validate_public_url(url)
            if normalized_origin(url) != self.initial_origin:
                raise WebNavigationError(
                    WEB_TEXT.main_frame_navigation_left_the_initial_origin_2.format(
                        url=url
                    ),
                    kind="cross_origin_navigation",
                )
            if not self._initial_request_seen:
                self._initial_request_seen = True
                return True
            self._navigation_count += 1
            if self._navigation_count > self._max_navigations:
                raise WebNavigationError(
                    WEB_TEXT.main_frame_navigation_exceeded_hops.format(
                        max_navigations=self._max_navigations
                    ),
                    kind="redirect_limit_exceeded",
                )
        except WebNavigationError as error:
            self.record_violation(error)
            return False
        return True

    def validate_action_url(self, url: str) -> str:
        """Validate an explicit action destination against the initial origin."""
        value = validate_public_url(url)
        if normalized_origin(value) != self.initial_origin:
            raise WebNavigationError(
                WEB_TEXT.navigation_action_left_the_initial_origin.format(value=value),
                kind="cross_origin_navigation",
            )
        return value

    def raise_if_violated(self) -> None:
        """Raise the first navigation-policy violation observed by route handling."""
        if self._violation is not None:
            raise self._violation


def validate_public_url(url: str) -> str:
    """Validate an HTTP(S) URL and require every resolved address to be global."""
    value = str(url).strip()
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError as error:
        raise WebNavigationError(
            WEB_TEXT.url_contains_an_invalid_port, kind="invalid_url"
        ) from error
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise WebNavigationError(
            WEB_TEXT.only_absolute_http_and_https_urls_are_supported, kind="invalid_url"
        )
    if parsed.username is not None or parsed.password is not None:
        raise WebNavigationError(
            WEB_TEXT.credentials_in_urls_are_not_supported, kind="invalid_url"
        )
    _validate_host(parsed.hostname, port)
    return value


def _validate_host(hostname: str, port: int | None) -> None:
    """Reject hostnames that resolve to any non-global network address."""
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        addresses = {literal}
    else:
        try:
            resolved = socket.getaddrinfo(
                hostname, port or 443, type=socket.SOCK_STREAM
            )
        except OSError as error:
            raise WebNavigationError(
                WEB_TEXT.could_not_resolve_url_hostname_2.format(hostname=hostname),
                kind="dns_error",
            ) from error
        addresses = {ipaddress.ip_address(item[4][0]) for item in resolved}
    if not addresses or any(not address.is_global for address in addresses):
        raise WebNavigationError(
            WEB_TEXT.url_hostname_is_not_publicly_routable_2.format(hostname=hostname),
            kind="blocked_url",
        )


def build_route_guard(
    *, page: object | None = None, policy: NavigationPolicy | None = None
) -> Callable[[object], Awaitable[None]]:
    """Build a route callback blocking unsafe requests and document navigation."""

    async def guard(route: object) -> None:
        """Continue safe requests and abort invalid or private destinations."""
        request = getattr(route, "request")
        url = str(getattr(request, "url"))
        is_main_navigation = _is_main_frame_navigation(request=request, page=page)
        if is_main_navigation and policy is not None:
            if not policy.observe_main_frame_navigation(url):
                await getattr(route, "abort")()
                return
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            await getattr(route, "abort")()
            return
        try:
            await asyncio.to_thread(validate_public_url, url)
        except WebNavigationError:
            await getattr(route, "abort")()
        else:
            if not is_main_navigation:
                await getattr(route, "continue_")()
                return
            response = await getattr(route, "fetch")(max_redirects=0)
            if not await _redirect_chain_is_allowed(
                route=route, request=request, response=response, url=url, policy=policy
            ):
                await getattr(route, "abort")()
                return
            await getattr(route, "fulfill")(response=response)

    return guard


async def _redirect_chain_is_allowed(
    *,
    route: object,
    request: object,
    response: object,
    url: str,
    policy: NavigationPolicy | None,
) -> bool:
    """Preflight GET redirects before the browser follows the first response."""
    current_url = url
    current_response = response
    hops = 0
    while 300 <= getattr(current_response, "status") < 400:
        location = getattr(current_response, "headers").get("location")
        if not location:
            return True
        hops += 1
        if hops > MAX_MAIN_FRAME_NAVIGATIONS:
            if policy is not None:
                policy.record_violation(
                    WebNavigationError(
                        WEB_TEXT.main_frame_navigation_exceeded_the_redirect_budget,
                        kind="redirect_limit_exceeded",
                    )
                )
            return False
        destination = urljoin(current_url, location)
        if policy is not None:
            if not policy.observe_main_frame_navigation(destination):
                return False
        else:
            try:
                await asyncio.to_thread(validate_public_url, destination)
            except WebNavigationError:
                return False
        if getattr(request, "method", "GET") != "GET":
            return True
        current_url = destination
        current_response = await getattr(route, "fetch")(
            url=destination, max_redirects=0
        )
    return True


def _is_main_frame_navigation(*, request: object, page: object | None) -> bool:
    """Return whether one routed request navigates the selected page's main frame."""
    is_navigation = getattr(request, "is_navigation_request", None)
    if not callable(is_navigation) or not bool(is_navigation()):
        return False
    if page is None:
        return True
    return getattr(request, "frame", None) is getattr(page, "main_frame", None)
