"""Read custom JSON/SSE through the configured SDK, closing every response."""

import codecs
import json
import re
from collections.abc import Iterator
from time import monotonic
from typing import Any

import httpx2
from openai import APIConnectionError, APIResponse, APITimeoutError, OpenAI

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.models.interactive import InteractiveProtocolError
from exmekh_lib.core.providers.interactive_contract import decode_object_pairs


class InteractiveTransport:
    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def events(self, body: dict[str, Any], timeout: float) -> Iterator[dict[str, Any]]:
        deadline = monotonic() + timeout
        client = self._client.with_options(timeout=timeout, max_retries=0)
        response = client.post(
            "/responses",
            body=body,
            cast_to=APIResponse[dict[str, Any]],
            options={"headers": {"X-Stainless-Raw-Response": "stream"}},
        )
        try:
            if "text/event-stream" not in response.headers.get("content-type", ""):
                raw = b"".join(
                    self._bounded_chunks(response, max(0, deadline - monotonic()))
                )
                value = json.loads(raw, object_pairs_hook=decode_object_pairs)
                if not isinstance(value, dict):
                    raise InteractiveProtocolError(
                        "Interactive response must be a JSON object"
                    )
                yield value
                return
            data: list[str] = []
            for line in self._bounded_lines(response, max(0, deadline - monotonic())):
                if not line:
                    if data:
                        raw = "\n".join(data)
                        data.clear()
                        if raw != "[DONE]":
                            value = json.loads(
                                raw, object_pairs_hook=decode_object_pairs
                            )
                            if not isinstance(value, dict):
                                raise InteractiveProtocolError(
                                    "SSE data must be a JSON object"
                                )
                            yield value
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
        except httpx2.TimeoutException as error:
            raise APITimeoutError(request=response.http_request) from error
        except httpx2.TransportError as error:
            raise APIConnectionError(request=response.http_request) from error
        finally:
            response.close()

    @staticmethod
    def _bounded_chunks(
        response: APIResponse[dict[str, Any]], timeout: float
    ) -> Iterator[bytes]:
        deadline = monotonic() + timeout
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > FILE_EDITING.max_response_bytes:
                raise InteractiveProtocolError(
                    "Interactive HTTP output exceeds the byte budget"
                )
            if monotonic() >= deadline:
                raise InteractiveProtocolError(
                    "Interactive logical-call timeout while reading HTTP output"
                )
            yield chunk

    @classmethod
    def _bounded_lines(
        cls, response: APIResponse[dict[str, Any]], timeout: float
    ) -> Iterator[str]:
        decoder = codecs.getincrementaldecoder("utf-8")()
        pending = ""
        for chunk in cls._bounded_chunks(response, timeout):
            pending += decoder.decode(chunk)
            while match := re.search(r"\r\n|\r|\n", pending):
                if match.group() == "\r" and match.end() == len(pending):
                    break
                yield pending[: match.start()]
                pending = pending[match.end() :]
        pending += decoder.decode(b"", final=True)
        if pending:
            yield pending.rstrip("\r")
