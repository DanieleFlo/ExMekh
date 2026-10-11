"""DS4 v1 transport using the configured OpenAI SDK's custom JSON/SSE API."""

from copy import deepcopy
from time import monotonic, sleep
from typing import Any
from uuid import uuid4

import openai
from openai import OpenAI
from openai.types.responses import Response

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
)
from exmekh_lib.core.models.interactive import (
    InteractiveProtocolError,
    InterruptedMessage,
)
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.reasoning_effort import ReasoningEffort
from exmekh_lib.core.providers.interactive_contract import InteractiveContract
from exmekh_lib.core.providers.interactive_transport import InteractiveTransport
from exmekh_lib.core.providers.response_openai import ResponseOpenAI
from exmekh_lib.core.providers.to_openai_response_request_mapper import (
    ToOpenAIResponseRequestMapper,
)

_TERMINAL_TYPES = {
    "interactive.rejected",
    "interactive.cancelled",
    "interactive.expired",
    "interactive.failed",
}


class InteractiveResponseOpenAI(ResponseOpenAI):
    """One logical inference may span multiple HTTP requests, without tool dispatch."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        super().__init__(client, model, reasoning_effort)
        self._transport = InteractiveTransport(client)

    def call(self, request: ProviderRequest) -> ProviderResponse:
        payload = ToOpenAIResponseRequestMapper.map_request(
            request, model=self._model, reasoning_effort=self._reasoning_effort
        )
        extra = payload.get("extra_body", {})
        scope = extra.get("agentic", {})
        tools = request.get_tools()
        definitions: dict[str, dict[str, Any]] = {}
        schemas: dict[str, dict[str, Any]] = {}
        for wire_tool in payload.get("tools", []):
            tool = tools.get_tool(wire_tool["name"]) if tools is not None else None
            if (
                tool is not None
                and tool.interactive is not None
                and tool.name in scope.get("allowed_tools", [])
            ):
                schemas[tool.name] = wire_tool["parameters"]
                definitions[tool.name] = tool.interactive.configuration(
                    tool.name, schemas[tool.name]
                )
        if not definitions:
            return super().call(request)
        handler = request.require_interactive_handler()
        snapshot_limit = self._negotiate()
        handler.begin()
        body = {key: value for key, value in payload.items() if key != "extra_body"}
        body.update(deepcopy(extra))
        scope = deepcopy(scope)
        body["agentic"] = {
            **scope,
            "interactive": {
                "version": 1,
                "operation": "configure",
                "request_id": uuid4().hex,
                "tools": list(definitions.values()),
            },
        }
        contract = InteractiveContract(definitions, schemas)
        return self._logical_call(request, body, scope, contract, snapshot_limit)

    def _negotiate(self) -> int:
        capabilities = self._client.get("/agentic/capabilities", cast_to=dict[str, Any])
        if not isinstance(capabilities, dict):
            raise InteractiveProtocolError("Malformed server capabilities")
        interactive = capabilities.get("interactive", {})
        if not isinstance(interactive, dict):
            raise InteractiveProtocolError("Malformed interactive capabilities")
        if any(
            not isinstance(interactive.get(field), list)
            or not all(isinstance(value, str) for value in interactive[field])
            for field in ("syntaxes", "apis", "behaviors")
        ):
            raise InteractiveProtocolError("Malformed interactive capability lists")
        if (
            type(interactive.get("version")) is not int
            or interactive["version"] != 1
            or interactive.get("model_supported") is not True
            or interactive.get("model_syntax") not in interactive.get("syntaxes", [])
            or "responses" not in interactive.get("apis", [])
            or "edit_file.v1" not in interactive.get("behaviors", [])
        ):
            raise InteractiveProtocolError(
                "Server/model does not support interactive Responses v1; no ordinary-call fallback is permitted"
            )
        limits = interactive.get("limits", {})
        if not isinstance(limits, dict):
            raise InteractiveProtocolError("Malformed interactive limits")
        snapshot_limit = limits.get("snapshot_bytes", FILE_EDITING.max_snapshot_bytes)
        if type(snapshot_limit) is not int or snapshot_limit < 1:
            raise InteractiveProtocolError("Invalid server snapshot byte limit")
        return min(snapshot_limit, FILE_EDITING.max_snapshot_bytes)

    def _logical_call(
        self,
        request: ProviderRequest,
        body: dict[str, Any],
        scope: dict[str, Any],
        contract: InteractiveContract,
        snapshot_limit: int,
    ) -> ProviderResponse:
        control_scope = self._control_scope(scope)
        deadline = monotonic() + FILE_EDITING.protocol_timeout_seconds
        streaming = request.get_streaming()
        if streaming:
            request.execute_on_stream_event(
                ProviderStreamEvent(ProviderStreamEventType.START)
            )
        try:
            for _ in range(FILE_EDITING.max_exchanges + 1):
                result = self._post_retry(
                    body, request, contract, deadline, control_scope
                )
                kind = result.get("type")
                contract.validate_next_result(result)
                if kind == "interactive.await_input":
                    body = self._resume_after_pause(
                        result, request, contract, control_scope, snapshot_limit
                    )
                    continue
                if kind in _TERMINAL_TYPES:
                    return self._interrupted_result(result, request, contract)
                return self._completed_result(result, request, contract)
            raise InteractiveProtocolError("Interactive exchange budget exhausted")
        except BaseException as error:
            if contract.is_accepted:
                # Cleanup is a separate idempotent operation. Failure remains visible;
                # no queued call can execute after an unconfirmed transport failure.
                try:
                    self._post_control(
                        self._control(
                            control_scope, "cancel", contract.continuation_id
                        ),
                        timeout=min(5.0, FILE_EDITING.protocol_timeout_seconds),
                    )
                except (
                    openai.APIError,
                    InteractiveProtocolError,
                    ValueError,
                ) as cleanup_error:
                    error.add_note(f"Cancellation was not confirmed: {cleanup_error}")
            raise
        finally:
            if streaming:
                request.execute_on_stream_event(
                    ProviderStreamEvent(ProviderStreamEventType.END)
                )

    def _resume_after_pause(
        self,
        result: dict[str, Any],
        request: ProviderRequest,
        contract: InteractiveContract,
        scope: dict[str, Any],
        snapshot_limit: int,
    ) -> dict[str, Any]:
        contract.record_pause(result)
        request.execute_on_stream_event(
            ProviderStreamEvent(ProviderStreamEventType.INTERACTIVE, interactive=result)
        )
        handler = request.require_interactive_handler()
        if handler.cancellation_requested():
            return self._control(scope, "cancel", contract.continuation_id)
        reply = handler.prepare(result, contract.definition(result["tool_name"]))
        reply = self._bounded_reply(reply, snapshot_limit)
        contract.record_reply(result, reply)
        return self._control(
            scope,
            "resume",
            contract.continuation_id,
            exchange_id=result["exchange_id"],
            reply=reply,
        )

    @staticmethod
    def _bounded_reply(
        reply: dict[str, object], snapshot_limit: int
    ) -> dict[str, object]:
        if reply.get("status") != "provide":
            return reply
        text = reply.get("text")
        if not isinstance(text, str):
            raise InteractiveProtocolError("Provide requires exact Unicode text")
        if len(text.encode("utf-8")) <= snapshot_limit:
            return reply
        return {
            "status": "reject",
            "code": "snapshot_limit",
            "message": "Snapshot exceeds the negotiated byte limit",
        }

    def _interrupted_result(
        self,
        result: dict[str, Any],
        request: ProviderRequest,
        contract: InteractiveContract,
    ) -> ProviderResponse:
        contract.record_acceptance(result)
        request.execute_on_stream_event(
            ProviderStreamEvent(ProviderStreamEventType.INTERACTIVE, interactive=result)
        )
        rejected = result["type"] == "interactive.rejected"
        if rejected:
            contract.validate_interruption(result)
        output = result.get("output", []) if rejected else []
        contract.validate_raw_calls(output)
        mapped = self._map_output(output, request, result)
        if rejected:
            contract.validate_rejected_calls(mapped.get_tool_calls() or [])
        else:
            contract.validate_calls(mapped.get_tool_calls() or [])
        record = {
            key: value
            for key, value in result.items()
            if key not in {"output", "usage"}
        }
        response = ProviderResponse(
            mapped,
            thinking=mapped.thinking,
            usage=mapped.usage,
            interruptions=(InterruptedMessage(record),),
        )
        if request.require_interactive_handler().cancellation_requested():
            response = self._steered_response(response, contract.continuation_id)
        self._report_usage(request, response)
        return response

    def _completed_result(
        self,
        result: dict[str, Any],
        request: ProviderRequest,
        contract: InteractiveContract,
    ) -> ProviderResponse:
        contract.record_acceptance(result.get("interactive", {}))
        contract.validate_raw_calls(result.get("output", []))
        response = self._parse_response(self._validated_response(result), request)
        calls = response.get_tool_calls() or []
        contract.validate_completed_calls(calls)
        self._report_usage(request, response)
        handler = request.require_interactive_handler()
        if handler.cancellation_requested():
            return self._steered_response(response, contract.continuation_id)
        handler.validate_completed(calls)
        return response

    @staticmethod
    def _steered_response(
        response: ProviderResponse, continuation_id: str
    ) -> ProviderResponse:
        record = {
            "type": "interactive.cancelled",
            "version": 1,
            "continuation_id": continuation_id,
            "code": "steered",
            "message": "Steering invalidated this inference before dispatch",
        }
        return ProviderResponse(
            AssistantMessage(None),
            thinking=response.thinking,
            usage=response.usage,
            interruptions=(*response.interruptions, InterruptedMessage(record)),
        )

    @staticmethod
    def _report_usage(request: ProviderRequest, response: ProviderResponse) -> None:
        if request.get_streaming() and response.usage is not None:
            request.execute_on_stream_event(
                ProviderStreamEvent(
                    ProviderStreamEventType.TOTALS, usage=response.usage
                )
            )

    @staticmethod
    def _control_scope(scope: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in scope.items()
            if key not in {"operation", "skill_call_id", "interactive"}
        }

    @staticmethod
    def _control(
        scope: dict[str, Any], operation: str, continuation_id: str, **values: Any
    ) -> dict[str, Any]:
        control = {
            "version": 1,
            "operation": operation,
            "continuation_id": continuation_id,
            **values,
        }
        if operation != "status":
            control["request_id"] = uuid4().hex
        return {"agentic": {**scope, "interactive": control}}

    def _post_control(
        self, body: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        events = self._transport.events(
            body,
            timeout if timeout is not None else FILE_EDITING.protocol_timeout_seconds,
        )
        result = None
        for event in events:
            result = (
                event.get("response", event)
                if event.get("type") == "interactive.result"
                else event
            )
        if not isinstance(result, dict):
            raise InteractiveProtocolError("Control operation returned no receipt")
        return result

    def status(
        self,
        scope: dict[str, Any],
        continuation_id: str,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._post_control(
            self._control(scope, "status", continuation_id), timeout=timeout
        )

    def _post_retry(
        self,
        body: dict[str, Any],
        request: ProviderRequest,
        contract: InteractiveContract,
        deadline: float,
        scope: dict[str, Any],
    ) -> dict[str, Any]:
        for attempt in range(3):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise InteractiveProtocolError("Interactive logical-call timeout")
            try:
                return self._post(body, request, contract, remaining)
            except openai.ConflictError:
                if not contract.is_accepted:
                    raise
                receipt = self.status(
                    scope,
                    contract.continuation_id,
                    timeout=max(0.001, deadline - monotonic()),
                )
                return self._unwrap_status(
                    receipt, request, contract, max(0, deadline - monotonic())
                )
            except (openai.APIConnectionError, openai.APITimeoutError):
                if attempt == 2:
                    raise
                # Retry the identical configure/resume request_id and payload. The
                # server's receipt prevents regeneration even if the reply was lost.
        raise AssertionError("unreachable")

    def _post(
        self,
        body: dict[str, Any],
        request: ProviderRequest,
        contract: InteractiveContract,
        remaining: float,
    ) -> dict[str, Any]:
        deadline = monotonic() + remaining
        terminal: dict[str, Any] | None = None
        for event in self._transport.events(body, remaining):
            kind = event.get("type")
            if kind == "interactive.accepted":
                contract.record_acceptance(event)
                request.execute_on_stream_event(
                    ProviderStreamEvent(
                        ProviderStreamEventType.INTERACTIVE, interactive=event
                    )
                )
            elif kind == "interactive.result":
                terminal = event["response"]
            elif kind == "interactive.await_input" or kind in _TERMINAL_TYPES:
                terminal = event
            elif kind in {
                "response.completed",
                "response.incomplete",
                "response.failed",
            }:
                terminal = event["response"]
            elif kind in {
                "response.output_text.delta",
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
            }:
                event_type = (
                    ProviderStreamEventType.RESPONSE
                    if kind == "response.output_text.delta"
                    else ProviderStreamEventType.REASONING
                )
                request.execute_on_stream_event(
                    ProviderStreamEvent(event_type, content=event["delta"])
                )
            elif kind == "interactive.status" or event.get("object") == "response":
                terminal = event
        if terminal is None:
            if contract.is_accepted:
                terminal = self.status(
                    self._control_scope(body["agentic"]),
                    contract.continuation_id,
                    timeout=max(0.001, deadline - monotonic()),
                )
            else:
                raise InteractiveProtocolError(
                    "Stream ended without acceptance or a confirmed result"
                )
        return self._unwrap_status(
            terminal, request, contract, max(0, deadline - monotonic())
        )

    def _unwrap_status(
        self,
        result: dict[str, Any],
        request: ProviderRequest,
        contract: InteractiveContract,
        remaining: float,
    ) -> dict[str, Any]:
        deadline = monotonic() + remaining
        while result.get("type") == "interactive.status":
            contract.record_acceptance(result)
            if isinstance(result.get("result"), dict):
                return result["result"]
            if monotonic() >= deadline:
                raise InteractiveProtocolError(
                    "No terminal receipt before the logical-call timeout"
                )
            sleep(min(0.1, max(0, deadline - monotonic())))
            scope = (request.get_extra() or {}).get("agentic", {})
            if not isinstance(scope, dict):
                raise InteractiveProtocolError("Invalid agentic scope")
            result = self.status(
                self._control_scope(scope),
                contract.continuation_id,
                timeout=max(0.001, deadline - monotonic()),
            )
        return result

    def _map_output(
        self,
        output: list[dict[str, Any]],
        request: ProviderRequest,
        terminal: dict[str, Any],
    ) -> ProviderResponse:
        return self._parse_response(
            self._validated_response(
                {
                    "id": "interactive-terminal",
                    "object": "response",
                    "created_at": 0,
                    "model": self._model,
                    "status": "completed",
                    "error": None,
                    "incomplete_details": None,
                    "instructions": None,
                    "metadata": {},
                    "output": output,
                    "parallel_tool_calls": False,
                    "temperature": 1,
                    "tool_choice": "auto",
                    "tools": [],
                    "top_p": 1,
                    "usage": terminal.get("usage"),
                }
            ),
            request,
        )

    @staticmethod
    def _validated_response(body: dict[str, Any]) -> Response:
        normalized = deepcopy(body)
        # DS4 omits these unused request-echo fields; they never drive dispatch,
        # argument validation, scope, or sampling in this adapter.
        normalized.setdefault("parallel_tool_calls", False)
        normalized.setdefault("tool_choice", "auto")
        normalized.setdefault("tools", [])
        usage = normalized.get("usage")
        if isinstance(usage, dict):
            details = usage.get("input_tokens_details")
            if isinstance(details, dict):
                # SDK 3.3 requires this counter even when a compatible response
                # omits it. This unused placeholder does not claim cache behavior;
                # the library reports only aggregate input/output/total usage.
                details.setdefault("cache_write_tokens", 0)
        return Response.model_validate(normalized)
