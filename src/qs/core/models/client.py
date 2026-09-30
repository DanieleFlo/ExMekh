from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

from pydantic import BaseModel

from qs.config.runtime import RUNTIME
from qs.core.models.assistant_message import ToolCall
from qs.core.models.client_request import ClientRequest
from qs.core.models.history import HistoryMessage, MessageHistory
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool_message import ToolMessage
from qs.core.models.tools import Tools
from qs.core.models.user_message import UserMessage
from qs.interfaces.event_handler import ProviderEventHandler
from qs.interfaces.provider import Provider
from qs.interfaces.tokenizer import Tokenizer

if TYPE_CHECKING:
    from qs.core.cache.session_cache import SessionCache

logger = logging.getLogger(__name__)


class Client:
    def __init__(
        self,
        provider: Provider,
        system_message: SystemMessage,
        tools: Tools | None = None,
    ) -> None:
        if not isinstance(system_message, SystemMessage):
            raise TypeError("Expected a SystemMessage instance")

        self._history = MessageHistory(history=[system_message])
        self._history.check_history()
        self._provider = provider
        self._validate_provider()

        if tools is not None and not isinstance(tools, Tools):
            raise TypeError("Tools must be a Tools instance")
        self.tools = tools

    def _validate_provider(self) -> None:
        if not isinstance(self._provider, Provider):
            raise TypeError("Expected a Provider instance")

    def get_tokenizer(self) -> Tokenizer:
        return self._provider.get_tokenizer()

    def generate_id(self) -> str:
        return self._provider.generate_id()

    def get_session(self) -> SessionCache:
        return self._provider.get_session()

    def get_history(self) -> list[HistoryMessage]:
        return self._history.get_history()

    def dump_history(self) -> list[dict[str, str]]:
        return self._history.dump_history()

    def copy_history(self) -> list[HistoryMessage]:
        return self._history.get_copy_of_history()

    def replace_history_from(
        self,
        start_index: int,
        messages: Sequence[HistoryMessage],
    ) -> None:
        self._history.replace_from(start_index, messages)

    def add_tool_calls(self, calls: Sequence[ToolCall]) -> None:
        self._history.add_tool_calls(calls)

    def replace_latest_tool_calls(self, calls: Sequence[ToolCall]) -> None:
        self._history.replace_latest_tool_calls(calls)

    def record_tool_call(
        self, call: ToolCall, *, batch_call_id: str | None = None
    ) -> AbstractContextManager[None]:
        return self._history.record_tool_call(call, batch_call_id=batch_call_id)

    def call(self, request: ClientRequest) -> ProviderResponse:
        return self._call_provider(self._prepare_request(request))

    def _call_provider(self, request: ProviderRequest) -> ProviderResponse:
        logger.info(
            "Calling provider: messages=%d tool_choice=%s",
            len(request.get_messages()),
            request.get_tool_choice(),
        )
        response = self._provider.call(request)
        self._history.add(response)
        logger.info(
            "Provider response recorded: tool_calls=%d history_size=%d",
            len(response.get_tool_calls() or ()),
            len(self._history.get_history()),
        )
        return response

    def continue_after_tools(
        self,
        *,
        extra: Mapping[str, object] | None = None,
        tool_choice: str | None = None,
        streaming: bool = False,
        on_event: ProviderEventHandler | None = None,
    ) -> ProviderResponse:
        messages = self._get_tool_continuation_messages()
        provider_request = ProviderRequest(
            messages=messages,
            extra=extra,
            tool_choice=tool_choice,
            tools=self.tools,
            stream=streaming,
            on_event=on_event,
        )
        return self._call_provider(provider_request)

    def continue_with_user_messages(
        self,
        messages: Sequence[UserMessage],
        *,
        extra: Mapping[str, object] | None = None,
        tool_choice: str | None = None,
        streaming: bool = False,
        on_event: ProviderEventHandler | None = None,
    ) -> ProviderResponse:
        """Append user input at a completed batch boundary and resume inference."""
        if not messages or not all(
            isinstance(message, UserMessage) for message in messages
        ):
            raise ValueError(RUNTIME.invalid_steering_messages)
        provider_request = ProviderRequest(
            messages=[*self._history.get_history(), *messages],
            extra=extra,
            tool_choice=tool_choice,
            tools=self.tools,
            stream=streaming,
            on_event=on_event,
        )
        self._history.add(list(messages))
        return self._call_provider(provider_request)

    def complete_tool_transition(self, *, extra: Mapping[str, object]) -> None:
        """Complete a provider transition before executing an already queued batch.

        The next batch supersedes this response, so it is neither streamed nor
        retained in the conversation.
        """
        history_size = len(self._history.get_history())
        self.continue_after_tools(extra=extra, tool_choice="none")
        self._history.replace_from(history_size, [])

    def call_structured[ResponseModel: BaseModel](
        self,
        request: ClientRequest,
        response_model: type[ResponseModel],
        *,
        strict: bool = True,
    ) -> StructuredProviderResponse[ResponseModel]:
        provider_request = self._prepare_structured_request(
            request,
            response_model,
            strict=strict,
        )
        return self._call_structured_provider(provider_request)

    def continue_after_tools_structured[ResponseModel: BaseModel](
        self,
        response_model: type[ResponseModel],
        *,
        strict: bool = True,
        extra: Mapping[str, object] | None = None,
        tool_choice: str | None = None,
        streaming: bool = False,
        on_event: ProviderEventHandler | None = None,
    ) -> StructuredProviderResponse[ResponseModel]:
        messages = self._get_tool_continuation_messages()
        provider_request = StructuredProviderRequest(
            messages,
            response_model,
            strict=strict,
            extra=extra,
            tool_choice=tool_choice,
            tools=self.tools,
            stream=streaming,
            on_event=on_event,
        )
        return self._call_structured_provider(provider_request)

    def _call_structured_provider[ResponseModel: BaseModel](
        self, request: StructuredProviderRequest[ResponseModel]
    ) -> StructuredProviderResponse[ResponseModel]:
        response = self._provider.call_structured(request)
        self._history.add(response)
        return response

    def add_tool_response(self, tool_response: ToolMessage) -> None:
        if not isinstance(tool_response, ToolMessage):
            raise TypeError("Expected a ToolMessage instance")
        self._history.add(tool_response)
        logger.debug(
            "Tool response recorded: name=%s call_id=%s history_size=%d",
            tool_response.get_name(),
            tool_response.get_tool_call_id(),
            len(self._history.get_history()),
        )

    def add_user_message(self, message: HistoryMessage) -> None:
        if message.get_role_name() != "user":
            raise TypeError("Expected a user message")
        self._history.add(message)

    def _get_tool_continuation_messages(self) -> list[HistoryMessage]:
        messages = self._history.get_history()
        if not isinstance(messages[-1], ToolMessage):
            raise TypeError("The last message must be a tool response")
        return messages

    def _prepare_request(self, request: ClientRequest) -> ProviderRequest:
        if not isinstance(request, ClientRequest):
            raise TypeError("Expected a ClientRequest instance")

        provider_request = ProviderRequest(
            messages=[*self._history.get_history(), request.message],
            extra=request.extra,
            tool_choice=request.tool_choice,
            tools=self.tools,
            stream=request.streaming,
            on_event=request.on_event,
        )
        self._history.add(request.message)
        return provider_request

    def _prepare_structured_request[ResponseModel: BaseModel](
        self,
        request: ClientRequest,
        response_model: type[ResponseModel],
        *,
        strict: bool,
    ) -> StructuredProviderRequest[ResponseModel]:
        if not isinstance(request, ClientRequest):
            raise TypeError("Expected a ClientRequest instance")

        if not isinstance(response_model, type) or not issubclass(
            response_model, BaseModel
        ):
            raise TypeError(
                f"response_model must be a subclass of BaseModel, "
                f"got {response_model!r}"
            )

        provider_request = StructuredProviderRequest(
            [*self._history.get_history(), request.message],
            response_model,
            strict=strict,
            extra=request.extra,
            tool_choice=request.tool_choice,
            tools=self.tools,
            stream=request.streaming,
            on_event=request.on_event,
        )
        self._history.add(request.message)
        return provider_request

    def create_history_checkpoint(self) -> MessageHistory:
        return MessageHistory(history=self._history.get_copy_of_history())
