from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.models.assistant_message import ToolCall
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.history import (
    HistoryMessage,
    MessageHistory,
    ToolBatchHistory,
)
from exmekh_lib.core.models.interactive import InteractiveHandler
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.interfaces.event_handler import ProviderEventHandler
from exmekh_lib.interfaces.provider import Provider, ProviderRequestRejectedError
from exmekh_lib.interfaces.tokenizer import Tokenizer

if TYPE_CHECKING:
    from exmekh_lib.core.cache.session_cache import SessionCache
    from exmekh_lib.core.models.native_suite import NativeSuiteClient

logger = logging.getLogger(__name__)


class Client:
    def __init__(
        self,
        provider: Provider,
        system_message: SystemMessage,
        tools: Tools | None = None,
        *,
        before_request: Callable[[ProviderRequest], ProviderRequest] | None = None,
    ) -> None:
        if not isinstance(system_message, SystemMessage):
            raise TypeError(MODEL_TEXT.expected_a_systemmessage_instance)

        self._history = MessageHistory(history=[system_message])
        self._history.check_history()
        self._provider = provider
        self._before_request = before_request
        self._validate_provider()

        if tools is not None and not isinstance(tools, Tools):
            raise TypeError(MODEL_TEXT.tools_must_be_a_tools_instance)
        self.tools = tools
        self.interactive_handler: InteractiveHandler | None = None

    def _validate_provider(self) -> None:
        if not isinstance(self._provider, Provider):
            raise TypeError(MODEL_TEXT.expected_a_provider_instance)

    def get_tokenizer(self) -> Tokenizer:
        return self._provider.get_tokenizer()

    def native_suite(self) -> NativeSuiteClient:
        return self._provider.native_suite()

    def generate_id(self) -> str:
        return self._provider.generate_id()

    def get_session(self) -> SessionCache:
        return self._provider.get_session()

    def get_history(self) -> list[HistoryMessage]:
        return self._history.get_history()

    def dump_history(self) -> list[dict[str, Any]]:
        return self._history.dump_history()

    def copy_history(self) -> list[HistoryMessage]:
        return self._history.get_copy_of_history()

    def get_latest_tool_batch(self) -> ToolBatchHistory:
        return self._history.get_latest_tool_batch()

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
        try:
            return self._call_provider(self._prepare_request(request))
        except ProviderRequestRejectedError:
            self.remove_rejected_input([request.message])
            raise

    def remove_rejected_input(self, messages: Sequence[HistoryMessage]) -> None:
        """Roll back admitted inputs even when a request hook compacted their prefix."""
        self._history.discard_latest(messages)

    def _call_provider(self, request: ProviderRequest) -> ProviderResponse:
        if self._before_request is not None:
            request = self._before_request(request)
        logger.info(
            MODEL_TEXT.calling_provider_messages_tool_choice,
            len(request.get_messages()),
            request.get_tool_choice(),
        )
        request.interactive_handler = self.interactive_handler
        response = self._provider.call(request)
        if not isinstance(response, ProviderResponse):
            raise TypeError(MODEL_TEXT.item_must_be_an_instance_of_message_or_a)
        messages = response.history_messages()
        if messages:
            self._history.add(messages)
        logger.info(
            MODEL_TEXT.provider_response_recorded_tool_calls_history_size,
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
        messages = self._history.get_tool_continuation_messages()
        provider_request = ProviderRequest(
            messages=messages,
            extra=extra,
            tool_choice=tool_choice,
            tools=self.tools,
            stream=streaming,
            on_event=on_event,
        )
        return self._call_provider(provider_request)

    def continue_after_interruption(
        self,
        *,
        extra: Mapping[str, object],
        tool_choice: str | None,
        streaming: bool,
        on_event: ProviderEventHandler,
    ) -> ProviderResponse:
        return self._call_provider(
            ProviderRequest(
                self._history.get_history(),
                extra=extra,
                tools=self.tools,
                tool_choice=tool_choice,
                stream=streaming,
                on_event=on_event,
            )
        )

    def continue_with_user_messages(
        self,
        messages: Sequence[UserMessage],
        *,
        extra: Mapping[str, object] | None = None,
        tool_choice: str | None = None,
        streaming: bool = False,
        on_event: ProviderEventHandler | None = None,
        initial_message: UserMessage | None = None,
    ) -> ProviderResponse:
        """Append steering at a completed batch boundary and resume inference.

        An optional initial turn message precedes steering without emitting an event.
        """
        messages = tuple(messages)
        if initial_message is not None and not isinstance(initial_message, UserMessage):
            raise TypeError(MODEL_TEXT.initial_message_must_be_a_usermessage)
        if (not messages and initial_message is None) or not all(
            isinstance(message, UserMessage) for message in messages
        ):
            raise ValueError(RUNTIME.invalid_steering_messages)
        added_messages: list[HistoryMessage] = [*messages]
        if initial_message is not None:
            added_messages.insert(0, initial_message)
        provider_request = ProviderRequest(
            messages=[*self._history.get_history(), *added_messages],
            extra=extra,
            tool_choice=tool_choice,
            tools=self.tools,
            stream=streaming,
            on_event=on_event,
        )
        self._history.add(added_messages)
        try:
            provider_request.notify_steering_added(messages)
            return self._call_provider(provider_request)
        except ProviderRequestRejectedError:
            self.remove_rejected_input(added_messages)
            raise

    def complete_tool_transition(self, *, extra: Mapping[str, object]) -> None:
        """Complete a provider transition before executing an already queued batch.

        The next batch supersedes this response, so it is neither streamed nor
        retained in the conversation.
        """
        response = self.continue_after_tools(extra=extra, tool_choice="none")
        appended = len(response.history_messages())
        if appended:
            self._history.replace_from(len(self._history.get_history()) - appended, [])

    def call_structured[ResponseModel: BaseModel](
        self,
        request: ClientRequest,
        response_model: type[ResponseModel],
        *,
        strict: bool = True,
    ) -> StructuredProviderResponse[ResponseModel]:
        try:
            provider_request = self._prepare_structured_request(
                request,
                response_model,
                strict=strict,
            )
            return self._call_structured_provider(provider_request)
        except ProviderRequestRejectedError:
            self.remove_rejected_input([request.message])
            raise

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
        messages = self._history.get_tool_continuation_messages()
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
            raise TypeError(MODEL_TEXT.expected_a_toolmessage_instance)
        self._history.add(tool_response)
        logger.debug(
            MODEL_TEXT.tool_response_recorded_name_call_id_history_size,
            tool_response.get_name(),
            tool_response.get_tool_call_id(),
            len(self._history.get_history()),
        )

    def add_user_message(self, message: HistoryMessage) -> None:
        if message.get_role_name() != "user":
            raise TypeError(MODEL_TEXT.expected_a_user_message)
        self._history.add(message)

    def _prepare_request(self, request: ClientRequest) -> ProviderRequest:
        if not isinstance(request, ClientRequest):
            raise TypeError(MODEL_TEXT.expected_a_clientrequest_instance)

        provider_request = ProviderRequest(
            messages=[*self._history.get_history(), request.message],
            extra=request.extra,
            tool_choice=request.tool_choice,
            tools=self.tools,
            stream=request.streaming,
            on_event=request.on_event,
            reasoning_effort_override=request.reasoning_effort_override,
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
            raise TypeError(MODEL_TEXT.expected_a_clientrequest_instance)

        if not isinstance(response_model, type) or not issubclass(
            response_model, BaseModel
        ):
            raise TypeError(
                MODEL_TEXT.response_model_must_be_a_subclass_of_basemodel_got.format(
                    response_model=response_model
                )
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
            reasoning_effort_override=request.reasoning_effort_override,
        )
        self._history.add(request.message)
        return provider_request

    def create_history_checkpoint(self) -> MessageHistory:
        return MessageHistory(history=self._history.get_copy_of_history())
