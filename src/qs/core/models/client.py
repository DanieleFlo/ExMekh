from __future__ import annotations

from pydantic import BaseModel

from qs.core.models.client_request import ClientRequest
from qs.core.models.history import MessageHistory
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool_message import ToolMessage
from qs.core.models.tools import Tools
from qs.interfaces.message import Message
from qs.interfaces.provider import Provider


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

    def get_history(self) -> list[Message]:
        return self._history.get_history()

    def dump_history(self) -> list[dict[str, str]]:
        return self._history.dump_history()

    def call(self, request: ClientRequest) -> ProviderResponse:
        provider_request = self._prepare_request(request)
        response = self._provider.call(provider_request)
        self._history.add(response.response)
        return response

    def continue_after_tools(
        self,
        *,
        extra: dict | None = None,
        tool_choice: str | None = None,
    ) -> ProviderResponse:
        messages = self._get_tool_continuation_messages()
        provider_request = ProviderRequest(
            messages=messages,
            extra=extra,
            tool_choice=tool_choice,
            tools=self.tools,
        )
        response = self._provider.call(provider_request)
        self._history.add(response.response)
        return response

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
        response = self._provider.call_structured(provider_request)
        self._history.add(response.response)
        return response

    def continue_after_tools_structured[ResponseModel: BaseModel](
        self,
        response_model: type[ResponseModel],
        *,
        strict: bool = True,
        extra: dict | None = None,
        tool_choice: str | None = None,
    ) -> StructuredProviderResponse[ResponseModel]:
        messages = self._get_tool_continuation_messages()
        provider_request = StructuredProviderRequest(
            messages,
            response_model,
            strict=strict,
            extra=extra,
            tool_choice=tool_choice,
            tools=self.tools,
        )
        response = self._provider.call_structured(provider_request)
        self._history.add(response.response)
        return response

    def add_tool_response(self, tool_response: ToolMessage) -> None:
        if not isinstance(tool_response, ToolMessage):
            raise TypeError("Expected a ToolMessage instance")
        self._history.add(tool_response)

    def _get_tool_continuation_messages(self) -> list[Message]:
        messages = self._history.get_history()
        if not isinstance(messages[-1], ToolMessage):
            raise TypeError("The last message must be a tool response")
        return messages

    def _prepare_request(self, request: ClientRequest) -> ProviderRequest:
        if not isinstance(request, ClientRequest):
            raise TypeError("Expected a ClientRequest instance")

        self._history.add(request.message)
        messages = self._history.get_history()

        provider_request = ProviderRequest(
            messages=messages,
            extra=request.extra,
            tool_choice=request.tool_choice,
            tools=self.tools,
        )
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

        self._history.add(request.message)
        messages = self._history.get_history()

        provider_request = StructuredProviderRequest(
            messages,
            response_model,
            strict=strict,
            extra=request.extra,
            tool_choice=request.tool_choice,
            tools=self.tools,
        )
        return provider_request
