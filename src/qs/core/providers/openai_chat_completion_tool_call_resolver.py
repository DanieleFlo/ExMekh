from __future__ import annotations

import inspect
import json
from typing import Any, get_type_hints

from qs.core.models.tool import Tool
from qs.core.models.tools import Tools
from qs.core.providers.openai_chat_completion_tool_call_data import (
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallData,
)
from qs.core.providers.openai_chat_completion_tool_call_reader import (
    OpenAIChatCompletionToolCallReader,
)
from qs.core.providers.openai_chat_completion_value_coercer import (
    OpenAIChatCompletionValueCoercer,
)


class OpenAIChatCompletionToolCallResolver:
    def __init__(self, coercer: OpenAIChatCompletionValueCoercer) -> None:
        self._coercer = coercer
        self._reader = OpenAIChatCompletionToolCallReader()

    def find_tool(
        self,
        tool_call: OpenAIChatCompletionToolCall,
        tools: Tools,
    ) -> Tool:
        call = self._reader.read(tool_call)
        return self._registered_tool(call.name, tools)

    def resolve(
        self,
        tool_call: OpenAIChatCompletionToolCall,
        tools: Tools,
    ) -> tuple[Tool, dict[str, Any]]:
        call = self._reader.read(tool_call)
        tool = self._registered_tool(call.name, tools)

        arguments = self._parse_arguments(call)
        bound_arguments = self._bind_arguments(tool, arguments)

        return tool, self._coerce_arguments(tool, bound_arguments)

    @staticmethod
    def _registered_tool(name: str, tools: Tools) -> Tool:
        tool = tools.get_tool(name)

        if tool is None:
            raise LookupError(f"Tool '{name}' is not registered")

        return tool

    @staticmethod
    def _parse_arguments(
        call: OpenAIChatCompletionToolCallData,
    ) -> dict[str, Any]:
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Tool '{call.name}' arguments are not valid JSON"
            ) from error

        if not isinstance(arguments, dict):
            raise TypeError(f"Tool '{call.name}' arguments must be a JSON object")

        return arguments

    @staticmethod
    def _bind_arguments(
        tool: Tool,
        arguments: dict[str, Any],
    ) -> inspect.BoundArguments:
        signature = inspect.signature(tool.tool)

        try:
            return signature.bind(**arguments)
        except TypeError as error:
            raise ValueError(
                f"Invalid arguments for tool '{tool.name}': {error}"
            ) from error

    def _coerce_arguments(
        self,
        tool: Tool,
        bound: inspect.BoundArguments,
    ) -> dict[str, Any]:
        signature = inspect.signature(tool.tool)
        hints = get_type_hints(tool.tool, include_extras=True)

        result: dict[str, Any] = {}

        for name, value in bound.arguments.items():
            annotation = hints.get(
                name,
                signature.parameters[name].annotation,
            )

            result[name] = self._coercer.coerce(
                value,
                annotation,
                path=name,
            )

        return result
