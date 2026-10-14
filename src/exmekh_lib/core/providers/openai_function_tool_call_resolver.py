from __future__ import annotations

import inspect
import json
from typing import Any

from exmekh_lib.config.provider_text import PROVIDER_TEXT
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.providers.openai_chat_completion_value_coercer import (
    OpenAIChatCompletionValueCoercer,
)
from exmekh_lib.core.providers.openai_function_tool_call import OpenAIFunctionToolCall


class OpenAIFunctionToolCallResolver:
    def __init__(self, coercer: OpenAIChatCompletionValueCoercer) -> None:
        self._coercer = coercer

    def find_tool(self, call: OpenAIFunctionToolCall, tools: Tools) -> Tool:
        return self._registered_tool(call.get_name(), tools)

    def resolve(
        self,
        call: OpenAIFunctionToolCall,
        tools: Tools,
    ) -> tuple[Tool, dict[str, Any]]:
        tool = self._registered_tool(call.get_name(), tools)
        arguments = tool.normalize_call_arguments(self._parse_arguments(call))
        bound_arguments = self._bind_arguments(tool, arguments)
        return tool, self._coerce_arguments(tool, bound_arguments)

    @staticmethod
    def _registered_tool(name: str, tools: Tools) -> Tool:
        tool = tools.get_tool(name)
        if tool is None:
            raise LookupError(PROVIDER_TEXT.tool_is_not_registered.format(name=name))
        return tool

    @staticmethod
    def _parse_arguments(call: OpenAIFunctionToolCall) -> dict[str, Any]:
        try:
            arguments = json.loads(call.get_arguments())
        except json.JSONDecodeError as error:
            raise ValueError(
                PROVIDER_TEXT.tool_arguments_are_not_valid_json.format(
                    call_get_name=call.get_name()
                )
            ) from error

        if not isinstance(arguments, dict):
            raise TypeError(
                PROVIDER_TEXT.tool_arguments_must_be_a_json_object.format(
                    call_get_name=call.get_name()
                )
            )
        return arguments

    @staticmethod
    def _bind_arguments(
        tool: Tool,
        arguments: dict[str, Any],
    ) -> inspect.BoundArguments:
        signature = tool.get_signature()
        try:
            return signature.bind(**arguments)
        except TypeError as error:
            raise ValueError(
                PROVIDER_TEXT.invalid_arguments_for_tool.format(
                    tool_get_name=tool.get_name(), error=error
                )
            ) from error

    def _coerce_arguments(
        self,
        tool: Tool,
        bound: inspect.BoundArguments,
    ) -> dict[str, Any]:
        signature = tool.get_signature()
        hints = tool.get_type_hints()

        return {
            name: self._coercer.coerce(
                value,
                hints.get(name, signature.parameters[name].annotation),
                path=name,
            )
            for name, value in bound.arguments.items()
        }
