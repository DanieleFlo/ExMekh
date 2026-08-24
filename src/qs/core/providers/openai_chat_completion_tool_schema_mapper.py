from __future__ import annotations

import inspect
import json
from typing import Any, ClassVar, get_type_hints

from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition

from qs.core.models.tool import Tool
from qs.core.models.tools import Tools
from qs.core.providers.openai_chat_completion_annotation_schema_builder import (
    OpenAIChatCompletionAnnotationSchemaBuilder,
    Schema,
)


class OpenAIChatCompletionToolSchemaMapper:
    _UNSUPPORTED_PARAMETER_KINDS: ClassVar[set[Any]] = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    }

    def __init__(
        self,
        schema_builder: OpenAIChatCompletionAnnotationSchemaBuilder,
    ) -> None:
        self._schema_builder = schema_builder

    def map(self, tool: Tool) -> ChatCompletionFunctionToolParam:
        self._ensure_tool(tool)

        parameters = self._build_parameters(tool)
        function = self._build_function_definition(tool, parameters)

        return ChatCompletionFunctionToolParam(
            type="function",
            function=function,
        )

    def map_all(self, tools: Tools) -> list[ChatCompletionFunctionToolParam]:
        if not isinstance(tools, Tools):
            raise TypeError("Expected a Tools instance")

        return [self.map(tool) for tool in tools.get_tools()]

    def _build_parameters(self, tool: Tool) -> Schema:
        signature = inspect.signature(tool.tool)
        type_hints = get_type_hints(tool.tool, include_extras=True)

        properties: Schema = {}
        required: list[str] = []

        for parameter in signature.parameters.values():
            self._ensure_supported_parameter(tool, parameter)

            annotation = self._parameter_annotation(parameter, type_hints)
            schema = self._schema_builder.build(annotation)

            self._add_default(schema, parameter)

            properties[parameter.name] = schema

            if self._is_required(parameter):
                required.append(parameter.name)

        return self._object_schema(properties, required)

    @staticmethod
    def _parameter_annotation(
        parameter: inspect.Parameter,
        type_hints: dict[str, Any],
    ) -> Any:
        annotation = type_hints.get(parameter.name, parameter.annotation)

        if annotation is inspect.Parameter.empty:
            raise TypeError(
                f"Tool parameter '{parameter.name}' must have a type annotation"
            )

        return annotation

    @classmethod
    def _ensure_supported_parameter(
        cls,
        tool: Tool,
        parameter: inspect.Parameter,
    ) -> None:
        if parameter.kind not in cls._UNSUPPORTED_PARAMETER_KINDS:
            return

        raise TypeError(
            f"Tool '{tool.name}' has unsupported parameter '{parameter.name}'"
        )

    @staticmethod
    def _is_required(parameter: inspect.Parameter) -> bool:
        return parameter.default is inspect.Parameter.empty

    @staticmethod
    def _add_default(schema: Schema, parameter: inspect.Parameter) -> None:
        if parameter.default is inspect.Parameter.empty:
            return

        try:
            json.dumps(parameter.default)
        except (TypeError, ValueError):
            return

        schema["default"] = parameter.default

    @staticmethod
    def _object_schema(
        properties: Schema,
        required: list[str],
    ) -> Schema:
        schema: Schema = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }

        if required:
            schema["required"] = required

        return schema

    @staticmethod
    def _build_function_definition(
        tool: Tool,
        parameters: Schema,
    ) -> FunctionDefinition:
        definition = FunctionDefinition(
            name=tool.name,
            parameters=parameters,
        )

        if tool.description:
            definition["description"] = tool.description

        return definition

    @staticmethod
    def _ensure_tool(tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("Expected a Tool instance")
