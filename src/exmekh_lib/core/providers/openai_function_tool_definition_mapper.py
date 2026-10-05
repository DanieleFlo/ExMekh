from __future__ import annotations

import inspect
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar

from openai.types.shared_params.function_definition import FunctionDefinition

from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.providers.openai_chat_completion_annotation_schema_builder import (
    OpenAIChatCompletionAnnotationSchemaBuilder,
    Schema,
)


@dataclass(frozen=True)
class OpenAIFunctionToolDefinition:
    _name: str
    _parameters: Schema
    _description: str

    def get_name(self) -> str:
        return self._name

    def get_parameters(self) -> Schema:
        return deepcopy(self._parameters)

    def get_description(self) -> str:
        return self._description

    def to_openai(self) -> FunctionDefinition:
        definition = FunctionDefinition(
            name=self._name,
            parameters=self.get_parameters(),
        )
        if self._description:
            definition["description"] = self._description
        return definition


class OpenAIFunctionToolDefinitionMapper:
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

    def map_all(
        self,
        tools: Tools,
        *,
        strict: bool = False,
    ) -> list[OpenAIFunctionToolDefinition]:
        if not isinstance(tools, Tools):
            raise TypeError("Expected a Tools instance")
        return [self.map(tool, strict=strict) for tool in tools.get_tools()]

    def map(
        self,
        tool: Tool,
        *,
        strict: bool = False,
    ) -> OpenAIFunctionToolDefinition:
        self._ensure_tool(tool)
        return OpenAIFunctionToolDefinition(
            _name=tool.get_name(),
            _parameters=self._build_parameters(tool, strict=strict),
            _description=tool.get_description(),
        )

    def _build_parameters(self, tool: Tool, *, strict: bool) -> Schema:
        signature = tool.get_signature()
        type_hints = tool.get_type_hints()
        properties: Schema = {}
        required: list[str] = []

        for parameter in signature.parameters.values():
            self._ensure_supported_parameter(tool, parameter)
            schema = self._schema_builder.build(
                self._parameter_annotation(parameter, type_hints)
            )
            if not strict:
                self._add_default(schema, parameter)
            elif tool.is_execution_argument(parameter.name):
                schema = {
                    "anyOf": [schema, {"type": "null"}],
                    "description": f"Null uses the default: {parameter.default}.",
                }
            properties[parameter.name] = schema
            if strict or self._is_required(parameter):
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
            f"Tool '{tool.get_name()}' has unsupported parameter '{parameter.name}'"
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
    def _object_schema(properties: Schema, required: list[str]) -> Schema:
        schema: Schema = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            schema["required"] = required
        return schema

    @staticmethod
    def _ensure_tool(tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("Expected a Tool instance")
