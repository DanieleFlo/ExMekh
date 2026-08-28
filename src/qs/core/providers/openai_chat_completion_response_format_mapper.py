from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from qs.core.providers.openai_json_schema_mapper import OpenAIJSONSchemaMapper


class OpenAIChatCompletionResponseFormatMapper:
    @classmethod
    def map[ResponseModel: BaseModel](
        cls,
        response_model: type[ResponseModel],
        *,
        strict: bool = True,
    ) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": OpenAIJSONSchemaMapper.map(
                response_model,
                strict=strict,
            ),
        }

    @staticmethod
    def _resolve_reference(root: dict[str, Any], reference: str) -> dict[str, Any]:
        return OpenAIJSONSchemaMapper.resolve_reference(root, reference)
