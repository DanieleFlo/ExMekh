from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from qs.core.providers.openai_json_schema_mapper import OpenAIJSONSchemaMapper


class OpenAIResponseFormatMapper:
    @staticmethod
    def map[ResponseModel: BaseModel](
        response_model: type[ResponseModel],
        *,
        strict: bool = True,
    ) -> dict[str, Any]:
        return {
            "format": {
                "type": "json_schema",
                **OpenAIJSONSchemaMapper.map(response_model, strict=strict),
            }
        }
