from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel

from exmekh_lib.config.provider_text import PROVIDER_TEXT


class OpenAIJSONSchemaMapper:
    @classmethod
    def map[ResponseModel: BaseModel](
        cls,
        response_model: type[ResponseModel],
        *,
        strict: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(response_model, type) or not issubclass(
            response_model,
            BaseModel,
        ):
            raise TypeError(
                PROVIDER_TEXT.response_model_must_be_a_pydantic_basemodel_type
            )
        if not isinstance(strict, bool):
            raise TypeError(PROVIDER_TEXT.strict_must_be_a_boolean)

        schema = response_model.model_json_schema()
        if strict:
            schema = cls._make_strict(deepcopy(schema), schema)

        return {
            "name": response_model.__name__,
            "schema": schema,
            "strict": strict,
        }

    @classmethod
    def _make_strict(
        cls,
        value: Any,
        root: dict[str, Any],
    ) -> Any:
        if isinstance(value, list):
            return [cls._make_strict(item, root) for item in value]
        if not isinstance(value, dict):
            return value

        reference = value.get("$ref")
        if isinstance(reference, str) and len(value) > 1:
            resolved = deepcopy(cls.resolve_reference(root, reference))
            value = {**resolved, **value}
            value.pop("$ref")

        if value.get("type") == "object":
            value.setdefault("additionalProperties", False)
            properties = value.get("properties")
            if isinstance(properties, dict):
                value["required"] = list(properties)

        if value.get("default") is None:
            value.pop("default", None)

        return {key: cls._make_strict(child, root) for key, child in value.items()}

    @staticmethod
    def resolve_reference(root: dict[str, Any], reference: str) -> dict[str, Any]:
        if not reference.startswith("#/"):
            raise ValueError(
                PROVIDER_TEXT.unsupported_json_schema_reference.format(
                    reference=reference
                )
            )

        resolved: Any = root
        for part in reference[2:].split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(resolved, dict) or key not in resolved:
                raise ValueError(
                    PROVIDER_TEXT.unresolvable_json_schema_reference.format(
                        reference=reference
                    )
                )
            resolved = resolved[key]

        if not isinstance(resolved, dict):
            raise TypeError(
                PROVIDER_TEXT.json_schema_reference_is_not_an_object.format(
                    reference=reference
                )
            )
        return resolved
