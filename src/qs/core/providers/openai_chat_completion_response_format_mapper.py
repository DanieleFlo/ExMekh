from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel


class OpenAIChatCompletionResponseFormatMapper:
    @classmethod
    def map[ResponseModel: BaseModel](
        cls,
        response_model: type[ResponseModel],
        *,
        strict: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(response_model, type) or not issubclass(
            response_model, BaseModel
        ):
            raise TypeError("Response model must be a Pydantic BaseModel type")
        if not isinstance(strict, bool):
            raise TypeError("Strict must be a boolean")

        schema = response_model.model_json_schema()
        if strict:
            schema = cls._make_strict(deepcopy(schema), schema)

        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
                "strict": strict,
            },
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
            resolved = deepcopy(cls._resolve_reference(root, reference))
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
    def _resolve_reference(root: dict[str, Any], reference: str) -> dict[str, Any]:
        if not reference.startswith("#/"):
            raise ValueError(f"Unsupported JSON schema reference: {reference}")

        resolved: Any = root
        for part in reference[2:].split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(resolved, dict) or key not in resolved:
                raise ValueError(f"Unresolvable JSON schema reference: {reference}")
            resolved = resolved[key]

        if not isinstance(resolved, dict):
            raise TypeError(f"JSON schema reference is not an object: {reference}")
        return resolved
