from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CapabilityMetadata:
    description: str
    sub_skills: tuple[str, ...]
    sub_tools: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MetadataFieldSpec:
    key: str
    required: bool = False


class CapabilityMetadataParser:
    DESCRIPTION_FIELD = MetadataFieldSpec("description", required=True)
    SUB_SKILLS_FIELD = MetadataFieldSpec("sub-skills")
    SUB_TOOLS_FIELD = MetadataFieldSpec("sub-tools")

    @property
    def field_specs(self) -> tuple[MetadataFieldSpec, ...]:
        return (
            self.DESCRIPTION_FIELD,
            self.SUB_SKILLS_FIELD,
            self.SUB_TOOLS_FIELD,
        )

    def parse(self, data: dict[str, Any]) -> CapabilityMetadata:
        self._validate_input(data)
        self._validate_fields(data)
        return self._build_metadata(data)

    def _build_metadata(self, data: dict[str, Any]) -> CapabilityMetadata:
        return CapabilityMetadata(
            description=self._parse_string(data, self.DESCRIPTION_FIELD),
            sub_skills=self._parse_string_collection(data, self.SUB_SKILLS_FIELD),
            sub_tools=self._parse_string_collection(data, self.SUB_TOOLS_FIELD),
        )

    @staticmethod
    def _validate_input(data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")

        if not all(isinstance(key, str) for key in data):
            raise ValueError("Metadata field names must be strings")

    def _validate_fields(self, data: dict[str, Any]) -> None:
        allowed_fields = {field.key for field in self.field_specs}
        unknown_fields = set(data) - allowed_fields

        if unknown_fields:
            fields = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Unknown metadata fields: {fields}")

        required_fields = {field.key for field in self.field_specs if field.required}
        missing_fields = required_fields - set(data)

        if missing_fields:
            fields = ", ".join(sorted(missing_fields))
            raise ValueError(f"Missing required metadata fields: {fields}")

    @staticmethod
    def _parse_string(
        data: dict[str, Any],
        field: MetadataFieldSpec,
    ) -> str:
        value = data.get(field.key)

        if not isinstance(value, str):
            raise TypeError(f"Metadata field '{field.key}' must be a string.")

        value = value.strip()

        if not value:
            raise ValueError(f"Metadata field '{field.key}' cannot be empty.")

        return value

    @staticmethod
    def _parse_boolean(
        data: dict[str, Any],
        field: MetadataFieldSpec,
        *,
        default: bool,
    ) -> bool:
        value = data.get(field.key, default)

        if not isinstance(value, bool):
            raise TypeError(f"Metadata field '{field.key}' must be a boolean.")

        return value

    def _parse_string_collection(
        self,
        data: dict[str, Any],
        field: MetadataFieldSpec,
    ) -> tuple[str, ...]:
        value = data.get(field.key)

        if value is None:
            return ()

        if isinstance(value, str):
            return self._normalize_string_collection(field, [value])

        if isinstance(value, list):
            return self._normalize_string_collection(field, value)

        raise TypeError(
            f"Metadata field '{field.key}' must be a string or a list of strings."
        )

    def _normalize_string_collection(
        self,
        field: MetadataFieldSpec,
        values: list[Any],
    ) -> tuple[str, ...]:
        normalized_values = tuple(
            self._normalize_collection_item(field, value) for value in values
        )

        if len(normalized_values) != len(set(normalized_values)):
            raise ValueError(
                f"Metadata field '{field.key}' cannot contain duplicate values."
            )

        return normalized_values

    @staticmethod
    def _normalize_collection_item(
        field: MetadataFieldSpec,
        value: Any,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(f"Metadata field '{field.key}' must contain only strings.")

        value = value.strip()

        if not value:
            raise ValueError(
                f"Metadata field '{field.key}' cannot contain empty values."
            )

        return value
