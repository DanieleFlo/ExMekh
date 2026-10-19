from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass, field
from typing import Any

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.config.capability_prompts import CAPABILITY_PROMPTS


@dataclass(frozen=True, slots=True)
class CapabilityMetadata:
    description: str
    sub_skills: tuple[str, ...]
    sub_tools: tuple[str, ...]
    optional_sub_skills: tuple[str, ...] = field(default=(), kw_only=True)
    optional_sub_tools: tuple[str, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        overlap = (set(self.sub_tools) & set(self.optional_sub_tools)) | (
            set(self.sub_skills) & set(self.optional_sub_skills)
        )
        if overlap:
            raise ValueError(
                CAPABILITY_PROMPTS.optional_overlap.format(
                    names=", ".join(sorted(overlap))
                )
            )

    def get_description(self) -> str:
        return self.description

    def get_sub_skill_names(self) -> tuple[str, ...]:
        return self.sub_skills

    def get_sub_tool_names(self) -> tuple[str, ...]:
        return self.sub_tools

    def get_optional_sub_skill_names(self) -> tuple[str, ...]:
        return self.optional_sub_skills

    def get_optional_sub_tool_names(self) -> tuple[str, ...]:
        return self.optional_sub_tools

    def resolve_sub_skill_names(self, available: Set[str]) -> tuple[str, ...]:
        return self.get_sub_skill_names() + tuple(
            name for name in self.get_optional_sub_skill_names() if name in available
        )

    def resolve_sub_tool_names(self, available: Set[str]) -> tuple[str, ...]:
        return self.get_sub_tool_names() + tuple(
            name for name in self.get_optional_sub_tool_names() if name in available
        )

    def get_direct_tool_names(self) -> set[str]:
        return {*self.get_sub_skill_names(), *self.get_sub_tool_names()}


@dataclass(frozen=True, slots=True)
class MetadataFieldSpec:
    key: str
    required: bool = False


class CapabilityMetadataParser:
    DESCRIPTION_FIELD = MetadataFieldSpec(
        CAPABILITY_FORMAT.description_field, required=True
    )
    SUB_SKILLS_FIELD = MetadataFieldSpec(CAPABILITY_FORMAT.sub_skills_field)
    SUB_TOOLS_FIELD = MetadataFieldSpec(CAPABILITY_FORMAT.sub_tools_field)
    OPTIONAL_SUB_SKILLS_FIELD = MetadataFieldSpec(
        CAPABILITY_FORMAT.optional_sub_skills_field
    )
    OPTIONAL_SUB_TOOLS_FIELD = MetadataFieldSpec(
        CAPABILITY_FORMAT.optional_sub_tools_field
    )

    @property
    def field_specs(self) -> tuple[MetadataFieldSpec, ...]:
        return (
            self.DESCRIPTION_FIELD,
            self.SUB_SKILLS_FIELD,
            self.SUB_TOOLS_FIELD,
            self.OPTIONAL_SUB_SKILLS_FIELD,
            self.OPTIONAL_SUB_TOOLS_FIELD,
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
            optional_sub_skills=self._parse_string_collection(
                data, self.OPTIONAL_SUB_SKILLS_FIELD
            ),
            optional_sub_tools=self._parse_string_collection(
                data, self.OPTIONAL_SUB_TOOLS_FIELD
            ),
        )

    @staticmethod
    def _validate_input(data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise TypeError(AGENT_TEXT.data_must_be_a_dictionary)

        if not all(isinstance(key, str) for key in data):
            raise ValueError(AGENT_TEXT.metadata_field_names_must_be_strings)

    def _validate_fields(self, data: dict[str, Any]) -> None:
        allowed_fields = {field.key for field in self.field_specs}
        unknown_fields = set(data) - allowed_fields

        if unknown_fields:
            fields = ", ".join(sorted(unknown_fields))
            raise ValueError(AGENT_TEXT.unknown_metadata_fields.format(fields=fields))

        required_fields = {field.key for field in self.field_specs if field.required}
        missing_fields = required_fields - set(data)

        if missing_fields:
            fields = ", ".join(sorted(missing_fields))
            raise ValueError(
                AGENT_TEXT.missing_required_metadata_fields.format(fields=fields)
            )

    @staticmethod
    def _parse_string(
        data: dict[str, Any],
        field: MetadataFieldSpec,
    ) -> str:
        value = data.get(field.key)

        if not isinstance(value, str):
            raise TypeError(
                AGENT_TEXT.metadata_field_must_be_a_string.format(field_key=field.key)
            )

        value = value.strip()

        if not value:
            raise ValueError(
                AGENT_TEXT.metadata_field_cannot_be_empty.format(field_key=field.key)
            )

        return value

    @staticmethod
    def _parse_optional_string(
        data: dict[str, Any],
        field: MetadataFieldSpec,
    ) -> str | None:
        if field.key not in data:
            return None

        value = data.get(field.key)
        if not isinstance(value, str):
            raise TypeError(
                AGENT_TEXT.metadata_field_must_be_a_string.format(field_key=field.key)
            )

        value = value.strip()
        if not value:
            raise ValueError(
                AGENT_TEXT.metadata_field_cannot_be_empty.format(field_key=field.key)
            )

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
            raise TypeError(
                AGENT_TEXT.metadata_field_must_be_a_boolean.format(field_key=field.key)
            )

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
            AGENT_TEXT.metadata_field_must_be_a_string_or_a_list.format(
                field_key=field.key
            )
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
                AGENT_TEXT.metadata_field_cannot_contain_duplicate_values.format(
                    field_key=field.key
                )
            )

        return normalized_values

    @staticmethod
    def _normalize_collection_item(
        field: MetadataFieldSpec,
        value: Any,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                AGENT_TEXT.metadata_field_must_contain_only_strings.format(
                    field_key=field.key
                )
            )

        value = value.strip()

        if not value:
            raise ValueError(
                AGENT_TEXT.metadata_field_cannot_contain_empty_values.format(
                    field_key=field.key
                )
            )

        return value
