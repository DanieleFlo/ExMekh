from __future__ import annotations

import inspect
from collections import abc
from dataclasses import MISSING, fields, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    NotRequired,
    Required,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)
from uuid import UUID

from exmekh_lib.utils.type_hints import typed_dict_required_keys

Schema = dict[str, Any]


class OpenAIChatCompletionAnnotationSchemaBuilder:
    _PRIMITIVE_SCHEMAS: ClassVar[dict[Any, Schema]] = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        NoneType: {"type": "null"},
        date: {"type": "string", "format": "date"},
        datetime: {"type": "string", "format": "date-time"},
        UUID: {"type": "string", "format": "uuid"},
        Decimal: {"type": "number"},
        Path: {"type": "string"},
    }

    _ARRAY_TYPES: ClassVar[set[Any]] = {
        list,
        set,
        frozenset,
        abc.Sequence,
        abc.Set,
    }

    _UNIQUE_ARRAY_TYPES: ClassVar[set[Any]] = {
        set,
        frozenset,
        abc.Set,
    }

    def build(
        self,
        annotation: Any,
        active_types: frozenset[Any] = frozenset(),
    ) -> Schema:
        origin = get_origin(annotation)
        arguments = get_args(annotation)

        if origin is Annotated:
            return self._annotated_schema(arguments, active_types)

        if origin in {Required, NotRequired}:
            return self.build(arguments[0], active_types)

        if self._is_unrestricted(annotation):
            return {}

        primitive = self._primitive_schema(annotation)
        if primitive is not None:
            return primitive

        if origin in {Union, UnionType}:
            return self._union_schema(arguments, active_types)

        if origin is Literal:
            return self._literal_schema(arguments, active_types)

        if self._is_enum(annotation):
            return self._enum_schema(annotation, active_types)

        collection_type = origin or annotation

        if collection_type in self._ARRAY_TYPES:
            return self._array_schema(
                collection_type,
                arguments,
                active_types,
            )

        if collection_type is tuple:
            return self._tuple_schema(arguments, active_types)

        if collection_type in {dict, abc.Mapping}:
            return self._mapping_schema(arguments, active_types)

        if is_typeddict(annotation):
            return self._typed_dict_schema(annotation, active_types)

        if self._is_dataclass(annotation):
            return self._dataclass_schema(annotation, active_types)

        supertype = self._new_type_supertype(annotation)
        if supertype is not None:
            return self.build(supertype, active_types)

        raise TypeError(f"Unsupported tool parameter type: {annotation!r}")

    def _annotated_schema(
        self,
        arguments: tuple[Any, ...],
        active_types: frozenset[Any],
    ) -> Schema:
        schema = self.build(arguments[0], active_types)

        for metadata in arguments[1:]:
            self._apply_metadata(schema, metadata)

        return schema

    @staticmethod
    def _apply_metadata(schema: Schema, metadata: Any) -> None:
        if isinstance(metadata, str):
            schema["description"] = metadata
        elif isinstance(metadata, dict):
            schema.update(metadata)

    @staticmethod
    def _is_unrestricted(annotation: Any) -> bool:
        return annotation in {Any, object} or isinstance(annotation, TypeVar)

    def _primitive_schema(self, annotation: Any) -> Schema | None:
        schema = self._PRIMITIVE_SCHEMAS.get(annotation)
        return schema.copy() if schema is not None else None

    def _union_schema(
        self,
        members: tuple[Any, ...],
        active_types: frozenset[Any],
    ) -> Schema:
        return {"anyOf": [self.build(member, active_types) for member in members]}

    def _literal_schema(
        self,
        values: tuple[Any, ...],
        active_types: frozenset[Any],
    ) -> Schema:
        return self._enum_values_schema(
            list(values),
            active_types,
        )

    def _enum_schema(
        self,
        annotation: type[Enum],
        active_types: frozenset[Any],
    ) -> Schema:
        values = [member.value for member in annotation]

        return self._enum_values_schema(
            values,
            active_types,
        )

    def _enum_values_schema(
        self,
        values: list[Any],
        active_types: frozenset[Any],
    ) -> Schema:
        schema: Schema = {"enum": values}

        value_types = {type(value) for value in values}

        if len(value_types) == 1:
            value_type = next(iter(value_types))
            schema.update(self.build(value_type, active_types))

        return schema

    def _array_schema(
        self,
        collection_type: Any,
        arguments: tuple[Any, ...],
        active_types: frozenset[Any],
    ) -> Schema:
        item_annotation = arguments[0] if arguments else Any

        schema: Schema = {
            "type": "array",
            "items": self.build(item_annotation, active_types),
        }

        if collection_type in self._UNIQUE_ARRAY_TYPES:
            schema["uniqueItems"] = True

        return schema

    def _tuple_schema(
        self,
        arguments: tuple[Any, ...],
        active_types: frozenset[Any],
    ) -> Schema:
        if not arguments:
            return {
                "type": "array",
                "items": {},
            }

        if self._is_variable_tuple(arguments):
            return {
                "type": "array",
                "items": self.build(arguments[0], active_types),
            }

        return {
            "type": "array",
            "prefixItems": [self.build(member, active_types) for member in arguments],
            "minItems": len(arguments),
            "maxItems": len(arguments),
        }

    @staticmethod
    def _is_variable_tuple(arguments: tuple[Any, ...]) -> bool:
        return len(arguments) == 2 and arguments[1] is Ellipsis

    def _mapping_schema(
        self,
        arguments: tuple[Any, ...],
        active_types: frozenset[Any],
    ) -> Schema:
        key_annotation, value_annotation = arguments if arguments else (str, Any)

        if key_annotation not in {str, Any}:
            raise TypeError("JSON object keys must be annotated as str")

        return {
            "type": "object",
            "additionalProperties": self.build(
                value_annotation,
                active_types,
            ),
        }

    def _typed_dict_schema(
        self,
        annotation: Any,
        active_types: frozenset[Any],
    ) -> Schema:
        nested_types = self._enter_type(annotation, active_types)

        hints = get_type_hints(annotation, include_extras=True)

        properties = {
            name: self.build(member, nested_types) for name, member in hints.items()
        }

        required_keys = typed_dict_required_keys(annotation, hints)

        required = [name for name in hints if name in required_keys]

        return self._object_schema(properties, required)

    def _dataclass_schema(
        self,
        annotation: Any,
        active_types: frozenset[Any],
    ) -> Schema:
        nested_types = self._enter_type(annotation, active_types)
        hints = get_type_hints(annotation, include_extras=True)

        dataclass_fields = fields(annotation)

        properties = {
            field.name: self.build(
                hints[field.name],
                nested_types,
            )
            for field in dataclass_fields
        }

        required = [
            field.name
            for field in dataclass_fields
            if field.default is MISSING and field.default_factory is MISSING
        ]

        return self._object_schema(properties, required)

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
    def _enter_type(
        annotation: Any,
        active_types: frozenset[Any],
    ) -> frozenset[Any]:
        if annotation in active_types:
            raise TypeError("Recursive tool parameter types are not supported")

        return active_types | {annotation}

    @staticmethod
    def _is_enum(annotation: Any) -> bool:
        return inspect.isclass(annotation) and issubclass(annotation, Enum)

    @staticmethod
    def _is_dataclass(annotation: Any) -> bool:
        return inspect.isclass(annotation) and is_dataclass(annotation)

    @staticmethod
    def _new_type_supertype(annotation: Any) -> Any | None:
        return getattr(annotation, "__supertype__", None)
