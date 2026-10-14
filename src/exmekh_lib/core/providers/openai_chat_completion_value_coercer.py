from __future__ import annotations

import inspect
from collections import abc
from dataclasses import is_dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
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

from exmekh_lib.config.provider_text import PROVIDER_TEXT
from exmekh_lib.utils.type_hints import typed_dict_required_keys


class OpenAIChatCompletionValueCoercer:
    _ARRAY_TYPES: ClassVar[set[Any]] = {
        list,
        set,
        frozenset,
        abc.Sequence,
        abc.Set,
    }

    def coerce(
        self,
        value: Any,
        annotation: Any,
        path: str,
    ) -> Any:
        origin = get_origin(annotation)
        arguments = get_args(annotation)

        if origin is Annotated:
            return self.coerce(value, arguments[0], path)

        if origin in {Required, NotRequired}:
            return self.coerce(value, arguments[0], path)

        if self._is_unrestricted(annotation):
            return value

        if origin in {Union, UnionType}:
            return self._coerce_union(
                value,
                arguments,
                path,
            )

        if origin is Literal:
            return self._coerce_literal(
                value,
                arguments,
                path,
            )

        if annotation is NoneType:
            return self._require_none(value, path)

        if annotation is bool:
            return self._require_bool(value, path)

        if annotation is int:
            return self._require_int(value, path)

        if annotation is float:
            return self._require_float(value, path)

        if annotation is str:
            return self._require_str(value, path)

        if self._is_enum(annotation):
            return self._coerce_enum(
                value,
                annotation,
                path,
            )

        if annotation in {date, datetime, UUID, Decimal, Path}:
            return self._coerce_scalar(
                value,
                annotation,
                path,
            )

        collection_type = origin or annotation

        if collection_type in self._ARRAY_TYPES:
            return self._coerce_array(
                value,
                collection_type,
                arguments,
                path,
            )

        if collection_type is tuple:
            return self._coerce_tuple(
                value,
                arguments,
                path,
            )

        if is_typeddict(annotation):
            return self._coerce_typed_dict(
                value,
                annotation,
                path,
            )

        if collection_type in {dict, abc.Mapping}:
            return self._coerce_mapping(
                value,
                arguments,
                path,
            )

        if self._is_dataclass(annotation):
            return self._coerce_dataclass(
                value,
                annotation,
                path,
            )

        supertype = getattr(annotation, "__supertype__", None)

        if supertype is not None:
            return self.coerce(value, supertype, path)

        raise TypeError(
            PROVIDER_TEXT.unsupported_type_for_argument.format(
                path=path, annotation=annotation
            )
        )

    def _coerce_union(
        self,
        value: Any,
        members: tuple[Any, ...],
        path: str,
    ) -> Any:
        for member in members:
            try:
                return self.coerce(value, member, path)
            except TypeError:
                pass

        raise TypeError(
            PROVIDER_TEXT.argument_does_not_match_any_allowed_type.format(path=path)
        )

    @staticmethod
    def _coerce_literal(
        value: Any,
        allowed: tuple[Any, ...],
        path: str,
    ) -> Any:
        matches = any(value == item and type(value) is type(item) for item in allowed)

        if not matches:
            raise TypeError(
                PROVIDER_TEXT.argument_is_not_an_allowed_literal.format(path=path)
            )

        return value

    @staticmethod
    def _require_none(value: Any, path: str) -> None:
        if value is not None:
            raise TypeError(PROVIDER_TEXT.argument_must_be_null.format(path=path))

    @staticmethod
    def _require_bool(value: Any, path: str) -> bool:
        if type(value) is not bool:
            raise TypeError(PROVIDER_TEXT.argument_must_be_a_boolean.format(path=path))

        return value

    @staticmethod
    def _require_int(value: Any, path: str) -> int:
        if type(value) is not int:
            raise TypeError(PROVIDER_TEXT.argument_must_be_an_integer.format(path=path))

        return value

    @staticmethod
    def _require_float(value: Any, path: str) -> float:
        if type(value) not in {int, float}:
            raise TypeError(PROVIDER_TEXT.argument_must_be_a_number.format(path=path))

        return float(value)

    @staticmethod
    def _require_str(value: Any, path: str) -> str:
        if not isinstance(value, str):
            raise TypeError(PROVIDER_TEXT.argument_must_be_a_string.format(path=path))

        return value

    @staticmethod
    def _coerce_enum(
        value: Any,
        annotation: type[Enum],
        path: str,
    ) -> Enum:
        try:
            return annotation(value)
        except ValueError as error:
            raise TypeError(
                PROVIDER_TEXT.argument_is_not_a_valid_enum_value.format(path=path)
            ) from error

    @staticmethod
    def _coerce_scalar(
        value: Any,
        annotation: Any,
        path: str,
    ) -> Any:
        try:
            if annotation is date:
                return date.fromisoformat(value)

            if annotation is datetime:
                return datetime.fromisoformat(value)

            if annotation is UUID:
                return UUID(value)

            if annotation is Decimal:
                return Decimal(value)

            if annotation is Path:
                return Path(value)

        except (InvalidOperation, TypeError, ValueError) as error:
            raise TypeError(
                PROVIDER_TEXT.argument_has_an_invalid_value.format(path=path)
            ) from error

        raise AssertionError(
            PROVIDER_TEXT.unsupported_scalar_annotation
        )  # pragma: no cover

    def _coerce_array(
        self,
        value: Any,
        collection_type: Any,
        arguments: tuple[Any, ...],
        path: str,
    ) -> Any:
        self._require_array(value, path)

        item_annotation = arguments[0] if arguments else Any

        items = [
            self.coerce(
                item,
                item_annotation,
                f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]

        if collection_type in {set, abc.Set}:
            return set(items)

        if collection_type is frozenset:
            return frozenset(items)

        return items

    def _coerce_tuple(
        self,
        value: Any,
        arguments: tuple[Any, ...],
        path: str,
    ) -> tuple[Any, ...]:
        self._require_array(value, path)

        if not arguments:
            return tuple(value)

        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                self.coerce(
                    item,
                    arguments[0],
                    f"{path}[{index}]",
                )
                for index, item in enumerate(value)
            )

        if len(value) != len(arguments):
            raise TypeError(
                PROVIDER_TEXT.argument_has_the_wrong_array_length.format(path=path)
            )

        return tuple(
            self.coerce(
                item,
                annotation,
                f"{path}[{index}]",
            )
            for index, (item, annotation) in enumerate(
                zip(value, arguments, strict=True)
            )
        )

    def _coerce_mapping(
        self,
        value: Any,
        arguments: tuple[Any, ...],
        path: str,
    ) -> dict[str, Any]:
        self._require_object(value, path)

        value_annotation = arguments[1] if arguments else Any

        return {
            key: self.coerce(
                item,
                value_annotation,
                f"{path}.{key}",
            )
            for key, item in value.items()
        }

    def _coerce_typed_dict(
        self,
        value: Any,
        annotation: Any,
        path: str,
    ) -> dict[str, Any]:
        self._require_object(value, path)

        hints = get_type_hints(
            annotation,
            include_extras=True,
        )

        required_keys = typed_dict_required_keys(annotation, hints)

        self._ensure_known_fields(
            value,
            hints,
            path,
        )

        self._ensure_required_fields(
            value,
            required_keys,
            path,
        )

        return {
            key: self.coerce(
                item,
                hints[key],
                f"{path}.{key}",
            )
            for key, item in value.items()
        }

    def _coerce_dataclass(
        self,
        value: Any,
        annotation: Any,
        path: str,
    ) -> Any:
        self._require_object(value, path)

        hints = get_type_hints(
            annotation,
            include_extras=True,
        )

        self._ensure_known_fields(
            value,
            hints,
            path,
        )

        converted = {
            key: self.coerce(
                item,
                hints[key],
                f"{path}.{key}",
            )
            for key, item in value.items()
        }

        try:
            return annotation(**converted)
        except TypeError as error:
            raise TypeError(
                PROVIDER_TEXT.argument_is_invalid.format(path=path)
            ) from error

    @staticmethod
    def _ensure_known_fields(
        value: dict[str, Any],
        hints: dict[str, Any],
        path: str,
    ) -> None:
        unknown = value.keys() - hints.keys()

        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(
                PROVIDER_TEXT.argument_has_unknown_fields.format(path=path, names=names)
            )

    @staticmethod
    def _ensure_required_fields(
        value: dict[str, Any],
        required: abc.Set[str],
        path: str,
    ) -> None:
        missing = required - value.keys()

        if missing:
            names = ", ".join(sorted(missing))
            raise TypeError(
                PROVIDER_TEXT.argument_is_missing_fields.format(path=path, names=names)
            )

    @staticmethod
    def _require_array(value: Any, path: str) -> None:
        if not isinstance(value, list):
            raise TypeError(PROVIDER_TEXT.argument_must_be_an_array.format(path=path))

    @staticmethod
    def _require_object(value: Any, path: str) -> None:
        if not isinstance(value, dict):
            raise TypeError(PROVIDER_TEXT.argument_must_be_an_object.format(path=path))

    @staticmethod
    def _is_unrestricted(annotation: Any) -> bool:
        return annotation in {Any, object} or isinstance(annotation, TypeVar)

    @staticmethod
    def _is_enum(annotation: Any) -> bool:
        return inspect.isclass(annotation) and issubclass(annotation, Enum)

    @staticmethod
    def _is_dataclass(annotation: Any) -> bool:
        return inspect.isclass(annotation) and is_dataclass(annotation)
