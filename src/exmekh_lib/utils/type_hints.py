"""Resolve required TypedDict fields without depending on annotation evaluation time."""

from typing import Annotated, Any, NotRequired, Required, get_args, get_origin


def typed_dict_required_keys(
    annotation: Any, resolved_hints: dict[str, Any]
) -> frozenset[str]:
    required: set[str] = set(getattr(annotation, "__required_keys__", ()))
    for name, hint in resolved_hints.items():
        while get_origin(hint) is Annotated:
            hint = get_args(hint)[0]
        qualifier = get_origin(hint)
        if qualifier is NotRequired:
            required.discard(name)
        elif qualifier is Required:
            required.add(name)
    return frozenset(required)
