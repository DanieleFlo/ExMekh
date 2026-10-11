"""Validate untrusted interactive IDs, frozen arguments and schema values."""

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from exmekh_lib.core.models.assistant_message import FunctionCall, ToolCall
from exmekh_lib.core.models.interactive import InteractiveProtocolError


def decode_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InteractiveProtocolError(f"Duplicate JSON property: {key}")
        result[key] = value
    return result


class InteractiveContract:
    """Own the authorization state of one logical call, independently of HTTP."""

    def __init__(
        self,
        definitions: dict[str, dict[str, Any]],
        schemas: dict[str, dict[str, Any]],
    ) -> None:
        self._definitions = deepcopy(definitions)
        self._schemas = deepcopy(schemas)
        self._continuation_id: str | None = None
        self._frozen: dict[str, tuple[str, dict[str, Any]]] = {}
        self._exchanges: set[str] = set()
        self._rejected_call_id: str | None = None

    @property
    def is_accepted(self) -> bool:
        return self._continuation_id is not None

    @property
    def continuation_id(self) -> str:
        if self._continuation_id is None:
            raise InteractiveProtocolError("No interactive continuation was accepted")
        return self._continuation_id

    def definition(self, tool_name: str) -> dict[str, Any]:
        return deepcopy(self._definitions[tool_name])

    def record_reply(self, event: dict[str, Any], reply: Mapping[str, object]) -> None:
        if reply.get("status") == "reject":
            self._rejected_call_id = event["call_id"]

    def validate_next_result(self, result: dict[str, Any]) -> None:
        if (
            self._rejected_call_id is not None
            and result.get("type") != "interactive.rejected"
        ):
            raise InteractiveProtocolError(
                "Server continued a call after its resource was rejected"
            )

    def record_acceptance(self, value: dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise InteractiveProtocolError("Missing interactive acceptance object")
        identifier = value.get("continuation_id")
        if (
            type(value.get("version")) is not int
            or value["version"] != 1
            or not isinstance(identifier, str)
            or not identifier
        ):
            raise InteractiveProtocolError("Missing or invalid interactive acceptance")
        if self.is_accepted and identifier != self.continuation_id:
            raise InteractiveProtocolError(
                "Continuation ID changed during the logical call"
            )
        self._continuation_id = identifier

    def record_pause(
        self,
        event: dict[str, Any],
    ) -> None:
        self.record_acceptance(event)
        for field in ("exchange_id", "call_id", "tool_name"):
            if not isinstance(event.get(field), str) or not event[field]:
                raise InteractiveProtocolError(f"Invalid pause {field}")
        name = event["tool_name"]
        if (
            name not in self._definitions
            or event.get("behavior") != self._definitions[name]["behavior"]
        ):
            raise InteractiveProtocolError(
                "Pause tool/behavior is outside the configured scope"
            )
        request = event.get("request")
        if not isinstance(request, dict):
            raise InteractiveProtocolError("Pause request must be a JSON object")
        values = request.get("values")
        if (
            request.get("kind") != "file_snapshot"
            or not isinstance(values, dict)
            or set(values) != set(self._definitions[name]["first_fields"])
        ):
            raise InteractiveProtocolError(
                "Pause must include all and only completed first_fields"
            )
        for field, value in values.items():
            self.validate_value(value, self._schemas[name]["properties"][field])
        previous = self._frozen.get(event["call_id"])
        if previous is not None and (
            previous[0] != name
            or json.dumps(previous[1], sort_keys=True)
            != json.dumps(values, sort_keys=True)
        ):
            raise InteractiveProtocolError("Frozen priority values changed")
        if event["exchange_id"] in self._exchanges:
            raise InteractiveProtocolError("Duplicate/consumed interactive exchange")
        self._exchanges.add(event["exchange_id"])
        self._frozen[event["call_id"]] = (name, deepcopy(values))

    @staticmethod
    def validate_value(value: Any, schema: dict[str, Any]) -> None:
        types = {
            "string": str,
            "boolean": bool,
            "integer": int,
            "number": (int, float),
            "object": dict,
            "array": list,
            "null": type(None),
        }
        expected = types.get(str(schema.get("type")))
        if (
            expected is None
            or not isinstance(value, expected)
            or (isinstance(value, bool) and schema.get("type") in {"integer", "number"})
        ):
            raise InteractiveProtocolError("Priority value has the wrong JSON type")

    def validate_interruption(self, result: dict[str, Any]) -> None:
        identifier = result.get("call_id")
        previous = self._frozen.get(identifier) if isinstance(identifier, str) else None
        if (
            previous is None
            or previous[0] != result.get("tool_name")
            or json.dumps(previous[1], sort_keys=True)
            != json.dumps(result.get("values"), sort_keys=True)
        ):
            raise InteractiveProtocolError(
                "Rejected call does not match the suspended call"
            )
        if self._rejected_call_id != identifier:
            raise InteractiveProtocolError(
                "Rejected result does not match the rejected exchange"
            )

    def validate_raw_calls(
        self,
        output: list[dict[str, Any]],
    ) -> None:
        for item in output:
            if (
                item.get("type") != "function_call"
                or item.get("name") not in self._definitions
            ):
                continue
            arguments = json.loads(
                item["arguments"], object_pairs_hook=decode_object_pairs
            )
            if not isinstance(arguments, dict):
                raise InteractiveProtocolError(
                    "Interactive arguments must be a JSON object"
                )
            for field in self._definitions[item["name"]]["fields"].values():
                if not isinstance(arguments.get(field), str):
                    raise InteractiveProtocolError(
                        "Interactive string arguments must not be coerced"
                    )
            self.validate_calls(
                [ToolCall(item["call_id"], FunctionCall(item["name"], arguments))],
            )

    def validate_calls(
        self,
        calls: list[ToolCall],
    ) -> None:
        seen: set[str] = set()
        for call in calls:
            if call.id in seen:
                raise InteractiveProtocolError(
                    "Duplicate call ID in interactive result"
                )
            seen.add(call.id)
            if call.get_name() not in self._definitions:
                continue
            prior = self._frozen.get(call.id)
            if (
                prior is None
                or prior[0] != call.get_name()
                or any(
                    json.dumps(call.get_arguments().get(key), sort_keys=True)
                    != json.dumps(value, sort_keys=True)
                    for key, value in prior[1].items()
                )
            ):
                raise InteractiveProtocolError(
                    "Complete interactive call has no matching frozen authorization"
                )

    def validate_completed_calls(self, calls: list[ToolCall]) -> None:
        self._validate_returned_calls(
            calls,
            set(self._frozen),
            "Completed turn omitted a suspended interactive call",
        )

    def validate_rejected_calls(self, calls: list[ToolCall]) -> None:
        self._validate_returned_calls(
            calls,
            set(self._frozen) - {self._rejected_call_id},
            "Rejected turn omitted an earlier complete interactive call",
        )

    def _validate_returned_calls(
        self, calls: list[ToolCall], expected_ids: set[str], message: str
    ) -> None:
        self.validate_calls(calls)
        returned_ids = {
            call.id for call in calls if call.get_name() in self._definitions
        }
        if returned_ids != expected_ids:
            raise InteractiveProtocolError(message)
