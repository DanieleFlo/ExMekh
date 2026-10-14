"""Validate untrusted interactive IDs, frozen arguments and schema values."""

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from exmekh_lib.config.interactive import INTERACTIVE_PROTOCOL
from exmekh_lib.config.provider_text import PROVIDER_TEXT
from exmekh_lib.core.models.assistant_message import FunctionCall, ToolCall
from exmekh_lib.core.models.edit_document import EditDocument
from exmekh_lib.core.models.interactive import InteractiveProtocolError


def decode_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InteractiveProtocolError(
                PROVIDER_TEXT.duplicate_json_property.format(key=key)
            )
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class _AuthorizedEditCall:
    tool_name: str
    paths: tuple[str, ...] = ()
    exchange_id: str = ""
    snapshots: tuple[str, ...] = ()

    def with_snapshot(self, text: str) -> "_AuthorizedEditCall":
        return replace(self, snapshots=(*self.snapshots, text))

    def with_pause(
        self, name: str, path: str, index: int, exchange_id: str
    ) -> "_AuthorizedEditCall":
        if self.tool_name != name or index != len(self.paths) + 1:
            raise InteractiveProtocolError(
                PROVIDER_TEXT.edit_indices_must_be_consecutive_within_the_same_call
            )
        return replace(self, paths=(*self.paths, path), exchange_id=exchange_id)

    def matches_call(self, name: str, document: EditDocument) -> bool:
        return self.tool_name == name and self.paths == tuple(
            element.path for element in document.elements
        )

    def matches_interruption(self, result: dict[str, Any]) -> bool:
        return (
            self.tool_name == result.get("tool_name")
            and result.get("values") == {"path": self.paths[-1]}
            and type(result.get("edit_index")) is int
            and result["edit_index"] == len(self.paths)
            and result.get("exchange_id") == self.exchange_id
        )


class InteractiveContract:
    """Own the authorization state of one logical call, independently of HTTP."""

    def __init__(
        self,
        definitions: dict[str, dict[str, Any]],
    ) -> None:
        self._definitions = deepcopy(definitions)
        self._continuation_id: str | None = None
        self._frozen: dict[str, _AuthorizedEditCall] = {}
        self._exchanges: set[str] = set()
        self._rejected_call_id: str | None = None

    @property
    def is_accepted(self) -> bool:
        return self._continuation_id is not None

    @property
    def continuation_id(self) -> str:
        if self._continuation_id is None:
            raise InteractiveProtocolError(
                PROVIDER_TEXT.no_interactive_continuation_was_accepted
            )
        return self._continuation_id

    def definition(self, tool_name: str) -> dict[str, Any]:
        return deepcopy(self._definitions[tool_name])

    def record_reply(self, event: dict[str, Any], reply: Mapping[str, object]) -> None:
        text = reply.get("text")
        if reply.get("status") == "reject":
            self._rejected_call_id = event["call_id"]
        elif reply.get("status") == "provide" and isinstance(text, str):
            prior = self._frozen[event["call_id"]]
            self._frozen[event["call_id"]] = prior.with_snapshot(text)

    def validate_next_result(self, result: dict[str, Any]) -> None:
        if (
            self._rejected_call_id is not None
            and result.get("type") != "interactive.rejected"
        ):
            raise InteractiveProtocolError(
                PROVIDER_TEXT.server_continued_a_call_after_its_resource_was_rejected
            )

    def record_acceptance(self, value: dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise InteractiveProtocolError(
                PROVIDER_TEXT.missing_interactive_acceptance_object
            )
        identifier = value.get("continuation_id")
        if (
            type(value.get("version")) is not int
            or value["version"] != INTERACTIVE_PROTOCOL.version
            or not isinstance(identifier, str)
            or not identifier
        ):
            raise InteractiveProtocolError(
                PROVIDER_TEXT.missing_or_invalid_interactive_acceptance
            )
        if self.is_accepted and identifier != self.continuation_id:
            raise InteractiveProtocolError(
                PROVIDER_TEXT.continuation_id_changed_during_the_logical_call
            )
        self._continuation_id = identifier

    def record_pause(
        self,
        event: dict[str, Any],
    ) -> None:
        self.record_acceptance(event)
        for field in ("exchange_id", "call_id", "tool_name"):
            if not isinstance(event.get(field), str) or not event[field]:
                raise InteractiveProtocolError(
                    PROVIDER_TEXT.invalid_pause.format(field=field)
                )
        name = event["tool_name"]
        if (
            name not in self._definitions
            or event.get("behavior") != self._definitions[name]["behavior"]
        ):
            raise InteractiveProtocolError(
                PROVIDER_TEXT.pause_tool_behavior_is_outside_the_configured_scope
            )
        request = event.get("request")
        if not isinstance(request, dict):
            raise InteractiveProtocolError(
                PROVIDER_TEXT.pause_request_must_be_a_json_object
            )
        values = request.get("values")
        index = request.get("edit_index")
        if (
            request.get("kind") != INTERACTIVE_PROTOCOL.snapshot_kind
            or not isinstance(values, dict)
            or set(values) != {"path"}
            or not isinstance(values["path"], str)
            or not values["path"]
            or type(index) is not int
            or index < 1
        ):
            raise InteractiveProtocolError(
                PROVIDER_TEXT.pause_requires_a_path_and_positive_edit_index
            )
        previous = self._frozen.get(event["call_id"], _AuthorizedEditCall(name))
        authorized = previous.with_pause(
            name, values["path"], index, event["exchange_id"]
        )
        if event["exchange_id"] in self._exchanges:
            raise InteractiveProtocolError(
                PROVIDER_TEXT.duplicate_consumed_interactive_exchange
            )
        self._exchanges.add(event["exchange_id"])
        self._frozen[event["call_id"]] = authorized

    def validate_interruption(self, result: dict[str, Any]) -> None:
        identifier = result.get("call_id")
        previous = self._frozen.get(identifier) if isinstance(identifier, str) else None
        if previous is None or not previous.matches_interruption(result):
            raise InteractiveProtocolError(
                PROVIDER_TEXT.rejected_call_does_not_match_the_suspended_edit
            )
        if self._rejected_call_id != identifier:
            raise InteractiveProtocolError(
                PROVIDER_TEXT.rejected_result_does_not_match_the_rejected_exchange
            )

    def validate_raw_calls(
        self,
        output: list[dict[str, Any]],
    ) -> None:
        if not isinstance(output, list) or any(
            not isinstance(item, dict) for item in output
        ):
            raise InteractiveProtocolError(
                PROVIDER_TEXT.interactive_output_must_be_a_list_of_json_objects
            )
        for item in output:
            if (
                item.get("type") != "function_call"
                or item.get("name") not in self._definitions
            ):
                continue
            if not isinstance(item.get("call_id"), str) or not item["call_id"]:
                raise InteractiveProtocolError(
                    PROVIDER_TEXT.interactive_call_id_must_be_a_nonempty_string
                )
            if not isinstance(item.get("arguments"), str):
                raise InteractiveProtocolError(
                    PROVIDER_TEXT.interactive_arguments_must_be_encoded_json
                )
            try:
                arguments = json.loads(
                    item["arguments"], object_pairs_hook=decode_object_pairs
                )
            except json.JSONDecodeError as error:
                raise InteractiveProtocolError(
                    PROVIDER_TEXT.malformed_interactive_argument_json
                ) from error
            if not isinstance(arguments, dict):
                raise InteractiveProtocolError(
                    PROVIDER_TEXT.interactive_arguments_must_be_a_json_object
                )
            field = self._definitions[item["name"]]["fields"]["document"]
            if set(arguments) != {field} or not isinstance(arguments[field], str):
                raise InteractiveProtocolError(
                    PROVIDER_TEXT.interactive_document_must_be_the_only_exact_string_argument
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
                    PROVIDER_TEXT.duplicate_call_id_in_interactive_result
                )
            seen.add(call.id)
            if call.get_name() not in self._definitions:
                continue
            prior = self._frozen.get(call.id)
            field = self._definitions[call.get_name()]["fields"]["document"]
            arguments = call.get_arguments()
            try:
                if set(arguments) != {field}:
                    raise ValueError(PROVIDER_TEXT.unexpected_interactive_arguments)
                document = EditDocument.parse(
                    arguments[field], snapshots=prior.snapshots if prior else ()
                )
            except (ValueError, KeyError, TypeError) as error:
                raise InteractiveProtocolError(
                    PROVIDER_TEXT.invalid_interactive_document.format(error=error)
                ) from error
            if prior is None or not prior.matches_call(call.get_name(), document):
                raise InteractiveProtocolError(
                    PROVIDER_TEXT.complete_interactive_call_has_no_matching_frozen_authorizations
                )

    def validate_completed_calls(self, calls: list[ToolCall]) -> None:
        self._validate_returned_calls(
            calls,
            set(self._frozen),
            PROVIDER_TEXT.completed_turn_omitted_a_suspended_interactive_call,
        )

    def validate_rejected_calls(self, calls: list[ToolCall]) -> None:
        self._validate_returned_calls(
            calls,
            set(self._frozen) - {self._rejected_call_id},
            PROVIDER_TEXT.rejected_turn_omitted_an_earlier_complete_interactive_call,
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
