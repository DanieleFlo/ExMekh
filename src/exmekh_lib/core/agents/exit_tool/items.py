from __future__ import annotations

import re
from collections.abc import Callable

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.config.validation import VALIDATION_PATTERNS
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.tool_calls.execution_history import ExecutionHistory
from exmekh_lib.core.tool_calls.output_view import ToolOutputView

_PLACEHOLDER = re.compile(VALIDATION_PATTERNS.exit_item_placeholder)


class ExitItemError(ValueError):
    """A correctable exit selection error to return to the model."""


class ExitItemResolver:
    """Select and render saved outputs from the current skill invocation."""

    def __init__(
        self,
        history: ExecutionHistory,
        cache_provider: Callable[[], NodeCache],
    ) -> None:
        self._history = history
        self._cache_provider = cache_provider

    def compose(
        self,
        text: str,
        items: list[str],
        skill_path: str,
        opening_position: int,
    ) -> str:
        self._validate_placeholder_syntax(text)
        if not items:
            self._require_items_for_placeholders(text)
            return text

        cache = self._cache_provider()
        selected, ordered = self._select_items(
            items, skill_path, opening_position, cache
        )
        placed: set[str] = set()

        def replace(match: re.Match[str]) -> str:
            identifier = self._placeholder_identifier(match)
            record = self._resolve(identifier, skill_path, opening_position, cache)
            if record.id not in selected:
                raise ExitItemError(
                    AGENT_TEXT.placeholder_is_not_listed_in_items_use_items_or.format(
                        identifier=identifier, identifier_2=identifier
                    )
                )
            placed.add(record.id)
            return ToolOutputView(record).format_embedded()

        content = _PLACEHOLDER.sub(replace, text)
        trailing = [
            ToolOutputView(record).format_embedded()
            for record in ordered
            if record.id not in placed
        ]
        return "\n\n".join((content, *trailing))

    @staticmethod
    def _validate_placeholder_syntax(text: str) -> None:
        remaining = _PLACEHOLDER.sub("", text)
        if "{{" in remaining or "}}" in remaining:
            raise ExitItemError(
                AGENT_TEXT.incomplete_placeholder_use_text_and_items_id_output_or
            )

    def _require_items_for_placeholders(self, text: str) -> None:
        match = _PLACEHOLDER.search(text)
        if match is None:
            return
        identifier = self._placeholder_identifier(match)
        raise ExitItemError(
            AGENT_TEXT.placeholder_requires_its_id_in_items_use_items_or.format(
                identifier=identifier, identifier_2=identifier
            )
        )

    @staticmethod
    def _placeholder_identifier(match: re.Match[str]) -> str:
        identifier = match.group(1).strip()
        if not identifier:
            raise ExitItemError(
                AGENT_TEXT.empty_placeholder_use_text_and_items_id_output
            )
        return identifier

    def _select_items(
        self,
        items: list[str],
        skill_path: str,
        opening_position: int,
        cache: NodeCache,
    ) -> tuple[dict[str, ToolCallRecord], list[ToolCallRecord]]:
        selected: dict[str, ToolCallRecord] = {}
        ordered: list[ToolCallRecord] = []
        for item in items:
            record = self._resolve(item, skill_path, opening_position, cache)
            if record.id in selected:
                raise ExitItemError(
                    AGENT_TEXT.output_appears_more_than_once_in_items_use_only.format(
                        record_id=record.id
                    )
                )
            selected[record.id] = record
            ordered.append(record)
        return selected, ordered

    def _resolve(
        self,
        identifier: str,
        skill_path: str,
        opening_position: int,
        cache: NodeCache,
    ) -> ToolCallRecord:
        matching_entries = self._history.find_by_identifier(identifier)
        eligible: dict[str, ToolCallRecord] = {}
        saw_out_of_scope = False
        for entry in matching_entries:
            record = entry.read_output(cache)
            if record is None:
                continue
            if entry.position <= opening_position or not self._is_in_skill(
                record.skill_path, skill_path
            ):
                saw_out_of_scope = True
                continue
            eligible[record.id] = record
        if len(eligible) > 1:
            ids = ", ".join(sorted(eligible))
            raise ExitItemError(
                AGENT_TEXT.call_id_identifies_multiple_outputs_use_a_specific_output.format(
                    identifier=identifier, ids=ids
                )
            )
        if eligible:
            return next(iter(eligible.values()))
        if saw_out_of_scope:
            raise ExitItemError(
                AGENT_TEXT.output_belongs_to_another_skill_or_invocation_find_an.format(
                    identifier=identifier
                )
            )
        if matching_entries:
            raise ExitItemError(
                AGENT_TEXT.id_has_no_saved_retrievable_output_use_get_execution.format(
                    identifier=identifier
                )
            )
        raise ExitItemError(
            AGENT_TEXT.id_is_not_present_in_this_node_s_history.format(
                identifier=identifier
            )
        )

    @staticmethod
    def _is_in_skill(record_path: str, skill_path: str) -> bool:
        if skill_path == "/":
            return record_path.startswith("/")
        return record_path == skill_path or record_path.startswith(skill_path + "/")
