from __future__ import annotations

import base64
import re
from collections.abc import Callable
from xml.etree import ElementTree as ET

from qs.config.validation import VALIDATION_PATTERNS
from qs.core.cache.node_cache import NodeCache
from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.tool_calls.execution_history import ExecutionHistory
from qs.core.tool_calls.output_view import ToolOutputView

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
                    f"Placeholder {{{{{identifier}}}}} is not listed in items. "
                    f'Use items=["{identifier}"] or remove the placeholder.'
                )
            placed.add(record.id)
            return self._render(record)

        content = _PLACEHOLDER.sub(replace, text)
        trailing = [
            self._render(record) for record in ordered if record.id not in placed
        ]
        return "\n\n".join((content, *trailing))

    @staticmethod
    def _validate_placeholder_syntax(text: str) -> None:
        remaining = _PLACEHOLDER.sub("", text)
        if "{{" in remaining or "}}" in remaining:
            raise ExitItemError(
                'Incomplete placeholder. Use text="… {{<id-output>}} …" '
                'and items=["<id-output>"], or remove the double braces.'
            )

    def _require_items_for_placeholders(self, text: str) -> None:
        match = _PLACEHOLDER.search(text)
        if match is None:
            return
        identifier = self._placeholder_identifier(match)
        raise ExitItemError(
            f"Placeholder {{{{{identifier}}}}} requires its ID in items. "
            f'Use items=["{identifier}"] or remove the placeholder.'
        )

    @staticmethod
    def _placeholder_identifier(match: re.Match[str]) -> str:
        identifier = match.group(1).strip()
        if not identifier:
            raise ExitItemError(
                'Empty placeholder. Use text="… {{<id-output>}} …" '
                'and items=["<id-output>"].'
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
                    f"Output {record.id!r} appears more than once in items. "
                    'Use only one entry: items=["<id-output>"].'
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
                f"Call ID {identifier!r} identifies multiple outputs: {ids}. "
                'Use a specific output ID in items=["<id-output>"].'
            )
        if eligible:
            return next(iter(eligible.values()))
        if saw_out_of_scope:
            raise ExitItemError(
                f"Output {identifier!r} belongs to another skill or invocation. "
                "Find an output from the current skill or a descendant "
                'and use items=["<id-output>"].'
            )
        if matching_entries:
            raise ExitItemError(
                f"ID {identifier!r} has no saved, retrievable output. "
                "Use get_execution_history to find a saved-output-id, "
                'then items=["<id-output>"].'
            )
        raise ExitItemError(
            f"ID {identifier!r} is not present in this node's history. "
            "Use get_execution_history to find a saved-output-id, "
            'then items=["<id-output>"].'
        )

    @staticmethod
    def _is_in_skill(record_path: str, skill_path: str) -> bool:
        if skill_path == "/":
            return record_path.startswith("/")
        return record_path == skill_path or record_path.startswith(skill_path + "/")

    @staticmethod
    def _render(record: ToolCallRecord) -> str:
        root = ET.Element(
            "tool-output",
            {"id": record.id, "name": record.name, "type": record.source},
        )
        content = ET.SubElement(root, "content")
        output = ToolOutputView(record).content()
        if any(not _is_xml_character(character) for character in output):
            content.set("encoding", "base64")
            content.text = base64.b64encode(output.encode("utf-8")).decode("ascii")
        else:
            content.text = output
        return ET.tostring(root, encoding="unicode", short_empty_elements=True)


def _is_xml_character(character: str) -> bool:
    code = ord(character)
    return (
        code in (9, 10, 13)
        or 0x20 <= code <= 0xD7FF
        or 0xE000 <= code <= 0xFFFD
        or 0x10000 <= code <= 0x10FFFF
    )
