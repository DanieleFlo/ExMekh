from __future__ import annotations

import base64
import re
from collections.abc import Callable
from xml.etree import ElementTree as ET

from qs.core.cache.node_cache import NodeCache
from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.tool_calls.memory_tools import ExecutionHistory, tool_output_content

_PLACEHOLDER = re.compile(r"{{([^{}]*)}}")


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
                    f"Il segnaposto {{{{{identifier}}}}} non è elencato in items. "
                    f'Usa items=["{identifier}"] oppure elimina il segnaposto.'
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
                'Segnaposto incompleto. Usa text="… {{<id-output>}} …" '
                'e items=["<id-output>"], oppure rimuovi le doppie parentesi.'
            )

    def _require_items_for_placeholders(self, text: str) -> None:
        match = _PLACEHOLDER.search(text)
        if match is None:
            return
        identifier = self._placeholder_identifier(match)
        raise ExitItemError(
            f"Il segnaposto {{{{{identifier}}}}} richiede l'ID in items. "
            f'Usa items=["{identifier}"] oppure elimina il segnaposto.'
        )

    @staticmethod
    def _placeholder_identifier(match: re.Match[str]) -> str:
        identifier = match.group(1).strip()
        if not identifier:
            raise ExitItemError(
                'Segnaposto vuoto. Usa text="… {{<id-output>}} …" '
                'e items=["<id-output>"].'
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
                    f"L'output {record.id!r} compare più volte in items. "
                    'Usa una sola voce: items=["<id-output>"].'
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
                f"L'ID di chiamata {identifier!r} identifica più output: {ids}. "
                'Usa un ID di output specifico in items=["<id-output>"].'
            )
        if eligible:
            return next(iter(eligible.values()))
        if saw_out_of_scope:
            raise ExitItemError(
                f"L'output {identifier!r} appartiene a un'altra skill o invocazione. "
                "Cerca un output della skill corrente o di una sua discendente "
                'e usa items=["<id-output>"].'
            )
        if matching_entries:
            raise ExitItemError(
                f"L'ID {identifier!r} non ha un output salvato e recuperabile. "
                "Usa get_execution_history per trovare un saved-output-id, "
                'poi items=["<id-output>"].'
            )
        raise ExitItemError(
            f"L'ID {identifier!r} non è presente nella cronologia di questo nodo. "
            "Usa get_execution_history per trovare un saved-output-id, "
            'poi items=["<id-output>"].'
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
        output = tool_output_content(record)
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
