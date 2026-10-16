"""Resolve visible and referenced outputs without changing cache or delivery."""

from collections.abc import Callable
from dataclasses import dataclass

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION
from exmekh_lib.core.cache.output_file_workspace import OutputFileWorkspace
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.models.external_tool_output import ExternalToolOutput
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.models.xml_header import XmlHeader
from exmekh_lib.core.tool_calls.output_view import ToolOutputView
from exmekh_lib.interfaces.tokenizer import Tokenizer


@dataclass(frozen=True, slots=True)
class OutputCandidate:
    entry_id: str
    output_id: str
    name: str
    message: ToolMessage
    body: str
    token_count: int
    external: ExternalToolOutput | None
    recoverable: bool
    used_tools: tuple[ToolOutputReference, ...] = ()
    visible: bool = True
    available: bool = True

    def preview(self, tokenizer: Tokenizer) -> dict[str, object]:
        limit = HISTORY_COMPACTION.preview_tokens
        text = (
            self.body
            if self.token_count <= 2 * limit
            else tokenizer.truncate_text(self.body, limit, limit)
        )
        return {
            "entry_id": self.entry_id,
            "output_id": self.output_id,
            "name": self.name,
            "token_count": self.token_count,
            "preview": text,
            "storage": "files" if self.external is not None else "message",
            "visibility": "message" if self.visible else "reference",
            "available": self.available,
        }


class CompactionOutputs:
    def __init__(
        self,
        tokenizer: Tokenizer,
        output_lookup: Callable[[str], ToolCallRecord | None] | None,
        files: OutputFileWorkspace | None,
    ) -> None:
        self._tokenizer = tokenizer
        self._output_lookup = output_lookup
        self._files = files

    def _record(self, output_id: str, name: str) -> ToolCallRecord | None:
        record = self._output_lookup(output_id) if self._output_lookup else None
        return record if record is not None and record.name == name else None

    def candidates(
        self, index: int, message: ToolMessage
    ) -> tuple[OutputCandidate, ...]:
        """Visit declared descendants only, ignoring cycles and unrelated cache entries."""
        opening = self._candidate(str(index), message)
        candidates = [opening]
        visited = {opening.output_id}
        position = 0
        while position < len(candidates):
            candidate = candidates[position]
            position += 1
            for reference in candidate.used_tools:
                if reference.id in visited:
                    continue
                visited.add(reference.id)
                record = self._record(reference.id, reference.name)
                if record is not None and record.source == "UserMessage":
                    continue
                entry_id = f"{index}:{reference.id}"
                if record is None:
                    candidates.append(
                        OutputCandidate(
                            entry_id,
                            reference.id,
                            reference.name,
                            ToolMessage("", reference.id, reference.name),
                            "",
                            0,
                            None,
                            False,
                            visible=False,
                            available=False,
                        )
                    )
                else:
                    candidates.append(self._referenced_candidate(entry_id, record))
        return tuple(candidates)

    def _referenced_candidate(
        self, entry_id: str, record: ToolCallRecord
    ) -> OutputCandidate:
        content = ToolOutputView(record).format_current()
        return self._candidate(
            entry_id,
            ToolMessage(
                content,
                record.tool_call_id or record.id,
                record.name,
                external_output=record.external_output,
            ),
            visible=False,
        )

    def _candidate(
        self, entry_id: str, message: ToolMessage, *, visible: bool = True
    ) -> OutputCandidate:
        content = message.get_message()
        parsed = XmlHeader.parse_content(content)
        output_id = (
            parsed.field("id") if parsed is not None else None
        ) or message.get_tool_call_id()
        name = (
            parsed.field("name") if parsed is not None else None
        ) or message.get_name()
        body = parsed.body if parsed is not None else content
        record = self._record(output_id, name)
        if record is not None and ToolOutputView(record).content() != body:
            record = None
        external = message.get_external_output()
        if external is None and record is not None:
            external = record.external_output
        if external is not None and body != external.render():
            raise ValueError(HISTORY_COMPACTION.mismatched_manifest)
        if external is not None:
            if self._files is None:
                raise ValueError(HISTORY_COMPACTION.unavailable_files)
            body = self._files.read_text(external)
        references = parsed.output_references() if parsed is not None else None
        if references is None and record is not None:
            references = record.used_tools
        return OutputCandidate(
            entry_id,
            output_id,
            name,
            message,
            body,
            self._tokenizer.count_tokens(body),
            external,
            record is not None or external is not None,
            references or (),
            visible,
        )
