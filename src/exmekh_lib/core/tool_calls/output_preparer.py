"""Materialize oversized text before publication or persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from exmekh_lib.config.cache import CACHE_FORMAT
from exmekh_lib.config.tool_output import TOOL_OUTPUT_TEXT
from exmekh_lib.core.cache.output_file_workspace import OutputFileWorkspace
from exmekh_lib.core.file_inspection.lines import SourceLines
from exmekh_lib.core.models.external_tool_output import (
    ExternalToolOutput,
    OutputTextFile,
)
from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.models.tool_output_presentation import ToolOutputPresentation
from exmekh_lib.core.models.xml_header import XmlHeader
from exmekh_lib.interfaces.tokenizer import Tokenizer
from exmekh_lib.utils.xml import stringify_tool_result

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreparedToolOutput:
    content: str
    external_output: ExternalToolOutput | None = None
    failure: dict[str, Any] | None = None

    @property
    def changed(self) -> bool:
        return self.external_output is not None or self.failure is not None

    def as_result(self, original: PreformattedToolResult) -> PreformattedToolResult:
        return PreformattedToolResult(
            self.content,
            attachments=original.attachments,
            external_output=self.external_output,
            delivery_failed=original.delivery_failed or self.failure is not None,
        )


class ToolOutputPreparer:
    def __init__(
        self,
        tokenizer: Tokenizer,
        files: OutputFileWorkspace,
    ) -> None:
        self._tokenizer = tokenizer
        self.files = files
        self.settings = files.settings

    def prepare(
        self,
        output: ToolOutputPresentation,
    ) -> PreparedToolOutput:
        if output.result.external_output is not None:
            # References may list many files; never recursively export their manifest.
            return PreparedToolOutput(
                output.result.content, output.result.external_output
            )
        text = output.body
        try:
            count = self._count(
                text if output.budget_text is None else output.budget_text
            )
            if count <= self.settings.max_tokens:
                return PreparedToolOutput(output.result.content)
            reference = self._export(text, output.identity.id)
            return PreparedToolOutput(
                output.replace_body(reference.render()),
                reference,
            )
        except Exception as error:
            logger.exception(TOOL_OUTPUT_TEXT.export_failed_log)
            failure = {
                "ok": False,
                "error_kind": "tool_output_export_failed",
                "error": TOOL_OUTPUT_TEXT.failed.format(error=error),
            }
            rendered = output.replace_body(stringify_tool_result(failure))
            rendered = XmlHeader.add_field_to_content(
                rendered, "response-type", "failed"
            )
            return PreparedToolOutput(rendered, failure=failure)

    def _export(self, text: str, output_id: str) -> ExternalToolOutput:
        parts = self._parts(text)
        counts = tuple(self._count(part) for part in parts)
        digest = sha256()
        for component in (output_id, str(self.settings.max_tokens), text):
            digest.update(
                component.encode(
                    CACHE_FORMAT.output_text_encoding, CACHE_FORMAT.output_text_errors
                )
            )
            digest.update(CACHE_FORMAT.output_digest_separator)
        paths = self.files.write(digest.hexdigest(), parts)
        files = tuple(
            OutputTextFile(
                path,
                tokens,
                SourceLines(part).physical_count,
            )
            for path, tokens, part in zip(paths, counts, parts, strict=True)
        )
        return ExternalToolOutput(
            self._count(text),
            SourceLines(text).physical_count,
            self.settings.max_tokens,
            files,
        )

    def _count(self, text: str) -> int:
        count = self._tokenizer.count_tokens(text)
        if type(count) is not int or count < 0:
            raise ValueError(TOOL_OUTPUT_TEXT.invalid_token_count)
        return count

    def _parts(self, text: str) -> tuple[str, ...]:
        if SourceLines(text).physical_count != 1:
            return (text,)
        parts = self._tokenizer.split_text(text, self.settings.max_tokens)
        if (
            not parts
            or "".join(parts) != text
            or any(
                not part or self._count(part) > self.settings.max_tokens
                for part in parts
            )
        ):
            raise ValueError(TOOL_OUTPUT_TEXT.invalid_prefix)
        return parts
