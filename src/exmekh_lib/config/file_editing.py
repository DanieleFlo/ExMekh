"""Shared edit-file contract and bounded client operations."""

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FileEditingSettings:
    max_snapshot_bytes: int = 1024 * 1024
    max_generated_bytes: int = 1024 * 1024
    protocol_timeout_seconds: float = 300
    max_exchanges: int = 64
    max_response_bytes: int = 16 * 1024 * 1024
    marker_prefix: str = "@ex_meckh"
    all_marker_name: str = "all"
    delete_marker_name: str = "delete"
    line_marker_name: str = "line"
    resource_description: str = (
        "Exact file path inside the workroot, frozen before generating old/new."
    )
    inspection_required: str = (
        "Inspect the current file first. The visible inspection is missing or stale; "
        "this edit was rejected. Read the current file, then make a new edit-file call."
    )
    read_required: str = (
        "Read the current file with read-file after its inspection. A complete, "
        "unchanged successful read must remain visible. This edit was rejected."
    )
    changed: str = "File or path changed during generation; inspect and read again. No edit was applied."
    selection_error: str = (
        "old_text must be nonempty, occur exactly once (including overlapping and "
        "mid-line matches), and start at a physical line boundary including indentation. "
        "Choose a larger or different literal block."
    )

    @property
    def reserved_prefix(self) -> str:
        return f"{self.marker_prefix}_"

    @property
    def reserved_marker_pattern(self) -> str:
        return re.escape(self.marker_prefix) + r"(?:_|(?!\w))"

    @property
    def all_marker(self) -> str:
        return f"{self.reserved_prefix}{self.all_marker_name}"

    @property
    def delete_marker(self) -> str:
        return f"{self.reserved_prefix}{self.delete_marker_name}"

    def line_marker(self, line: int | str) -> str:
        return f"{self.reserved_prefix}{self.line_marker_name}[{line}]"

    @property
    def line_family(self) -> str:
        return self.line_marker("<N>")

    @property
    def line_example(self) -> str:
        return self.line_marker("N")

    @property
    def first_line_marker(self) -> str:
        return self.line_marker(1)

    @property
    def line_pattern(self) -> str:
        return re.escape(self.line_family).replace(re.escape("<N>"), r"([1-9][0-9]*)")

    @property
    def selection_description(self) -> str:
        return (
            f"Exact unique literal from a physical line start, {self.line_example} for "
            f"insertion, or {self.all_marker} for the entire file. Include indentation. "
            "Markers have no surrounding spaces or newlines."
        )

    @property
    def replacement_description(self) -> str:
        return (
            "Exact replacement content, including any intended newline characters. "
            f"To delete the file after old_text={self.all_marker}, use exactly "
            f"{self.delete_marker} as the entire argument. "
            "Do not add native framing newlines around a marker. Empty content keeps "
            "the file; it does not delete it."
        )

    @property
    def invalid_marker(self) -> str:
        return (
            f"Reserved {self.reserved_prefix} markers must occupy the entire correct field. "
            f"Use {self.all_marker} or {self.line_example} in old_text, and "
            f"{self.delete_marker} only in new_text with {self.all_marker} on an existing file."
        )

    @property
    def creation_required(self) -> str:
        return (
            f"Creation requires old_text={self.all_marker} or {self.first_line_marker}."
        )

    @property
    def description(self) -> str:
        return (
            "Edit a text file inside the workroot synchronously. Supply path, old_text and "
            "new_text as exact strings. Existing files require a successful inspect-file "
            "followed by read-file of the unchanged content, both fully visible. "
            "Literal old_text must start at a physical line boundary including indentation "
            "and occur exactly once globally; new_text replaces that occurrence, and may "
            f"be longer or empty. Use {self.line_example} as the entire old_text to insert "
            "before physical line N (1..L+1, L+1 appends); include your own newlines. "
            f"Use {self.all_marker} to rewrite the whole file. To DELETE an existing file, "
            f'use old_text="{self.all_marker}" and new_text="{self.delete_marker}". '
            "Each marker must occupy the entire value, with no leading or trailing "
            "space or newline, including native parameter framing newlines. "
            "Empty new_text removes content but keeps the file. To create a missing file "
            f"use {self.all_marker} or {self.first_line_marker}, including empty new_text, "
            "without inspect/read; its parent "
            f"must already exist. The {self.reserved_prefix} family is reserved even inside ordinary "
            "text. No implicit escaping, normalization or truncation. The server pauses "
            "after path for authorization; missing/stale inspection triggers automatic "
            "inspection, while a missing read requires a new explicit read. Changed files "
            "and invalid paths/markers/selections are rejected without writes."
        )


FILE_EDITING = FileEditingSettings()
