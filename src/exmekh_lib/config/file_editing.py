"""Shared XML edit contract and bounded client operations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FileEditingSettings:
    max_snapshot_bytes: int = 1024 * 1024
    max_generated_bytes: int = 1024 * 1024
    protocol_timeout_seconds: float = 300
    max_exchanges: int = 64
    max_response_bytes: int = 16 * 1024 * 1024
    inspection_required: str = (
        "Inspect the current file first. The visible inspection is missing or stale; "
        "this edit was rejected. Read the current file, then make a new edit-file call."
    )
    read_required: str = (
        "Read the current file with read-file after its inspection. A complete, "
        "unchanged successful read must remain visible. This edit was rejected."
    )
    changed: str = "File or path changed during generation; inspect and read again."
    selection_error: str = (
        "old_text must be nonempty, occur exactly once (including overlapping and "
        "mid-line matches), and start at a physical line boundary including indentation. "
        "Choose a larger or different literal block."
    )
    range_error: str = "Line numbers must be positive, ordered and within the physical lines of the initial snapshot."
    conflict_error: str = (
        "Edits of the same canonical file must use the initial snapshot without overlapping "
        "replacements, interior insertions or duplicate insertion positions. "
        "Whole-file rewrite or deletion must be that file's only edit."
    )
    creation_required: str = 'Creation requires <all/> or <line number="1"/> in old_text and cannot delete a file.'
    document_description: str = (
        'One exact XML string with one or more consecutive <edit path="file.txt">'
        "<old_text>...</old_text><new_text>...</new_text></edit> fragments. "
        "Each path is authorized before old/new generation. Use entities or CDATA for literal markup. "
        "The final </new_text> before each </edit> closes its replacement; earlier bare </new_text> is literal. "
        "A literal </new_text></edit> pair requires entities or CDATA. Preserve intended CR/LF and spaces. "
        'Use <all/> as the sole old_text content to rewrite/create; <line number="N"/> to insert before '
        "physical line N (1..L+1, L+1 appends); <delete/> as sole new_text content with <all/> to delete "
        "an existing file. Empty new_text keeps the file. For extremely long selections only, use "
        '<old_text start_line="N" end_line="M"></old_text> for inclusive physical lines. '
        "Escaped/CDATA command tags are literal text."
    )

    @property
    def description(self) -> str:
        return (
            "Edit workspace text files synchronously using the single edits document. "
            "Existing files require successful inspect-file and subsequent read-file of unchanged "
            "content, both fully visible. Missing inspection rejects the call and triggers inspection; "
            "missing read requires an explicit read and a new edit call. Missing files need no inspect/read; "
            "their parent must exist. Literal selections must start at a physical line boundary, include "
            "indentation and occur exactly once globally. All edits of each canonical file refer to one "
            "initial snapshot. Any rejected resource invalidates the entire call. The client validates "
            "the complete document after hooks before writing, preserves encoding/BOM, and checks concurrency. "
            "Each file is committed once; distinct files have no transactional rollback guarantee. "
            + self.document_description
        )


FILE_EDITING = FileEditingSettings()
