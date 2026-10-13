"""Shared literal edit contract and bounded filesystem operations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FileEditingSettings:
    min_new_file_characters: int = 2
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
        "old_text must occur exactly once globally and start at a physical line boundary, "
        "including indentation. It may be empty only when the existing file is empty."
    )
    conflict_error: str = "Edits of the same canonical file must use the initial snapshot without overlapping selections."
    document_description: str = (
        'One string: <exmekh_edit path="file.txt" id="123" />old text'
        '<exmekh_new_text id="123"/>new text. The server supplies the opening ID; '
        "copy it into the separator. An omitted ID is accepted only with unambiguous framing. "
        "Both bodies are literal text, with exact whitespace and newlines; no CDATA or XML entity decoding. "
        "New text ends at the end of the string or the next edit opening with a fresh ID. "
        "Empty new text removes the selection but keeps the file. To insert, retain an existing "
        "anchor in the replacement and add text before or after it. For an empty existing file, "
        "use an empty old body. No line, range, all, create or delete commands."
    )

    noneditable_inspection: str = (
        "Inspection identifies a format without editable physical text."
    )
    unsupported_text_format: str = (
        "This file format is not supported by the text edit tool."
    )
    existing_file_required: str = "edit-file requires an existing file; use new-file."
    unicode_text_required: str = (
        "Only supported Unicode text without NUL can be edited."
    )
    decoded_snapshot_limit: str = (
        "Decoded snapshot exceeds the UTF-8 control byte limit."
    )
    authorization_mismatch: str = (
        "Every path and edit index must match its authorization, including after "
        "Before hooks."
    )
    alias_snapshot_mismatch: str = "Aliases must share the same initial snapshot."
    replacement_byte_limit: str = (
        "Result exceeds the configured file byte limit; no edit was applied."
    )
    interactive_exchange_required: str = (
        "An accepted interactive snapshot exchange is required; ordinary tool-call "
        "fallback is forbidden."
    )
    parent_required: str = "The parent directory must already exist."
    snapshot_byte_limit: str = (
        "File exceeds the edit snapshot byte limit; no truncated snapshot is allowed."
    )
    minimum_content: str = "content must contain at least {minimum} Unicode characters."
    content_nul: str = "content must not contain NUL characters."
    creation_byte_limit: str = "Content exceeds the configured file byte limit."
    path_exists: str = "The path already exists; new-file cannot overwrite it."
    deletion_regular_file: str = "delete-file requires an existing regular file."
    changed_before_deletion: str = "File or path changed before deletion."
    unsupported_platform: str = (
        "This platform lacks descriptor-relative atomic edit operations; use a "
        "supported filesystem/runtime."
    )
    file_not_writable: str = "File is not writable; check permissions before retrying."
    parent_not_writable: str = (
        "Parent directory is not writable/searchable; check permissions before "
        "retrying."
    )
    selection_required: str = "At least one file edit is required."
    document_string: str = "The edit document must be an exact string"
    document_unicode: str = "The edit document contains invalid Unicode text"
    document_byte_limit: str = "Edit document exceeds the generated UTF-8 byte limit"
    opening_marker_required: str = "Expected an exmekh_edit opening marker"
    duplicate_edit_id: str = "Duplicate edit ID"
    separator_required: str = "Missing matching exmekh_new_text separator"
    separator_ambiguous: str = (
        "Ambiguous exmekh_new_text separator; include a distinct ID"
    )
    document_element_limit: str = "Edit document exceeds the element limit"
    empty_document: str = "Edit document must contain at least one edit"
    runtime_handler_required: str = (
        "Interactive behavior has no registered runtime handler"
    )
    authorization_sequence: str = "Missing or duplicate edit authorization"
    cancelled_before_dispatch: str = (
        "Interactive call invalidated by steering before dispatch"
    )
    cancelled_before_write: str = (
        "Interactive edit invalidated by cancellation or scope change before writing."
    )
    definition_required: str = (
        "Literal edit behavior requires exactly one required document string"
    )

    @property
    def description(self) -> str:
        return (
            "Edit existing workspace text files synchronously. Inspect and subsequently read "
            "unchanged content first; both results must remain visible. Each path is authorized "
            "before body generation. All selections refer to one initial snapshot; any rejected "
            "resource invalidates the entire call. Encoding/BOM and concurrency checks are preserved. "
            "Use new-file to create and delete-file to remove a file. "
            + self.document_description
        )

    delete_file_description: str = (
        "Delete one existing regular file inside the workroot. Requires path. A "
        "missing file or directory is an error. No inspect-file or read-file is "
        "required; file contents are not decoded."
    )

    argument_guidance: str = (
        "Supply one literal two-marker edits string. Inspect and read existing files "
        "first."
    )

    @property
    def new_file_description(self) -> str:
        return (
            "Create a new file inside the workroot. Requires path and content of at "
            f"least {self.min_new_file_characters} Unicode characters. The parent "
            "directory must exist. An existing path is an error. No inspect-file or "
            "read-file is required."
        )


FILE_EDITING = FileEditingSettings()


@dataclass(frozen=True, slots=True)
class FileEditFormat:
    """Fixed framing and behavior negotiated with the interactive provider."""

    behavior: str = "edit_file.v3"
    opening_pattern: str = r'<exmekh_edit path="([^"<>]+)"(?: id="([0-9]{3})")?\s*/>'
    separator_pattern: str = r'<exmekh_new_text(?: id="([0-9]{3})")?\s*/>'


FILE_EDIT_FORMAT = FileEditFormat()
