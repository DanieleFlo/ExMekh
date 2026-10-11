"""Authorize snapshots and revalidate every edit after execution hooks."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.config.file_inspection import FILE_INSPECTION

from .decoding import decode_text
from .edit_filesystem import (
    commit_edit,
    edit_snapshot,
    encode_replacement,
    resolve_edit_target,
    resource_lock,
)
from .edit_selection import select_replacement
from .errors import FileInspectionError
from .evidence import InspectionEvidence, hash_content
from .lines import physical_lines
from .workspace import FileSnapshot


@dataclass(frozen=True, slots=True)
class EditApproval:
    resource: str
    target: Path
    snapshot: FileSnapshot | None
    text: str
    encoding: str

    def reply(self) -> dict[str, object]:
        if self.snapshot is None:
            return {"status": "create"}
        return {
            "status": "provide",
            "text": self.text,
            "encoding": self.encoding,
            "line_count": len(physical_lines(self.text)),
        }


class EditPrerequisiteError(FileInspectionError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require_edit_evidence(
    target: Path, content: bytes, evidence: tuple[InspectionEvidence, ...]
) -> None:
    digest = hash_content(content)
    inspected = False
    for item in evidence:
        if (
            item.path == target
            and item.digest == digest
            and item.source == "inspect"
            and item.successful
            and not item.readable
        ):
            raise FileInspectionError(
                "Inspection identifies a format without editable physical text."
            )
        if (
            item.path != target
            or item.digest != digest
            or not item.successful
            or not item.readable
        ):
            continue
        if item.source == "inspect":
            inspected = True
        elif item.source == "read" and inspected:
            return
    if not inspected:
        raise EditPrerequisiteError("uninspected", FILE_EDITING.inspection_required)
    raise EditPrerequisiteError("unread", FILE_EDITING.read_required)


def approve_edit(
    work_root: Path, resource: str, evidence: tuple[InspectionEvidence, ...]
) -> EditApproval:
    target = resolve_edit_target(work_root, resource)
    if target.suffix.casefold() in {
        *FILE_INSPECTION.image_extensions,
        *FILE_INSPECTION.archive_extensions,
        *FILE_INSPECTION.zip_extensions,
        ".pdf",
    }:
        raise FileInspectionError(
            "This file format is not supported by the text edit tool."
        )
    with resource_lock(target):
        try:
            target.lstat()
        except FileNotFoundError:
            return EditApproval(resource, target, None, "", "utf-8")
        snapshot = edit_snapshot(work_root, resource)
        assert snapshot.content is not None
        require_edit_evidence(target, snapshot.content, evidence)
        decoded = decode_text(snapshot.content, target.suffix.casefold())
        if decoded is None or "\x00" in decoded[0]:
            raise FileInspectionError(
                "Only supported Unicode text without NUL can be edited."
            )
        text, encoding = decoded
        if len(text.encode("utf-8")) > FILE_EDITING.max_snapshot_bytes:
            raise FileInspectionError(
                "Decoded snapshot exceeds the UTF-8 control byte limit."
            )
        return EditApproval(resource, target, snapshot, text, encoding)


type EditEvidence = (
    tuple[InspectionEvidence, ...] | Callable[[], tuple[InspectionEvidence, ...]]
)

_approval: ContextVar[tuple[EditApproval | None, EditEvidence] | None] = ContextVar(
    "edit_approval", default=None
)


@contextmanager
def file_edit_context(
    approval: EditApproval | None, evidence: EditEvidence
) -> Generator[None]:
    token = _approval.set((approval, evidence))
    try:
        yield
    finally:
        _approval.reset(token)


def edit_file(
    work_root: Path, path: str, old_text: str, new_text: str
) -> dict[str, object]:
    context = _approval.get()
    if context is None or context[0] is None:
        raise FileInspectionError(
            "An accepted interactive snapshot exchange is required; ordinary tool-call fallback is forbidden."
        )
    approval, evidence_source = context
    evidence = evidence_source() if callable(evidence_source) else evidence_source
    assert approval is not None
    if path != approval.resource:
        raise FileInspectionError(
            "path must equal the resource approved before generation, including after Before hooks."
        )
    if (
        not isinstance(new_text, str)
        or len(new_text.encode("utf-8")) > FILE_EDITING.max_generated_bytes
    ):
        raise FileInspectionError(
            "new_text must be a string within the generated UTF-8 byte limit."
        )
    with resource_lock(approval.target):
        snapshot = approval.snapshot
        if snapshot is not None:
            assert snapshot.content is not None
            require_edit_evidence(approval.target, snapshot.content, evidence)
        replacement = select_replacement(
            approval.text, old_text, new_text, exists=snapshot is not None
        )
        encoded = (
            None
            if replacement is None
            else encode_replacement(
                replacement,
                snapshot.content if snapshot and snapshot.content is not None else b"",
                approval.encoding,
            )
        )
        if encoded is not None and len(encoded) > FILE_EDITING.max_snapshot_bytes:
            raise FileInspectionError(
                "Result exceeds the configured file byte limit; no edit was applied."
            )
        commit_edit(work_root, path, approval.target, snapshot, encoded)
    return {
        "ok": True,
        "path": approval.target.relative_to(work_root.resolve()).as_posix(),
        "operation": "remove"
        if encoded is None
        else "create"
        if snapshot is None
        else "edit",
        "size_bytes": len(encoded) if encoded is not None else 0,
    }
