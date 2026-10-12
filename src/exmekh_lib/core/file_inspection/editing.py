"""Authorize snapshots and revalidate every edit after execution hooks."""

from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.core.models.edit_document import EditDocument, EditElement

from .decoding import decode_text
from .edit_filesystem import (
    commit_edit,
    edit_snapshot,
    encode_replacement,
    resolve_edit_target,
    resource_lock,
    validate_edit_target,
)
from .edit_selection import select_replacements
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

    def authorize_alias(self, work_root: Path, resource: str) -> Self:
        """Reuse the initial snapshot only while the alias resolves to its target."""
        validate_edit_target(
            work_root,
            resource,
            self.target,
            self.snapshot,
            self.snapshot.content if self.snapshot else b"",
        )
        return replace(self, resource=resource)

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

_approval: ContextVar[tuple[tuple[EditApproval, ...], EditEvidence] | None] = (
    ContextVar("edit_approvals", default=None)
)


@contextmanager
def file_edit_context(
    approvals: tuple[EditApproval, ...], evidence: EditEvidence
) -> Generator[None]:
    token = _approval.set((approvals, evidence))
    try:
        yield
    finally:
        _approval.reset(token)


@dataclass(frozen=True, slots=True)
class _PreparedEdit:
    approval: EditApproval
    resources: tuple[str, ...]
    content: bytes | None

    def result(self, work_root: Path) -> dict[str, object]:
        return {
            "path": self.approval.target.relative_to(work_root.resolve()).as_posix(),
            "operation": "remove"
            if self.content is None
            else "create"
            if self.approval.snapshot is None
            else "edit",
            "size_bytes": len(self.content) if self.content is not None else 0,
        }


def _prepare_edits(
    document: EditDocument,
    approvals: tuple[EditApproval, ...],
    evidence: tuple[InspectionEvidence, ...],
) -> tuple[_PreparedEdit, ...]:
    if len(document.elements) != len(approvals) or any(
        element.path != approval.resource
        for element, approval in zip(document.elements, approvals, strict=True)
    ):
        raise FileInspectionError(
            "Every path and edit index must match its authorization, including after Before hooks."
        )
    groups: dict[Path, list[tuple[EditElement, EditApproval]]] = {}
    for element, approval in zip(document.elements, approvals, strict=True):
        groups.setdefault(approval.target, []).append((element, approval))
    prepared: list[_PreparedEdit] = []
    for group in groups.values():
        approval = group[0][1]
        if any(
            other.snapshot != approval.snapshot or other.text != approval.text
            for _, other in group
        ):
            raise FileInspectionError("Aliases must share the same initial snapshot.")
        snapshot = approval.snapshot
        if snapshot is not None:
            assert snapshot.content is not None
            require_edit_evidence(approval.target, snapshot.content, evidence)
        replacement = select_replacements(
            approval.text,
            tuple(element for element, _ in group),
            exists=snapshot is not None,
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
        prepared.append(
            _PreparedEdit(
                approval, tuple(element.path for element, _ in group), encoded
            )
        )
    return tuple(prepared)


def edit_files(work_root: Path, edits: str) -> dict[str, object]:
    """Validate the entire post-hook document, then commit each canonical file once."""
    context = _approval.get()
    if context is None or not context[0]:
        raise FileInspectionError(
            "An accepted interactive snapshot exchange is required; ordinary tool-call fallback is forbidden."
        )
    document = EditDocument.parse(edits)
    approvals, evidence_source = context
    committed: list[dict[str, object]] = []
    with ExitStack() as locks:
        for target in sorted({approval.target for approval in approvals}):
            locks.enter_context(resource_lock(target))
        evidence = evidence_source() if callable(evidence_source) else evidence_source
        prepared = _prepare_edits(document, approvals, evidence)
        # Validate all resources before the first write, then recheck each commit.
        for item in prepared:
            for resource in item.resources:
                validate_edit_target(
                    work_root,
                    resource,
                    item.approval.target,
                    item.approval.snapshot,
                    item.content,
                )
        for item in prepared:
            try:
                # Steering can arrive while waiting on locks or between commits.
                if callable(evidence_source):
                    current_evidence = evidence_source()
                    if item.approval.snapshot is not None:
                        assert item.approval.snapshot.content is not None
                        require_edit_evidence(
                            item.approval.target,
                            item.approval.snapshot.content,
                            current_evidence,
                        )
                commit_edit(
                    work_root,
                    item.approval.resource,
                    item.approval.target,
                    item.approval.snapshot,
                    item.content,
                )
            except (FileInspectionError, OSError, UnicodeError, ValueError) as error:
                if not committed:
                    raise
                return {
                    "ok": False,
                    "error_kind": "partial_file_edit",
                    "error": str(error),
                    "failed_path": item.approval.resource,
                    "files": committed,
                }
            committed.append(item.result(work_root))
    return {"ok": True, "files": committed}
