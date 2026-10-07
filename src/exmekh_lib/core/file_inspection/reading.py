"""Read images directly or inspection-gated text ranges and PDF pages."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from exmekh_lib.config.file_inspection import FILE_INSPECTION

from .decoding import decode_text
from .errors import FileInspectionError
from .evidence import InspectionEvidence, hash_content
from .image_reading import read_image
from .lines import physical_lines
from .pdf_reading import read_pdf, validate_pdf_read_arguments
from .workspace import FileSnapshot, read_snapshot, validate_file_path

_NO_READABLE_LINES = (
    "This format has no readable physical text lines; use inspect-file metadata."
)


class FileReadContext:
    """Expose visible proof and an explicit request for runtime recovery."""

    def __init__(self, evidence: tuple[InspectionEvidence, ...]) -> None:
        self._evidence = evidence
        self._inspection_path: str | None = None

    def visible_inspections(self) -> tuple[InspectionEvidence, ...]:
        return self._evidence

    def request_inspection(self, path: str) -> None:
        self._inspection_path = path

    def pending_inspection_path(self) -> str | None:
        return self._inspection_path


_read_context: ContextVar[FileReadContext | None] = ContextVar(
    "visible_file_inspections", default=None
)


@contextmanager
def file_read_context(
    evidence: tuple[InspectionEvidence, ...],
) -> Iterator[FileReadContext]:
    context = FileReadContext(evidence)
    token = _read_context.set(context)
    try:
        yield context
    finally:
        _read_context.reset(token)


def require_inspection(path: str) -> dict[str, object]:
    """Request runtime recovery separately from the result passed through hooks."""
    context = _read_context.get()
    if context is not None:
        context.request_inspection(path)
    indices = (
        "page indices" if Path(path).suffix.casefold() == ".pdf" else "line indices"
    )
    return {
        "ok": False,
        "error_kind": "file_inspection_required",
        "error": (
            "inspect-file(path) and its successful result are missing from the "
            "visible context, or the file has changed. Inspect the current file, "
            f"then choose start_line and end_line as {indices} again from that analysis and "
            "make a new read-file call. This read was not performed."
        ),
    }


def validate_read_arguments(path: str, start_line: int, end_line: int) -> None:
    validate_file_path(path)
    for name, value in (("start_line", start_line), ("end_line", end_line)):
        if type(value) is not int or value < 1:
            raise FileInspectionError(
                f"{name} must be a positive integer (1-based); choose it from inspect-file."
            )
    if start_line > end_line:
        raise FileInspectionError(
            "start_line must be <= end_line; supply an inclusive interval from inspect-file."
        )


def read_file(
    work_root: Path, path: str, start_line: int, end_line: int
) -> dict[str, object]:
    validate_file_path(path)
    is_pdf = Path(path).suffix.casefold() == ".pdf"
    if Path(path).suffix.casefold() in FILE_INSPECTION.image_extensions:
        return read_image(
            read_snapshot(work_root, path, FILE_INSPECTION), FILE_INSPECTION
        )
    if is_pdf:
        validate_pdf_read_arguments(start_line, end_line, FILE_INSPECTION)
    else:
        validate_read_arguments(path, start_line, end_line)
    # No content is decoded or selected before the prerequisite is satisfied.
    context = _read_context.get()
    evidence = context.visible_inspections() if context is not None else ()
    if not evidence:
        return require_inspection(path)
    try:
        snapshot = read_snapshot(work_root, path, FILE_INSPECTION)
    except FileInspectionError:
        return require_inspection(path)
    inspection = _find_matching_inspection(work_root, snapshot, evidence)
    if inspection is None:
        return require_inspection(path)
    if is_pdf:
        return read_pdf(snapshot, start_line, end_line, FILE_INSPECTION)
    return _read_line_range(snapshot, inspection, start_line, end_line)


def _find_matching_inspection(
    work_root: Path, snapshot: FileSnapshot, evidence: tuple[InspectionEvidence, ...]
) -> InspectionEvidence | None:
    if snapshot.content is None:
        return None
    digest = hash_content(snapshot.content)
    for inspection in evidence:
        try:
            current = read_snapshot(work_root, str(inspection.path), FILE_INSPECTION)
        except FileInspectionError:
            continue
        if (
            current.content is not None
            and hash_content(current.content) == inspection.digest == digest
            and current.path == snapshot.path
        ):
            return inspection
    return None


def _read_line_range(
    snapshot: FileSnapshot,
    inspection: InspectionEvidence,
    start_line: int,
    end_line: int,
) -> dict[str, object]:
    if not inspection.readable:
        raise FileInspectionError(_NO_READABLE_LINES)
    assert snapshot.content is not None
    decoded = decode_text(snapshot.content, snapshot.path.suffix.casefold())
    if decoded is None:
        raise FileInspectionError(_NO_READABLE_LINES)
    text, encoding = decoded
    lines = physical_lines(text)
    if not lines:
        raise FileInspectionError(
            "File is empty; no positive line interval can be read."
        )
    if end_line > len(lines):
        raise FileInspectionError(
            f"start_line/end_line are outside the file (1..{len(lines)}); "
            "choose both inclusive indices from inspect-file."
        )
    return {
        "ok": True,
        "path": snapshot.relative_path,
        "encoding": encoding,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": len(lines),
        "lines": [
            {"line": number, "text": lines[number - 1]}
            for number in range(start_line, end_line + 1)
        ],
    }
