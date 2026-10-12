"""Plan non-overlapping XML edits against one immutable physical-text snapshot."""

from dataclasses import dataclass

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.models.edit_document import EditElement

from .errors import FileInspectionError
from .lines import SourceLines


def _line_number(digits: str, maximum: int) -> int:
    significant = digits.lstrip("0")
    if not significant or len(significant) > len(str(maximum)):
        raise FileInspectionError(FILE_EDITING.range_error)
    number = int(significant)
    if number > maximum:
        raise FileInspectionError(FILE_EDITING.range_error)
    return number


@dataclass(frozen=True, slots=True)
class _Replacement:
    start: int
    end: int
    text: str


def _selection(text: str, element: EditElement, lines: SourceLines) -> _Replacement:
    selection = element.selection
    replacement = element.replacement
    assert replacement is not None
    if selection.kind == "all":
        return _Replacement(0, len(text), replacement)
    if selection.kind == "line":
        if selection.start_line.startswith("0"):
            raise FileInspectionError(FILE_EDITING.range_error)
        line = _line_number(selection.start_line, lines.physical_count + 1)
        offset = lines.start_offset(line)
        return _Replacement(offset, offset, replacement)
    if selection.kind == "range":
        start = _line_number(selection.start_line, lines.physical_count)
        end = _line_number(selection.end_line, lines.physical_count)
        if start > end:
            raise FileInspectionError(FILE_EDITING.range_error)
        return _Replacement(
            lines.start_offset(start),
            lines.start_offset(end + 1),
            replacement,
        )
    old = selection.text
    offset = text.find(old)
    if (
        not old
        or offset < 0
        or text.find(old, offset + 1) >= 0
        or not lines.is_line_start(offset)
    ):
        raise FileInspectionError(FILE_EDITING.selection_error)
    return _Replacement(offset, offset + len(old), replacement)


def _conflict(left: _Replacement, right: _Replacement) -> bool:
    if left.start == left.end and right.start == right.end:
        return left.start == right.start
    if left.start == left.end:
        return right.start < left.start < right.end
    if right.start == right.end:
        return left.start < right.start < left.end
    return max(left.start, right.start) < min(left.end, right.end)


def select_replacements(
    text: str, elements: tuple[EditElement, ...], *, exists: bool
) -> str | None:
    """Validate every selection before composing replacements in original offsets."""
    if not elements:
        raise FileInspectionError("At least one file edit is required.")
    if (
        any(
            element.selection.kind == "all" or element.replacement is None
            for element in elements
        )
        and len(elements) != 1
    ):
        raise FileInspectionError(FILE_EDITING.conflict_error)
    for element in elements:
        if not exists and not (
            element.selection.kind == "all"
            or (
                element.selection.kind == "line"
                and _line_number(element.selection.start_line, 1) == 1
            )
        ):
            raise FileInspectionError(FILE_EDITING.creation_required)
        if element.replacement is None:
            if not exists or element.selection.kind != "all":
                raise FileInspectionError(FILE_EDITING.creation_required)
            return None
    lines = SourceLines(text)
    replacements = [_selection(text, element, lines) for element in elements]
    for index, replacement in enumerate(replacements):
        if any(_conflict(replacement, other) for other in replacements[:index]):
            raise FileInspectionError(FILE_EDITING.conflict_error)
    result = text
    # At a boundary, replacement precedes insertion in this reverse traversal,
    # so the insertion remains immediately before the replacement.
    for replacement in sorted(
        replacements, key=lambda item: (item.start, item.end), reverse=True
    ):
        result = (
            result[: replacement.start] + replacement.text + result[replacement.end :]
        )
    return result
