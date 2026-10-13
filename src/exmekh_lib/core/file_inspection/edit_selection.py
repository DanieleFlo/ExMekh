"""Compose unique literal selections against one immutable file snapshot."""

from dataclasses import dataclass

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.models.edit_document import EditElement, literal_selection_offset

from .errors import FileInspectionError


@dataclass(frozen=True, slots=True)
class _Replacement:
    start: int
    end: int
    text: str


def select_replacements(text: str, elements: tuple[EditElement, ...]) -> str:
    if not elements:
        raise FileInspectionError(FILE_EDITING.selection_required)
    if not text:
        if len(elements) != 1 or elements[0].old_text:
            raise FileInspectionError(FILE_EDITING.selection_error)
        return elements[0].replacement
    replacements: list[_Replacement] = []
    for element in elements:
        old = element.old_text
        offset = literal_selection_offset(text, old)
        if offset is None:
            raise FileInspectionError(FILE_EDITING.selection_error)
        replacement = _Replacement(offset, offset + len(old), element.replacement)
        if any(
            max(replacement.start, other.start) < min(replacement.end, other.end)
            for other in replacements
        ):
            raise FileInspectionError(FILE_EDITING.conflict_error)
        replacements.append(replacement)
    result = text
    for replacement in sorted(replacements, key=lambda item: item.start, reverse=True):
        result = (
            result[: replacement.start] + replacement.text + result[replacement.end :]
        )
    return result
