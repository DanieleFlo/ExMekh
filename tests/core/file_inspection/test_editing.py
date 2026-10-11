"""Independent client validation against forged or unconstrained server selections."""

import codecs

import pytest

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.file_inspection.edit_filesystem import encode_replacement
from exmekh_lib.core.file_inspection.edit_selection import select_replacement
from exmekh_lib.core.file_inspection.errors import FileInspectionError


@pytest.mark.parametrize(
    "text,old,new,expected",
    [
        (
            "  indented\r\nend",
            "  indented",
            "longer replacement",
            "longer replacement\r\nend",
        ),
        ("one\r\ntwo\rthree\n", "one\r\ntwo", "", "\rthree\n"),
        ("one\ntwo", FILE_EDITING.line_marker(2), "insert\n", "one\ninsert\ntwo"),
        ("one\ntwo", FILE_EDITING.line_marker(3), "APPEND", "one\ntwoAPPEND"),
        ("one\n", FILE_EDITING.line_marker(2), "tail", "one\ntail"),
        ("", FILE_EDITING.line_marker(1), "", ""),
        ("one", FILE_EDITING.all_marker, "", ""),
        ("one", FILE_EDITING.all_marker, FILE_EDITING.delete_marker, None),
    ],
)
def test_literal_and_marker_selection_preserves_exact_text(text, old, new, expected):
    assert select_replacement(text, old, new, exists=True) == expected


@pytest.mark.parametrize(
    "text,old,new",
    [
        ("  indented", "indented", "new"),
        ("prefix value", "value", "new"),
        ("value\nprefix value", "value", "new"),
        ("aaa", "aa", "new"),
        ("xabc\nabc", "abc", "new"),
        ("one", "", "new"),
        ("one", "missing", "new"),
        ("one", FILE_EDITING.line_marker("0"), "new"),
        ("one", FILE_EDITING.line_marker("-1"), "new"),
        ("one", FILE_EDITING.line_marker("01"), "new"),
        ("one", FILE_EDITING.line_marker(3), "new"),
        ("one", FILE_EDITING.line_marker("1.0"), "new"),
        ("one", (" " + FILE_EDITING.all_marker), "new"),
        ("one", (FILE_EDITING.all_marker + FILE_EDITING.all_marker), "new"),
        ("one", FILE_EDITING.delete_marker, "new"),
        ("one", "one", FILE_EDITING.delete_marker),
        ("one", FILE_EDITING.all_marker, ("text " + FILE_EDITING.line_marker(1))),
        ("one", FILE_EDITING.all_marker, FILE_EDITING.marker_prefix),
        ("one", FILE_EDITING.all_marker, ("\n" + FILE_EDITING.marker_prefix + "\n")),
        (FILE_EDITING.marker_prefix, FILE_EDITING.marker_prefix, "new"),
        ("one", "one", "\x00"),
        ("one", True, "new"),
    ],
)
def test_selection_cannot_bypass_boundaries_uniqueness_or_reserved_markers(
    text, old, new
):
    with pytest.raises(FileInspectionError):
        select_replacement(text, old, new, exists=True)


@pytest.mark.parametrize("old", [FILE_EDITING.all_marker, FILE_EDITING.line_marker(1)])
def test_creation_allows_only_the_two_creation_markers(old):
    assert select_replacement("", old, "created", exists=False) == "created"
    with pytest.raises(FileInspectionError):
        select_replacement("", old, FILE_EDITING.delete_marker, exists=False)


@pytest.mark.parametrize(
    "codec,bom",
    [
        ("utf-16-be", codecs.BOM_UTF16_BE),
        ("utf-16-le", codecs.BOM_UTF16_LE),
        ("utf-32-be", codecs.BOM_UTF32_BE),
        ("utf-32-le", codecs.BOM_UTF32_LE),
    ],
)
def test_encoding_preserves_the_original_bom_and_byte_order(codec, bom):
    assert encode_replacement(
        "uno\r\ndue", bom + "old".encode(codec), codec.split("-")[0]
    ) == bom + "uno\r\ndue".encode(codec)
