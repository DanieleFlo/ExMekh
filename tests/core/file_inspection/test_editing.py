"""Independent client validation against forged or unconstrained server selections."""

import codecs

import pytest

from exmekh_lib.core.file_inspection.edit_filesystem import encode_replacement
from exmekh_lib.core.file_inspection.edit_selection import select_replacements
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.models.edit_document import EditDocument
from tests.interactive_support import edit_fragment


@pytest.mark.parametrize(
    "text,old,new,expected",
    [
        ("  indented\r\nend", "  indented", "longer", "longer\r\nend"),
        ("one\r\ntwo\rthree\n", "one\r\ntwo", "", "\rthree\n"),
        ("one\ntwo", "two", "insert\ntwo", "one\ninsert\ntwo"),
        ("one\ntwo", "two", "twoAPPEND", "one\ntwoAPPEND"),
        ("one\n", "one\n", "one\ntail", "one\ntail"),
        ("", "", "", ""),
        ("one", "one", "", ""),
        ("@ex_meckh_all", "@ex_meckh_all", "@ex_meckh_delete", "@ex_meckh_delete"),
    ],
)
def test_selection_preserves_exact_text(text, old, new, expected):
    document = EditDocument.parse(edit_fragment("file.txt", old, new))
    assert select_replacements(text, document.elements) == expected


@pytest.mark.parametrize(
    "text,old",
    [
        ("  indented", "indented"),
        ("prefix value", "value"),
        ("value\nprefix value", "value"),
        ("aaa", "aa"),
        ("xabc\nabc", "abc"),
        ("one", ""),
        ("one", "missing"),
        ("one", '<line number="0"/>'),
        ("one", '<line number="3"/>'),
    ],
)
def test_selection_rejects_ambiguity_midline_empty_and_invalid_lines(text, old):
    document = EditDocument.parse(edit_fragment("file.txt", old, "new"))
    with pytest.raises(FileInspectionError):
        select_replacements(text, document.elements)


@pytest.mark.parametrize(
    "text,old,code,correction",
    [
        ("one", "", "old_text_empty", "anchor"),
        ("one", "missing", "old_text_not_found", "exactly"),
        ("value\nprefix value", "value", "old_text_ambiguous", "adjacent"),
        ("aaa", "aa", "old_text_ambiguous", "adjacent"),
        ("  value", "value", "old_text_not_at_line_start", "indentation"),
    ],
)
def test_selection_diagnostics_explain_distinct_corrections(
    text, old, code, correction
):
    document = EditDocument.parse(edit_fragment("file.txt", old, "new"))
    with pytest.raises(FileInspectionError) as caught:
        select_replacements(text, document.elements)
    assert caught.value.code == code
    assert correction in caught.value.hint
    corrected = EditDocument.parse(edit_fragment("file.txt", text, "new"))
    assert select_replacements(text, corrected.elements) == "new"


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
