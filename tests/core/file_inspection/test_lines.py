"""Physical line positions preserve CR/LF and ignore Unicode display separators."""

import pytest

from exmekh_lib.core.file_inspection.lines import SourceLines, physical_lines


@pytest.mark.parametrize(
    "text,expected_lines,starts,eof_line",
    [
        ("", [], [], 1),
        ("one", ["one"], [0], 1),
        ("a\r\nb\rc\n", ["a", "b", "c"], [0, 3, 5], 4),
        ("\r\n\r\n", ["", ""], [0, 2], 3),
        ("a\u2028b\u0085c", ["a\u2028b\u0085c"], [0], 1),
    ],
)
def test_physical_lines_and_offsets_agree_without_normalizing_text(
    text, expected_lines, starts, eof_line
):
    source = SourceLines(text)
    assert physical_lines(text) == expected_lines
    assert source.physical_count == len(expected_lines)
    assert [source.start_offset(line) for line in range(1, len(starts) + 1)] == starts
    assert source.start_offset(source.physical_count + 1) == len(text)
    assert source.line_number(len(text)) == eof_line
    for offset in range(-1, len(text) + 2):
        assert source.is_line_start(offset) == (offset in starts)


@pytest.mark.parametrize("line", [-1, 0, 4])
def test_line_offsets_reject_positions_beyond_the_eof_insertion_point(line):
    with pytest.raises(ValueError, match="outside"):
        SourceLines("one\ntwo").start_offset(line)
