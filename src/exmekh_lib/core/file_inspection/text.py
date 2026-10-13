"""Decode bounded text and dispatch format-specific navigation."""

import csv
from xml.parsers.expat import ExpatError

import yaml

from exmekh_lib.config.file_inspection import FILE_INSPECTION_TEXT

from .decoding import decode_text
from .lines import physical_lines
from .models import Analysis, RegionIndex
from .navigation import markdown_navigation, python_navigation
from .structured import analyze_mapping, analyze_table, analyze_toml, analyze_xml


def analyze_text(raw: bytes, extension: str, index: RegionIndex) -> Analysis:
    decoded = decode_text(raw, extension)
    if decoded is None:
        return Analysis(
            "binary",
            "binary",
            index,
            status="unsupported",
            warnings=[FILE_INSPECTION_TEXT.unsupported_binary_warning],
        )
    text, encoding = decoded
    format_name = {
        ".py": "python",
        ".pyi": "python",
        ".md": "markdown",
        ".markdown": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".xml": "xml",
        ".svg": "xml",
        ".csv": "csv",
        ".tsv": "tsv",
    }.get(extension, "text")
    result = Analysis(format_name, "text", index)
    lines = physical_lines(text)
    crlf = text.count("\r\n")
    newlines = {
        "crlf": crlf,
        "lf": text.count("\n") - crlf,
        "cr": text.count("\r") - crlf,
    }
    active = [name for name, count in newlines.items() if count]
    result.details.update(
        encoding=encoding,
        total_lines=len(lines),
        characters=len(text),
        non_empty_lines=sum(bool(line.strip()) for line in lines),
        maximum_line_length=max(map(len, lines), default=0),
        newline_style=active[0] if len(active) == 1 else "mixed" if active else "none",
    )
    try:
        if format_name == "python":
            result.details.update(python_navigation(text, index))
        elif format_name == "markdown":
            markdown_navigation(text, index)
        elif format_name in {"json", "yaml"}:
            analyze_mapping(text, result)
        elif format_name == "toml":
            analyze_toml(text, result)
        elif format_name == "xml":
            analyze_xml(text, result)
        elif format_name in {"csv", "tsv"}:
            analyze_table(text, result)
    except (ValueError, SyntaxError, csv.Error, yaml.YAMLError, ExpatError) as error:
        result.status = "invalid_format"
        result.warnings.append(
            FILE_INSPECTION_TEXT.invalid_text_warning.format(
                format_name=format_name,
                error_type=type(error).__name__,
            )
        )
        if isinstance(error, SyntaxError):
            result.details.update(syntax_valid=False, error_line=error.lineno)
    except RecursionError:
        result.status = "skipped_limit"
        result.warnings.append(FILE_INSPECTION_TEXT.parser_depth_warning)
    if index.is_empty() and lines:
        index.add("content", FILE_INSPECTION_TEXT.content_region_label, 1, len(lines))
    return result
