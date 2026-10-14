"""Structured references to temporary text, without keeping the original output."""

from dataclasses import dataclass
from typing import Any, Self

from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.config.tool_output import TOOL_OUTPUT_TEXT
from exmekh_lib.utils.xml import stringify_tool_result


@dataclass(frozen=True, slots=True)
class OutputTextFile:
    path: str
    token_count: int
    line_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError(MODEL_TEXT.output_file_path_must_be_non_empty)
        for value in (self.token_count, self.line_count):
            if type(value) is not int or value < 0:
                raise ValueError(
                    MODEL_TEXT.output_file_counts_must_be_non_negative_integers
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "token_count": self.token_count,
            "line_count": self.line_count,
        }


@dataclass(frozen=True, slots=True)
class ExternalToolOutput:
    token_count: int
    line_count: int
    limit: int
    files: tuple[OutputTextFile, ...]

    def __post_init__(self) -> None:
        if type(self.limit) is not int or self.limit < 1:
            raise ValueError(MODEL_TEXT.output_limit_must_be_positive)
        for value in (self.token_count, self.line_count):
            if type(value) is not int or value < 0:
                raise ValueError(MODEL_TEXT.output_counts_must_be_non_negative_integers)
        if (
            not isinstance(self.files, tuple)
            or not self.files
            or not all(isinstance(item, OutputTextFile) for item in self.files)
        ):
            raise ValueError(MODEL_TEXT.external_output_must_contain_output_text_files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": TOOL_OUTPUT_TEXT.exported.format(limit=self.limit),
            "token_count": self.token_count,
            "line_count": self.line_count,
            "limit": self.limit,
            "files": [item.to_dict() for item in self.files],
        }

    def render(self) -> str:
        return stringify_tool_result(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        return cls(
            token_count=value["token_count"],
            line_count=value["line_count"],
            limit=value["limit"],
            files=tuple(OutputTextFile(**item) for item in value["files"]),
        )
