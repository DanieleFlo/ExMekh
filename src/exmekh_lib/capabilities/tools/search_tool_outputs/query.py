"""Validated filters and pagination for a node's output search."""

import json
from dataclasses import dataclass

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.utils.tool_errors import ToolInputError, ToolTypeError


@dataclass(frozen=True, slots=True)
class ToolOutputQuery:
    query: str = ""
    tool_name: str | None = None
    skill_path: str | None = None
    offset: int = 0
    limit: int = RUNTIME.memory_default_page_limit

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise ToolTypeError(
                RUNTIME.output_query_required, code="invalid_output_filter"
            )
        if self.tool_name is not None and (
            not isinstance(self.tool_name, str) or not self.tool_name.strip()
        ):
            raise ToolInputError(
                RUNTIME.output_tool_name_required, code="invalid_output_filter"
            )
        if self.skill_path is not None and (
            not isinstance(self.skill_path, str) or not self.skill_path.startswith("/")
        ):
            raise ToolInputError(
                RUNTIME.output_skill_path_required, code="invalid_output_filter"
            )
        if type(self.offset) is not int or self.offset < 0:
            raise ToolInputError(
                RUNTIME.output_offset_required, code="invalid_output_filter"
            )
        if (
            type(self.limit) is not int
            or not 1 <= self.limit <= RUNTIME.memory_max_page_limit
        ):
            raise ToolInputError(
                RUNTIME.output_limit_required.format(
                    limit=RUNTIME.memory_max_page_limit
                ),
                code="invalid_output_filter",
            )

    def matches(self, record: ToolCallRecord) -> bool:
        return (
            (self.tool_name is None or record.name == self.tool_name)
            and (self.skill_path is None or record.skill_path == self.skill_path)
            and self.query.casefold()
            in json.dumps(record.to_dict(), ensure_ascii=False).casefold()
        )

    def select(
        self, records: tuple[ToolCallRecord, ...]
    ) -> tuple[list[ToolCallRecord], int, int | None]:
        matches = [record for record in records if self.matches(record)]
        end = self.offset + self.limit
        return (
            matches[self.offset : end],
            len(matches),
            end if end < len(matches) else None,
        )
