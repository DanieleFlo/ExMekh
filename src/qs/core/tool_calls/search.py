from __future__ import annotations

import json
from typing import Any

from qs.core.cache.node_cache import NodeCache
from qs.core.models.tool import Tool


class ToolCallSearch:
    """Expose a read-only search tool scoped to one node's saved outputs."""

    def __init__(self, cache: NodeCache) -> None:
        self._cache = cache

    def as_tool(self) -> Tool:
        return Tool(self.search, name="search_saved_tool_outputs")

    def search(
        self,
        query: str = "",
        skill_path: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search saved calls by text and exact skill path, with offset pagination.

        Searches names, IDs, arguments and outputs. Returns only this node's
        records. Omit skill_path to search all its skill paths.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if skill_path is not None and (
            not isinstance(skill_path, str) or not skill_path.startswith("/")
        ):
            raise ValueError("skill_path must be an absolute skill path")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        needle = query.casefold()
        matches = [
            record.to_dict()
            for record in self._cache.get_all()
            if (skill_path is None or record.skill_path == skill_path)
            and needle in json.dumps(record.to_dict(), ensure_ascii=False).casefold()
        ]
        end = offset + limit
        return {
            "records": matches[offset:end],
            "total": len(matches),
            "next_offset": end if end < len(matches) else None,
        }
