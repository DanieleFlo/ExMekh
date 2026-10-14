from __future__ import annotations

import logging
import re
from pathlib import Path

from exmekh_lib.config.cache import CACHE_FORMAT
from exmekh_lib.config.cache_text import CACHE_TEXT
from exmekh_lib.core.cache.paths import cache_component
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.utils.file import read_json_from_path, write_json_atomically

logger = logging.getLogger(__name__)


class ToolCallStore:
    """Read and atomically replace records in one explicit directory.

    SessionCache owns lifecycle locking; standalone callers own their directory.
    Reads never create directories and ignore malformed records.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def save(self, record: ToolCallRecord) -> Path:
        if not isinstance(record, ToolCallRecord):
            raise TypeError(CACHE_TEXT.record_must_be_a_toolcallrecord_instance)
        payload = record.to_dict()
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / self._filename(record.id)
        write_json_atomically(path, payload)
        return path

    def get(self, call_id: str) -> ToolCallRecord | None:
        if not isinstance(call_id, str) or not call_id.strip():
            return None
        call_id = call_id.strip()
        for filename in (
            self._filename(call_id),
            self._legacy_filename(call_id),
        ):
            record = self._read(self.directory / filename)
            if record is not None and record.id == call_id:
                return record
        return None

    def get_all(self) -> list[ToolCallRecord]:
        records: dict[str, ToolCallRecord] = {}
        for path in self.directory.glob(f"*{CACHE_FORMAT.record_suffix}"):
            record = self._read(path)
            if record is None:
                continue
            if record.id not in records or path.name == self._filename(record.id):
                records[record.id] = record
        return sorted(records.values(), key=lambda record: record.id)

    def list_ids(self) -> list[str]:
        return [record.id for record in self.get_all()]

    @classmethod
    def _read(cls, path: Path) -> ToolCallRecord | None:
        if path.is_symlink():
            return None
        try:
            record = ToolCallRecord.from_dict(read_json_from_path(path))
            expected_names = (
                cls._filename(record.id),
                cls._legacy_filename(record.id),
            )
            return record if path.name in expected_names else None
        except (OSError, ValueError, TypeError, KeyError) as error:
            logger.debug(CACHE_TEXT.cannot_read_tool_call_record, path, error)
            return None

    @staticmethod
    def _filename(call_id: str) -> str:
        return f"{cache_component(call_id)}{CACHE_FORMAT.record_suffix}"

    @staticmethod
    def _legacy_filename(call_id: str) -> str:
        """Read old archives only after checking the full ID inside the record."""
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", call_id) + CACHE_FORMAT.record_suffix
