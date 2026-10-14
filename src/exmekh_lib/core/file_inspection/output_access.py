"""Bind optional runtime file access without depending on tool execution/cache."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from exmekh_lib.config.cache import CACHE_FORMAT
from exmekh_lib.interfaces.file_read_access import FileReadAccess

_output_access: ContextVar[FileReadAccess | None] = ContextVar(
    CACHE_FORMAT.output_access_context_name, default=None
)


@contextmanager
def output_file_access(access: FileReadAccess | None) -> Generator[None]:
    token = _output_access.set(access)
    try:
        yield
    finally:
        _output_access.reset(token)


def output_file_read_limit(path: Path, default: int) -> int:
    access = _output_access.get()
    return default if access is None else access.read_limit(path, default)
