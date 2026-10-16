"""Explicit history compaction without mutating the supplied conversation."""

from .compactor import HistoryCompactor
from .models import CompactionResult

__all__ = ["CompactionResult", "HistoryCompactor"]
