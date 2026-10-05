"""Analysis records and an encapsulated, bounded navigation index."""

from dataclasses import dataclass, field
from typing import Literal

from exmekh_lib.config.file_inspection import FileInspectionSettings

from .selection import query_terms


@dataclass(frozen=True, slots=True)
class Region:
    kind: str
    name: str
    start_line: int | None = None
    end_line: int | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"kind": self.kind, "name": self.name}
        if self.start_line is not None:
            result.update(start_line=self.start_line, end_line=self.end_line)
        return result


@dataclass(frozen=True, slots=True)
class RegionPage:
    regions: tuple[Region, ...]
    offset: int
    max_items: int
    indexed_items: int
    index_truncated: bool
    labels_truncated: bool
    ranking: Literal["lexical", "document_order"]

    def to_dict(self) -> dict[str, object]:
        next_offset = self.offset + len(self.regions)
        has_more = next_offset < self.indexed_items
        return {
            "regions": [region.to_dict() for region in self.regions],
            "selection": {
                "offset": self.offset,
                "max_items": self.max_items,
                "indexed_items": self.indexed_items,
                "index_truncated": self.index_truncated,
                "labels_truncated": self.labels_truncated,
                "has_more": has_more,
                "next_offset": next_offset if has_more else None,
                "ranking": self.ranking,
            },
        }


class RegionIndex:
    """Keep navigation bounded; expose selection without mutable list access."""

    def __init__(self, settings: FileInspectionSettings) -> None:
        self._settings = settings
        self._regions: list[Region] = []
        self._truncated = False
        self._labels_truncated = False

    @property
    def settings(self) -> FileInspectionSettings:
        return self._settings

    def is_empty(self) -> bool:
        return not self._regions

    def mark_truncated(self) -> None:
        self._truncated = True

    def add(
        self, kind: str, name: str, start: int | None = None, end: int | None = None
    ) -> None:
        """Add a region in discovery order, retaining at most max_regions."""
        if len(self._regions) >= self._settings.max_regions:
            self.mark_truncated()
            return
        self._labels_truncated |= len(name) > self._settings.max_label_chars
        label = name[: self._settings.max_label_chars]
        self._regions.append(Region(kind, label, start, end))

    def select(self, *, max_items: int, offset: int, query: str | None) -> RegionPage:
        regions = sorted(
            self._regions,
            key=lambda region: (region.start_line or 0, region.end_line or 0),
        )
        if query is not None:
            terms = query_terms(query)
            regions.sort(
                key=lambda region: (
                    -len(terms & query_terms(region.name + " " + region.kind))
                )
            )
        return RegionPage(
            tuple(regions[offset : offset + max_items]),
            offset,
            max_items,
            len(regions),
            self._truncated,
            self._labels_truncated,
            "lexical" if query is not None else "document_order",
        )


@dataclass(slots=True)
class Analysis:
    format: str
    family: str
    index: RegionIndex
    status: Literal["complete", "invalid_format", "unsupported", "skipped_limit"] = (
        "complete"
    )
    details: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
