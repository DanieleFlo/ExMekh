"""Resource budgets for read-only workspace file analysis."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FileInspectionSettings:
    max_file_bytes: int = 8 * 1024 * 1024
    max_regions: int = 10_000
    default_max_items: int = 50
    max_items: int = 500
    max_label_chars: int = 200
    max_query_chars: int = 1_000
    max_structure_nodes: int = 100_000
    max_pdf_pages: int = 2_000
    max_structure_depth: int = 64
    max_archive_entries: int = 10_000
    max_expanded_bytes: int = 64 * 1024 * 1024
    table_rows_per_region: int = 100
    max_image_pixels: int = 40_000_000


FILE_INSPECTION = FileInspectionSettings()
