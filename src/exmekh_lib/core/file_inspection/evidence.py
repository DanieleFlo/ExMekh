"""Private proof of the exact bytes and report produced by an inspection."""

from dataclasses import dataclass
from pathlib import Path

from exmekh_lib.utils.xml import stringify_tool_result

from .workspace import FileSnapshot


def hash_content(content: bytes) -> bytes:
    """Load the fast hash dependency at execution, like the inspection parsers."""
    from xxhash import xxh3_128_digest

    return xxh3_128_digest(content)


@dataclass(frozen=True, slots=True)
class InspectionEvidence:
    path: Path
    digest: bytes
    report_content: str
    readable: bool
    successful: bool


class InspectionReport(dict[str, object]):
    """A normal structured report; proof is never serialized to the model/cache."""

    def __init__(self, report: dict[str, object], snapshot: FileSnapshot) -> None:
        super().__init__(report)
        assert snapshot.content is not None
        self._evidence = InspectionEvidence(
            snapshot.path,
            hash_content(snapshot.content),
            stringify_tool_result(report),
            report.get("family") == "text",
            report.get("analysis_status") in {"complete", "unsupported"},
        )

    @property
    def evidence(self) -> InspectionEvidence:
        return self._evidence
