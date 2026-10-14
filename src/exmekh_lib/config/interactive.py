"""Fixed wire identity and endpoints for interactive Responses calls."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InteractiveProtocolSettings:
    version: int = 2
    snapshot_kind: str = "file_snapshot"
    capabilities_path: str = "/agentic/capabilities"
    responses_path: str = "/responses"


INTERACTIVE_PROTOCOL = InteractiveProtocolSettings()
