from exmekh_lib.core.agents.capability.parser import CapabilityMarkdownParser
from exmekh_lib.core.agents.hds.markdown import HDSMarkdown
from exmekh_lib.core.agents.hds.metadata import HDSMetadata, HDSMetadataParser


class HDSMarkdownParser(CapabilityMarkdownParser[HDSMetadata, HDSMarkdown]):
    def __init__(
        self,
        metadata_parser: HDSMetadataParser | None = None,
    ) -> None:
        super().__init__(metadata_parser or HDSMetadataParser(), HDSMarkdown)
