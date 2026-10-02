from exmekh_lib.core.agents.capability.parser import CapabilityMarkdownParser
from exmekh_lib.core.agents.deep_skill.markdown import SkillMarkdown
from exmekh_lib.core.agents.deep_skill.metadata import (
    DeepSkillMetadata,
    DeepSkillMetadataParser,
)


class SkillMarkdownParser(CapabilityMarkdownParser[DeepSkillMetadata, SkillMarkdown]):
    def __init__(
        self,
        metadata_parser: DeepSkillMetadataParser | None = None,
    ) -> None:
        super().__init__(metadata_parser or DeepSkillMetadataParser(), SkillMarkdown)
