from qs.core.agents.capability.parser import CapabilityMarkdownParser
from qs.core.agents.deep_skill.markdown import SkillMarkdown
from qs.core.agents.deep_skill.metadata import (
    DeepSkillMetadata,
    DeepSkillMetadataParser,
)


class SkillMarkdownParser(CapabilityMarkdownParser[DeepSkillMetadata, SkillMarkdown]):
    def __init__(
        self,
        metadata_parser: DeepSkillMetadataParser | None = None,
    ) -> None:
        super().__init__(metadata_parser or DeepSkillMetadataParser(), SkillMarkdown)
