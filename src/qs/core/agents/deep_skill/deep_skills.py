from qs.core.agents.capability.collection import MarkdownCapabilityCollection
from qs.core.agents.deep_skill.deep_skill import DeepSkill
from qs.core.agents.deep_skill.parser import SkillMarkdownParser


class DeepSkills(MarkdownCapabilityCollection[DeepSkill]):
    resource_label = "Deep-skill"
    duplicate_label = "deep-skill"

    @property
    def resource_class(self) -> type[DeepSkill]:
        return DeepSkill

    @property
    def parser_factory(self) -> type[SkillMarkdownParser]:
        return SkillMarkdownParser
