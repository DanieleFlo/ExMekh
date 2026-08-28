from qs.core.agents.capability.collection import MarkdownCapabilityCollection
from qs.core.agents.deep_skill.parser import SkillMarkdownParser
from qs.core.agents.exit_skill.exit_skill import ExitSkill


class ExitSkills(MarkdownCapabilityCollection[ExitSkill]):
    resource_class = ExitSkill
    parser_factory = SkillMarkdownParser
    resource_label = "Exit-skill"
    duplicate_label = "exit-skill"
