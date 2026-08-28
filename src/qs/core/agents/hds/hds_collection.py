from qs.core.agents.capability.collection import MarkdownCapabilityCollection
from qs.core.agents.hds.hds import HDS
from qs.core.agents.hds.parser import HDSMarkdownParser


class HDSCollection(MarkdownCapabilityCollection[HDS]):
    resource_class = HDS
    parser_factory = HDSMarkdownParser
    resource_label = "HDS"
    duplicate_label = "HDS"
