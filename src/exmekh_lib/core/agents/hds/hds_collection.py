from exmekh_lib.core.agents.capability.collection import MarkdownCapabilityCollection
from exmekh_lib.core.agents.hds.hds import HDS
from exmekh_lib.core.agents.hds.parser import HDSMarkdownParser


class HDSCollection(MarkdownCapabilityCollection[HDS]):
    resource_label = "HDS"
    duplicate_label = "HDS"

    @property
    def resource_class(self) -> type[HDS]:
        return HDS

    @property
    def parser_factory(self) -> type[HDSMarkdownParser]:
        return HDSMarkdownParser
