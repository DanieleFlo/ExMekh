from enum import StrEnum

from exmekh_lib.config.capabilities import CAPABILITY_FORMAT


class ResourceType(StrEnum):
    MARKDOWN_CAPABILITY = CAPABILITY_FORMAT.markdown_resource_type
    DEEP_SKILL = CAPABILITY_FORMAT.deep_skill_resource_type
    HDS = CAPABILITY_FORMAT.hds_resource_type
