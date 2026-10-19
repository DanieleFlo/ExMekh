"""Syntax and diagnostics for capability-dependent instruction templates."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CapabilityPromptSettings:
    tokens: str = r"</?if-(?:tool|skill)\b[^>]*>|</?if-(?:tool|skill)\b[^>]*$"
    literal_start: str = r"(?m)^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*(?:\n|$)|(?P<comment><!--)|(?P<inline>`+)"
    invalid_condition: str = 'Invalid capability condition: {tag}. Use <if-tool> or <if-skill> with exactly one of name="name" or any-of="name1 name2", an optional scope="direct|reachable|selected", and a matching closing tag.'
    unclosed_condition: str = "unclosed if-{kind}"
    unknown_condition: str = "Unknown {kind} in capability condition: {name}"
    unavailable_reference: str = "Instruction references unavailable capability: {name}. Declare it in this scope, use a reachable skill, or guard the text with a capability condition."
    dependency_missing: str = "Capability '{name}' requires {dependencies}. Include those capabilities or exclude/replace '{name}'; excluded dependencies are never enabled automatically."
    optional_overlap: str = "Capabilities cannot be both required and optional: {names}"
    invalid_contract: str = "Override for '{name}' must accept the runtime arguments {arguments} without additional required arguments."
    invalid_result: str = (
        "Override for '{name}' returned {actual}; the runtime requires {expected}."
    )


CAPABILITY_PROMPTS = CapabilityPromptSettings()
