from typing import cast

import pytest

from exmekh_lib.core.agents.deep_skill.markdown import SkillMarkdown
from exmekh_lib.core.agents.deep_skill.parser import SkillMarkdownParser


def test_parse_returns_markdown_with_references() -> None:
    result = SkillMarkdownParser().parse(
        "---\ndescription: Example skill\n---\nUse {{!other}} then {{!other}}"
    )

    assert isinstance(result, SkillMarkdown)
    assert result.body == "Use other then other"
    assert result.references == ("other",)


def test_does_not_execute_jinja_from_markdown_body() -> None:
    result = SkillMarkdownParser().parse(
        "---\n"
        "description: Example skill\n"
        "---\n"
        "Keep {{ 7 * 7 }} and {% set value = 1 %} literal."
    )

    assert result.body == "Keep {{ 7 * 7 }} and {% set value = 1 %} literal."
    assert result.references == ()


def test_rejects_non_string_markdown() -> None:
    with pytest.raises(TypeError, match="markdown must be a string"):
        SkillMarkdownParser().parse(cast(str, object()))


def test_rejects_non_string_metadata_keys() -> None:
    with pytest.raises(
        ValueError,
        match="Metadata field names must be strings",
    ):
        SkillMarkdownParser().parse(
            "---\n1: invalid\ndescription: Example skill\n---\nBody"
        )


@pytest.mark.parametrize(
    ("markdown", "error", "message"),
    [
        ("Body only", ValueError, "valid YAML front matter"),
        ("---\n[invalid\n---\nBody", ValueError, "Invalid YAML metadata"),
        ("---\n\n---\nBody", ValueError, "Metadata section cannot be empty"),
        ("---\nvalue\n---\nBody", TypeError, "must be a YAML mapping"),
    ],
)
def test_rejects_invalid_front_matter(
    markdown: str,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        SkillMarkdownParser().parse(markdown)
