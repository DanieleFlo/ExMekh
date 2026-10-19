"""Reserved conditional tags leave ordinary Markdown and XML examples intact."""

import pytest

from exmekh_lib.core.agents.capability.conditions import (
    CapabilityPromptContext,
    render_conditions,
)
from exmekh_lib.core.agents.capability.references import parse_references


@pytest.fixture
def context():
    return CapabilityPromptContext(
        frozenset({"yes", "no"}),
        frozenset({"skill"}),
        frozenset({"yes"}),
        frozenset({"skill"}),
        frozenset({"yes"}),
        frozenset(),
        frozenset({"yes"}),
        frozenset({"skill"}),
    )


def test_nested_conditions_preserve_whitespace_and_nonconditional_markup(context):
    text = 'a<&>\n<if-tool name="yes"> x <if-tool name="no">hidden</if-tool> y </if-tool>\r\nz'
    assert render_conditions(text, context) == "a<&>\n x  y \r\nz"


@pytest.mark.parametrize(
    "text",
    [
        '<if-tool name="yes">unclosed',
        "</if-tool>",
        '<if-tool name="yes"></if-skill>',
        "<if-tool/>",
        '<if-tool name="yes" unknown="x">bad</if-tool>',
        '<if-tool name="yes" scope="other">bad</if-tool>',
        '<if-tool any-of="">bad</if-tool>',
        '<if-tool any-of="   ">bad</if-tool>',
        '<if-tool name="yes" any-of="no">bad</if-tool>',
        "<if-tool name=yes>bad</if-tool>",
        '<if-tool name="yes"',
        '<if-tool-other name="skill">bad</if-tool-other>',
    ],
)
def test_malformed_tags_fail_explicitly(text, context):
    with pytest.raises(ValueError, match="Invalid capability condition"):
        render_conditions(text, context)


@pytest.mark.parametrize(
    "text",
    [
        '```xml\n<if-tool name="typo">{{!unknown}}</if-tool>\n```\n',
        'Example: `<if-tool name="typo">{{!unknown}}</if-tool>`.',
        '<!-- <if-tool name="typo">{{!unknown}}</if-tool> -->',
        '``Example: `<if-tool name="typo">{{!unknown}}</if-tool>` ``',
        '``<if-tool name="typo">`{{!unknown}}</if-tool>``',
        '```xml\n<if-tool name="typo">{{!unknown}}</if-tool>\n',
        '~~~xml\n<if-tool name="typo">{{!unknown}}</if-tool>\n',
        '```xml\n```not a closing fence\n<if-tool name="typo">{{!unknown}}</if-tool>\n```',
        '<!-- <if-tool name="typo">{{!unknown}}</if-tool>',
    ],
)
def test_code_and_comments_are_literal(text, context):
    rendered = render_conditions(text, context)
    assert rendered == text
    assert parse_references(rendered) == (text, ())


@pytest.mark.parametrize(
    "literal",
    [
        '```xml\n<if-tool name="typo">{{!unknown}}</if-tool>\n````\n',
        '``<if-tool name="typo">`{{!unknown}}</if-tool>``\n',
        '<!-- <if-tool name="typo">{{!unknown}}</if-tool> -->\n',
    ],
)
def test_conditions_and_references_after_literals_are_still_active(literal, context):
    rendered = render_conditions(
        literal + '<if-tool name="yes">{{!yes}}</if-tool>', context
    )
    assert parse_references(rendered) == (literal + "yes", ("yes",))


def test_inactive_references_do_not_require_capabilities(context):
    rendered = render_conditions(
        '<if-tool name="no">{{!absent}}</if-tool> {{!yes}}', context
    )
    assert parse_references(rendered) == (" yes", ("yes",))


@pytest.mark.parametrize(
    ("kind", "names", "scope", "expected"),
    [
        ("tool", "no yes", "direct", "visible"),
        ("tool", "yes no", "direct", "visible"),
        ("tool", "no", "direct", ""),
        ("skill", "skill", "direct", ""),
        ("skill", "skill", "reachable", "visible"),
        ("skill", "skill", "selected", "visible"),
    ],
)
def test_any_of_conditions_follow_availability_and_scope(
    context, kind, names, scope, expected
):
    text = f'<if-{kind} any-of="{names}" scope="{scope}">visible</if-{kind}>'
    assert render_conditions(text, context) == expected


@pytest.mark.parametrize(
    "text",
    [
        '<if-tool any-of="yes typo">bad</if-tool>',
        '<if-tool name="no"><if-tool any-of="yes typo">bad</if-tool></if-tool>',
    ],
)
def test_any_of_conditions_validate_all_names_even_when_inactive(context, text):
    with pytest.raises(ValueError, match="Unknown tool.*typo"):
        render_conditions(text, context)
