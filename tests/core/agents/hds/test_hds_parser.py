from qs.core.agents.hds.markdown import HDSMarkdown
from qs.core.agents.hds.parser import HDSMarkdownParser


def test_parses_hds_body_references_and_exit_tool() -> None:
    markdown = HDSMarkdownParser().parse(
        "---\ndescription: Instructions\nexit-tool: finish\n---\n"
        "Use {{!child}} then {{!search}}"
    )

    assert isinstance(markdown, HDSMarkdown)
    assert markdown.body == "Use child then search"
    assert markdown.references == ("child", "search")
    assert markdown.metadata.exit_tool == "finish"
