"""Selection, replacement and instructions must describe the same callable tools."""

from dataclasses import FrozenInstanceError, fields

import pytest

from exmekh_lib.config.default_capabilities import DEFAULT_CAPABILITIES as names
from exmekh_lib.core.agents.capabilities.capabilities import Capabilities
from exmekh_lib.core.agents.node import DefaultCapabilities, Node, ToolOverride
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.tool import PreformattedToolResult, Tool
from tests.core.agents.capabilities.test_default_capabilities import (
    SequenceProvider,
    write_hds,
    write_tool,
)
from tests.core.agents.deep_agent.test_deep_agent import _write_capabilities


def initialize(
    tmp_path, defaults, *, paths=None, name="main", children=(), responses=None
):
    provider = SequenceProvider(
        tmp_path / "cache", responses or [AssistantMessage("done")]
    )
    node = Node(paths, name_hds=name, default_capabilities=defaults, children=children)
    data = node.initialize(provider, work_root=tmp_path)
    return node, data, provider


def test_discoverable_names_are_frozen_and_cover_the_public_catalog():
    assert {getattr(names, field.name) for field in fields(names)} == set(
        DefaultCapabilities.catalog
    )
    assert names.delete_file == "delete-file"
    with pytest.raises(FrozenInstanceError):
        names.bash = "other"


def test_boolean_false_keeps_only_application_capabilities(tmp_path):
    path = _write_capabilities(tmp_path, "custom")
    node, data, provider = initialize(tmp_path, False, paths=[path], name="custom")
    assert node.get_capability_inventory().selected_names == {"custom", "exit_agent"}
    assert data.deep_skills == () and data.tools == ()
    node.run("hello")
    request = provider.requests[0]
    assert request.get_extra()["agentic"]["allowed_tools"] == []
    prompt = request.get_messages()[0].get_message()
    for name in (
        names.read_tool_output,
        names.bash,
        names.call_sub_agent,
        names.control_tool_execution,
    ):
        assert name not in prompt


def test_boolean_true_selects_main_without_application_paths(tmp_path):
    node, data, _ = initialize(tmp_path, True)
    assert data.get_hds("main").get_sub_skill_names() == (
        names.deep_thinking,
        names.file_processing,
    )
    assert names.bash in node.get_capability_inventory().selected_names
    assert names.call_sub_agent not in node.get_capability_inventory().selected_names


def test_excluded_optional_tools_disappear_from_schema_catalog_body_and_description(
    tmp_path,
):
    node, data, provider = initialize(
        tmp_path,
        DefaultCapabilities(
            include={names.main},
            exclude={names.delete_file, names.bash, names.search_web},
        ),
    )
    files = next(
        skill for skill in data.deep_skills if skill.name == names.file_processing
    )
    thinking = next(
        skill for skill in data.deep_skills if skill.name == names.deep_thinking
    )
    assert names.delete_file not in files.render_instructions("")
    assert "Remove files" not in files.get_description()
    assert names.search_web not in thinking.render_instructions("")
    assert "Search web" not in thinking.get_description()
    assert names.bash not in node.get_capability_inventory().selected_names
    node.run("hello")
    assert names.delete_file not in provider.requests[0].get_tools().list_tools()


def test_excluded_skill_disappears_from_main_instructions(tmp_path):
    _, data, _ = initialize(
        tmp_path,
        DefaultCapabilities(include={names.main}, exclude={names.deep_thinking}),
    )
    assert names.deep_thinking not in data.get_hds("main").get_hds()
    assert data.get_hds("main").get_sub_skill_names() == (names.file_processing,)


@pytest.mark.parametrize(
    "available_logic_tools",
    [
        (),
        (names.logic_check,),
        (names.logic_consistency,),
        (names.logic_counterexample,),
        (names.logic_check, names.logic_consistency),
        (names.logic_check, names.logic_counterexample),
        (names.logic_consistency, names.logic_counterexample),
        (names.logic_check, names.logic_consistency, names.logic_counterexample),
    ],
)
def test_shared_formula_instructions_require_at_least_one_logic_tool(
    tmp_path, available_logic_tools
):
    logic_tools = {
        names.logic_check,
        names.logic_consistency,
        names.logic_counterexample,
    }
    _, data, _ = initialize(
        tmp_path,
        DefaultCapabilities(
            include={names.main},
            exclude=logic_tools - set(available_logic_tools),
        ),
    )
    thinking = next(
        skill for skill in data.deep_skills if skill.name == names.deep_thinking
    )
    instructions = thinking.render_instructions("")
    for text in (
        "These tools verify the supplied formulas",
        "For formulas, declare each variable as Bool, Int, or Real.",
        "Supported operators are",
        "If a tool reports unavailable dependencies or an error",
        "saved output IDs needed by the caller.",
    ):
        assert instructions.count(text) == int(bool(available_logic_tools))
    for tool in logic_tools:
        assert (f"Use `{tool}`" in instructions) == (tool in available_logic_tools)
    assert "Use `calculate`" in instructions


@pytest.mark.parametrize("excluded", [{names.inspect_file}, {names.read_file}])
def test_missing_builtin_prerequisites_explain_how_to_correct_selection(
    tmp_path, excluded
):
    with pytest.raises(ValueError, match="requires .*Include.*exclude/replace"):
        initialize(
            tmp_path, DefaultCapabilities(include={names.main}, exclude=excluded)
        )


def test_required_application_references_still_fail_when_excluded(tmp_path):
    path = write_hds(tmp_path, "custom", tools=(names.calculate,))
    with pytest.raises(ValueError, match="invalid sub-tools: calculate"):
        initialize(
            tmp_path,
            DefaultCapabilities(exclude={names.calculate}),
            paths=[path],
            name="custom",
        )


def test_hds_conditions_distinguish_direct_reachable_and_selected(tmp_path):
    path = write_hds(
        tmp_path,
        "custom",
        skills=(names.file_processing,),
        body=(
            '<if-tool name="read-file">DIRECT</if-tool>'
            '<if-tool name="read-file" scope="reachable">REACHABLE</if-tool>'
            '<if-tool name="calculate" scope="selected">SELECTED</if-tool>'
            '<if-tool name="calculate" scope="reachable">UNREACHABLE</if-tool>'
            '<if-skill name="file-processing">OPEN {{!file-processing}}</if-skill>'
        ),
    )
    node, data, _ = initialize(tmp_path, True, paths=[path], name="custom")
    assert data.get_hds("custom").get_hds() == "REACHABLESELECTEDOPEN file-processing"
    entries = {entry.name: entry for entry in node.get_capability_inventory().entries}
    assert entries[names.read_file].reachable and not entries[names.read_file].direct
    assert not entries[names.calculate].reachable


def test_runtime_conditions_use_actual_child_availability_and_node_isolation(tmp_path):
    path = write_hds(
        tmp_path, "custom", body='<if-tool name="call_sub_agent">DELEGATE</if-tool>'
    )
    leaf, leaf_data, _ = initialize(
        tmp_path / "leaf", True, paths=[path], name="custom"
    )
    parent, parent_data, _ = initialize(
        tmp_path / "parent", True, paths=[path], name="custom", children=[leaf]
    )
    assert leaf_data.get_hds("custom").get_hds() == ""
    assert parent_data.get_hds("custom").get_hds() == "DELEGATE"
    assert names.call_sub_agent not in leaf.get_capability_inventory().selected_names
    assert names.call_sub_agent in parent.get_capability_inventory().selected_names


def test_named_override_uses_discoverable_target_and_replacement_schema(tmp_path):
    path = write_hds(tmp_path, "custom", tools=(names.search_web,))
    write_tool(path, "company-search")
    node, _, provider = initialize(
        tmp_path,
        DefaultCapabilities(
            overrides=ToolOverride(names.search_web, "company-search"),
        ),
        paths=[path],
        name="custom",
    )
    node.run("hello")
    tool = provider.requests[0].get_tools().get_tool(names.search_web)
    assert "text" in tool.get_signature().parameters
    assert tool.get_description() == "company-search application tool"
    assert next(
        entry
        for entry in node.get_capability_inventory().entries
        if entry.name == names.search_web
    ).replacement


def test_reader_override_rejects_incompatible_runtime_signature(tmp_path):
    with pytest.raises(
        ValueError,
        match="must accept the runtime arguments output_id, cursor, summary_mode",
    ):
        initialize(
            tmp_path,
            DefaultCapabilities(
                include={names.main},
                overrides={
                    names.read_tool_output: Tool(
                        lambda text: text, name=names.read_tool_output
                    ),
                },
            ),
        )


def test_reader_override_checks_result_contract_after_hooks(tmp_path):
    def read(output_id, cursor=None, summary_mode="none"):
        return "incorrect protocol"

    node, _, provider = initialize(
        tmp_path,
        DefaultCapabilities(
            include={names.main},
            overrides={
                names.read_tool_output: Tool(read, name=names.read_tool_output),
            },
        ),
        responses=[
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "read",
                        FunctionCall(names.read_tool_output, {"output_id": "missing"}),
                    )
                ],
            ),
            AssistantMessage("done"),
        ],
    )
    response = node.run("read")
    assert "runtime requires PreformattedToolResult" in str(response.history)
    assert (
        not provider.requests[0]
        .get_tools()
        .get_tool(names.read_tool_output)
        .background_enabled
    )


def test_compatible_reader_override_runs(tmp_path):
    def read(output_id, cursor=None, summary_mode="none"):
        return PreformattedToolResult("replacement output")

    node, _, _ = initialize(
        tmp_path,
        DefaultCapabilities(
            include={names.main},
            overrides={
                names.read_tool_output: Tool(read, name=names.read_tool_output),
            },
        ),
        responses=[
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "read",
                        FunctionCall(names.read_tool_output, {"output_id": "saved"}),
                    )
                ],
            ),
            AssistantMessage("done"),
        ],
    )
    assert "replacement output" in str(node.run("read").history)


def test_scope_conditions_follow_inherited_tools_on_skill_entry(tmp_path):
    path = write_hds(
        tmp_path, "custom", tools=(names.calculate,), skills=("custom-skill",)
    )
    directory = path / "deep-skill"
    directory.mkdir()
    (directory / "custom-skill.md").write_text(
        "---\ndescription: Custom\ninherits-parent-tools: true\n---\n"
        '<if-tool name="calculate">INHERITED</if-tool>'
    )
    node, _, provider = initialize(
        tmp_path,
        True,
        paths=[path],
        name="custom",
        responses=[
            AssistantMessage(
                None, tool_calls=[ToolCall("open", FunctionCall("custom-skill", {}))]
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "exit",
                        FunctionCall(names.exit_agent, {"text": "done", "items": []}),
                    )
                ],
            ),
            AssistantMessage("done"),
        ],
    )
    node.run("open")
    assert any(
        "INHERITED" in (message.get_message() or "")
        for message in provider.requests[1].get_messages()
    )


def test_invalid_condition_name_is_reported_even_in_inactive_block(tmp_path):
    path = write_hds(
        tmp_path,
        "custom",
        body='<if-tool name="bash"><if-tool name="typo">bad</if-tool></if-tool>',
    )
    with pytest.raises(ValueError, match="Unknown tool.*typo"):
        Capabilities(
            [path], default_capabilities=DefaultCapabilities(exclude={names.bash})
        ).load(work_root=tmp_path)


def test_disabling_defaults_does_not_load_the_bundle(tmp_path, monkeypatch):
    path = _write_capabilities(tmp_path, "custom")
    monkeypatch.setattr(
        DefaultCapabilities, "CAPABILITIES_PATH", tmp_path / "missing-bundle"
    )
    node, _, _ = initialize(tmp_path, False, paths=[path], name="custom")
    assert node.get_capability_inventory().selected_names == {
        "custom",
        names.exit_agent,
    }


def test_minimal_main_can_omit_both_default_skills(tmp_path):
    node, data, _ = initialize(
        tmp_path,
        DefaultCapabilities(
            harness=False,
            include={names.main, names.exit_agent},
        ),
    )
    assert data.get_hds(names.main).get_sub_skill_names() == ()
    assert node.get_capability_inventory().selected_names == {
        names.main,
        names.exit_agent,
    }
    assert (
        names.get_execution_history
        not in data.get_exit_tool(names.exit_agent).render_prompt_section()
    )


def test_conditional_body_references_are_validated_only_when_active(tmp_path):
    path = write_hds(
        tmp_path, "custom", body='<if-tool name="bash">{{!missing}}</if-tool>'
    )
    _, data, _ = initialize(
        tmp_path / "off",
        DefaultCapabilities(exclude={names.bash}),
        paths=[path],
        name="custom",
    )
    assert data.get_hds("custom").get_hds() == ""
    with pytest.raises(ValueError, match="invalid references: missing"):
        initialize(tmp_path / "on", True, paths=[path], name="custom")


def test_optional_and_required_declarations_cannot_overlap(tmp_path):
    path = write_hds(tmp_path, "custom", tools=(names.calculate,))
    hds = path / "hds/custom.md"
    hds.write_text(
        hds.read_text().replace(
            "sub-tools:", "optional-sub-tools: [calculate]\nsub-tools:"
        )
    )
    with pytest.raises(ValueError, match="both required and optional: calculate"):
        initialize(tmp_path, True, paths=[path], name="custom")


def test_exit_instruction_conditions_use_the_real_runtime_selection(tmp_path):
    node, data, provider = initialize(
        tmp_path,
        DefaultCapabilities(
            include={names.main},
            exclude={names.get_execution_history},
        ),
    )
    assert (
        names.get_execution_history
        not in data.get_exit_tool(names.exit_agent).render_prompt_section()
    )
    node.run("hello")
    assert (
        names.get_execution_history
        not in provider.requests[0].get_messages()[0].get_message()
    )


def test_shared_object_override_keeps_its_description_template_private_per_node(
    tmp_path,
):
    replacement = Tool(
        lambda text: text,
        name=names.calculate,
        description='<if-tool name="bash">SHELL</if-tool>Calculator',
    )
    path = write_hds(tmp_path, "custom", tools=(names.calculate,))
    _, first, _ = initialize(
        tmp_path / "first",
        DefaultCapabilities(overrides={names.calculate: replacement}),
        paths=[path],
        name="custom",
    )
    _, second, _ = initialize(
        tmp_path / "second",
        DefaultCapabilities(
            overrides={names.calculate: replacement},
            exclude={names.bash},
        ),
        paths=[path],
        name="custom",
    )
    assert (
        next(
            tool for tool in first.tools if tool.name == names.calculate
        ).get_description()
        == "SHELLCalculator"
    )
    assert (
        next(
            tool for tool in second.tools if tool.name == names.calculate
        ).get_description()
        == "Calculator"
    )
    assert "<if-tool" in replacement.get_description()


def test_unknown_condition_in_tool_description_fails_before_inference(tmp_path):
    path = write_hds(tmp_path, "custom", tools=(names.calculate,))
    replacement = Tool(
        lambda: 1,
        name=names.calculate,
        description='<if-tool name="typo">bad</if-tool>',
    )
    with pytest.raises(ValueError, match="Unknown tool.*typo"):
        initialize(
            tmp_path,
            DefaultCapabilities(overrides={names.calculate: replacement}),
            paths=[path],
            name="custom",
        )


def test_unknown_reference_in_exit_instructions_fails_before_inference(tmp_path):
    path = _write_capabilities(tmp_path, "custom")
    (path / "exit-tool/finish/instructions.md").write_text(
        '<if-tool name="bash">{{!missing}}</if-tool>'
    )
    with pytest.raises(ValueError, match="unavailable capability: missing"):
        initialize(
            tmp_path,
            DefaultCapabilities(harness=False, include={names.bash}),
            paths=[path],
            name="custom",
        )


def test_inventory_includes_exit_tool_in_positive_initial_depth(tmp_path):
    provider = SequenceProvider(tmp_path / "cache", [AssistantMessage("done")])
    node = Node(name_hds=names.main, depth=1, default_capabilities=True)
    node.initialize(provider, work_root=tmp_path)
    entry = next(
        entry
        for entry in node.get_capability_inventory().entries
        if entry.name == names.exit_agent
    )
    assert entry.direct and entry.reachable


@pytest.mark.parametrize("with_skills", [True, False])
def test_inventory_exit_reachability_follows_skill_availability(tmp_path, with_skills):
    excluded = set() if with_skills else {names.deep_thinking, names.file_processing}
    node, _, _ = initialize(
        tmp_path, DefaultCapabilities(include={names.main}, exclude=excluded)
    )
    entry = next(
        entry
        for entry in node.get_capability_inventory().entries
        if entry.name == names.exit_agent
    )
    assert not entry.direct
    assert entry.reachable == with_skills


@pytest.mark.parametrize("depth", [0, 1])
def test_hds_description_conditions_follow_initial_depth(tmp_path, depth):
    path = write_hds(tmp_path, "custom")
    hds_path = path / "hds/custom.md"
    hds_path.write_text(
        hds_path.read_text().replace(
            "description: custom",
            "description: '<if-tool name=\"exit_agent\">EXIT </if-tool>Custom'",
        )
    )
    provider = SequenceProvider(tmp_path / "cache", [AssistantMessage("done")])
    node = Node([path], name_hds="custom", depth=depth, default_capabilities=True)
    data = node.initialize(provider, work_root=tmp_path)
    assert data.get_hds("custom").get_description() == (
        "EXIT Custom" if depth else "Custom"
    )
