"""Advertised instructions and defaults must agree with configured values."""

from dataclasses import replace

import pytest

from exmekh_lib.config import file_editing, file_inspection, reasoning
from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.capabilities.capabilities import Capabilities
from exmekh_lib.core.file_inspection import edit_filesystem, reading
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.providers.openai_chat_completion_annotation_schema_builder import (
    OpenAIChatCompletionAnnotationSchemaBuilder,
)
from exmekh_lib.core.providers.openai_function_tool_definition_mapper import (
    OpenAIFunctionToolDefinitionMapper,
)


def test_bundled_prompts_and_signatures_follow_changed_defaults(tmp_path, monkeypatch):
    inspection = replace(
        file_inspection.FILE_INSPECTION,
        default_max_items=7,
        max_pdf_render_pages=2,
        pdf_render_dpi=72,
    )
    date_time = replace(
        reasoning.DATE_TIME, default_timezone="Europe/Rome", default_unit="minutes"
    )
    monkeypatch.setattr(file_inspection, "FILE_INSPECTION", inspection)
    monkeypatch.setattr(reasoning, "DATE_TIME", date_time)
    capabilities = Capabilities([CAPABILITY_FORMAT.bundled_root]).load(
        work_root=tmp_path
    )
    tools = {tool.name: tool for tool in capabilities.tools}
    skills = {skill.name: skill for skill in capabilities.deep_skills}
    assert tools["inspect-file"].get_signature().parameters["max_items"].default == 7
    assert "max_items defaults to 7" in tools["inspect-file"].get_description()
    assert "at most 2 pages" in tools["read-file"].get_description()
    file_prompt = skills["file-processing"].render_instructions("")
    assert "at most 2 pages" in file_prompt and "72 DPI" in file_prompt
    assert (
        tools["date-time"].get_signature().parameters["timezone"].default
        == "Europe/Rome"
    )
    assert tools["date-time"].get_signature().parameters["unit"].default == "minutes"
    assert "timezone defaults to Europe/Rome" in tools["date-time"].get_description()
    assert "when Europe/Rome is not intended" in skills[
        "deep-thinking"
    ].render_instructions("")
    assert "{{config." not in file_prompt


def test_harness_guidance_uses_the_configured_sub_agent_name():
    settings = replace(RUNTIME, sub_agent_tool_name="delegate_child")
    assert settings.harness_output_guidance.count("delegate_child") == 2
    assert RUNTIME.sub_agent_tool_name not in settings.harness_output_guidance


def test_new_file_minimum_agrees_in_runtime_schema_and_instructions(
    tmp_path, monkeypatch
):
    settings = replace(file_editing.FILE_EDITING, min_new_file_characters=4)
    monkeypatch.setattr(file_editing, "FILE_EDITING", settings)
    monkeypatch.setattr(edit_filesystem, "FILE_EDITING", settings)
    loaded = Capabilities([CAPABILITY_FORMAT.bundled_root]).load(work_root=tmp_path)
    tool = next(item for item in loaded.tools if item.get_name() == "new-file")
    skill = next(item for item in loaded.deep_skills if item.name == "file-processing")
    mapper = OpenAIFunctionToolDefinitionMapper(
        OpenAIChatCompletionAnnotationSchemaBuilder()
    )
    schema = mapper.map(tool).get_parameters()
    assert schema["properties"]["content"]["minLength"] == 4
    assert "at least 4 Unicode characters" in tool.get_description()
    assert "at least 4 characters" in skill.render_instructions("")
    with pytest.raises(FileInspectionError, match="at least 4"):
        edit_filesystem.new_file(tmp_path, "notes.txt", "abc")
    result = edit_filesystem.new_file(tmp_path, "notes.txt", "abcd")
    assert result["ok"] and (tmp_path / "notes.txt").read_text() == "abcd"


def test_read_validation_uses_configured_diagnostics(tmp_path, monkeypatch):
    messages = replace(
        file_inspection.FILE_INSPECTION_TEXT,
        positive_index="Configured index error: {name}",
    )
    monkeypatch.setattr(reading, "FILE_INSPECTION_TEXT", messages)
    with pytest.raises(
        FileInspectionError, match="Configured index error: start_index"
    ):
        reading.read_file(tmp_path, "notes.txt", -1, 1)


def test_tool_descriptions_use_configured_execution_and_output_names():
    settings = replace(
        RUNTIME, read_tool_name="get_output", execution_tool_name="control_work"
    )
    assert "get_output(output_id, cursor)" in settings.bash_description
    assert "control_work(tool_call_id)" in settings.bash_description
    assert "get_output" in settings.control_tool_execution_description
    assert RUNTIME.read_tool_name not in settings.bash_description
    assert RUNTIME.execution_tool_name not in settings.bash_description
