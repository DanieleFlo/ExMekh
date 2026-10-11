"""Advertised instructions and defaults must agree with configured values."""

from dataclasses import replace

from exmekh_lib.config import file_inspection, reasoning
from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.capabilities.capabilities import Capabilities


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
