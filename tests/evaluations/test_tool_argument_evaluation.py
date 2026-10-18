import json
from dataclasses import replace
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import httpx2
import pytest
from openai import APIConnectionError
from openai.types.responses import Response

from evaluations import tool_arguments
from evaluations.tool_argument_cases import build_cases
from evaluations.tool_argument_grading import RawToolCall, grade_arguments
from evaluations.tool_arguments import (
    EvaluationSettings,
    TrialResult,
    build_request,
    execute_trial,
    summarize,
    trial_schedule,
    write_report,
)
from exmekh_lib.core.models.json_schema import (
    schema_descriptions,
    without_schema_descriptions,
)
from exmekh_lib.core.providers.to_openai_response_request_mapper import (
    ToOpenAIResponseRequestMapper,
)
from exmekh_lib.utils.xml import read_text_element
from tests.tokenizer_support import StubTokenizer


@pytest.fixture
def settings(tmp_path):
    return EvaluationSettings(
        "fake-model",
        1,
        17,
        tuple(c.name for c in build_cases()),
        "low",
        5.0,
        1000,
        tmp_path / "report.xml",
    )


@pytest.mark.parametrize("case", build_cases(), ids=lambda case: case.name)
def test_case_gold_arguments_receive_full_credit(case):
    assert (
        grade_arguments(
            case, (RawToolCall(case.tool.name, json.dumps(case.expected_arguments())),)
        ).score
        == 100
    )


@pytest.mark.parametrize("case", build_cases(), ids=lambda case: case.name)
def test_paired_requests_change_only_documentation_location(case):
    full = ToOpenAIResponseRequestMapper.map_request(
        build_request(case, "full"), model="model"
    )
    scoped = ToOpenAIResponseRequestMapper.map_request(
        build_request(case, "scoped"), model="model"
    )
    full_tool, scoped_tool = full["tools"][0], scoped["tools"][0]
    assert full_tool["name"] == scoped_tool["name"] == case.tool.name
    assert full_tool["description"] == scoped_tool["description"]
    assert (
        without_schema_descriptions(full_tool["parameters"])
        == scoped_tool["parameters"]
    )
    assert schema_descriptions(scoped_tool["parameters"]) == ()
    assert full["input"][-1] == scoped["input"][-1]
    assert "<parameter-descriptions" not in full["input"][0]["content"]
    instructions = scoped["input"][0]["content"]
    for description in schema_descriptions(full_tool["parameters"]):
        if (
            "/properties/execution_mode" in description.path
            or "/properties/wait_seconds" in description.path
        ):
            continue
        catalog_text = instructions[
            instructions.index("<catalog>") : instructions.index("</catalog>")
            + len("</catalog>")
        ]
        parameters = ET.fromstring(catalog_text).findall(
            "./parameter-descriptions/tool/parameter"
        )
        assert description.text in [
            read_text_element(parameter) for parameter in parameters
        ]


def test_dataset_and_pair_order_are_reproducible(settings):
    assert len(settings.cases) >= 10
    first = trial_schedule(settings)
    assert [(r, c.name, v) for r, c, v in first] == [
        (r, c.name, v) for r, c, v in trial_schedule(settings)
    ]
    for index in range(0, len(first), 2):
        left, right = first[index : index + 2]
        assert left[1] is right[1]
        assert left[1].expected_arguments() == right[1].expected_arguments()
        assert {left[2], right[2]} == {"full", "scoped"}
    assert first[0][2] != first[2][2]


def test_verification_gold_is_available_only_in_documentation():
    case = next(c for c in build_cases(29) if c.category == "documentation_lookup")
    code = case.expected["code"]
    assert code not in case.prompt
    for variant in ("full", "scoped"):
        payload = ToOpenAIResponseRequestMapper.map_request(
            build_request(case, variant), model="model"
        )
        assert code in json.dumps(payload)
    assert (
        next(
            c for c in build_cases(30) if c.category == "documentation_lookup"
        ).expected
        != case.expected
    )


@pytest.mark.parametrize(
    "arguments", ['{"x":1,"x":2}', "[]", "null", "{", '{"x":NaN}', '{"x":1e999}']
)
def test_invalid_json_is_a_model_failure(arguments):
    case = build_cases()[0]
    grade = grade_arguments(case, (RawToolCall(case.tool.name, arguments),))
    assert grade.score == 0 and not grade.exact
    assert "Invalid argument JSON" in grade.issue


@pytest.mark.parametrize(
    "calls",
    [
        (),
        (RawToolCall("wrong", "{}"),),
        (RawToolCall("lookup_customer", "{}"), RawToolCall("lookup_customer", "{}")),
    ],
)
def test_wrong_tool_or_call_count_receives_zero(calls):
    assert grade_arguments(build_cases()[0], calls).score == 0


@pytest.mark.parametrize("change", ["wrong_value", "wrong_type", "missing", "extra"])
def test_business_errors_reduce_score_without_coercion(change):
    case = next(c for c in build_cases() if c.name == "money_in_minor_units")
    arguments = case.expected_arguments()
    if change == "wrong_value":
        arguments["amount_minor"] = 12
    elif change == "wrong_type":
        arguments["amount_minor"] = "1234"
    elif change == "missing":
        arguments.pop("amount_minor")
    else:
        arguments["unexpected"] = True
    grade = grade_arguments(case, (RawToolCall(case.tool.name, json.dumps(arguments)),))
    assert 0 < grade.score < 100
    assert not grade.exact


def test_boolean_is_not_accepted_as_integer_and_business_null_stays_null():
    case = next(c for c in build_cases() if c.name == "null_folder")
    arguments = case.expected_arguments()
    arguments["limit"] = True
    grade = grade_arguments(case, (RawToolCall(case.tool.name, json.dumps(arguments)),))
    assert not next(check for check in grade.checks if check.path == "/limit").passed
    assert next(check for check in grade.checks if check.path == "/folder").passed


def test_runtime_nulls_use_defaults_but_missing_fields_are_not_repaired():
    case = build_cases()[0]
    arguments = case.expected_arguments()
    arguments.update(execution_mode=None, wait_seconds=None)
    assert grade_arguments(
        case, (RawToolCall(case.tool.name, json.dumps(arguments)),)
    ).exact
    arguments.pop("execution_mode")
    assert not grade_arguments(
        case, (RawToolCall(case.tool.name, json.dumps(arguments)),)
    ).exact


def test_equal_integer_json_numbers_are_valid_without_accepting_numeric_strings():
    case = next(c for c in build_cases() if c.name == "money_in_minor_units")
    arguments = case.expected_arguments()
    arguments["amount_minor"] = 1234.0
    assert grade_arguments(
        case, (RawToolCall(case.tool.name, json.dumps(arguments)),)
    ).exact


def test_large_integer_in_float_field_is_a_failed_check_without_crashing():
    case = next(c for c in build_cases() if c.name == "signed_coordinate_order")
    arguments = case.expected_arguments()
    arguments["coordinates"][0] = 10**400

    grade = grade_arguments(case, (RawToolCall(case.tool.name, json.dumps(arguments)),))

    assert not grade.exact
    assert any(
        not check.passed and check.path == "/coordinates/0" for check in grade.checks
    )


def test_nested_gold_arguments_are_not_mutated_by_callers():
    case = next(c for c in build_cases() if c.name == "nested_order_lines")
    arguments = case.expected_arguments()
    arguments["lines"][0]["quantity"] = 99
    assert case.expected_arguments()["lines"][0]["quantity"] == 2


def test_transport_failure_has_no_model_score(settings):
    class FailingClient:
        def with_options(self, **kwargs):
            return self

        @property
        def responses(self):
            return self

        def create(self, **kwargs):
            raise APIConnectionError(
                request=httpx2.Request("POST", "http://provider.test/v1/responses")
            )

    provider = SimpleNamespace(
        client=FailingClient(), get_tokenizer=lambda: StubTokenizer()
    )
    result, trace = execute_trial(provider, settings, build_cases()[0], "full", 1)
    assert result.grade is None
    assert "APIConnectionError" in result.infrastructure_error
    assert trace["result"]["response_status"] == "request_error"
    assert summarize([result])["full"]["mean_score"] is None


def test_listing_cases_does_not_contact_a_provider(monkeypatch, capsys):
    def unexpected_provider(**kwargs):
        raise AssertionError("Listing must not contact a provider")

    monkeypatch.setattr(tool_arguments, "LocalProvider", unexpected_provider)
    monkeypatch.setattr(tool_arguments, "load_dotenv", lambda path: False)
    monkeypatch.setattr("sys.argv", ["tool_arguments", "--list"])
    assert tool_arguments.main() == 0
    assert len(ET.fromstring(capsys.readouterr().out).findall("cases/item")) == 14


def test_report_cannot_overwrite_its_own_jsonl_trace(settings, monkeypatch):
    def unexpected_provider(**kwargs):
        raise AssertionError("Invalid output must be rejected before connecting")

    monkeypatch.setattr(tool_arguments, "LocalProvider", unexpected_provider)
    settings = replace(settings, output=settings.output.with_suffix(".jsonl"))

    with pytest.raises(ValueError, match="XML report"):
        tool_arguments.run_evaluation(settings)

    assert not settings.output.exists()


def make_result(case, variant, *, score=100, error=None):
    grade = grade_arguments(
        case, (RawToolCall(case.tool.name, json.dumps(case.expected_arguments())),)
    )
    grade = replace(grade, score=score, exact=score == 100)
    return TrialResult(
        case.name,
        case.category,
        variant,
        1,
        case.expected_arguments(),
        (),
        None if error else grade,
        0.1,
        100,
        200,
        "request_error" if error else "completed",
        None,
        error,
    )


def test_summary_excludes_transport_errors_and_uses_only_complete_pairs():
    first, second = build_cases()[:2]
    results = [
        make_result(first, "full"),
        make_result(first, "scoped", score=80),
        make_result(second, "full", error="offline"),
    ]
    summary = summarize(results)
    assert summary["full"]["mean_score"] == 100
    assert summary["full"]["infrastructure_errors"] == 1
    assert summary["comparison"] == {
        "paired_samples": 1,
        "mean_score_delta_scoped_minus_full": -20,
    }
    assert summarize([])["scoped"]["mean_score"] is None


def test_report_preserves_typed_scores_and_expected_data(settings):
    result = make_result(build_cases()[0], "scoped")
    write_report(settings, [result], complete=True)
    root = ET.parse(settings.output).getroot()
    assert root.findtext("complete") == "true"
    assert root.findtext("summary/scoped/mean_score") == "100.0"
    assert root.findtext("trials/item/expected/account_id") == "C-1042"


def test_sdk_arguments_are_graded_before_provider_coercion(settings):
    case = next(c for c in build_cases() if c.name == "money_in_minor_units")
    arguments = case.expected_arguments()
    arguments["amount_minor"] = "1234"
    response = Response.model_validate(
        {
            "id": "response-1",
            "created_at": 0,
            "model": "fake-model",
            "object": "response",
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "required",
            "tools": [],
            "output": [
                {
                    "id": "function-1",
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": case.tool.name,
                    "arguments": json.dumps(arguments),
                }
            ],
        }
    )

    class FakeClient:
        responses = None

        def with_options(self, **kwargs):
            self.responses = self
            return self

        def create(self, **kwargs):
            return response

    class FakeProvider:
        client = FakeClient()

        def get_tokenizer(self):
            return StubTokenizer()

    result, trace = execute_trial(FakeProvider(), settings, case, "scoped", 1)
    assert result.grade.score < 100
    assert '"1234"' in result.calls[0].arguments
    assert trace["response"]["output"][0]["arguments"] == result.calls[0].arguments


def test_incomplete_model_response_is_scored_zero_even_with_correct_arguments(settings):
    case = build_cases()[0]
    response = Response.model_validate(
        {
            "id": "response-incomplete",
            "created_at": 0,
            "model": "fake-model",
            "object": "response",
            "status": "incomplete",
            "parallel_tool_calls": False,
            "tool_choice": "required",
            "tools": [],
            "output": [
                {
                    "id": "function-1",
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": case.tool.name,
                    "arguments": json.dumps(case.expected_arguments()),
                }
            ],
        }
    )
    response_client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: response)
    )
    provider = SimpleNamespace(
        client=SimpleNamespace(with_options=lambda **kwargs: response_client),
        get_tokenizer=lambda: StubTokenizer(),
    )
    result, _ = execute_trial(provider, settings, case, "full", 1)
    assert result.infrastructure_error is None
    assert result.grade.score == 0
    assert not result.grade.exact
    assert "incomplete" in result.grade.issue
