import json

from pydantic import BaseModel, ConfigDict

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.structured_assistant_message import (
    StructuredAssistantMessage,
)
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.xml_header import XmlHeader
from exmekh_lib.utils.json import serialize_json
from tests.core.agents.node.test_node import SequenceProvider
from tests.history_validation import validate_request_history
from tests.tool_result_support import parse_tool_result


def count_request(request):
    text = serialize_json(
        [message.message_dump() for message in request.get_messages()]
    )
    if isinstance(request, StructuredProviderRequest):
        text += serialize_json(request.get_response_model().model_json_schema())
    return len(text)


class UnvalidatedScores(BaseModel):
    model_config = ConfigDict(extra="allow")


class CompactionProvider(SequenceProvider):
    def __init__(self, responses=(), score=60, summary="preserved facts"):
        super().__init__(list(responses))
        self.score = score
        self.summary = summary
        self.score_requests = []
        self.summary_requests = []
        self.catalogs = []
        self.on_call = None
        self.on_score = None
        self.score_transform = lambda outputs: outputs
        self.summary_error = None

    def call(self, request):
        validate_request_history(request)
        if request.get_reasoning_effort_override() == "none":
            self.summary_requests.append(request)
            if self.summary_error is not None:
                raise self.summary_error
            return ProviderResponse(AssistantMessage(self.summary))
        if self.on_call is not None:
            self.on_call(request)
        return super().call(request)

    def call_structured(self, request):
        validate_request_history(request)
        self.score_requests.append(request)
        content = request.get_messages()[-1].get_message()
        catalog = json.loads(content[len(HISTORY_COMPACTION.scoring_prompt) :])
        self.catalogs.append(catalog)
        if self.on_score is not None:
            self.on_score(request)
        scores = self.score_transform(
            {field: self.score for field in request.get_response_model().model_fields}
        )
        return StructuredProviderResponse(
            StructuredAssistantMessage(UnvalidatedScores.model_validate(scores))
        )


def output_batch(body, call_id="call", output_id="output", name="work", **kwargs):
    header = XmlHeader("Tool", name, output_id, fields={"status": "completed"})
    return [
        AssistantMessage(None, tool_calls=[ToolCall(call_id, FunctionCall(name, {}))]),
        ToolMessage(header.prepend_to(body), call_id, name, **kwargs),
    ]


def compacted_payload(message):
    assert message.get_name() == HISTORY_COMPACTION.tool_name
    return parse_tool_result(message.get_message())
