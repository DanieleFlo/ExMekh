# Function Tools & Structured Outputs

`exmekh_lib` features type-safe function tool registration, dynamic JSON Schema generation, and strict response parsing against Pydantic models.

---

## 1. Function Tools (`Tool` & `Tools`)

The [`Tool`](../src/exmekh_lib/core/models/tool.py) wrapper converts standard Python functions into OpenAI-compatible tool specifications.

### Function Registration & Parameter Reflection

```python
from typing import Literal
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tools import Tools


def calculate_discount(
    price: float,
    percentage: float = 10.0,
    customer_tier: Literal["standard", "premium", "vip"] = "standard",
) -> float:
    """Calculate discounted price for a customer."""
    discount = price * (percentage / 100.0)
    if customer_tier == "vip":
        discount += 5.0
    return max(0.0, price - discount)


tools = Tools([Tool(calculate_discount, background_enabled=False)])
```

### Supported Type Annotations
In `NodeRuntime`, work tools default to managed execution and add
`execution_mode` and `wait_seconds` to the model schema without changing the
Python function signature. The standalone `Client` example above disables
this protocol because it does not execute the agent loop. See
[Tool Executions](13-tool-executions.md) for background work and custom handles.

The schema builder inspects standard Python type hints and docstrings:
- **Primitives**: `int`, `float`, `str`, `bool`, `None`
- **Containers**: `list[T]`, `set[T]`, `tuple[T, ...]`, `dict[K, V]`
- **Enumerations & Choices**: `Literal[...]`, `enum.Enum`
- **Data Structures**: Pydantic models, `dataclass`, `TypedDict`
- **Special Types**: `Annotated`, `NewType`, `datetime.date`, `uuid.UUID`, `decimal.Decimal`, `pathlib.Path`

> [!WARNING]
> Positional-only parameters (without `@deep_tool`) and variadic parameters (`*args`, `**kwargs`) are rejected because model tool-call payloads are JSON objects with key-value key pairs.

---

## 2. Agent Capabilities Tools (`@deep_tool`)

Within agent capability directories, function tools are implemented in a `main.py` entrypoint and decorated with [`@deep_tool`](../src/exmekh_lib/core/agents/tools/decorator.py).

```python
from pathlib import Path
from exmekh_lib.core.agents.tools import deep_tool


@deep_tool(
    name="read_workspace_file",
    description="Reads a text file relative to the project workspace directory.",
)
def read_workspace_file(work_root: Path, relative_path: str) -> str:
    target_path = (work_root / relative_path).resolve()
    if not target_path.is_relative_to(work_root):
        raise ValueError("Access outside workspace is prohibited.")
    return target_path.read_text(encoding="utf-8")
```

### Key `@deep_tool` Rules:
1. **`work_root` Binding**: Every `@deep_tool` entrypoint MUST declare `work_root: Path` as its first positional argument.
2. **Internal Binding**: The `DeepAgent` runtime injects `work_root` internally before invocation. `work_root` is automatically removed from the JSON schema generated for the LLM.

Bundled harness tools follow the same decorated entrypoint convention. Runtime
dependencies follow `work_root`; memory, control, and delegation tools bind
public methods of their owning classes. Binding hides these dependencies and
the work root from the model schema. The regular capability loader
handles ordinary tools, while the node assembles its stateful harness tools.
Specialized wrappers may add cancellation, progress reads, or a dynamic
description, but their callable and metadata still come from `@deep_tool`.

### Before and after hooks

`Tool` and `@deep_tool` also accept `argument_guidance`, a correction message
used when argument binding/validation fails. It is runtime metadata, not a model
argument, and survives description/name copies. For example,
`@deep_tool(description="Read a range", argument_guidance="Inspect the file first.")`
keeps file-specific advice with the tool definition instead of the agent loop.
Binding errors are `invalid_tool_arguments`; exceptions from the function or
hooks are `tool_execution_failed`. Both inline and managed work tools return a
failed XML response and let the model continue. A valid result containing
`ok=false` remains a final result. Low-level direct Python calls still raise
their exceptions; recovery belongs to the node runtime.

`@deep_tool` and `@exit_tool` accept `before` and `after` instances. Plain
`Tool` objects accept the same constructor arguments. Hooks run for agent tool
calls and are kept out of the model's tool schema.

```python
from pathlib import Path
from typing import Any
from exmekh_lib.interfaces.tool_hook import (
    AfterTool, BeforeTool, ToolArguments, ToolContext, ToolResponse,
)

class NormalizeInput(BeforeTool):
    def handle(self, context: ToolContext) -> ToolArguments | ToolResponse | None:
        # context also contains work_root and the conversation history so far.
        return ToolArguments({**context.arguments, "query": context.arguments["query"].strip()})

class FormatOutput(AfterTool):
    def handle(self, context: ToolContext, result: Any) -> Any:
        return {"answer": result}

@deep_tool(description="Search the project", before=NormalizeInput(), after=FormatOutput())
def search(work_root: Path, query: str) -> str:
    ...
```

`BeforeTool.handle` returns `None` to continue unchanged, `ToolArguments` to
replace the arguments, or `ToolResponse` to skip the tool and supply a result.
`AfterTool.handle` receives the effective arguments and the result, including
one supplied by `BeforeTool`, and returns the final result. The history is a
snapshot of messages available when the tool starts. Exit tool results must
still contain valid `text` and `items` fields. Hook instances must implement
synchronous `handle` methods with the shown number of parameters.

Direct `execute_with_hooks` calls and managed `create_execution_with_hooks`
operations apply each hook once and use the same result finalization. Exit tools
validate and normalize `text` and `items` after the after hook in both paths.
Calling `execute` directly runs the callable without hooks.

---

## 3. Tool Call Resolution & Value Coercion

When an LLM requests a tool call, `exmekh_lib` handles argument extraction and type conversion through `OpenAIChatCompletionToolMapper` and `OpenAIChatCompletionValueCoercer`:

```python
from exmekh_lib.core.providers.openai_chat_completion_tool_mapper import OpenAIToolMapper

# Resolve matching tool from model response payload
tool_call = assistant_message.get_tool_calls()[0]
selected_tool, parsed_arguments = OpenAIToolMapper.resolve_tool_call(
    tool_call=tool_call,
    tools=tools,
)

# Execute tool with coerced arguments
result = selected_tool.execute(**parsed_arguments)
```

Values passed as strings in JSON are automatically coerced to target types (e.g. converting ISO strings `"2026-09-16"` into `datetime.date` objects, UUID strings into `UUID` instances).

---

## 4. Structured Outputs (`call_structured`)

To force the model to output validated Pydantic objects instead of arbitrary prose, use `Client.call_structured`:

```python
from pydantic import BaseModel, Field, ValidationError
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.user_message import UserMessage


class FinancialSummary(BaseModel):
    company_name: str
    revenue: float = Field(description="Annual revenue in USD")
    growth_rate: float = Field(description="YoY growth percentage")


client = Client(provider=provider, system_message=system_message)

try:
    response = client.call_structured(
        ClientRequest(UserMessage("Analyze financial data for Acme Corp...")),
        FinancialSummary,
        strict=True,
    )
    summary: FinancialSummary = response.response.get_message()
    print(f"Company: {summary.company_name}, Revenue: ${summary.revenue}M")
except ValidationError as err:
    print("Schema validation failed:", err.errors())
```

### Strict vs Non-Strict Mode
- **`strict=True` (Default)**: Enforces `additionalProperties: false` in the generated JSON schema and validates strict compliance.
- **`strict=False`**: Allows Pydantic to perform loose coercion while still validating field shapes and types.
