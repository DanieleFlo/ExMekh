# Function Tools & Structured Outputs

`qs` features type-safe function tool registration, dynamic JSON Schema generation, and strict response parsing against Pydantic models.

---

## 1. Function Tools (`Tool` & `Tools`)

The [`Tool`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/models/tool.py) wrapper converts standard Python functions into OpenAI-compatible tool specifications.

### Function Registration & Parameter Reflection

```python
from typing import Literal
from qs.core.models.tool import Tool
from qs.core.models.tools import Tools


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


tools = Tools([Tool(calculate_discount)])
```

### Supported Type Annotations
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

Within agent capability directories, function tools are implemented in a `main.py` entrypoint and decorated with [`@deep_tool`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/agents/tools/decorator.py).

```python
from pathlib import Path
from qs.core.agents.tools import deep_tool


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

---

## 3. Tool Call Resolution & Value Coercion

When an LLM requests a tool call, `qs` handles argument extraction and type conversion through `OpenAIChatCompletionToolMapper` and `OpenAIChatCompletionValueCoercer`:

```python
from qs.core.providers.openai_chat_completion_tool_mapper import OpenAIToolMapper

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
from qs.core.models.client import Client
from qs.core.models.client_request import ClientRequest
from qs.core.models.user_message import UserMessage


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
