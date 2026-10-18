# Capabilities System & Catalog

The `exmekh_lib` capabilities framework allows agent behaviors, system prompts, skills, and tools to be modularly defined on the filesystem in Markdown and Python files, and dynamically loaded at runtime.

---

## Capabilities Directory Layout

A capability repository follows a structured directory layout under a `capabilities/` folder:

```
capabilities/
├── hds/
│   └── main.md                 # Hierarchical Domain Specification (Root System Prompt)
├── deep-skill/
│   ├── mock-skill.md           # Domain skill definition
│   └── nested-skill.md         # Nested sub-skill definition
├── tools/
│   ├── mock-tool/
│   │   └── main.py             # Entrypoint with @deep_tool function
│   ├── shared-tool/
│   │   └── main.py
│   └── bash/
│       ├── main.py             # Decorated entrypoint
│       ├── bash_tool.py        # Runtime tool wrapper
│       └── bash_process.py     # Bash process lifecycle
└── exit-tool/
    └── finish/
        ├── main.py             # Entrypoint with @exit_tool function
        └── instructions.md     # Plain Markdown exit instructions
```

The [`CapabilitiesLayout`](../src/exmekh_lib/core/agents/capabilities/layout.py) class verifies directory integrity and automatically creates missing subfolders when initialized.

Bundled runtime tools also live under `src/exmekh_lib/capabilities/tools/<name>/main.py`:
`bash`, `get_execution_history`, `search_tool_outputs`, `read_tool_output`,
`control_tool_execution`, and `call_sub_agent`. They need node-local state, so
the node runtime binds their decorated entrypoints to node-local operations
after loading the ordinary `@deep_tool` resources. Search, history rendering,
and output reading live with their tool implementations. Class-only modules
use the class name in snake case (for example, `bash_tool.py`). Tool factories
live in `tool.py` next to their decorated entrypoints. The ordinary tool
loader excludes only these bundled runtime directories; application capability
roots still load their own tools through `@deep_tool`.
The bundled root is resolved relative to the `exmekh_lib` package, like HDS, deep
skills, and exit tools.

## Per-node default capabilities

`DefaultCapabilities` selects bundled resources independently for each `Node`.
Its default is `mocks=False, harness=True`. `Node` accepts
`capabilities_paths=None`; when `DefaultCapabilities` is supplied, it adds the
bundled directory exposed by `DefaultCapabilities.capabilities_path` automatically.
The bundled directory can also appear explicitly in `capabilities_paths` and is
loaded only once. A harness-only node needs an application HDS in
`capabilities_paths` (or an HDS override). Without `DefaultCapabilities`, nodes
with application paths keep their existing behavior; a node with no paths loads
the bundled directory in full.

| Group | Bundled names |
| --- | --- |
| Mock | HDS `main`, `mock-sub-agent`; skills `mock-skill`, `nested-skill`; tools `mock-tool`, `mock-sub-tool`, `nested-sub-tool`, `shared-tool` |
| Harness | Skill `deep-thinking` with web, calculation, date/time, and logic tools; exit tool `exit_agent`; memory tools `get_execution_history`, `search_tool_outputs`, `read_tool_output`; execution control `control_tool_execution`; skill `file-processing` with file analysis `inspect-file`; Bash tool `bash`; delegation tool `call_sub_agent` on nodes with children |

The `file-processing` skill must be declared in the application HDS
`sub-skills`; the bundled main HDS already declares it. Its `inspect-file`
tool is available inside that skill. See
[Workspace File Inspection](14-file-inspection.md) for supported formats and limits.

Group switches select all names in that group. `include` and `exclude` take names
that adjust the switches;
`overrides` accepts `ToolOverride(target_name, replacement_name)` and
`DeepSkillOverride(target_name, replacement_name)`. The replacement name refers
to a resource loaded from any `capabilities_paths` directory, including the
bundled directory. The target may be a bundled or application tool or skill;
runtime harness tools can also be tool targets. The replacement is exposed under
the target name in this node. Missing targets or replacement names, and
ambiguous replacement names, raise errors during loading. An override selects a
bundled target even when its group is off. An explicit exclusion cannot also be
included or overridden.

```python
from pathlib import Path

from exmekh_lib.core.agents.node import (
    DeepSkillOverride,
    DefaultCapabilities,
    Node,
    ToolOverride,
)

node = Node(
    capabilities_paths=[Path("my-app/capabilities")],
    name_hds="main",
    default_capabilities=DefaultCapabilities(
        mocks=True,
        harness=True,
        exclude={"read_tool_output"},
        overrides=[
            ToolOverride("mock-tool", "example-echo"),
            DeepSkillOverride("mock-skill", "my-skill"),
        ],
    ),
)
```

Here `my-app/capabilities` defines the tool `example-echo` and the deep skill
`my-skill`. The tool runs as `mock-tool`, and the skill opens as `mock-skill`.
Named overrides also work when the target is an application tool or skill. The
target name stays stable in provider schemas, allowed scopes, and calls. The
existing mapping form remains available for direct Python object replacements.

To customize only the hooks of a selected bundled tool, pass `tool_hooks`.
`ToolHooks` accepts the existing `BeforeTool` and `AfterTool` interfaces:

```python
from typing import Any

from exmekh_lib.core.agents.node import DefaultCapabilities, ToolHooks
from exmekh_lib.interfaces.tool_hook import (
    AfterTool,
    BeforeTool,
    ToolArguments,
    ToolContext,
)


class NormalizeCommand(BeforeTool):
    def handle(self, context: ToolContext) -> ToolArguments:
        return ToolArguments({"command": context.arguments["command"].strip()})


class TrimOutput(AfterTool):
    def handle(self, context: ToolContext, result: Any) -> Any:
        return {**result, "stdout": result["stdout"].strip()}


defaults = DefaultCapabilities(
    tool_hooks={
        "bash": ToolHooks(before=NormalizeCommand(), after=TrimOutput()),
        "read-web-page": ToolHooks(after=None),
    },
)
```

Omitting a field preserves its original hook; an explicit `None` removes it;
an instance replaces it. Hooks are applied after object or named overrides,
under the target's exposed name, and are not automatically chained. This works
for loaded tools, runtime harness tools, `call_sub_agent`, and exit tools.
Tools retain their signatures, cache policies, and specialized execution behavior.
Runtime overrides also retain the default tool's execution restrictions: control,
memory, and delegation tools stay synchronous even if their replacement supports
background execution.
For direct Python tool objects, `tool.with_hooks(ToolHooks(after=None))` returns
a copy with the same rules.

Hook configuration does not select capabilities. Unknown names, HDS or skill
targets, and unselected or excluded tools raise errors at configuration time.
Configuring `call_sub_agent` hooks on a node without children raises an error
during initialization. Tools are copied without changing shared overrides;
the supplied hook instances are reused. Use separate instances when a hook
needs mutable state specific to one node. See
[tool hook execution](03-tools-and-structured-outputs.md#before-and-after-hooks) for the
argument and result contracts.

For a smaller selection, use `mocks=False` and `include={"mock-tool"}` with an
application HDS that declares `mock-tool` in `sub-tools`. Likewise,
`harness=False` with `include={"exit_agent", "get_execution_history"}` keeps
only those harness capabilities. Each HDS must still name a selected exit tool;
an application can provide a different one. `call_sub_agent` requires children
and cannot be overridden on a leaf node.

Composition happens before reference validation and catalog attachment. A
selected HDS or skill that references an excluded capability fails loading with
the invalid reference named; no dependencies are silently re-enabled. Active
names must be unique across application and bundled resources unless the name
is replaced in `overrides`. Only selected runtime tools enter the provider tool
list and every skill scope. Internal execution history and caching continue when
their reader tools are excluded.

---

## Resource Types ([`CapabilityType`](../src/exmekh_lib/core/agents/capabilities/layout.py))

### 1. HDS (Hierarchical Domain Specification)
HDS files are Markdown files located in `capabilities/hds/`. They define the base identity, prompt, available tools, and exit strategy for a [`Node`](../src/exmekh_lib/core/agents/node/node.py).

```markdown
---
description: Main agent system prompt.
exit-tool: exit_agent
agent-context-id: main
sub-tools:
  - mock-tool
sub-skills:
  - mock-skill
---
# Main System Agent

You are the lead orchestrator agent responsible for analyzing user commands and delegating to tools and sub-skills.
```

- **`agent-context-id`** (optional): Non-empty string identifying the agent context for inference engines like DS4. If omitted, defaults to the HDS file name (e.g. `main`). In DS4, this opaque key binds live KV cache, system prefix frontiers, and automatic context checkpointing/restoring during sub-agent delegation.

### 2. Deep Skills (`deep-skill`)
Deep skills in `capabilities/deep-skill/` allow an agent to switch into a focused execution mode with dedicated instructions and subset of tools.

```markdown
---
description: Performs refactoring and codebase analysis.
sub-tools:
  - read_workspace_file
sub-skills:
  - nested-skill
---
# Refactoring Skill Instructions

Follow these guidelines when refactoring code:
1. Always preserve public API contracts.
2. Run test suites before and after changes.
```

### 3. Exit Tools (`exit-tool`)
Each HDS must select exactly one exit tool by its exposed name (`exit-tool: exit_agent`). Multiple HDS files can select the same exit tool. The folder name is only the default tool name when `name` is omitted from the decorator.

```python
from exmekh_lib.core.agents.exit_tool import ExitToolData, exit_tool


@exit_tool(
    name="exit_agent",
    description="Return the current skill result to its caller.",
    instructions_path="instructions.md",
)
def finish(text: str, items: list[str]) -> ExitToolData:
    return ExitToolData(text=text, items=items)
```

`instructions_path` is required and accepts a string or `Path`. Relative paths are relative to `main.py`, not the working directory; absolute paths are also supported. The loader requires a readable `.md` file, rejects symbolic links and blank or whitespace-only content, and reads it once during capability loading. File errors abort loading and unload imported modules.

The file is plain Markdown, without YAML parsing or skill metadata:

```markdown
When the current skill is complete, call {{!exit_agent}}.
Put the result in text and saved output IDs to include in items (or items=[]).
```

For the current `items` contract, XML placement, and correctable errors, see
[Exit Tool Items](10-exit-tool-items.md).

References such as `{{!exit_agent}}` are replaced by capability names and validated after all configured capability roots have loaded. Unknown references fail loading. References do not grant permission to call a capability.

The selected tool's instructions are appended after each normal skill's body and catalog, inside `<exit-tool name="exit_agent">...</exit-tool>`. For nodes initialized at positive depth, they also appear in the initial system prompt. Instructions have no callable capability or separate catalog entry, and are not added to the tool's parameter schema.
The runtime also appends common output-selection guidance to every HDS system
prompt; its exact text and purpose are documented in
[Exit Tool Items](10-exit-tool-items.md).

Only the selected exit tool is added to `allowed_tools`, and only at `depth > 0`. It returns from the current skill; it does not terminate every enclosing skill. `ExitToolData`, history compaction, and saved tool results keep their existing behavior.

#### Migration from exit skills

Move the instruction body into the exit tool directory, remove the old YAML front matter, set `instructions_path` on the decorator, and replace the HDS field `exit-skill` with `exit-tool` using the tool's exposed name. Remove the old `exit-skill` directory and any calls that requested its instructions. The loader rejects the legacy directory with a migration hint. Exit tool names must be unique across all roots and must not collide with tools, deep skills, or HDS names.

---

## Capability Loading & Validation Pipeline

When [`Capabilities.load(...)`](../src/exmekh_lib/core/agents/capabilities/capabilities.py) is called, it executes a multi-stage validation and attachment pipeline:

```mermaid
flowchart TD
    A[Layout Discovery] --> B[Load Markdown & Python Tools]
    B --> C[Select defaults and apply overrides]
    C --> D[Build CapabilitiesIndex]
    D --> E[CapabilitiesValidator]
    E --> F[SkillCatalogAttacher]
    F --> G[Return CapabilitiesData]
```

1. **Discovery & Parsing**: Scans filesystem, parses YAML frontmatter in HDS and deep-skill files, reads plain exit instruction Markdown, and imports Python modules under `tools/` and `exit-tool/`.
2. **Optional Default Composition**: Selects bundled resources and applies per-node named or object overrides before indexing. Without `DefaultCapabilities`, this step leaves loaded resources unchanged.
3. **Index Building**: Indexes resources by exposed name within each capability type using `CapabilitiesIndex`.
4. **Validation (`CapabilitiesValidator`)**: Ensures:
   - Referenced tools and skills exist in the index.
   - No duplicate tool or skill names across capability folders.
   - Each HDS selects an existing exit tool; exit instruction references resolve.
5. **Catalog Attachment (`SkillCatalogAttacher`)**: Adds descriptions of declared sub-skills and tools, including parameter documentation, to deep-skill catalogs.
6. **Runtime Composition**: The HDS-selected exit tool supplies instructions when a skill is opened.

### Scoped parameter documentation

Tools available only inside skills keep their structural parameter schemas in
the initial provider request, with a short `Available in skill ...` description.
Their parameter `description` annotations are omitted, including those on nested
objects, arrays, mappings and union branches. Types, enum values, required fields,
defaults and other validation keywords remain unchanged. A property named
`description`, or literal data in defaults, examples and enum values, is preserved.

When a skill opens, its catalog automatically includes the original tool
descriptions and a `<parameter-descriptions path-format="json-pointer">` section.
Each tool's documented parameters have paths relative to its `parameters` schema,
for example `/properties/request/items/properties/query`. Documentation comes
from the same annotation schema builder as the provider definitions; no Markdown
files need to be generated or maintained. The shared XML renderer escapes text
and losslessly encodes characters that XML cannot represent.

The injected `execution_mode` and `wait_seconds` descriptions are omitted from
these catalogs: their semantics, defaults, null handling and limits are already
explained once in the HDS's `<harness-output-guidance>`. Synchronous business
parameters with those names retain their own documentation.

HDS-direct tools, skill descriptions, exit tools, runtime tools and standalone
`Client` tools retain their schema documentation. Contextual tool copies do not
mutate the originals, so the same tool can be direct in one node and internal in
another. Skill documentation follows the existing history lifecycle: it is
available before work calls, remains available through nested skill execution,
and disappears when that skill's history is replaced on exit. Reopening the
skill loads its documentation again.

`DeepAgent.get_tools_text()` returns these same contextual definitions for either
provider API. Token counts of this JSON are useful for comparisons; they do not
measure the backend's rendered prompt or establish model accuracy. Compare
task success, argument corrections and total tokens separately when evaluating
the behavior with a real model.

An offline measurement on 2026-10-18 used the cached Qwen tokenizer and the bundled
`main` HDS with one `mock-sub-agent` child (28 definitions):

| API | Full parameter descriptions | Scoped descriptions | Saved |
| --- | ---: | ---: | ---: |
| Responses | 4,108 tokens | 3,246 tokens | 862 (21.0%) |
| Chat Completions | 3,606 tokens | 3,032 tokens | 574 (15.9%) |

Before catalog serialization was unified on the shared XML renderer, generated
parameter-documentation sections cost 88 tokens in `deep-thinking` and 210 in
`file-processing`, loaded only when those skills open. These historical counts
may differ from the current XML indentation. The comparison
keeps the existing contextual function descriptions in both variants. It excludes
system text, messages, backend prompt rendering and model reasoning.

The focused live regression in `tests/live/test_parameter_documentation_live.py`
checks both Responses and Chat Completions: a real model fills a nested argument from the generated catalog
while its function schema omits that description. It makes one required tool
call per API and verifies the actual Python result; it does not measure autonomous skill
selection or end-to-end task completion.

An exploratory nested-skill workflow on the same date read the documentation
correctly, executed the check and returned through both skills, but reopened the
completed workflow. Restoring full parameter descriptions reproduced the same
loop. Both runs reached their turn limit; they do not establish end-to-end
quality equivalence. See the [recorded observations](test-evidence/scoped-parameter-documentation-2026-10-18.xml).
