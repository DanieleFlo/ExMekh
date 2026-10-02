When the current skill is complete, call {{!exit_agent}} with `text` and `items`.
`text` is the useful result for the caller. `items` contains zero or more IDs of outputs saved during this skill invocation or its descendant skills. Use `items=[]` when full outputs are unnecessary.

Prefer the `saved-output-id` from `get_execution_history`. You may also use a call ID if it identifies exactly one output. A memory tool without saved output cannot be included. Do not repeat an output in `items`.

To place an output within the text, use `{{output-id}}` and include its ID in `items`. The ID in the text and the ID in the list may be different IDs for the same output. Listed outputs without placeholders are appended at the end in `items` order as `<tool-output>` XML blocks.

If `exit_agent` returns an error, the skill remains open. Correct the indicated IDs or placeholders and call the tool again. A valid call finishes only the current skill and returns control to the previous level.
