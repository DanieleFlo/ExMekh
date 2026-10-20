Call {{!exit_agent}} to close only this skill. Its inner transcript leaves active context:
`text` must preserve the outcome, completed actions, essential facts/references and next step.
Promote full outputs in `items` only when the caller needs their details now; otherwise use `items=[]`.
For use inside a later skill, put the ID/path and purpose in `text` and retrieve there.

`items` accepts saved outputs only from this invocation or descendants. Prefer canonical
`saved-output-id`; an unambiguous call ID is accepted. No duplicate aliases or unsaved memory reads.
<if-tool name="get_execution_history">Find retrievable IDs with `get_execution_history`.</if-tool>

Place with `{{output-id}}` and list that output in `items` (either alias may identify the same record).
Unplaced items append in list order as `<tool-output>` XML; large outputs stay file manifests.

An exit error keeps the skill open: correct the arguments and retry the exit, not the completed work.
