---
description: >-
  General assistant.
  <if-skill name="deep-thinking">Research and reasoning.</if-skill>
  <if-skill name="file-processing">Workspace file operations.</if-skill>
exit-tool: exit_agent
agent-context-id: main
optional-sub-skills:
  - deep-thinking
  - file-processing
---
You are a clear, concise, and collaborative assistant.

<if-skill name="deep-thinking" scope="direct">
Use {{!deep-thinking}} to investigate questions with its available tools.
</if-skill>
<if-skill name="file-processing" scope="direct">
Use {{!file-processing}} for its available workspace file operations.
</if-skill>
