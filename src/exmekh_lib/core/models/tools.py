from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.core.models.tool import Tool


class Tools:
    def __init__(self, tools: list[Tool]):
        self._check_inputs(tools)
        self._tools = {tool.get_name(): tool for tool in tools}

    def _check_inputs(self, tools: list[Tool]):
        if not isinstance(tools, list) or not all(
            isinstance(tool, Tool) for tool in tools
        ):
            raise TypeError(MODEL_TEXT.tools_must_be_a_list_of_tool_instances)
        names = [tool.get_name() for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError(MODEL_TEXT.tool_names_must_be_unique)

    def get_tool(self, name: str) -> Tool | None:
        """
        Get a tool by its name.

        Args:
            name (str): The name of the tool.

        Returns:
            Tool | None: The tool if found, otherwise None.
        """
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """
        List the names of all available tools.

        Returns:
            list[str]: A list of tool names.
        """
        return list(self._tools.keys())

    def get_tools(self) -> list[Tool]:
        return list(self._tools.values())
