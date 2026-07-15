"""Tool registry for managing available research tools."""

from tools.base import BaseTool


class ToolRegistry:
    """Simple in-memory registry for research tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool by its unique name."""

        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """Return a registered tool by name."""

        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Tool is not registered: {name}") from exc

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools."""

        return list(self._tools.values())

    def run_all(self, **kwargs) -> list[dict]:
        """Run every registered tool and concatenate their results."""

        results: list[dict] = []
        for tool in self.list_tools():
            results.extend(tool.run(**kwargs))
        return results


def create_default_tool_registry() -> ToolRegistry:
    """Create the default registry used by the research node."""

    from tools.arxiv_tool import ArxivTool
    from tools.github_tool import GithubTool

    registry = ToolRegistry()
    registry.register(ArxivTool())
    registry.register(GithubTool())
    return registry
