"""Tool registry for managing available research tools."""

from __future__ import annotations

import os

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
    from tools.daily_brief_tool import DailyBriefTool
    from tools.fault_injection_tool import FaultInjectionTool
    from tools.github_tool import GithubTool
    from tools.recorded_tool import RecordedTool

    registry = ToolRegistry()
    mode = os.getenv("RESEARCH_TOOL_MODE", "live").strip().lower()
    if mode == "recorded":
        recordings_dir = os.getenv("TOOL_RECORDINGS_DIR", "data/recordings")
        registry.register(RecordedTool("arxiv", recordings_dir))
        registry.register(RecordedTool("github", recordings_dir))
    elif mode == "fault_injection":
        component = os.getenv("TOOL_FAULT_COMPONENT", "arxiv").strip().lower()
        failure_type = os.getenv("TOOL_FAULT_TYPE", "timeout").strip().lower()
        registry.register(FaultInjectionTool(component, failure_type))
        if component != "arxiv":
            registry.register(RecordedTool("arxiv"))
        if component != "github":
            registry.register(RecordedTool("github"))
    else:
        registry.register(ArxivTool())
        registry.register(GithubTool())
    registry.register(DailyBriefTool())
    return registry
