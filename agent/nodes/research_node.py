"""ReAct-style research node that selects and calls external tools."""

from agent.state import ResearchState
from tools.base import ToolPermanentError, ToolTransientError
from tools.registry import create_default_tool_registry


def research_node(state: ResearchState) -> dict:
    """Run selected research tools and record a compact reasoning trace."""

    registry = create_default_tool_registry()
    query = state.get("query", "")
    selected_tools = state.get("selected_tools")

    if not hasattr(registry, "get"):
        results = registry.run_all(query=query)
        return {
            "research_results": results,
            "reasoning_trace": [
                {
                    "thought": "No explicit tool plan was available, so all registered tools were used.",
                    "action": "run_all",
                    "observation": f"{len(results)} results returned.",
                }
            ],
        }

    if selected_tools is None:
        selected_tools = ["arxiv", "github"]

    if not selected_tools:
        return {
            "research_results": [],
            "reasoning_trace": [
                {
                    "thought": "The planner selected a local-RAG-only answer path.",
                    "action": "local_rag_only",
                    "observation": "External tools were skipped.",
                }
            ],
        }

    results: list[dict] = []
    trace: list[dict] = []
    for tool_name in selected_tools:
        try:
            tool = registry.get(tool_name)
        except KeyError:
            trace.append(
                {
                    "thought": f"Tool '{tool_name}' was selected but is not registered.",
                    "action": "skip",
                    "observation": "No results returned.",
                }
            )
            continue

        tool_results, tool_trace = _run_tool_with_recovery(tool, query)
        results.extend(tool_results)
        trace.append(
            {
                "thought": f"Use {tool_name} because the plan selected it for this query.",
                "action": tool_name,
                "observation": f"{len(tool_results)} results returned. {tool_trace}",
            }
        )

    return {"research_results": results, "reasoning_trace": trace}


def _run_tool_with_recovery(tool, query: str) -> tuple[list[dict], str]:
    """Execute a tool with one retry and a deterministic degraded result."""

    try:
        return tool.run(query=query), "status=ok"
    except ToolTransientError as exc:
        try:
            retry_results = tool.run(query=query)
            return retry_results, f"status=recovered_after_retry; error={exc}"
        except Exception as retry_exc:
            return [_degraded_result(tool.name, query, str(retry_exc))], (
                f"status=degraded_after_retry; error={retry_exc}"
            )
    except ToolPermanentError as exc:
        return [_degraded_result(tool.name, query, str(exc))], f"status=degraded_no_retry; error={exc}"
    except Exception as exc:
        return [_degraded_result(tool.name, query, str(exc))], f"status=degraded_unknown; error={exc}"


def _degraded_result(tool_name: str, query: str, error: str) -> dict:
    return {
        "type": "degraded_tool_result",
        "source": tool_name,
        "title": f"{tool_name} degraded result",
        "url": "",
        "query": query,
        "error": error,
        "degraded": True,
    }
