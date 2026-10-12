"""GSSC-style context builder for report generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent.state import ResearchState
from context.packet import ContextPacket, estimate_tokens


DEFAULT_CONTEXT_BUDGET_TOKENS = 2800


@dataclass(frozen=True)
class BuiltContext:
    """Structured prompt context and accounting metadata."""

    prompt: str
    packets: list[ContextPacket]
    estimated_tokens: int
    budget_tokens: int

    def summary(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        for packet in self.packets:
            by_type[packet.source_type] = by_type.get(packet.source_type, 0) + 1
        return {
            "estimated_tokens": self.estimated_tokens,
            "budget_tokens": self.budget_tokens,
            "packet_count": len(self.packets),
            "by_type": by_type,
        }


class ContextBuilder:
    """Build a bounded, structured context for one LLM report call."""

    def __init__(self, budget_tokens: int = DEFAULT_CONTEXT_BUDGET_TOKENS) -> None:
        self.budget_tokens = max(800, int(budget_tokens))

    def build(self, state: ResearchState) -> BuiltContext:
        packets = self.gather(state)
        selected = self.select(packets)
        prompt = self.structure(selected)
        if estimate_tokens(prompt) > self.budget_tokens:
            selected = self.compress(selected)
            prompt = self.structure(selected)
        return BuiltContext(
            prompt=prompt,
            packets=selected,
            estimated_tokens=estimate_tokens(prompt),
            budget_tokens=self.budget_tokens,
        )

    def gather(self, state: ResearchState) -> list[ContextPacket]:
        packets = [
            ContextPacket(
                source_type="role_policy",
                title="Role & Policies",
                content=(
                    "你是一名严谨的医学 AI 科研助理。必须基于给定信息生成中文 Markdown 回答；"
                    "不要编造未提供的信息；区分本地实验记录、PDF 原文、外部论文、代码和推测；"
                    "保留 URL、PDF 文件名、页码、file、heading_path 和 Source。"
                ),
                relevance_score=1.0,
                importance=1.0,
            ),
            ContextPacket(
                source_type="task",
                title="Task",
                content=str(state.get("query", "")),
                relevance_score=1.0,
                importance=1.0,
            ),
            ContextPacket(
                source_type="output_requirements",
                title="Output Requirements",
                content=(
                    "回答必须短而清楚，优先直接回答用户问题。不要使用一级标题。"
                    "必须使用结构化格式，每条结论都绑定证据和来源：\n"
                    "### 结论\n"
                    "- 结论1：...\n"
                    "  - 证据：引用给定上下文中的具体句子或摘要，不要编造。\n"
                    "  - 来源：file: 文件名；page: 页码或N/A；heading_path: ...；Source: URL或N/A\n"
                    "- 结论2：...\n"
                    "  - 证据：...\n"
                    "  - 来源：file: ...；page: ...；heading_path: ...；Source: ...\n"
                    "### 信息不足与拒答\n"
                    "- 如果证据不足，明确说明无法回答，并列出缺失证据。"
                ),
                relevance_score=1.0,
                importance=1.0,
            ),
        ]
        packets.extend(self._plan_packets(state))
        packets.extend(self._episodic_memory_packets(state))
        packets.extend(self._rag_packets(state))
        packets.extend(self._tool_trace_packets(state))
        packets.extend(self._external_result_packets(state))
        return packets

    def select(self, packets: list[ContextPacket]) -> list[ContextPacket]:
        fixed = [
            packet
            for packet in packets
            if packet.source_type in {"role_policy", "task", "output_requirements", "plan"}
        ]
        candidates = [
            packet
            for packet in packets
            if packet.source_type not in {"role_policy", "task", "output_requirements", "plan"}
        ]

        selected = list(fixed)
        remaining_budget = self.budget_tokens - sum(packet.estimated_tokens for packet in selected)
        type_budgets = {
            "rag_evidence": int(self.budget_tokens * 0.48),
            "episodic_memory": int(self.budget_tokens * 0.16),
            "tool_trace": int(self.budget_tokens * 0.16),
            "external_result": int(self.budget_tokens * 0.12),
        }
        used_by_type = {key: 0 for key in type_budgets}

        for packet in sorted(candidates, key=lambda item: item.priority, reverse=True):
            packet_tokens = packet.estimated_tokens
            type_budget = type_budgets.get(packet.source_type, int(self.budget_tokens * 0.1))
            if used_by_type.get(packet.source_type, 0) + packet_tokens > type_budget:
                compacted = self._compress_packet(packet, max_tokens=max(80, type_budget - used_by_type.get(packet.source_type, 0)))
                packet_tokens = compacted.estimated_tokens
                packet = compacted
            if packet_tokens <= 0 or packet_tokens > remaining_budget:
                continue
            selected.append(packet)
            remaining_budget -= packet_tokens
            used_by_type[packet.source_type] = used_by_type.get(packet.source_type, 0) + packet_tokens

        return selected

    def structure(self, packets: list[ContextPacket]) -> str:
        sections: list[str] = []
        grouped = _group_packets(packets)
        order = [
            ("role_policy", "[Role & Policies]"),
            ("task", "[Task]"),
            ("plan", "[Plan]"),
            ("rag_evidence", "[Evidence: Local RAG / PDF]"),
            ("episodic_memory", "[Context: Episodic Memory]"),
            ("tool_trace", "[Tool Trace]"),
            ("external_result", "[External Results]"),
            ("output_requirements", "[Output Requirements]"),
        ]
        for source_type, heading in order:
            items = grouped.get(source_type, [])
            if not items:
                continue
            sections.append(heading)
            for packet in items:
                if len(items) == 1 and packet.title == heading.strip("[]"):
                    sections.append(packet.content)
                else:
                    sections.append(f"### {packet.title}\n{packet.content}")
        return "\n\n".join(sections).strip()

    def compress(self, packets: list[ContextPacket]) -> list[ContextPacket]:
        fixed_types = {"role_policy", "task", "output_requirements"}
        compressed: list[ContextPacket] = []
        for packet in packets:
            if packet.source_type in fixed_types:
                compressed.append(packet)
                continue
            compressed.append(self._compress_packet(packet, max_tokens=180))
        return self.select(compressed)

    def _plan_packets(self, state: ResearchState) -> list[ContextPacket]:
        plan = state.get("plan", [])
        if not plan:
            return []
        return [
            ContextPacket(
                source_type="plan",
                title="Plan-and-Solve",
                content=json.dumps(plan, ensure_ascii=False, indent=2),
                relevance_score=0.95,
                importance=0.8,
            )
        ]

    def _episodic_memory_packets(self, state: ResearchState) -> list[ContextPacket]:
        packets: list[ContextPacket] = []
        for record in state.get("memory_records", []) or []:
            refs = "; ".join(record.get("source_refs", []))
            content = "\n".join(
                [
                    f"query: {record.get('query', '')}",
                    f"summary: {record.get('summary', '')}",
                    f"refs: {refs}",
                    f"created_at: {record.get('created_at', '')}",
                ]
            )
            packets.append(
                ContextPacket(
                    source_type="episodic_memory",
                    title=f"{record.get('memory_type', 'interaction')} #{record.get('id', '')}",
                    content=content,
                    relevance_score=float(record.get("memory_score", 0.4)),
                    importance=float(record.get("importance", 0.5)),
                    metadata=record,
                )
            )
        return packets

    def _rag_packets(self, state: ResearchState) -> list[ContextPacket]:
        packets: list[ContextPacket] = []
        for document in state.get("retrieved_documents", []) or []:
            source = document.get("filename") or document.get("source", "local document")
            page = document.get("page_start", "")
            title = f"{source}" + (f" p.{page}" if page else "")
            provenance = [
                f"file: {source}",
                f"document_type: {document.get('document_type', '')}",
                f"heading_path: {document.get('heading_path', '')}",
                f"page: {document.get('page_start', '')}",
                f"Source: {document.get('abs_url') or document.get('pdf_url') or ''}",
            ]
            content = "\n".join(
                [
                    *[line for line in provenance if line.split(": ", 1)[-1]],
                    "content:",
                    str(document.get("content", "")).strip(),
                ]
            )
            packets.append(
                ContextPacket(
                    source_type="rag_evidence",
                    title=title,
                    content=content,
                    relevance_score=float(document.get("memory_score", document.get("score", 0.4))),
                    importance=float(document.get("importance", 0.7)),
                    metadata=document,
                )
            )
        return packets

    def _tool_trace_packets(self, state: ResearchState) -> list[ContextPacket]:
        trace = state.get("reasoning_trace", [])
        if not trace:
            return []
        return [
            ContextPacket(
                source_type="tool_trace",
                title="ReAct trace",
                content=json.dumps(trace, ensure_ascii=False, indent=2),
                relevance_score=0.7,
                importance=0.6,
            )
        ]

    def _external_result_packets(self, state: ResearchState) -> list[ContextPacket]:
        packets: list[ContextPacket] = []
        for index, result in enumerate(state.get("research_results", []) or [], start=1):
            title = result.get("title") or result.get("name") or result.get("type") or f"result {index}"
            packets.append(
                ContextPacket(
                    source_type="external_result",
                    title=str(title),
                    content=json.dumps(result, ensure_ascii=False, indent=2),
                    relevance_score=0.65,
                    importance=0.65 if result.get("url") or result.get("html_url") else 0.5,
                    metadata=result,
                )
            )
        return packets

    def _compress_packet(self, packet: ContextPacket, max_tokens: int) -> ContextPacket:
        if packet.estimated_tokens <= max_tokens:
            return packet
        max_chars = max(240, int(max_tokens * 2.2))
        preserved = _preserved_metadata_lines(packet.content)
        content_body = " ".join(packet.content.split())
        if preserved:
            reserved = "\n".join(preserved)
            remaining = max(120, max_chars - len(reserved))
            compacted = f"{reserved}\nsummary: {content_body[:remaining].rstrip()}..."
        else:
            compacted = content_body[:max_chars].rstrip() + "..."
        return ContextPacket(
            source_type=packet.source_type,
            title=packet.title,
            content=compacted,
            relevance_score=packet.relevance_score,
            importance=packet.importance,
            metadata={**packet.metadata, "compressed": True},
        )


def _group_packets(packets: list[ContextPacket]) -> dict[str, list[ContextPacket]]:
    grouped: dict[str, list[ContextPacket]] = {}
    for packet in packets:
        grouped.setdefault(packet.source_type, []).append(packet)
    return grouped


def _preserved_metadata_lines(content: str) -> list[str]:
    prefixes = ("file:", "document_type:", "heading_path:", "page:", "Source:", "refs:")
    lines = []
    for line in content.splitlines():
        if line.startswith(prefixes):
            lines.append(line)
    return lines[:8]
