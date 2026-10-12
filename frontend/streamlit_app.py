"""Streamlit chat UI for the medical research assistant."""

from __future__ import annotations

import re
import sys
from html import escape
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.chat_service import ask_agent_with_progress, ensure_knowledge_base  # noqa: E402
from config.settings import get_settings  # noqa: E402
from llm.health import recent_failure  # noqa: E402


SUGGESTIONS = {
    ":material/experiment: 总结实验进度": "总结当前实验进度记录中 StereoVLA 和 LeWM 相关工作的主要结论",
    ":material/picture_as_pdf: 询问 PDF": "In the LeWorldModel paper, how is CEM used for planning?",
    ":material/newspaper: 生成日报": "generate daily brief",
}


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 5rem;
        }
        h1 {
            font-size: 2rem !important;
            line-height: 1.25 !important;
            letter-spacing: 0 !important;
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-size: 1.05rem !important;
            line-height: 1.35 !important;
            letter-spacing: 0 !important;
        }
        .stChatMessage h1,
        .stChatMessage h2 {
            font-size: 1.28rem !important;
            line-height: 1.45 !important;
            letter-spacing: 0 !important;
        }
        .stChatMessage h3 {
            font-size: 1.08rem !important;
            line-height: 1.45 !important;
            letter-spacing: 0 !important;
        }
        .stChatMessage p,
        .stChatMessage li {
            font-size: 0.98rem !important;
            line-height: 1.65 !important;
        }
        .stChatMessage code,
        [data-testid="stCodeBlock"] code {
            font-size: 0.9rem !important;
        }
        [data-testid="stCaptionContainer"] {
            font-size: 0.86rem !important;
        }
        .agent-step {
            padding: 0.48rem 0;
            border-bottom: 1px solid rgba(49, 51, 63, 0.08);
        }
        .agent-step:last-child {
            border-bottom: 0;
        }
        .agent-step-title {
            font-size: 0.94rem;
            font-weight: 650;
            line-height: 1.4;
        }
        .agent-step-detail {
            color: rgba(49, 51, 63, 0.72);
            font-size: 0.88rem;
            line-height: 1.55;
            margin-top: 0.12rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _normalize_markdown_for_chat(text: str) -> str:
    """Demote headings so model Markdown cannot dominate the chat page."""

    def demote(match: re.Match[str]) -> str:
        hashes = match.group(1)
        return "#" * min(len(hashes) + 2, 6) + " "

    return re.sub(r"^(#{1,6})\s+", demote, text.strip(), flags=re.MULTILINE)


def _render_progress(records: list[dict]) -> None:
    if not records:
        st.caption("等待 Agent 开始执行。")
        return

    for index, item in enumerate(records, start=1):
        title = escape(str(item.get("title", "步骤完成")))
        detail = escape(str(item.get("detail", "")))
        st.markdown(
            f"""
            <div class="agent-step">
              <div class="agent-step-title">{index}. {title}</div>
              <div class="agent-step-detail">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _show_sidebar() -> None:
    settings = get_settings()
    with st.sidebar:
        st.subheader("运行状态")
        st.caption(f"LLM base：{settings.deepseek_base_url}")
        st.caption(f"模型：{settings.llm_model}")

        reason = recent_failure(settings.llm_failure_cooldown_seconds)
        if reason:
            st.warning(
                "远程 LLM 最近调用失败，当前会优先走本地 fallback。"
                "请检查模型名、中转服务可用模型或稍后重试。",
                icon=":material/warning:",
            )
            with st.expander("最近失败摘要", expanded=False):
                st.caption(reason or "未记录具体原因。")

        if st.button("检查或构建知识库", icon=":material/database:"):
            with st.spinner("正在检查知识库..."):
                stats = ensure_knowledge_base()
            if stats:
                st.success(
                    f"已构建索引：{stats['markdown_files']} 个文档，"
                    f"{stats['chunks']} 个分块，新增 {stats['added']} 个向量。"
                )
            else:
                st.info("知识库已存在，无需重建。")

        if st.button("清空对话", icon=":material/delete:"):
            st.session_state.messages = []
            st.rerun()

        st.subheader("示例问题")
        st.code("总结当前实验进度记录中 StereoVLA 和 LeWM 相关工作的主要结论")
        st.code("In the LeWorldModel paper, how is CEM used for planning?")
        st.code("generate daily brief")


st.set_page_config(
    page_title="Medical Research Agent",
    page_icon=":material/science:",
    layout="wide",
)
_inject_style()

st.title("Medical Research Agent")
st.caption("面向个人实验记录、论文 PDF、每日 AI 简报和外部工具的交互式科研问答助手。")

st.session_state.setdefault("messages", [])

_show_sidebar()

if not st.session_state.messages:
    selected = st.pills(
        "可以先试试：",
        list(SUGGESTIONS.keys()),
        label_visibility="visible",
    )
    if selected:
        st.session_state.pending_prompt = SUGGESTIONS[selected]
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(_normalize_markdown_for_chat(message["content"]))
        progress = message.get("progress") or []
        if message["role"] == "assistant" and progress:
            with st.expander("过程回放", expanded=False):
                _render_progress(progress)

prompt = st.session_state.pop("pending_prompt", None)
chat_prompt = st.chat_input(
    "输入科研问题，或询问 data/papers 中的 PDF 内容",
    submit_mode="disable",
)
prompt = prompt or chat_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    progress_records: list[dict] = []
    answer = ""

    with st.chat_message("assistant"):
        st.markdown("### 执行过程")
        progress_placeholder = st.empty()
        answer_placeholder = st.empty()

        try:
            with progress_placeholder.container(border=True):
                _render_progress(progress_records)

            for event in ask_agent_with_progress(prompt):
                if event.kind == "progress":
                    progress_records.append({"title": event.title, "detail": event.detail})
                    with progress_placeholder.container(border=True):
                        _render_progress(progress_records)
                else:
                    answer = event.answer
                    progress_records.append({"title": event.title, "detail": event.detail})
                    with progress_placeholder.container(border=True):
                        _render_progress(progress_records)
                    answer_placeholder.markdown(_normalize_markdown_for_chat(answer))

        except Exception as exc:
            answer = f"Agent error: {exc}"
            progress_records.append({"title": "运行失败", "detail": str(exc)})
            with progress_placeholder.container(border=True):
                _render_progress(progress_records)
            answer_placeholder.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "progress": progress_records,
        }
    )
