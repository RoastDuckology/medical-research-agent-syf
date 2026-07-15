"""Mock LLM provider for local development without API credentials."""

from llm.base import BaseLLM


class MockLLM(BaseLLM):
    """Deterministic fallback implementation used when DeepSeek is not configured."""

    def generate(self, prompt: str) -> str:
        """Return a local mock report."""

        return (
            "# LLM Research Report\n\n"
            "当前使用 MockLLM 生成报告，因为未配置 DeepSeek API Key。\n\n"
            "## 摘要\n\n"
            "系统已完成个人工作日志检索和外部研究信息收集。"
            "真实报告生成将在配置 DeepSeek 后启用。\n\n"
            "## 说明\n\n"
            "本结果不调用任何外部 LLM API。"
        )
