"""Lightweight multi-query expansion for research RAG retrieval."""

from __future__ import annotations


DOMAIN_EXPANSIONS = {
    "baseline": ["基线", "当前判断", "实验结论", "对照实验"],
    "基线": ["baseline", "当前判断", "对照实验"],
    "stereovla": ["StereoVLA", "视觉语言动作模型", "手术场景", "双目视觉"],
    "foundationstereo": ["FoundationStereo", "stereo matching", "disparity", "视差"],
    "双目": ["stereo", "FoundationStereo", "StereoVLA", "视差"],
    "视差": ["disparity", "FoundationStereo", "stereo matching"],
    "lewm": ["LeWM", "latent world model", "world model", "潜在世界模型"],
    "jepa": ["JEPA", "joint embedding predictive architecture", "预测表征学习"],
    "endotapp": ["EndoTAP", "endoscopic", "surgical video"],
    "synthetic": ["合成数据", "synthetic stereo", "仿真数据"],
    "合成": ["synthetic", "仿真数据", "domain gap"],
    "shift": ["shift10", "domain shift", "分布偏移"],
    "日报": ["每日简报", "近7天", "近15天", "AI论文"],
    "简报": ["日报", "每日AI简报", "近15天"],
    "论文": ["paper", "arxiv", "文献", "顶会"],
    "代码": ["github", "repository", "开源实现"],
    "pdf": ["PDF", "原文", "页码", "论文全文"],
}


QUESTION_EXPANSION = "当前判断 证据 来源 下一步 风险 信息不足"


def expand_query(query: str, max_queries: int = 4) -> list[str]:
    """Return a small set of query variants for chapter-8 style MQE retrieval."""

    normalized = query.strip()
    if not normalized:
        return []

    variants = [normalized]
    lowered = normalized.lower()
    for keyword, expansions in DOMAIN_EXPANSIONS.items():
        if keyword.lower() not in lowered:
            continue
        for expansion in expansions:
            candidate = f"{normalized} {expansion}"
            if candidate not in variants:
                variants.append(candidate)
            if len(variants) >= max_queries:
                return variants

    if len(variants) == 1 and _looks_like_question(normalized):
        variants.append(f"{normalized} {QUESTION_EXPANSION}")

    return variants[:max_queries]


def _looks_like_question(query: str) -> bool:
    markers = ["是否", "如何", "为什么", "总结", "当前", "下一步", "?", "？", "where", "how", "why"]
    lowered = query.lower()
    return any(marker in lowered for marker in markers)
