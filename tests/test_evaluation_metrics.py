"""Tests for local evaluation metrics."""

from evaluation.metrics import citation_coverage, evaluate_retrieval_case
from rag.query_expander import expand_query


def test_evaluate_retrieval_case_hit_and_mrr():
    case = {
        "question": "StereoVLA 记录在哪里？",
        "expected_sources": ["2026-07-10.md"],
        "expected_keywords": ["StereoVLA"],
    }
    retrieved = [
        {"filename": "other.md", "content": "nothing"},
        {"filename": "2026-07-10.md", "content": "StereoVLA feature extraction"},
    ]

    metrics = evaluate_retrieval_case(case, retrieved, k_values=(1, 3))

    assert metrics["hit@1"] == 0.0
    assert metrics["hit@3"] == 1.0
    assert metrics["mrr"] == 0.5


def test_evaluate_retrieval_case_can_check_expected_page():
    case = {
        "question": "Where is CEM discussed?",
        "expected_sources": ["leworldmodel.pdf"],
        "expected_keywords": ["CEM"],
        "expected_pages": [12],
    }
    retrieved = [
        {"filename": "leworldmodel.pdf", "content": "CEM planning", "page_start": 11},
        {"filename": "leworldmodel.pdf", "content": "CEM planning", "page_start": 12},
    ]

    metrics = evaluate_retrieval_case(case, retrieved, k_values=(1, 2))

    assert metrics["hit@1"] == 0.0
    assert metrics["hit@2"] == 1.0


def test_citation_coverage_counts_linked_points():
    answer = "StereoVLA 需要继续验证 https://example.com\nLeWM 尚未验证"

    assert citation_coverage(answer, ["StereoVLA", "LeWM"]) == 0.5


def test_expand_query_adds_domain_terms():
    variants = expand_query("StereoVLA baseline")

    assert variants[0] == "StereoVLA baseline"
    assert any("基线" in variant for variant in variants)
