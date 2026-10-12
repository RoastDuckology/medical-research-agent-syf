"""Tests for QA evaluation helpers."""

from evaluation.evaluate_qa import _source_blob


def test_source_blob_includes_pdf_and_abs_url():
    blob = _source_blob(
        {
            "retrieved_documents": [
                {
                    "filename": "2603.19312_leworldmodel.pdf",
                    "abs_url": "https://arxiv.org/abs/2603.19312",
                    "page_start": 12,
                }
            ],
            "research_results": [],
        }
    )

    assert "2603.19312_leworldmodel.pdf" in blob
    assert "https://arxiv.org/abs/2603.19312" in blob
