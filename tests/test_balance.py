"""Unit tests for corpus balancing and concentration metrics."""

from architectai_pretraining.balance import CorpusBalancer, calculate_concentration_metrics
from architectai_pretraining.models import CorpusDocument
from architectai_pretraining.scoring import DocumentQualityScore


def test_calculate_concentration_metrics() -> None:
    doc1 = CorpusDocument(
        id="d1", title="T1", text="Text 1", source_id="s1", category="c1", license_id="MIT"
    )
    doc2 = CorpusDocument(
        id="d2", title="T2", text="Text 2", source_id="s1", category="c1", license_id="MIT"
    )

    tokens_map = {"d1": 100, "d2": 100}
    metrics = calculate_concentration_metrics([doc1, doc2], tokens_map)

    assert metrics.total_tokens == 200
    assert metrics.top_1_source_share == 1.0
    assert metrics.hhi_source_index == 1.0


def test_corpus_balancer_downsampling() -> None:
    # Create 10 docs for source s1 and 2 docs for source s2
    docs = []
    tokens_map = {}
    quality_scores = {}

    for i in range(10):
        did = f"s1_doc_{i}"
        doc = CorpusDocument(
            id=did, title=f"Title {i}", text=f"Text content for s1 document {i}", source_id="s1", category="c1", license_id="MIT"
        )
        docs.append(doc)
        tokens_map[did] = 100
        quality_scores[did] = DocumentQualityScore(0.5 + (i * 0.04), "medium")

    for j in range(2):
        did = f"s2_doc_{j}"
        doc = CorpusDocument(
            id=did, title=f"Title {j}", text=f"Text content for s2 document {j}", source_id="s2", category="c2", license_id="MIT"
        )
        docs.append(doc)
        tokens_map[did] = 100
        quality_scores[did] = DocumentQualityScore(0.8, "high")

    # Balancer with max 0.30 source share
    balancer = CorpusBalancer(max_source_token_share=0.30)
    res = balancer.balance(docs, tokens_map, quality_scores)

    assert len(res.kept_documents) < len(docs)
    assert len(res.balanced_out_documents) > 0
    assert res.concentration_after.top_1_source_share <= res.concentration_before.top_1_source_share
