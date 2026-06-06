"""Tests for retrieval pipeline and rankers."""

import pytest

from hermes_next.retrieval.ranker import _cosine_sim, mmr_diversify, rrf_merge


class TestRRF:
    """Reciprocal Rank Fusion."""

    def test_empty_input(self):
        assert rrf_merge([]) == []
        assert rrf_merge([[], []]) == []

    def test_single_list(self):
        results = rrf_merge([[{"id": "a"}, {"id": "b"}]])
        assert len(results) == 2
        assert results[0]["id"] == "a"
        assert results[1]["id"] == "b"

    def test_merge_two_lists(self):
        list1 = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        list2 = [{"id": "b"}, {"id": "c"}, {"id": "d"}]
        merged = rrf_merge([list1, list2])
        assert len(merged) == 4
        # 'b' and 'c' should rank higher (appear in both lists)
        assert merged[0]["id"] in ("b", "c")
        assert merged[1]["id"] in ("b", "c")

    def test_rrf_score(self):
        list1 = [{"id": "a"}]
        list2 = [{"id": "a"}]
        merged = rrf_merge([list1, list2])
        # RRF score: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.0328
        assert abs(merged[0]["score"] - 2.0 / 61.0) < 1e-6


class TestMMR:
    """Maximum Marginal Relevance."""

    def test_empty_input(self):
        assert mmr_diversify([], k=5) == []

    def test_no_embedding_fallback(self):
        results = [
            {"id": "a", "score": 0.9},
            {"id": "b", "score": 0.5},
            {"id": "c", "score": 0.3},
        ]
        diversified = mmr_diversify(results, k=2)
        assert len(diversified) == 2
        assert diversified[0]["id"] == "a"
        assert diversified[1]["id"] == "b"

    def test_with_embeddings(self):
        results = [
            {"id": "a", "score": 0.9, "embedding": [1.0, 0.0]},
            {"id": "b", "score": 0.8, "embedding": [0.9, 0.1]},
            {"id": "c", "score": 0.7, "embedding": [0.0, 1.0]},
        ]
        query_emb = [1.0, 0.0]
        diversified = mmr_diversify(results, query_embedding=query_emb, lambda_=0.7, k=2)
        assert len(diversified) == 2


class TestCosineSim:
    """Cosine similarity utility."""

    def test_identical(self):
        assert _cosine_sim([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite(self):
        assert _cosine_sim([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_empty(self):
        assert _cosine_sim([], [1.0, 0.0]) == 0.0
        assert _cosine_sim([1.0, 0.0], []) == 0.0
