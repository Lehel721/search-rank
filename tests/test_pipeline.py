from __future__ import annotations

import numpy as np

from search_rank.pipeline import Document, SemanticSearchEngine, TrainingQuery


class FakeFaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = None

    def add(self, vectors):
        self.vectors = vectors

    def search(self, query, k):
        scores = np.dot(self.vectors, query[0])
        order = np.argsort(scores)[::-1][:k]
        return np.asarray([scores[order]], dtype=np.float32), np.asarray([order], dtype=np.int64)


class FakeRanker:
    def fit(self, X, y, group):
        self.fitted_ = True
        return self

    def predict(self, X):
        # Prefer lower L2 distance (feature index 2), independent of stage-1 score.
        return -X[:, 2]


def test_two_stage_search_reranks_results():
    engine = SemanticSearchEngine(
        embedding_dim=2,
        retrieve_k=3,
        rerank_k=2,
        faiss_index=FakeFaissIndex(dim=2),
        ranker=FakeRanker(),
    )
    docs = [
        Document("a", "doc a", [1.0, 0.0]),
        Document("b", "doc b", [0.8, 0.2]),
        Document("c", "doc c", [0.0, 1.0]),
    ]
    engine.build_index(docs)

    train = [
        TrainingQuery("q1", [1.0, 0.0], {"a": 3, "b": 2, "c": 0}),
    ]
    engine.fit_ranker(train)

    results = engine.search([0.7, 0.3], top_k=2)
    assert [result.doc_id for result in results] == ["b", "a"]


def test_evaluate_returns_ndcg_and_mrr():
    engine = SemanticSearchEngine(
        embedding_dim=2,
        retrieve_k=3,
        rerank_k=3,
        faiss_index=FakeFaissIndex(dim=2),
        ranker=FakeRanker(),
    )
    docs = [
        Document("a", "doc a", [1.0, 0.0]),
        Document("b", "doc b", [0.8, 0.2]),
        Document("c", "doc c", [0.0, 1.0]),
    ]
    engine.build_index(docs)
    engine.fit_ranker([TrainingQuery("q1", [1.0, 0.0], {"a": 3, "b": 2, "c": 0})])

    report = engine.evaluate([TrainingQuery("q2", [0.7, 0.3], {"b": 3, "a": 1})], k=2)
    assert set(report.keys()) == {"ndcg@2", "mrr@2"}
    assert 0 <= report["ndcg@2"] <= 1
    assert 0 <= report["mrr@2"] <= 1
