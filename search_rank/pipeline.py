from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .metrics import mean_mrr, mean_ndcg


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    embedding: Sequence[float]


@dataclass(frozen=True)
class TrainingQuery:
    query_id: str
    embedding: Sequence[float]
    relevance_by_doc_id: Dict[str, float]


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    text: str
    stage1_score: float
    rerank_score: float


class SemanticSearchEngine:
    def __init__(
        self,
        embedding_dim: int,
        retrieve_k: int = 100,
        rerank_k: int = 20,
        faiss_index=None,
        ranker=None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.retrieve_k = retrieve_k
        self.rerank_k = rerank_k
        self._index = faiss_index if faiss_index is not None else self._create_default_index(embedding_dim)
        self._ranker = ranker if ranker is not None else self._create_default_ranker()
        self._is_ranker_fitted = False
        self._documents: List[Document] = []

    @staticmethod
    def _create_default_index(embedding_dim: int):
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss is required to build the retrieval index") from exc
        return faiss.IndexFlatIP(embedding_dim)

    @staticmethod
    def _create_default_ranker():
        try:
            from lightgbm import LGBMRanker
        except ImportError as exc:
            raise ImportError("lightgbm is required to train the reranker") from exc
        return LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=50,
            learning_rate=0.1,
            num_leaves=31,
            min_data_in_leaf=5,
        )

    def build_index(self, documents: Sequence[Document]) -> None:
        self._documents = list(documents)
        vectors = self._to_matrix([doc.embedding for doc in documents], normalize=True)
        self._index.add(vectors)

    def fit_ranker(self, queries: Iterable[TrainingQuery]) -> None:
        if not self._documents:
            raise ValueError("build_index must be called before fit_ranker")

        features: List[List[float]] = []
        labels: List[float] = []
        groups: List[int] = []

        for query in queries:
            candidates = self._retrieve(query.embedding, self.retrieve_k)
            if not candidates:
                continue
            groups.append(len(candidates))
            for index_position, doc_index, stage1_score in candidates:
                doc = self._documents[doc_index]
                labels.append(float(query.relevance_by_doc_id.get(doc.doc_id, 0.0)))
                features.append(self._extract_features(query.embedding, doc.embedding, stage1_score, index_position))

        if not features:
            raise ValueError("no candidates found for ranker training")

        self._ranker.fit(np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32), group=groups)
        self._is_ranker_fitted = True

    def search(self, query_embedding: Sequence[float], top_k: int = 10) -> List[SearchResult]:
        candidates = self._retrieve(query_embedding, self.retrieve_k)
        if not candidates:
            return []

        candidate_features = []
        for index_position, doc_index, stage1_score in candidates:
            doc = self._documents[doc_index]
            candidate_features.append(
                self._extract_features(query_embedding, doc.embedding, stage1_score, index_position)
            )

        rerank_scores = self._predict_scores(candidate_features)
        ranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda item: item[1],
            reverse=True,
        )[: min(top_k, self.rerank_k)]

        return [
            SearchResult(
                doc_id=self._documents[doc_index].doc_id,
                text=self._documents[doc_index].text,
                stage1_score=stage1_score,
                rerank_score=float(rerank_score),
            )
            for (_, doc_index, stage1_score), rerank_score in ranked
        ]

    def evaluate(self, queries: Iterable[TrainingQuery], k: int = 10) -> Dict[str, float]:
        ranked_labels: List[List[float]] = []
        for query in queries:
            results = self.search(query.embedding, top_k=k)
            ranked_labels.append([query.relevance_by_doc_id.get(result.doc_id, 0.0) for result in results])

        return {f"ndcg@{k}": mean_ndcg(ranked_labels, k), f"mrr@{k}": mean_mrr(ranked_labels, k)}

    def _to_matrix(self, vectors: Sequence[Sequence[float]], normalize: bool) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected vectors with shape (n, {self.embedding_dim})")
        if normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        return matrix

    def _retrieve(self, query_embedding: Sequence[float], k: int):
        if not self._documents:
            return []
        query = self._to_matrix([query_embedding], normalize=True)
        scores, indexes = self._index.search(query, min(k, len(self._documents)))
        candidates = []
        for idx, (doc_idx, score) in enumerate(zip(indexes[0], scores[0]), start=1):
            if doc_idx < 0:
                continue
            candidates.append((idx, int(doc_idx), float(score)))
        return candidates

    def _extract_features(
        self,
        query_embedding: Sequence[float],
        doc_embedding: Sequence[float],
        stage1_score: float,
        index_position: int,
    ) -> List[float]:
        q = np.asarray(query_embedding, dtype=np.float32)
        d = np.asarray(doc_embedding, dtype=np.float32)
        dot = float(np.dot(q, d))
        l2 = float(np.linalg.norm(q - d))
        return [stage1_score, dot, l2, 1.0 / index_position]

    def _predict_scores(self, features: List[List[float]]) -> np.ndarray:
        if self._is_ranker_fitted:
            return np.asarray(self._ranker.predict(np.asarray(features, dtype=np.float32)))
        return np.asarray([feature[0] for feature in features], dtype=np.float32)
