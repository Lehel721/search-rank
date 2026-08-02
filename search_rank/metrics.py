from __future__ import annotations

import math
from typing import Iterable, Sequence


def _dcg(labels: Sequence[float], k: int) -> float:
    return sum((2**label - 1) / math.log2(idx + 2) for idx, label in enumerate(labels[:k]))


def ndcg_at_k(labels_in_rank_order: Sequence[float], k: int) -> float:
    if not labels_in_rank_order:
        return 0.0
    ideal = sorted(labels_in_rank_order, reverse=True)
    ideal_dcg = _dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(labels_in_rank_order, k) / ideal_dcg


def mrr_at_k(labels_in_rank_order: Sequence[float], k: int) -> float:
    for idx, label in enumerate(labels_in_rank_order[:k], start=1):
        if label > 0:
            return 1.0 / idx
    return 0.0


def mean_ndcg(ranked_labels: Iterable[Sequence[float]], k: int) -> float:
    ranked_labels = list(ranked_labels)
    if not ranked_labels:
        return 0.0
    return sum(ndcg_at_k(labels, k) for labels in ranked_labels) / len(ranked_labels)


def mean_mrr(ranked_labels: Iterable[Sequence[float]], k: int) -> float:
    ranked_labels = list(ranked_labels)
    if not ranked_labels:
        return 0.0
    return sum(mrr_at_k(labels, k) for labels in ranked_labels) / len(ranked_labels)
