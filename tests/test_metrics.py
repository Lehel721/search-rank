from search_rank.metrics import mean_mrr, mean_ndcg, mrr_at_k, ndcg_at_k


def test_ndcg_perfect_order_is_one():
    assert ndcg_at_k([3, 2, 1, 0], 4) == 1.0


def test_mrr_finds_first_relevant():
    assert mrr_at_k([0, 0, 1, 0], 4) == 1 / 3


def test_mean_metrics_across_queries():
    labels = [[3, 0, 0], [0, 1, 0]]
    assert 0 <= mean_ndcg(labels, 3) <= 1
    assert mean_mrr(labels, 3) == (1.0 + 0.5) / 2
