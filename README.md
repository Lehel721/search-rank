# search-rank

Two-stage semantic search engine over Wikipedia-style documents:

1. **Stage 1 retrieval** with a FAISS dense index
2. **Stage 2 reranking** with a LightGBM learning-to-rank model

The package also includes ranking metrics for **NDCG@k** and **MRR@k**.

## Quick start

```python
from search_rank.pipeline import SemanticSearchEngine

engine = SemanticSearchEngine(embedding_dim=384, retrieve_k=50, rerank_k=10)
engine.build_index(documents)  # list[Document]
engine.fit_ranker(train_queries)  # list[TrainingQuery]

results = engine.search(query_embedding, top_k=10)
report = engine.evaluate(eval_queries, k=10)
print(report)  # {"ndcg@10": ..., "mrr@10": ...}
```