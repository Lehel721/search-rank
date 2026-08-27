# Evaluation Methodology

## Metric

Mean Reciprocal Rank (MRR): for each query, reciprocal rank = `1 / (position of the first correct passage)`. MRR is the average across all queries. Chosen because it directly measures what matters here — how quickly a user finds the right answer, with fast falloff for burying it lower in the list.

## Train/test split

Evaluating on training data measures memorization, not generalization. The 3,452 queries were split 80/20 by query ID (not by row, so a query's positive + hard negatives always stay together). The ranker was retrained from scratch on the 2,761 training queries only, then evaluated on the 691 held-out test queries it had never seen.

## The retrieval_rank bug and fix

An earlier version of this evaluation returned a suspicious MRR of **1.0000**. Traced to a real bug: the `retrieval_rank` feature was hardcoded to `0` for every positive example (since positives come from qrels, not live search), giving the model a trivial shortcut instead of requiring it to learn from the other three features.

**Fix:** every positive was re-run through actual FAISS search and given its real retrieval rank (0-9 if found in the top 10, or 10 as a placeholder if not found at all).

| Positive breakdown by true retrieval rank | Count | % |
|---|---|---|
| Rank 0 (retrieval's #1 pick) | 2,133 | 51% |
| Rank 1-9 (retrieved, not top) | 1,386 | 33% |
| Not retrieved in top-10 | 682 | 16% |

## Easy vs. hard query split

Since half the positives were already retrieval's top pick, a single MRR partly reflects "did reranking avoid breaking easy cases" rather than "did reranking add value." Test queries were split into:

- **Easy** — retrieval's raw top pick was already correct
- **Hard** — it wasn't (misranked lower, or missing from top-10 entirely)

**Results (test set, corrected retrieval_rank):**

| | MRR | Queries |
|---|---|---|
| Overall | **0.9006** | 691 |
| Easy | 0.9746 | 389 |
| Hard | 0.8052 | 302 |

The hard-query MRR is the more meaningful number for judging whether reranking genuinely helps, since it excludes cases retrieval already solved.

## Why this was run on Google Colab

Generating corrected training data required live FAISS search for all 4,201 positives, plus recomputing embedding/TF-IDF features. This repeatedly caused process kills and segfaults locally (8GB RAM), even after optimization (memory-mapped FAISS, on-disk passage lookup, cached TF-IDF vectorizer). Files were uploaded to Colab, where regeneration, training, and evaluation completed successfully; results were downloaded back and saved in `Evaluation/evaluation.json`.

## Known limitations

- **Dataset scale** (3,452 queries, ~6 candidates each) is small relative to production benchmarks, due to hardware constraints on generating a larger corpus/query set. Read results as a positive signal on a constrained setup, not a state-of-the-art claim.
- **Hard negatives** come only from FAISS's own top-10, so evaluation doesn't test the full diversity of wrong answers a real user might see.
- **`retrieval_rank`** remains the most informative feature by construction (retrieval was already correct 51% of the time); the fix removed the hardcoded-zero shortcut, but this feature's real strength should be kept in mind when reading how much credit goes to the other three features.