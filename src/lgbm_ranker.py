import lightgbm as lgb
import numpy as np
import pandas as pd
from features import build_feature_table, model, fit_tfidf
from prepare_labels import qrels_data, queries_dict, passages_dict, passages, build_labeled_data

labeled_data = build_labeled_data(qrels_data, queries_dict, passages_dict)
vectorizer = fit_tfidf(passages)
feature_table = build_feature_table(labeled_data, model, vectorizer)

df = pd.DataFrame(feature_table)

df = df.sort_values("query_id").reset_index(drop=True)
group_sizes = df.groupby("query_id").size().tolist()

feature_cols = ["embedding_similarity", "tfidf_similarity", "passage_length"]
X = df[feature_cols]
y = df["relevance"]

ranker = lgb.LGBMRanker(objective="lambdarank")
ranker.fit(X, y, group=group_sizes)