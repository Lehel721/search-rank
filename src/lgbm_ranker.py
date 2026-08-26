import lightgbm as lgb
import json
import pandas as pd

feature_table = []
with open("data/processed/feature_table.jsonl", "r") as f:
    for line in f:
        feature_table.append(json.loads(line))

df = pd.DataFrame(feature_table)

df = df.sort_values("query_id").reset_index(drop=True)
group_sizes = df.groupby("query_id").size().tolist()

feature_cols = ["embedding_similarity", "tfidf_similarity", "passage_length", "retrieval_rank"]
X = df[feature_cols]
y = df["relevance"]

ranker = lgb.LGBMRanker(objective="lambdarank")
ranker.fit(X, y, group=group_sizes)

print("Training complete")
print(f"Trained on {len(df)} rows across {len(group_sizes)} queries")

ranker.booster_.save_model("models/lgbm_ranker.txt")
print("Model saved to models/lgbm_ranker.txt")