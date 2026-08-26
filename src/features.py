from fastembed import TextEmbedding
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from prepare_labels import build_labeled_data, build_full_training_data, qrels_data, queries_dict, passages_dict, passages

model = TextEmbedding("BAAI/bge-small-en-v1.5")

def compute_similarity(query_text, passage_text, model):
    query_emb = list(model.embed([query_text]))[0]
    passage_emb = list(model.embed([passage_text]))[0]
    
    similarity = np.dot(query_emb, passage_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(passage_emb))
    return similarity

def fit_tfidf(passages):
    texts = [p["text"] for p in passages]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    return vectorizer

def compute_tfidf_similarity(query_text, passage_text, vectorizer):
    query_vec = vectorizer.transform([query_text])
    passage_vec = vectorizer.transform([passage_text])
    
    similarity = cosine_similarity(query_vec, passage_vec)[0][0]
    return similarity

def compute_passage_length(passage_text):
    return len(passage_text.split())

def extract_features(query_text, passage_text, retrieval_rank):
    return {
        "embedding_similarity": compute_similarity(query_text, passage_text, model),
        "tfidf_similarity": compute_tfidf_similarity(query_text, passage_text, vectorizer),
        "passage_length": compute_passage_length(passage_text),
        "retrieval_rank": retrieval_rank
    }

def build_feature_table(labeled_data, model, vectorizer):
    feature_rows = []
    
    for row in labeled_data:
        features = extract_features(row["query_text"], row["passage_text"], row["retrieval_rank"])
        features["query_id"] = row["query_id"]
        features["passage_id"] = row["passage_id"]
        features["relevance"] = row["relevance"]
        feature_rows.append(features)
    
    return feature_rows

if __name__ == "__main__":
    full_data = []
    with open("data/processed/full_training_data.jsonl", "r") as f:
        for line in f:
            full_data.append(json.loads(line))
    
    vectorizer = fit_tfidf(passages)
    feature_table = build_feature_table(full_data, model, vectorizer)
    
    print(f"Built {len(feature_table)} feature rows")
    print(feature_table[0])
    
    with open("data/processed/feature_table.jsonl", "w") as f:
        for row in feature_table:
            row_serializable = {k: float(v) if hasattr(v, 'item') else v for k, v in row.items()}
            f.write(json.dumps(row_serializable) + "\n")
    print("Saved feature table to data/processed/feature_table.jsonl")