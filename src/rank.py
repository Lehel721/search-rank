import lightgbm as lgb
import json
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from fastembed import TextEmbedding
import faiss


model = TextEmbedding("BAAI/bge-small-en-v1.5")
index = faiss.read_index("models/faiss_index_100k.bin", faiss.IO_FLAG_MMAP)
ranker = lgb.Booster(model_file="models/lgbm_ranker.txt")

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

CORPUS_PATH = "data/processed/corpus_100k.jsonl"



def build_line_offsets(path=CORPUS_PATH):
    offsets = []
    with open(path, "rb") as f:
        offset = f.tell()
        line = f.readline()
        while line:
            offsets.append(offset)
            offset = f.tell()
            line = f.readline()
    return offsets

line_offsets = build_line_offsets()

def get_passage_by_index(idx, path=CORPUS_PATH):
    with open(path, "rb") as f:
        f.seek(line_offsets[idx])
        line = f.readline()
    return json.loads(line)



def search(query, top_k=10):
    query_embedding = list(model.embed([query]))
    query_vector = np.array(query_embedding)

    distances, indices = index.search(query_vector, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        passage = get_passage_by_index(idx)
        passage["retrieval_rank"] = rank
        results.append(passage)

    return results



def compute_similarity(query_text, passage_text):
    query_emb = list(model.embed([query_text]))[0]
    passage_emb = list(model.embed([passage_text]))[0]
    similarity = np.dot(query_emb, passage_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(passage_emb))
    return similarity


def compute_tfidf_similarity(query_text, passage_text):
    query_vec = vectorizer.transform([query_text])
    passage_vec = vectorizer.transform([passage_text])
    similarity = cosine_similarity(query_vec, passage_vec)[0][0]
    return similarity


def compute_passage_length(passage_text):
    return len(passage_text.split())


def extract_features(query_text, passage_text, retrieval_rank):
    return {
        "embedding_similarity": compute_similarity(query_text, passage_text),
        "tfidf_similarity": compute_tfidf_similarity(query_text, passage_text),
        "passage_length": compute_passage_length(passage_text),
        "retrieval_rank": retrieval_rank
    }



def rerank(query, top_k=10):
    candidates = search(query, top_k=top_k)
    
    scored_candidates = []
    for candidate in candidates:
        features = extract_features(query, candidate["text"], candidate["retrieval_rank"])
        feature_values = [[features["embedding_similarity"], features["tfidf_similarity"], features["passage_length"], features["retrieval_rank"]]]
        
        score = ranker.predict(feature_values)[0]
        
        scored_candidates.append({
            "id": candidate["id"],
            "title": candidate["title"],
            "text": candidate["text"],
            "score": score
        })
    
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    return scored_candidates

if __name__ == "__main__":
    query = "what is non controlling interest on balance sheet"
    results = rerank(query, top_k=10)
    
    for r in results[:5]:
        print(f"Score: {r['score']:.4f}")
        print(r["title"])
        print(r["text"][:150])
        print("---")