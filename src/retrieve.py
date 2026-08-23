from fastembed import TextEmbedding
import faiss
import json
import numpy as np

model = TextEmbedding("BAAI/bge-small-en-v1.5")
index = faiss.read_index("models/faiss_index.bin")

def load_passages(path="data/processed/corpus_subset.jsonl"):
    passages = []
    with open(path, "r") as f:
        for line in f:
            passages.append(json.loads(line))
    return passages

passages = load_passages()

def search(query, top_k=5):
    query_embedding = list(model.embed([query]))
    query_vector = np.array(query_embedding)
    
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for idx in indices[0]:
        results.append(passages[idx])
    
    return results

if __name__ == "__main__":
    query = "what is minority interest in accounting"
    results = search(query, top_k=3)
    
    for r in results:
        print(r["title"])
        print(r["text"][:200])
        print("---")