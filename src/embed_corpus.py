from fastembed import TextEmbedding
import json
import numpy as np

model = TextEmbedding("BAAI/bge-small-en-v1.5")

def load_passages(path="data/processed/corpus_subset.jsonl"):
    passages = []
    with open(path, "r") as f:
        for line in f:
            passages.append(json.loads(line))
    return passages

def embed_passages_batched(passages, model, batch_size=100):
    all_embeddings = []
    texts = [p["text"] for p in passages]
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = list(model.embed(batch))
        all_embeddings.extend(batch_embeddings)
        print(f"Processed {i + len(batch)} / {len(texts)} passages")
    
    return np.array(all_embeddings)

def save_embeddings(embeddings, passages, save_path="data/processed/embeddings.npy"):
    np.save(save_path, embeddings)
    print(f"Saved {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]} to {save_path}")

if __name__ == "__main__":
    passages = load_passages(path="data/processed/corpus_100k.jsonl")
    embeddings = embed_passages_batched(passages, model)
    save_embeddings(embeddings, passages, save_path="data/processed/embeddings_100k.npy")