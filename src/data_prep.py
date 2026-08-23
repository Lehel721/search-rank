from datasets import load_dataset
import json
import os

def load_corpus_subset(n_passages=3000, save_path="data/processed/corpus_subset.jsonl"):
    """
    Load a slice of BeIR/nq corpus and save it locally as JSONL
    for downstream embedding.
    """
    corpus=load_dataset("BeIR/nq", "corpus", split=f"corpus[:{n_passages}]")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for doc in corpus:
            record = {
                "id": doc["_id"],
                "title": doc["title"],
                "text": doc["text"]
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(corpus)} passages to {save_path}")
    return save_path

if __name__ == "__main__":
    load_corpus_subset(n_passages=3000)