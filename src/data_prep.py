from datasets import load_dataset
import json
import os
import random

def get_qrels_corpus_ids():
    qrels = load_dataset("BeIR/nq-qrels")
    return set(row["corpus-id"] for row in qrels["test"])

def load_full_corpus():
    return load_dataset("BeIR/nq", "corpus", split="corpus")

def filter_qrels_passages(full_corpus, qrels_corpus_ids):
    return [doc for doc in full_corpus if doc["_id"] in qrels_corpus_ids]

def get_padding_passages(full_corpus, qrels_corpus_ids, target_total, already_have):
    remaining_pool = [doc for doc in full_corpus if doc["_id"] not in qrels_corpus_ids]
    n_needed = target_total - already_have
    padding = random.sample(remaining_pool, n_needed)
    return padding

def save_passages(passages, save_path="data/processed/corpus_100k.jsonl"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for doc in passages:
            record = {"id": doc["_id"], "title": doc["title"], "text": doc["text"]}
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(passages)} passages to {save_path}")

if __name__ == "__main__":
    qrels_corpus_ids = get_qrels_corpus_ids()
    print(f"Qrels passages needed: {len(qrels_corpus_ids)}")

    full_corpus = load_full_corpus()
    print(f"Full corpus loaded: {len(full_corpus)}")

    qrels_passages = filter_qrels_passages(full_corpus, qrels_corpus_ids)
    print(f"Qrels passages found: {len(qrels_passages)}")

    padding = get_padding_passages(full_corpus, qrels_corpus_ids, target_total=100000, already_have=len(qrels_passages))
    print(f"Padding passages selected: {len(padding)}")

    all_passages = qrels_passages + padding
    print(f"Total passages: {len(all_passages)}")

    save_passages(all_passages)