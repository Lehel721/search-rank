from datasets import load_dataset
import json
from retrieve import search

qrels = load_dataset("BeIR/nq-qrels")

def load_passages(path="data/processed/corpus_100k.jsonl"):
    passages = []
    with open(path, "r") as f:
        for line in f:
            passages.append(json.loads(line))
    return passages

passages = load_passages()
corpus_ids_in_subset = set(p["id"] for p in passages)
qrels_data = qrels["test"]

def load_queries():
    queries_dataset = load_dataset("BeIR/nq", "queries")
    queries_dict = {}
    for row in queries_dataset["queries"]:
        queries_dict[row["_id"]] = row["text"]
    return queries_dict

queries_dict = load_queries()

def passages_to_dict(passages):
    return {p["id"]: p for p in passages}

passages_dict = passages_to_dict(passages)

def build_labeled_data(qrels_split, queries_dict, passages_dict):
    labeled_rows = []
    for row in qrels_split:
        query_id = row["query-id"]
        corpus_id = row["corpus-id"]
        score = row["score"]
        if corpus_id in passages_dict:
            labeled_rows.append({
                "query_id": query_id,
                "query_text": queries_dict[query_id],
                "passage_id": corpus_id,
                "passage_text": passages_dict[corpus_id]["text"],
                "relevance": score
            })
    return labeled_rows

def get_hard_negatives(query_id, query_text, positive_passage_ids, top_k=10, n_negatives=5):
    candidates = search(query_text, top_k=top_k)
    
    negatives = []
    for candidate in candidates:
        if candidate["id"] not in positive_passage_ids:
            negatives.append(candidate)
        if len(negatives) >= n_negatives:
            break
    
    return negatives

def get_positive_ids_by_query(labeled_data):
    positive_ids = {}
    for row in labeled_data:
        qid = row["query_id"]
        if qid not in positive_ids:
            positive_ids[qid] = set()
        positive_ids[qid].add(row["passage_id"])
    return positive_ids

def build_full_training_data(labeled_data, n_negatives=5):
    positive_ids_by_query = get_positive_ids_by_query(labeled_data)
    full_data = []
    seen_queries = set()

    for row in labeled_data:
        row["retrieval_rank"] = 0  
        full_data.append(row)
        
        qid = row["query_id"]
        if qid not in seen_queries:
            seen_queries.add(qid)
            negatives = get_hard_negatives(
                query_id=qid,
                query_text=row["query_text"],
                positive_passage_ids=positive_ids_by_query[qid],
                n_negatives=n_negatives
            )
            for neg in negatives:
                full_data.append({
                    "query_id": qid,
                    "query_text": row["query_text"],
                    "passage_id": neg["id"],
                    "passage_text": neg["text"],
                    "relevance": 0,
                    "retrieval_rank": neg["retrieval_rank"]
                })

    return full_data

if __name__ == "__main__":
    labeled_data = build_labeled_data(qrels_data, queries_dict, passages_dict)
    full_data = build_full_training_data(labeled_data)
    
    print(f"Total rows (positives + hard negatives): {len(full_data)}")
    print(f"Positive rows: {sum(1 for r in full_data if r['relevance'] == 1)}")
    print(f"Negative rows: {sum(1 for r in full_data if r['relevance'] == 0)}")
    
    with open("data/processed/full_training_data.jsonl", "w") as f:
        for row in full_data:
            f.write(json.dumps(row) + "\n")
    print("Saved full_data to data/processed/full_training_data.jsonl")