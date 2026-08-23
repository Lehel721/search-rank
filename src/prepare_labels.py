from datasets import load_dataset
import json
qrels = load_dataset("BeIR/nq-qrels")

def load_passages(path="data/processed/corpus_subset.jsonl"):
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

def build_labeled_data(qrels, queries_dict, passages_dict):
    labeled_rows = []
    
    for row in qrels:
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

if __name__ == "__main__":
    labeled_data = build_labeled_data(qrels_data, queries_dict, passages_dict)
    print(f"Total labeled rows: {len(labeled_data)}")
    print(labeled_data[0])