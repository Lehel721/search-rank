import json

passages = []
with open("data/processed/corpus_subset.jsonl", "r") as f:
    for line in f:
        passages.append(json.loads(line))

lengths = [len(p["text"].split()) for p in passages]
long_passages = [l for l in lengths if l > 350]
print("Number of passages over 350 words:", len(long_passages))
print("Percentage:", len(long_passages) / len(lengths) * 100)