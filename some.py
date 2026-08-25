from datasets import load_dataset

qrels = load_dataset("BeIR/nq-qrels")
qrels_corpus_ids = set(row["corpus-id"] for row in qrels["test"])
full_corpus = load_dataset("BeIR/nq", "corpus", split="corpus")
print(f"Total corpus size: {len(full_corpus)}")