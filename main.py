import os
import subprocess
import sys


def run_step(name, script_path, output_check_path):
    if os.path.exists(output_check_path):
        print(f"[SKIP] {name} — output already exists at {output_check_path}")
        return

    print(f"[RUNNING] {name} ...")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"[FAILED] {name} — stopping pipeline")
        sys.exit(1)
    print(f"[DONE] {name}")


if __name__ == "__main__":
    run_step(
        name="Data Preparation (qrels-first 100k corpus)",
        script_path="src/data_prep.py",
        output_check_path="data/processed/corpus_100k.jsonl",
    )

    run_step(
        name="Embedding (fastembed, batched — long-running)",
        script_path="src/embed_corpus.py",
        output_check_path="data/processed/embeddings_100k.npy",
    )

    run_step(
        name="FAISS Index Build",
        script_path="src/build_index.py",
        output_check_path="models/faiss_index_100k.bin",
    )

    run_step(
        name="TF-IDF Vectorizer Fit",
        script_path="src/fit_and_save_vectorizer.py",
        output_check_path="models/tfidf_vectorizer.pkl",
    )

    run_step(
        name="Label Preparation (qrels + hard negatives via FAISS)",
        script_path="src/prepare_labels.py",
        output_check_path="data/processed/full_training_data.jsonl",
    )

    run_step(
        name="Feature Engineering",
        script_path="src/features.py",
        output_check_path="data/processed/feature_table.jsonl",
    )

    run_step(
        name="Train Ranker",
        script_path="src/lgbm_ranker.py",
        output_check_path="models/lgbm_ranker.txt",
    )

    print("\nPipeline complete. Run `python src/rank.py` to try a search.")