import faiss
import numpy as np

def load_embeddings(path="data/processed/embeddings.npy"):
    return np.load(path)

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_index(index, save_path="models/faiss_index.bin"):
    faiss.write_index(index, save_path)
    print(f"Saved FAISS index to {save_path}")

if __name__ == "__main__":
    embeddings = load_embeddings(path="data/processed/embeddings_100k.npy")
    index = build_faiss_index(embeddings)
    save_index(index, save_path="models/faiss_index_100k.bin")