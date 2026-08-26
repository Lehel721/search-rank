import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

passages = []
with open("data/processed/corpus_100k.jsonl", "r") as f:
    for line in f:
        passages.append(json.loads(line))

texts = [p["text"] for p in passages]

vectorizer = TfidfVectorizer()
vectorizer.fit(texts)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Vectorizer saved to models/tfidf_vectorizer.pkl")