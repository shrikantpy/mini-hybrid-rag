"""
A minimal VECTOR STORE.

Idea: turn every text into a list of numbers (a vector). Texts that use similar
words get similar vectors. To search, turn the question into a vector and return
the documents whose vectors are closest (cosine similarity).

We use TF-IDF vectors so the project runs offline with no model download.
Real projects use neural embeddings (OpenAI, sentence-transformers, ...) and a
vector DB (Chroma, pgvector, Qdrant, ...), but the search logic is the same.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    def __init__(self, documents: dict[str, str]):
        self.doc_ids = list(documents.keys())
        self.texts = list(documents.values())
        self.vectorizer = TfidfVectorizer(stop_words="english")
        # One row (vector) per document
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, top_k: int = 2) -> list[tuple[str, float]]:
        """Return the top_k (doc_id, similarity score) pairs for the query."""
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.doc_vectors)[0]
        ranked = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
