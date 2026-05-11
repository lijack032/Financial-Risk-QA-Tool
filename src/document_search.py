from pathlib import Path
import json
import joblib
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = BASE_DIR / "artifacts" / "document_index"


class FinancialDocumentRetriever:
    """
    Search financial document chunks using TF-IDF similarity.
    """

    def __init__(self) -> None:
        chunks_path = INDEX_DIR / "chunks.json"
        vectorizer_path = INDEX_DIR / "tfidf_vectorizer.joblib"
        matrix_path = INDEX_DIR / "tfidf_matrix.joblib"

        if not chunks_path.exists():
            raise FileNotFoundError(
                "Document index not found. Run src/04_build_document_retriever.py first."
            )

        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.vectorizer = joblib.load(vectorizer_path)
        self.matrix = joblib.load(matrix_path)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()

        top_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in top_indices:
            chunk = self.chunks[index]

            results.append(
                {
                    "score": float(scores[index]),
                    "source": chunk["source"],
                    "chunk_number": chunk["chunk_number"],
                    "text": chunk["text"],
                }
            )

        return results


def print_results(results: list[dict]) -> None:
    for i, result in enumerate(results, start=1):
        print("=" * 90)
        print(f"Result {i}")
        print(f"Source: {result['source']}")
        print(f"Chunk: {result['chunk_number']}")
        print(f"Score: {result['score']:.4f}")
        print("-" * 90)
        print(result["text"][:1000])
        print()


def main() -> None:
    retriever = FinancialDocumentRetriever()

    test_queries = [
        "What does RBC say about liquidity risk?",
        "What are the liquidity coverage ratio and net stable funding ratio?",
        "What does the annual report say about market risk?",
        "What regulatory requirements apply to bank liquidity?",
    ]

    for query in test_queries:
        print("\n\n")
        print("#" * 90)
        print(f"QUERY: {query}")
        print("#" * 90)

        results = retriever.search(query, top_k=3)
        print_results(results)


if __name__ == "__main__":
    main()