from pathlib import Path
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DOC_DIR = BASE_DIR / "data" / "processed_docs"
INDEX_DIR = BASE_DIR / "artifacts" / "document_index"

INDEX_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def load_documents() -> list[dict]:
    """
    Load processed text documents from data/processed_docs.
    """

    documents = []

    for path in sorted(PROCESSED_DOC_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")

        documents.append(
            {
                "source": path.name,
                "text": text,
            }
        )

    if not documents:
        raise FileNotFoundError(
            f"No .txt files found in {PROCESSED_DOC_DIR}. "
            "Run src/03_download_documents.py first."
        )

    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping character chunks.
    """

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def build_chunks(documents: list[dict]) -> list[dict]:
    """
    Create searchable chunks with source metadata.
    """

    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"])

        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_id": len(all_chunks),
                    "source": doc["source"],
                    "chunk_number": i,
                    "text": chunk,
                }
            )

    return all_chunks


def build_tfidf_index(chunks: list[dict]) -> tuple[TfidfVectorizer, object]:
    """
    Fit a TF-IDF vectorizer on document chunks.
    """

    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


def save_index(chunks: list[dict], vectorizer: TfidfVectorizer, matrix: object) -> None:
    """
    Save chunks, vectorizer, and TF-IDF matrix.
    """

    chunks_path = INDEX_DIR / "chunks.json"
    vectorizer_path = INDEX_DIR / "tfidf_vectorizer.joblib"
    matrix_path = INDEX_DIR / "tfidf_matrix.joblib"

    chunks_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(matrix, matrix_path)

    print(f"Saved chunks to: {chunks_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")
    print(f"Saved matrix to: {matrix_path}")


def main() -> None:
    documents = load_documents()
    chunks = build_chunks(documents)
    vectorizer, matrix = build_tfidf_index(chunks)

    save_index(chunks, vectorizer, matrix)

    print("\nDocument retrieval index created.")
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    print("\nExample sources:")
    for doc in documents:
        print(f"- {doc['source']}")


if __name__ == "__main__":
    main()