from pathlib import Path
import joblib

from importlib.util import spec_from_file_location, module_from_spec


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "risk_classifier" / "risk_text_classifier.joblib"

# Import the document retriever from src/05_search_documents.py
RETRIEVER_PATH = BASE_DIR / "src" / "05_search_documents.py"
spec = spec_from_file_location("search_documents", RETRIEVER_PATH)
search_documents = module_from_spec(spec)
spec.loader.exec_module(search_documents)

FinancialDocumentRetriever = search_documents.FinancialDocumentRetriever


def load_classifier():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Risk classifier not found. Run src/06_train_risk_classifier.py first."
        )

    return joblib.load(MODEL_PATH)


def classify_retrieved_chunks(query: str, top_k: int = 5) -> None:
    retriever = FinancialDocumentRetriever()
    classifier = load_classifier()

    results = retriever.search(query, top_k=top_k)

    print("=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    for i, result in enumerate(results, start=1):
        text = result["text"]
        predicted_label = classifier.predict([text])[0]

        print(f"\nResult {i}")
        print("-" * 100)
        print(f"Source: {result['source']}")
        print(f"Chunk: {result['chunk_number']}")
        print(f"Retrieval score: {result['score']:.4f}")
        print(f"Predicted risk category: {predicted_label}")
        print("\nText excerpt:")
        print(text[:900])


def main() -> None:
    test_queries = [
        "What does RBC say about liquidity risk and funding?",
        "What does the annual report say about market risk?",
        "What does OSFI say about LCR and NSFR requirements?",
        "What does RBC say about credit risk and counterparty exposure?",
    ]

    for query in test_queries:
        classify_retrieved_chunks(query, top_k=3)
        print("\n\n")


if __name__ == "__main__":
    main()