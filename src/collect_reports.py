from pathlib import Path
import requests
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DOC_DIR = BASE_DIR / "data" / "raw_docs"
PROCESSED_DOC_DIR = BASE_DIR / "data" / "processed_docs"

RAW_DOC_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DOC_DIR.mkdir(parents=True, exist_ok=True)


DOCUMENTS = {
    "rbc_2024_annual_report": {
        "url": "https://www.rbc.com/investor-relations/_assets-custom/pdf/ar_2024_e.pdf",
        "type": "pdf",
    },
    "rbc_2024_pillar_3_report": {
        "url": "https://www.rbc.com/investor-relations/_assets-custom/pdf/2024q4pillar3.pdf",
        "type": "pdf",
    },
    "osfi_liquidity_adequacy_requirements_2026": {
        "url": "https://www.osfi-bsif.gc.ca/en/print/pdf/node/2660",
        "type": "pdf",
    },
}


def download_file(url: str, output_path: Path) -> None:
    """
    Download a file from a public URL.
    """

    print(f"Downloading: {url}")

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    output_path.write_bytes(response.content)

    print(f"Saved to: {output_path}")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF using pypdf.
    """

    reader = PdfReader(str(pdf_path))
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"\n\n--- Page {i + 1} ---\n{text}")

    return "\n".join(pages)


def clean_text(text: str) -> str:
    """
    Light text cleaning for retrieval.
    """

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    cleaned = "\n".join(lines)

    return cleaned


def main() -> None:
    for doc_name, meta in DOCUMENTS.items():
        pdf_path = RAW_DOC_DIR / f"{doc_name}.pdf"
        txt_path = PROCESSED_DOC_DIR / f"{doc_name}.txt"

        download_file(meta["url"], pdf_path)

        print(f"Extracting text from: {pdf_path}")
        raw_text = extract_text_from_pdf(pdf_path)
        cleaned_text = clean_text(raw_text)

        txt_path.write_text(cleaned_text, encoding="utf-8")

        print(f"Saved extracted text to: {txt_path}")
        print(f"Extracted characters: {len(cleaned_text):,}")
        print("-" * 80)


if __name__ == "__main__":
    main()