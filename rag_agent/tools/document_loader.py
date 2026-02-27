
import sys
from pathlib import Path

# Handle imports whether run directly or as module
try:
    # Try relative import first (when imported as module)
    from config import DATA_RAW
except ImportError:
    # Fall back to absolute import (when run directly)
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import DATA_RAW

from pypdf import PdfReader

file_path = DATA_RAW / "Beta_vae.pdf"


class DocumentLoader:

    @staticmethod
    def load_doc(file_path):
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() + "\n"

        return text


# print(Document_Loader(file_path))

if __name__ == "__main__":

    # Test loading a single document
    loader = DocumentLoader()

    # Example: Load a PDF
    pdf_path = DATA_RAW / "Beta_vae.pdf"

    if pdf_path.exists():
        text = loader.load_doc(pdf_path)
        print(f"Loaded {len(text)} characters from {pdf_path.name}")
        print(f"Preview: {text[:200]}...")
