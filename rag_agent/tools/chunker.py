import sys
from pathlib import Path
try:
    from config import CHUNK_SIZE, CHUNK_OVERLAP
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import CHUNK_SIZE, CHUNK_OVERLAP


class Chunker:
    def __init__(self, chunk_size=CHUNK_SIZE, overlap_size=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def chunk_text(self, text):
        chunks = []
        start = 0
        end = len(text)
        while start < end:
            end_inx = start + self.chunk_size
            if end_inx > end:
                end_inx = end
            chunks.append(text[start:end_inx])
            start += self.chunk_size - self.overlap_size

        return chunks


# print(chunker(text))
if __name__ == "__main__":
    from document_loader import DocumentLoader
    from config import DATA_RAW
    # Test chunking
    loader = DocumentLoader()
    chunker = Chunker(chunk_size=200, overlap_size=50)

    # Load a document
    pdf_path = DATA_RAW / "Beta_vae.pdf"
    if pdf_path.exists():
        text = loader.load_doc(str(pdf_path))

        # Chunk it
        chunks = chunker.chunk_text(text)

        print(f"Total chunks: {len(chunks)}")
        print(f"\nFirst chunk ({len(chunks[0])} chars):")
        print(chunks[0])
