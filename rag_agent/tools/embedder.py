
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import sys
from pathlib import Path


try:
    from config import EMBEDDING_MODEL, VECTOR_DB_PATH, DATA_RAW
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import EMBEDDING_MODEL, VECTOR_DB_PATH, DATA_RAW


class Embedder:

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.dimension = None
        self.chunks = []

        self.index_path = VECTOR_DB_PATH / "index.faiss"
        self.chunks_path = VECTOR_DB_PATH / "chunks.plk"

        self.load_exixting_index()

    def load_exixting_index(self):
        if self.index_path.exists() and self.chunks_path.exists():
            print(f'Loading existing index from {str(self.index_path.name)}')
            self.index = faiss.read_index(str(self.index_path))
            self.dimension = self.index.d

            with open(self.chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f'Loaded {len(self.chunks)} chunks from {str(self.chunks_path.name)}')

    def create_index(self, dim):
        self.dimension = dim
        self.index = faiss.IndexFlatL2(dim)
        print('Faiss index are created')

    def _normalize_embeddings(self, embedding):
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding / np.maximum(norm, 1e-10)

    def embed_texts(self, text):
        embedded_text = self.model.encode(text)
        return self._normalize_embeddings(embedded_text)

    def add_documents(self, text):
        if not text:
            print("No text provided for embedding.")
            return

        embeddings = self.embed_texts(text)

        if self.index is None:
            self.create_index(embeddings.shape[1])
            print(f"Created new index with dimension: {embeddings.shape[1]}")

        if self.dimension != embeddings.shape[1]:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}")

        self.index.add(embeddings.astype(np.float32))

        self.chunks.extend(text)
        print(f"Added {len(text)} documents → total chunks now: {self.index.ntotal}")
        self.save_index()

    def save_index(self):
        if self.index is not None and self.dimension is not None:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)


if __name__ == "__main__":
    from document_loader import DocumentLoader
    from chunker import Chunker

    # Test chunking
    loader = DocumentLoader()
    chunker = Chunker(chunk_size=200, overlap_size=50)
    embedder = Embedder()

    # Load a document
    pdf_path = DATA_RAW / "Beta_vae.pdf"
    if pdf_path.exists():
        text = loader.load_doc(str(pdf_path))

        # Chunk it
        chunks = chunker.chunk_text(text)

        print(f"loaded document with {len(text)} characters")
        print(f"Total chunks: {len(chunks)}")
        print(f"First chunk ({len(chunks[0])} chars):")
        embedder.add_documents(chunks)
