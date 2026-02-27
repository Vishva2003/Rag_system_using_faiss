from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import sys
from pathlib import Path

try:
    from config import EMBEDDING_MODEL, VECTOR_DB_PATH, TOP_K_RESULTS, DATA_RAW
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import EMBEDDING_MODEL, VECTOR_DB_PATH, TOP_K_RESULTS, DATA_RAW


class Retriever:

    def __init__(self, embedder=None, top_k=None):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.top_k = top_k if top_k is not None else TOP_K_RESULTS
        self.embedder = embedder
        if embedder is None:
            raise ValueError("Retriever requires a valid Embedder with index")

    def set_embedder(self, embedder):
        self.embedder = embedder

    def _normalize_embeddings(self, embedding):
        norm = np.linalg.norm(embedding)
        return embedding / np.maximum(norm, 1e-10)

    def retrieve(self, query):

        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        query_embedding = self._normalize_embeddings(query_embedding)

        distances, indices = self.embedder.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            self.top_k
        )
        print(f'Results shapes: Distance:{distances.shape}, Indices:{indices.shape}')

        distances = distances.flatten()
        indices = indices.flatten()

        valid_indices = indices[indices != -1]
        valid_distances = distances[indices != -1]

        retrieved_docs = [self.embedder.chunks[i] for i in valid_indices]

        metadatas = [{"index": int(idx)} for idx in valid_indices]
        ids = [f"chunk_{idx}" for idx in valid_indices]

        return {
            'documents': retrieved_docs,
            'metadatas': metadatas,
            'distances': valid_distances.tolist(),
            'ids': ids
        }

    def format_context(self, retrieved_data):
        formatted_context = []

        for i, (doc, dist) in enumerate(zip(retrieved_data['documents'], retrieved_data['distances'])):
            formatted_context.append(f"Document {i+1} (Distance: {dist:.2f}):\n{doc}\n")
        return '\n'.join(formatted_context)


if __name__ == "__main__":
    from document_loader import DocumentLoader
    from chunker import Chunker
    from embedder import Embedder

    # Test chunking
    loader = DocumentLoader()
    chunker = Chunker(chunk_size=200, overlap_size=50)
    embedder = Embedder()
    retriever = Retriever()

    # Load a document
    pdf_path = DATA_RAW / "Beta_vae.pdf"

    if pdf_path.exists():

        text = loader.load_doc(str(pdf_path))

        chunks = chunker.chunk_text(text)

        embedder.add_documents(chunks)

        query = 'What is beta VAE?'

        print(f"loaded document with {len(text)} characters")
        print(f"Total chunks: {len(chunks)}")
        print(f"First chunk ({len(chunks[0])} chars):")

        retrieved_data = retriever.retrieve(query)
        context = retriever.format_context(retrieved_data)
        print(f"\nRetrieved Context:\n{context}")
