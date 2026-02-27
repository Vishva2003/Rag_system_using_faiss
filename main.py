from rag_agent.tools.document_loader import DocumentLoader
from rag_agent.tools.chunker import Chunker
from rag_agent.tools.embedder import Embedder
from rag_agent.tools.retriever import Retriever
from rag_agent.tools.generator import Generator
from config import DATA_RAW, CHUNK_OVERLAP, CHUNK_SIZE

# use your document path here ('Beta_vae.pdf' is just an example, you can replace it with any document you want to test with)
file_path = DATA_RAW / "Beta_vae.pdf"

if __name__ == "__main__":

    loader = DocumentLoader()
    chunker = Chunker(chunk_size=CHUNK_SIZE, overlap_size=CHUNK_OVERLAP)
    embedder = Embedder()
    retriever = Retriever(embedder=embedder)
    generator = Generator()

    collection_name = 'beta_vae_collection' # you can change this to any name you like related to the document

    if file_path.exists():
        text = loader.load_doc(str(file_path))
        chunks = chunker.chunk_text(text)
        embedder.add_documents(chunks)
        query = input("Enter your question: ")
        retrieved_data = retriever.retrieve(query)
        context = retriever.format_context(retrieved_data)
        print(f"\nRetrieved Context:\n{context}")
        answer = generator.generate(query, context)
        print(f"\nGenerated Answer:\n{answer}")
