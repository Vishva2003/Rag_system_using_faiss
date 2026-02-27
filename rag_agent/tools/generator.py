from google import genai
import sys
from pathlib import Path

try:
    from config import GEMINI_API_KEY, GEMINI_MODEL, DATA_RAW
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import GEMINI_API_KEY, GEMINI_MODEL, DATA_RAW


class Generator:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = GEMINI_MODEL

    def generate(self, query, context):
        prompt = f"""
        You are an expert AI assistant answering questions using retrieved documents.

        You MUST follow these rules:
        1. Use ONLY the information in the provided context.
        2. Do NOT use outside knowledge.
        3. If the answer is not fully supported by the context, say:
        "I don't have enough information to answer that question."
        4. If multiple documents disagree, report all viewpoints.

        ----------------------
        CONTEXT:
        {context}
        ----------------------

        QUESTION:
        {query}

        INSTRUCTIONS:
        - Read the context carefully.
        - Identify the most relevant parts.
        - Synthesize a clear, factual answer.
        - Quote or paraphrase directly from the context.
        - Do not guess or assume anything.

        ANSWER:
        """

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text


if __name__ == "__main__":
    from document_loader import DocumentLoader
    from chunker import Chunker
    from embedder import Embedder
    from retriever import Retriever

    # Test chunking
    loader = DocumentLoader()
    chunker = Chunker(chunk_size=200, overlap_size=50)
    embedder = Embedder()
    retriever = Retriever(embedder=embedder)
    generator = Generator()

    # Load a document
    pdf_path = DATA_RAW / "Beta_vae.pdf"

    if pdf_path.exists():

        text = loader.load_doc(str(pdf_path))

        chunks = chunker.chunk_text(text)

        embedder.add_documents(chunks)

        query = 'what is the topic of the document?'

        print(f"loaded document with {len(text)} characters")
        print(f"Total chunks: {len(chunks)}")
        print(f"First chunk ({len(chunks[0])} chars):")
        
        retrieved_data = retriever.retrieve(query)
        context = retriever.format_context(retrieved_data)
        print(f"\nRetrieved Context:\n{context}")
        answer = generator.generate(query, context)
        print("Generated Answer:")
        print(answer)
