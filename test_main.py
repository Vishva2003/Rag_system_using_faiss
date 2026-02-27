from .tools.document_loader import DocumentLoader
from .tools.chunker import Chunker


loader = DocumentLoader()
chunker = Chunker()

file = "./basic_rag_system/data/raw/Beta_vae.pdf"

text = loader.load_doc(file)
chunks = chunker.chunk_text(text)

print(chunks[0])
