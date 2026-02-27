import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()


@st.cache_resource()
def get_rag_tools():
    from rag_agent.tools.document_loader import DocumentLoader
    from rag_agent.tools.embedder import Embedder
    from rag_agent.tools.generator import Generator

    loader = DocumentLoader()
    embedder = Embedder()
    generator = Generator()

    return loader, embedder, generator


def create_chunker(chunk_size, overlap_size):
    from rag_agent.tools.chunker import Chunker
    return Chunker(chunk_size=chunk_size, overlap_size=overlap_size)


def create_retriever(embedder, top_k):
    from rag_agent.tools.retriever import Retriever
    return Retriever(embedder=embedder, top_k=top_k)


st.set_page_config(page_title="Rag system with Faiss Index", layout="wide")
st.title("RAG System with Faiss Index")

with st.sidebar:
    st.header("Document Loader")
    uploadfile = st.file_uploader("Upload a PDF document", type=["pdf"])
    chunks = st.slider("Chunk size (characters)", 100, 1000, 200, step=50)
    chunks_overlap = st.slider("Chunk overlap (characters)", 0, 100, 20, step=10)
    top_K_results = st.slider("Top K results to retrieve", 1, 20, 5)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if uploadfile is not None:
    temp_file_path = f'temp_{uploadfile.name}'
    with open(temp_file_path, "wb") as f:
        f.write(uploadfile.getbuffer())

    with st.status("Processing document...", expanded=True) as status:
        try:
            loader, embedder, generator = get_rag_tools()
            chunker = create_chunker(chunks, chunks_overlap)

            text = loader.load_doc(temp_file_path)
            chunk_list = chunker.chunk_text(text)
            embedder = embedder.add_documents(chunk_list)

            status.update(label="Document processed successfully!", state='complete', expanded=False)
            st.success('Ready to answer questions about the document!')

            if chunks:
                with st.expander("Document Chunks"):
                    for i, chunk in enumerate(chunk_list[:3]):
                        st.markdown(f"**Chunk {i+1}:** {chunk[:200]}...")

        except Exception as e:
            status.update(
                label=f"Error processing document: {str(e)}",
                state="error",
                expanded=True
            )
            st.error("Processing failed. Please check the file and try again.")

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

if prompt := st.chat_input("Ask a question about the document.."):

    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        with st.spinner("generating answer..."):
            try:
                _, embedder, generator = get_rag_tools()
                retriever = create_retriever(embedder, top_K_results)
                print(f"Retrieving with top_k={top_K_results}...")
                retrieved_data = retriever.retrieve(prompt)

                if not retrieved_data["documents"]:
                    answer = "No relevant information found in the document."
                else:
                    context = retriever.format_context(retrieved_data)
                    answer = generator.generate(prompt, context)

                    with st.expander("Retrieved context"):
                        for i, chunk in enumerate(retrieved_data["documents"]):
                            st.markdown(f"**Doc {i+1}:** {chunk[:200]}...")
                            print(f"Retrieved Doc {i+1}: {chunk[:10]}...")

                st.markdown(answer)

            except Exception as e:
                st.error(f"Error initializing RAG tools: {e}")
                st.stop()

    st.session_state.messages.append({'role': 'assistant', 'content': answer})
