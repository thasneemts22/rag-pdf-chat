import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Load .env file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Support both local .env and Streamlit Cloud Secrets
API_KEY = os.environ.get("GEMINI_API_KEY") or (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else "")

from google import genai
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------------------------------------------------------
# Streamlit UI Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PDF Chat AI (RAG)", page_icon="📄", layout="wide")

st.title("📄 Chat with any PDF / Document")
st.caption("Upload a document, and Google Gemini will answer questions strictly based on its contents.")

# Initialize Gemini Client
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=API_KEY)

ai_client = get_gemini_client()

# In-memory ChromaDB
@st.cache_resource
def get_db():
    client = chromadb.Client()
    return client.get_or_create_collection(name="user_uploaded_docs")

collection = get_db()

# -----------------------------------------------------------------------------
# Sidebar: Upload Document
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF, TXT, or MD file", type=["pdf", "txt", "md"])
    
    if uploaded_file and st.button("⚡ Process & Index Document", type="primary", use_container_width=True):
        with st.spinner("Reading and embedding document into Vector DB..."):
            # 1. Extract text
            if uploaded_file.name.lower().endswith(".pdf"):
                reader = PdfReader(uploaded_file)
                raw_text = "\n".join([page.extract_text() or "" for page in reader.pages])
            else:
                raw_text = uploaded_file.read().decode("utf-8", errors="ignore")

            # 2. Chunking
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            chunks = splitter.split_text(raw_text)

            # 3. Store in Vector DB
            # Reset existing collection
            client = chromadb.Client()
            try:
                client.delete_collection("user_uploaded_docs")
            except Exception:
                pass
            collection = client.create_collection(name="user_uploaded_docs")
            st.cache_resource.clear()

            collection.add(
                documents=chunks,
                ids=[f"chunk_{i}" for i in range(len(chunks))]
            )
            st.session_state["doc_name"] = uploaded_file.name
            st.session_state["messages"] = [
                {"role": "assistant", "content": f"✅ **{uploaded_file.name}** processed successfully ({len(chunks)} chunks). Ask me anything about it!"}
            ]
            st.success(f"Indexed {len(chunks)} chunks!")

    if "doc_name" in st.session_state:
        st.info(f"Active Document: **{st.session_state['doc_name']}**")

# -----------------------------------------------------------------------------
# Main Chat Area
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hello! Upload a PDF on the left sidebar to start chatting."}
    ]

# Display conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "context" in msg:
            with st.expander("🔍 View Retrieved Document Chunks (RAG Context)"):
                for idx, c in enumerate(msg["context"]):
                    st.info(f"**Chunk {idx+1}:**\n{c}")

# User input
user_query = st.chat_input("Ask a question about your document...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Check if document has been indexed
    fresh_collection = get_db()
    if fresh_collection.count() == 0:
        ans = "⚠️ Please upload and click **'Process & Index Document'** in the left sidebar first!"
        with st.chat_message("assistant"):
            st.warning(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
    else:
        with st.spinner("Searching document & generating answer with Gemini..."):
            # 1. RETRIEVE
            results = fresh_collection.query(query_texts=[user_query], n_results=3)
            retrieved_chunks = results["documents"][0]

            # 2. AUGMENT
            context_str = "\n\n".join(retrieved_chunks)
            prompt = f"""You are a helpful and accurate assistant.
Answer the user's question strictly based ONLY on the context below.
If the answer is not in the context, say "This information is not mentioned in the uploaded document."

<Context>
{context_str}
</Context>

Question: {user_query}

Answer:"""

            # 3. GENERATE (Gemini API)
            try:
                response = ai_client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )
                answer_text = response.text.strip()
            except Exception as e:
                answer_text = f"Error generating answer: {e}"

            with st.chat_message("assistant"):
                st.markdown(answer_text)
                with st.expander("🔍 View Retrieved Document Chunks (RAG Context)"):
                    for idx, c in enumerate(retrieved_chunks):
                        st.info(f"**Chunk {idx+1}:**\n{c}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "context": retrieved_chunks
            })
