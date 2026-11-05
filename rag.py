import os
import sys
from dotenv import load_dotenv

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Load .env file automatically
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ==============================================================================
# Production-Style RAG Pipeline with Gemini Cloud API
# ==============================================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("=" * 60)
    print("🔑 GEMINI API KEY SETUP")
    print("=" * 60)
    API_KEY = input("Enter your Gemini API Key: ").strip()
    if not API_KEY:
        print("API Key required. Exiting.")
        sys.exit(1)

from google import genai
import chromadb
from pypdf import PdfReader

# Initialize Gemini Client
llm_client = genai.Client(api_key=API_KEY)


def load_document(file_path: str) -> str:
    """Reads text from either a .pdf or .txt/.md file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages_text)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


def split_text_into_chunks(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Splits a document into smaller chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


def build_vector_db(chunks: list[str]):
    """Stores text chunks into an in-memory ChromaDB vector store."""
    client = chromadb.Client()
    try:
        client.delete_collection("rag_collection")
    except Exception:
        pass

    collection = client.create_collection(name="rag_collection")
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection


def generate_gemini_answer(prompt: str) -> str:
    """Calls Gemini API with fast fallback across models."""
    candidate_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemma-4-26b-a4b-it"]
    last_err = None
    for model_name in candidate_models:
        try:
            response = llm_client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All Gemini models failed: {last_err}")


def ask_rag(collection, question: str, top_k: int = 3):
    """Retrieves relevant chunks and generates a grounded response using Gemini."""
    print("\n" + "=" * 60)
    print(f"❓ User Question: {question}")
    print("-" * 60)

    # 1. RETRIEVAL (Vector Similarity Search)
    results = collection.query(query_texts=[question], n_results=top_k)
    retrieved_chunks = results["documents"][0]

    print(f"🔍 [Retrieved {len(retrieved_chunks)} Relevant Chunks from Vector DB]:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  [{i+1}] {chunk.strip()}")
    print("-" * 60)

    # 2. AUGMENTATION (Context + Question)
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""You are a helpful and precise AI assistant.
Answer the question strictly based ONLY on the context provided below.
If the context does not contain the answer, say "I cannot find this information in the document."

<Context>
{context}
</Context>

Question: {question}

Answer:"""

    # 3. GENERATION
    print("🤖 Calling Google Gemini API (gemini-3.5-flash-lite)...")
    try:
        ans = generate_gemini_answer(prompt)
        print("\n✨ Grounded Answer from Gemini:")
        print(ans.strip())
    except Exception as e:
        print(f"Error calling Gemini: {e}")


if __name__ == "__main__":
    doc_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "knowledge.txt")

    print(f"📄 Loading document: {doc_path}")
    raw_text = load_document(doc_path)
    
    chunks = split_text_into_chunks(raw_text)
    print(f"✂️  Split document into {len(chunks)} chunks.")

    print("⚡ Storing chunks in Vector Database...")
    collection = build_vector_db(chunks)
    print("✅ Ready! Connected to Google Gemini API.\n")

    # Demo questions to demonstrate RAG immediately
    print("--- Running Test Queries ---")
    ask_rag(collection, "Who is the head architect for Project Phoenix?")
    ask_rag(collection, "What is the budget and beta deadline?")

    print("\n" + "=" * 60)
    print("💬 You can now ask any question! (Type 'exit' to quit)")
    print("=" * 60)

    while True:
        try:
            q = input("\nEnter your question: ").strip()
            if not q or q.lower() in ["exit", "quit", "q"]:
                print("👋 Bye!")
                break
            ask_rag(collection, q)
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break
