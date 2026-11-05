# 🚀 Clean RAG with Gemini API

A clean, production-style Retrieval-Augmented Generation (RAG) script in Python.

## How it works:
1. **Document Loading**: Supports `.pdf`, `.txt`, and `.md` files.
2. **Chunking**: Uses `RecursiveCharacterTextSplitter` to create clean chunks.
3. **Vector Database**: Indexes chunks in ChromaDB for fast similarity retrieval.
4. **LLM Generation**: Injects the retrieved chunks into Gemini API (`gemini-2.5-flash`) to generate grounded, hallucination-free answers.

## Usage:

### 1. Set your Gemini API Key (or paste it when prompted)
```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
```
*(Get a free key from https://aistudio.google.com/)*

### 2. Run with default text document:
```powershell
python rag.py
```

### 3. Run with ANY of your own PDF or TXT files:
```powershell
python rag.py "path\to\your\document.pdf"
```
