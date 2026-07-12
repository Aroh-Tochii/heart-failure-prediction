"""
Heart Failure RAG Knowledge Base Builder
==========================================
Loads 4 clinical knowledge documents into ChromaDB
using Gemini embeddings for semantic search.

Run this ONCE to build the knowledge base.
After that the agent uses it automatically.

Author: Tochukwu Aroh
"""

import os
import sys
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
KNOWLEDGE_DIR  = Path(__file__).parent / "knowledge"
CHROMA_DIR     = Path(__file__).parent / "chroma_db"

# ── Gemini embedding function ─────────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)

def embed_text(text: str) -> list:
    """Convert text to embedding vector using Gemini."""
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values


# ── Split document into chunks ────────────────────────────────────────────────
def chunk_document(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split a document into overlapping chunks.
    Smaller chunks = more precise retrieval.
    Overlap = context is not lost at boundaries.
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ── Build the knowledge base ──────────────────────────────────────────────────
def build_knowledge_base():
    print("\n" + "="*60)
    print("  HEART FAILURE RAG KNOWLEDGE BASE BUILDER")
    print("="*60)

    # Check knowledge directory exists
    if not KNOWLEDGE_DIR.exists():
        print(f"ERROR: Knowledge directory not found: {KNOWLEDGE_DIR}")
        print("Please create the knowledge/ folder with the 4 clinical documents.")
        sys.exit(1)

    # List available documents
    docs = list(KNOWLEDGE_DIR.glob("*.txt"))
    if not docs:
        print("ERROR: No .txt files found in knowledge/ directory")
        sys.exit(1)

    print(f"\nFound {len(docs)} knowledge documents:")
    for doc in docs:
        print(f"  • {doc.name} ({doc.stat().st_size // 1024} KB)")

    # Initialize ChromaDB
    print(f"\nInitializing ChromaDB at: {CHROMA_DIR}")
    CHROMA_DIR.mkdir(exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if rebuilding
    try:
        chroma_client.delete_collection("clinical_knowledge")
        print("Deleted existing collection — rebuilding fresh")
    except Exception:
        pass

    # Create collection with custom embedding function
    collection = chroma_client.create_collection(
        name="clinical_knowledge",
        metadata={"description": "Heart failure clinical guidelines and protocols"}
    )

    # Process each document
    total_chunks = 0
    all_ids       = []
    all_docs      = []
    all_embeddings = []
    all_metadata  = []

    for doc_path in docs:
        print(f"\nProcessing: {doc_path.name}")
        text   = doc_path.read_text(encoding="utf-8")
        chunks = chunk_document(text, chunk_size=200, overlap=30)
        print(f"  Split into {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 20:  # skip very short chunks
                continue

            chunk_id = f"{doc_path.stem}_{i}"

            print(f"  Embedding chunk {i+1}/{len(chunks)}...", end="\r")
            try:
                embedding = embed_text(chunk)
                all_ids.append(chunk_id)
                all_docs.append(chunk)
                all_embeddings.append(embedding)
                all_metadata.append({
                    "source":   doc_path.name,
                    "doc_name": doc_path.stem,
                    "chunk_id": i
                })
                total_chunks += 1
            except Exception as e:
                print(f"\n  Warning: Could not embed chunk {i}: {e}")
                continue

        print(f"  Done — {len(chunks)} chunks from {doc_path.name}")

    # Add all chunks to ChromaDB in one batch
    print(f"\nStoring {total_chunks} chunks in ChromaDB...")
    collection.add(
        ids=all_ids,
        documents=all_docs,
        embeddings=all_embeddings,
        metadatas=all_metadata
    )

    print(f"\n{'='*60}")
    print(f"  Knowledge base built successfully!")
    print(f"  Total chunks stored: {total_chunks}")
    print(f"  Location: {CHROMA_DIR}")
    print(f"{'='*60}\n")

    # Test retrieval
    print("Testing retrieval...")
    test_query   = "What should I do if ejection fraction is below 30%?"
    test_embed   = embed_text(test_query)
    test_results = collection.query(
        query_embeddings=[test_embed],
        n_results=2
    )

    print(f"Query: '{test_query}'")
    print(f"Top result from: {test_results['metadatas'][0][0]['source']}")
    print(f"Content preview: {test_results['documents'][0][0][:200]}...")
    print("\nRAG knowledge base is ready for the agent.")


if __name__ == "__main__":
    build_knowledge_base()
