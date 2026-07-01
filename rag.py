import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

from config import (
    GOOGLE_API_KEY,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    GEMINI_MODEL
)

# -----------------------------
# Load Embedding Model
# -----------------------------
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# -----------------------------
# Load ChromaDB
# -----------------------------
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

collection = client.get_collection("documents")

# -----------------------------
# Configure Gemini
# -----------------------------
gemini = genai.Client(api_key=GOOGLE_API_KEY)


def ask_question(question):
    """
    Ask a question to the RAG system.
    """

    # Create embedding for user question
    question_embedding = embedding_model.encode(question).tolist()

    # Retrieve top 3 relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    chunks = results["documents"][0]
    metadata = results["metadatas"][0]

    # Build context
    context = ""
    citations = set()

    for chunk, meta in zip(chunks, metadata):

        context += (
            f"Source: {meta['source']} "
            f"(Page {meta['page']})\n"
        )

        context += chunk + "\n\n"

        citations.add(
            f"{meta['source']} (Page {meta['page']})"
        )

    prompt = f"""
You are an AI Document Question Answering Assistant.

Answer ONLY from the provided context.

Rules:

1. Do not use outside knowledge.

2. If the answer is not found in the context, reply exactly:

I could not find this information in the documents.

3. Keep the answer clear and concise.

Context:

{context}

Question:

{question}

Answer:
"""

    # Gemini Response
    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return {
        "answer": response.text.strip(),
        "citations": sorted(list(citations)),
        "retrieved_chunks": chunks
    }


if __name__ == "__main__":

    print("=" * 60)
    print("Document QA using RAG")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:

        question = input("\nQuestion: ")

        if question.lower() == "exit":
            break

        result = ask_question(question)

        print("\n" + "=" * 60)
        print("Answer")
        print("=" * 60)

        print(result["answer"])

        print("\n" + "=" * 60)
        print("Sources")
        print("=" * 60)

        for citation in result["citations"]:
            print("-", citation)

        print("\n" + "=" * 60)
        print("Retrieved Chunks")
        print("=" * 60)

        for i, chunk in enumerate(result["retrieved_chunks"], start=1):
            print(f"\nChunk {i}:\n")
            print(chunk[:300])
            print("...")