import os

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loader import load_documents
from config import (
    DOCUMENT_FOLDER,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

# Create Chroma client
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# Delete old collection if it exists
try:
    client.delete_collection("documents")
except:
    pass

collection = client.create_collection("documents")


def ingest():

    print("Loading documents...")

    documents = load_documents(DOCUMENT_FOLDER)

    print(f"Loaded {len(documents)} document(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunk_count = 0

    for doc in documents:

        chunks = splitter.split_text(doc["text"])

        for chunk in chunks:

            embedding = model.encode(chunk).tolist()

            collection.add(
                ids=[str(chunk_count)],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "source": doc["source"],
                    "page": doc["page"]
                }]
            )

            chunk_count += 1

    print(f"\nStored {chunk_count} chunks in ChromaDB")


if __name__ == "__main__":
    ingest()