import time
import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DB_DIR, EMBEDDING_MODEL

# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer(EMBEDDING_MODEL)

# -----------------------------
# Load ChromaDB
# -----------------------------
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_collection("documents")

# -----------------------------
# Test Questions
# -----------------------------
# expected_page = page where answer exists

test_data = [
    ("What law governs the eBay User Agreement?", 13),
    ("What is Informal Dispute Resolution?", 14),
    ("Who pays arbitration fees?", 15),
    ("What is Batch Arbitration?", 17),
    ("Where should an Opt-Out Notice be mailed?", 19),
]

k = 3

hits = 0
mrr = 0
context_precision = 0
total_latency = 0

print("=" * 70)
print("Retrieval Evaluation")
print("=" * 70)

for question, expected_page in test_data:

    start = time.time()

    embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    latency = time.time() - start
    total_latency += latency

    pages = [m["page"] for m in results["metadatas"][0]]

    print(f"\nQuestion: {question}")
    print("Retrieved Pages:", pages)

    # ---------------------
    # Recall / Hit Rate
    # ---------------------

    if expected_page in pages:
        hits += 1

    # ---------------------
    # MRR
    # ---------------------

    reciprocal = 0

    for rank, page in enumerate(pages, start=1):

        if page == expected_page:
            reciprocal = 1 / rank
            break

    mrr += reciprocal

    # ---------------------
    # Context Precision
    # ---------------------

    relevant = pages.count(expected_page)

    context_precision += relevant / k

print("\n" + "=" * 70)

num = len(test_data)

print(f"Questions              : {num}")
print(f"k used                 : {k}")
print(f"Recall@{k}             : {hits/num:.2f}")
print(f"MRR                    : {mrr/num:.2f}")
print(f"Context Precision      : {context_precision/num:.2f}")
print(f"Average Latency (sec)  : {total_latency/num:.3f}")

print("=" * 70)