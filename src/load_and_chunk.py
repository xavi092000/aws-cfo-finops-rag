from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = Path("data/chunks_general_questions.json")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# -----------------------------
# 1. Modèles
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -----------------------------
# 2. Question
# -----------------------------
question = "What is FinOps maturity for CFOs?"

# -----------------------------
# 3. Recherche sémantique
# -----------------------------
question_embedding = embedding_model.encode(question)

chunk_texts = [chunk["text"] for chunk in chunks]
chunk_embeddings = embedding_model.encode(chunk_texts)

semantic_similarities = np.dot(chunk_embeddings, question_embedding)
top_semantic_indices = np.argsort(semantic_similarities)[-5:][::-1]

# -----------------------------
# 4. Recherche lexicale BM25
# -----------------------------
tokenized_chunks = [text.lower().split() for text in chunk_texts]
bm25 = BM25Okapi(tokenized_chunks)

tokenized_query = question.lower().split()
bm25_scores = bm25.get_scores(tokenized_query)
top_bm25_indices = np.argsort(bm25_scores)[::-1][:5]

# -----------------------------
# 5. Fusion des résultats
# -----------------------------
combined_indices = list(set(list(top_semantic_indices) + list(top_bm25_indices)))

# -----------------------------
# 6. Reranking
# -----------------------------
pairs = [(question, chunks[i]["text"]) for i in combined_indices]
rerank_scores = reranker.predict(pairs)

reranked_results = []
for idx, rerank_score in zip(combined_indices, rerank_scores):
    reranked_results.append({
        "index": idx,
        "filename": chunks[idx]["filename"],
        "chunk_id": chunks[idx]["chunk_id"],
        "text": chunks[idx]["text"],
        "semantic_score": float(semantic_similarities[idx]),
        "bm25_score": float(bm25_scores[idx]),
        "rerank_score": float(rerank_score),
    })

reranked_results = sorted(
    reranked_results,
    key=lambda x: x["rerank_score"],
    reverse=True
)

top_final_results = reranked_results[:5]

# -----------------------------
# 7. Contexte pour le LLM
# -----------------------------
selected_chunks = []
for item in top_final_results:
    selected_chunks.append(
        f"Source file: {item['filename']}\n"
        f"Chunk ID: {item['chunk_id']}\n"
        f"Semantic score: {item['semantic_score']:.4f}\n"
        f"BM25 score: {item['bm25_score']:.4f}\n"
        f"Rerank score: {item['rerank_score']:.4f}\n"
        f"Content:\n{item['text']}"
    )

context = "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(selected_chunks)

# -----------------------------
# 8. Prompt LLM
# -----------------------------
system_prompt = """
You are a FinOps assistant answering broad CFO-level questions.
Use only the provided context.
If the context is insufficient, say so clearly.
Be concise, structured, and executive-friendly.
"""

user_prompt = f"""
Question:
{question}

Retrieved context:
{context}

Please answer the question using only the retrieved context.

Return:
1. A short executive answer
2. 3 to 5 key best practices
3. A short note on what a CFO should prioritize first
"""

# -----------------------------
# 9. Appel OpenAI
# -----------------------------
client = OpenAI()

response = client.responses.create(
    model="gpt-5.4",
    input=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
)

# -----------------------------
# 10. Affichage
# -----------------------------
print("\nQUESTION:")
print(question)

print("\nTOP CHUNKS RETRIEVED (HYBRID + RERANK):")
for item in top_final_results:
    print(
        f"- {item['filename']} | "
        f"chunk {item['chunk_id']} | "
        f"semantic={item['semantic_score']:.4f} | "
        f"bm25={item['bm25_score']:.4f} | "
        f"rerank={item['rerank_score']:.4f}"
    )

print("\nFINAL ANSWER:\n")
print(response.output_text)