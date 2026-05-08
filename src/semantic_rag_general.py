from pathlib import Path
import json
from datetime import datetime
import os
import re
import time
import numpy as np
import boto3
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = Path("data/chunks_general_questions.json")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

AWS_PROFILE = "terraform-runner"
AWS_REGION = "us-east-1"
VOICE_ID = "Joanna"
ENGINE = "neural"

SYSTEM_PROMPT = """
You are a FinOps AI assistant answering CFO-level questions using Retrieval-Augmented Generation (RAG).

STRICT RULES:
1. Only use the provided retrieved context
2. Do not use outside knowledge
3. Do not invent facts, numbers, explanations, or examples
4. Do not guess
5. Do not infer beyond what is explicitly supported by the retrieved context
6. Do not mention specific cloud services, platforms, tools, products, numbers, percentages, or examples unless they appear explicitly in the retrieved context
7. If the answer is not clearly supported by the retrieved context, respond exactly with:
   "I don't have enough information in the retrieved documents."
8. If rule 7 applies, STOP immediately and do not add any extra explanation, bullets, recommendations, or priorities
9. If only part of the question is supported, answer only the supported part and clearly state what is missing
10. Stay grounded in the retrieved documents only

ANSWER STYLE:
- Be concise
- Be structured
- Be executive-friendly
- Focus on factual FinOps guidance
- Do not add speculation

OUTPUT FORMAT:
1. Short executive answer
2. 3 to 5 key best practices
3. What a CFO should prioritize first
"""


# =========================
# LOAD / RETRIEVE
# =========================
def load_chunks(input_file: Path):
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_bm25(chunks):
    chunk_texts = [chunk["text"] for chunk in chunks]
    tokenized_chunks = [text.lower().split() for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25, chunk_texts


def semantic_search(question, chunks, embedding_model, top_k=5):
    question_embedding = embedding_model.encode(question)
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embedding_model.encode(chunk_texts)

    semantic_similarities = np.dot(chunk_embeddings, question_embedding)
    top_indices = np.argsort(semantic_similarities)[-top_k:][::-1]

    return semantic_similarities, top_indices


def lexical_search(question, bm25, top_k=5):
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(bm25_scores)[::-1][:top_k]

    return bm25_scores, top_indices


def rerank_results(question, candidate_indices, chunks, semantic_similarities, bm25_scores, reranker):
    pairs = [(question, chunks[i]["text"]) for i in candidate_indices]
    rerank_scores = reranker.predict(pairs)

    reranked_results = []
    for idx, rerank_score in zip(candidate_indices, rerank_scores):
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

    return reranked_results


def build_context(top_results):
    selected_chunks = []

    for item in top_results:
        selected_chunks.append(
            f"Source file: {item['filename']}\n"
            f"Chunk ID: {item['chunk_id']}\n"
            f"Semantic score: {item['semantic_score']:.4f}\n"
            f"BM25 score: {item['bm25_score']:.4f}\n"
            f"Rerank score: {item['rerank_score']:.4f}\n"
            f"Content:\n{item['text']}"
        )

    return "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(selected_chunks)


def get_confidence_info(top_results):
    if not top_results:
        return {
            "label": "Low",
            "reason": "No retrieved results",
            "best_rerank_score": None,
            "best_semantic_score": None,
            "best_bm25_score": None,
        }

    best_result = top_results[0]
    best_rerank_score = best_result["rerank_score"]
    best_semantic_score = best_result["semantic_score"]
    best_bm25_score = best_result["bm25_score"]

    if best_rerank_score >= 2.0 and best_semantic_score >= 0.60:
        label = "High"
        reason = "Strong rerank match and strong semantic relevance"
    elif best_rerank_score >= -4.0 and best_semantic_score >= 0.45:
        label = "Medium"
        reason = "Moderate retrieval relevance"
    else:
        label = "Low"
        reason = "Weak retrieval relevance"

    return {
        "label": label,
        "reason": reason,
        "best_rerank_score": best_rerank_score,
        "best_semantic_score": best_semantic_score,
        "best_bm25_score": best_bm25_score,
    }


def should_fallback(top_results, rerank_threshold=-5.0):
    if not top_results:
        return True

    best_rerank_score = top_results[0]["rerank_score"]
    return best_rerank_score < rerank_threshold


def build_sources(top_results):
    lines = []
    for item in top_results:
        lines.append(f"- {item['filename']} (chunk {item['chunk_id']})")
    return "\n".join(lines)


def ask_llm(question, context):
    client = OpenAI()

    user_prompt = f"""
Question:
{question}

Retrieved context:
{context}

Answer the question using only the retrieved context.
"""

    response = client.responses.create(
        model="gpt-5.4",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.output_text


def rag_answer(question, chunks, embedding_model, reranker):
    bm25, _ = build_bm25(chunks)

    semantic_similarities, top_semantic_indices = semantic_search(
        question, chunks, embedding_model, top_k=5
    )

    bm25_scores, top_bm25_indices = lexical_search(
        question, bm25, top_k=5
    )

    combined_indices = list(set(list(top_semantic_indices) + list(top_bm25_indices)))

    reranked_results = rerank_results(
        question,
        combined_indices,
        chunks,
        semantic_similarities,
        bm25_scores,
        reranker
    )

    top_final_results = reranked_results[:5]
    confidence = get_confidence_info(top_final_results)
    fallback = should_fallback(top_final_results)
    context = build_context(top_final_results)
    sources = build_sources(top_final_results)

    if fallback:
        final_answer = "I don't have enough information in the retrieved documents."
    else:
        final_answer = ask_llm(question, context)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "question": question,
        "answer": final_answer,
        "top_results": top_final_results,
        "confidence": confidence,
        "sources": sources,
        "timestamp": timestamp,
        "fallback": fallback,
    }


# =========================
# DISPLAY / SPOKEN TEXT
# =========================
def build_display_text(result):
    return result["answer"]


def build_spoken_text(answer: str) -> str:
    fallback_text = "I don't have enough information in the retrieved documents."

    if answer.strip() == fallback_text:
        return (
            "I do not have enough information in the retrieved documents "
            "to answer confidently."
        )

    spoken = answer.strip()

    # Basic cleanup
    spoken = spoken.replace("%", " percent")
    spoken = spoken.replace("&", " and ")
    spoken = spoken.replace("/", " slash ")
    spoken = spoken.replace("•", "")
    spoken = spoken.replace(" - ", ". ")

    # Acronyms
    acronym_map = {
        r"\bCFO\b": "C F O",
        r"\bAI\b": "A I",
        r"\bAWS\b": "A W S",
        r"\bSQL\b": "S Q L",
        r"\bRAG\b": "R A G",
        r"\bFinOps\b": "Fin Ops",
        r"\bKPI\b": "K P I",
        r"\bROI\b": "R O I",
        r"\bETL\b": "E T L",
    }

    for pattern, replacement in acronym_map.items():
        spoken = re.sub(pattern, replacement, spoken)

    # Decimals
    spoken = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", spoken)

    # Bullets / numbered lists
    spoken = re.sub(r"(?m)^\s*[-*]\s+", "Next point. ", spoken)
    spoken = re.sub(r"(?m)^\s*\d+\.\s+", "Step. ", spoken)

    # Section labels
    spoken = re.sub(r"(?i)executive summary\s*:?", "Executive summary.", spoken)
    spoken = re.sub(r"(?i)key best practices\s*:?", "Key best practices.", spoken)
    spoken = re.sub(
        r"(?i)what a cfo should prioritize first\s*:?",
        "What the C F O should prioritize first.",
        spoken
    )

    # Normalize spaces
    spoken = re.sub(r"\s+", " ", spoken).strip()

    # Premium intro
    spoken = f"Here is your executive briefing. {spoken}"

    return spoken


def escape_ssml_text(text: str) -> str:
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def apply_ssml_formatting(text: str) -> str:
    safe_text = escape_ssml_text(text)

    safe_text = safe_text.replace(". ", ". <break time='650ms'/> ")
    safe_text = safe_text.replace(": ", ": <break time='400ms'/> ")
    safe_text = safe_text.replace("; ", "; <break time='350ms'/> ")
    safe_text = safe_text.replace("? ", "? <break time='700ms'/> ")

    return safe_text


# =========================
# TEXT TO SPEECH
# =========================
def text_to_speech(text, output_file=None):
    if output_file is None:
        timestamp_for_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"rag_answer_{timestamp_for_file}.mp3"
    else:
        output_file = Path(output_file)

    session = boto3.Session(profile_name=AWS_PROFILE)
    polly = session.client("polly", region_name=AWS_REGION)

    formatted_text = apply_ssml_formatting(text)

    ssml_text = f"""
    <speak>
        <prosody rate="90%">
            {formatted_text}
        </prosody>
    </speak>
    """

    response = polly.synthesize_speech(
        Text=ssml_text,
        TextType="ssml",
        OutputFormat="mp3",
        VoiceId=VOICE_ID,
        Engine=ENGINE
    )

    with open(output_file, "wb") as f:
        f.write(response["AudioStream"].read())

    return str(output_file)


def speak_response(answer_text: str):
    spoken_text = build_spoken_text(answer_text)
    audio_file = text_to_speech(spoken_text)
    return spoken_text, audio_file


# =========================
# MAIN
# =========================
def main():
    question = input("Enter your FinOps question: ").strip()

    if not question:
        print("No question entered.")
        return

    print("Loading chunks...")
    chunks = load_chunks(INPUT_FILE)

    print("Loading models...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("Running RAG pipeline...")
    result = rag_answer(question, chunks, embedding_model, reranker)

    display_text = build_display_text(result)
    spoken_text, audio_file = speak_response(display_text)

    print(f"\nTIMESTAMP: {result['timestamp']}")

    print("\nQUESTION:")
    print(result["question"])

    print("\nTOP CHUNKS RETRIEVED (HYBRID + RERANK):")
    for item in result["top_results"]:
        print(
            f"- {item['filename']} | "
            f"chunk {item['chunk_id']} | "
            f"semantic={item['semantic_score']:.4f} | "
            f"bm25={item['bm25_score']:.4f} | "
            f"rerank={item['rerank_score']:.4f}"
        )

    print("\nDISPLAY TEXT:\n")
    print(display_text)

    print("\nSPOKEN TEXT:\n")
    print(spoken_text)

    print("\nSOURCES:")
    print(result["sources"])

    print("\nCONFIDENCE:")
    print(f"Label: {result['confidence']['label']}")
    print(f"Reason: {result['confidence']['reason']}")

    if result["confidence"]["best_rerank_score"] is not None:
        print(f"Best rerank score: {result['confidence']['best_rerank_score']:.4f}")
    else:
        print("Best rerank score: None")

    if result["confidence"]["best_semantic_score"] is not None:
        print(f"Best semantic score: {result['confidence']['best_semantic_score']:.4f}")
    else:
        print("Best semantic score: None")

    if result["confidence"]["best_bm25_score"] is not None:
        print(f"Best BM25 score: {result['confidence']['best_bm25_score']:.4f}")
    else:
        print("Best BM25 score: None")

    print("\nAUDIO FILE:")
    print(audio_file)

    try:
        os.startfile(audio_file)
        time.sleep(0.5)
    except Exception as e:
        print(f"Audio auto-play skipped: {e}")


if __name__ == "__main__":
    main()