# 💼 AI FinOps CFO Assistant — Decision Intelligence System

## 🚀 Overview

This project is a production-grade **AI-powered FinOps assistant** designed to support CFO-level decision-making.

It combines:
- deterministic financial analytics
- retrieval-augmented generation (RAG)
- hybrid reasoning (data + AI interpretation)

The system delivers **real-time financial insights, risk analysis, and operational recommendations** based on cloud cost data.

---

## 🧠 Key Capabilities

### 📊 Financial Analytics Engine
- Total cost, budget, variance ($ and %)
- Top / bottom cost drivers
- Ranking by product, service, division
- Average cost analysis
- Period-based aggregation (days / weeks / months)

### 🤖 RAG (Retrieval-Augmented Generation)
- Context-aware FinOps explanations
- Strict anti-hallucination guardrails
- Safe fallback for unsupported questions

### 🔀 Hybrid Reasoning (Advanced)
- Combines:
  - quantitative financial analysis
  - FinOps interpretation
  - CFO-level recommendations

---

## 🏗️ Architecture

- **Data Layer**
  - AWS Athena
  - AWS Glue
  - S3 (data lake)

- **Analytics Layer**
  - Python (pandas)
  - dbt transformations

- **AI Layer**
  - Embeddings (SentenceTransformers)
  - RAG pipeline (BM25 + semantic + reranker)
  - OpenAI for reasoning

- **Visualization**
  - Power BI dashboard

- **Experience Layer**
  - Guided CFO workflow
  - Voice narration (Amazon Polly)

---

## ⚙️ End-to-End Workflow

1. User selects analysis period
2. Data is filtered and validated
3. Power BI dashboard is refreshed
4. CFO receives narrated insights
5. User asks strategic question
6. System responds with:
   - analytical results
   - AI interpretation
   - business recommendation

---

## 📊 📈 Proven Results (Evaluation)

### 🔬 Test Coverage

- Total scenarios: **50**
- Analytical: 30
- RAG: 10
- Hybrid: 10

### ✅ Performance

- **Overall pass rate:** 96%
- **Analytical accuracy:** 100%
- **Hybrid reasoning:** 100%
- **RAG accuracy:** 80% (production-safe, no shortcuts)
- **Hallucination rate:** 0%
- **Fallback accuracy:** 100%

### ⏱️ Latency

- Average response time: **~4.2 seconds**

---

## 📁 Evidence

Evaluation outputs:

outputs/evaluation/
├── cfo_eval_summary_latest.json
├── cfo_eval_results_.json
├── cfo_eval_results_.csv


Pipeline outputs:


outputs/
├── agent_prompt_.wav
├── cfo_answer_.mp3


---

## 💡 Example Insight


Total actual cost: $16,859.58
Budget: $15,668.98
Variance: +$1,190.60 (+7.60%)


Top cost drivers:
- AI Assistant
- Bedrock

➡️ Recommendation:
Focus on governance, cost allocation, and optimization of primary drivers.

---

## 🎯 Business Impact

This system enables:

- real-time cost visibility
- proactive risk detection
- structured financial decision-making
- alignment between engineering and finance

---

## 🧩 What Makes This Project Strong

- Full **data + AI + system design integration**
- Production-like evaluation framework
- Zero hallucination tolerance
- CFO-oriented outputs (not generic AI answers)

---

## 🧑‍💻 Félix Brillant

AI Cloud Data Engineer specializing in:
- AWS data architecture
- FinOps optimization
- AI-driven decision systems
