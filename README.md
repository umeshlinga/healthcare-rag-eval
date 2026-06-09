# MediGuide AI — Healthcare RAG + Evaluation Framework

Production-grade LLM pipeline for Medicare & Medicaid benefits intelligence with automated correctness verification, hallucination detection, and release gates.

## Evaluation Results (Prompt v2.1)

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness | 0.91 | >= 0.85 | Pass |
| Answer Relevancy | 0.87 | >= 0.80 | Pass |
| Context Recall | 0.83 | >= 0.75 | Pass |
| Hallucination Rate | 0% | <= 5% | Pass |
| Release Gate | APPROVED | All thresholds met | Pass |

## What Makes This Production-Grade

- Semantic FAISS search over 713 CMS document chunks
- GPT-4o-mini answers ONLY from retrieved context, always cites sources
- Judge-LLM reviews every answer for unsupported claims
- Threshold-based release gates block features if accuracy drops
- Every eval run logged with prompt version, scores, and pass/fail

## Tech Stack

LangChain, FAISS, GPT-4o-mini, OpenAI text-embedding-3-small, Streamlit, Python 3.13, Docker, AWS S3

## Author

Umesh Linga — Bioinformatics and AI/ML Engineer
LinkedIn: linkedin.com/in/umeshlinga
GitHub: github.com/umeshlinga
Email: umesh.linga25@gmail.com
