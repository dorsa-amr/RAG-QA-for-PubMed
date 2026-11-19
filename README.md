# LLM-RAG

[![Tests](https://github.com/dorsa-amr/RAG-QA-for-PubMed/actions/workflows/tests.yml/badge.svg)](https://github.com/dorsa-amr/RAG-QA-for-PubMed/actions/workflows/tests.yml)

This project implements a tool-calling, evidence-grounded Retrieval-Augmented Generation (RAG) agent for PubMed articles. It uses LangChain, ChromaDB, and OpenAI APIs to provide reliable answers while making bounded decisions about retrieval, re-querying, and citation verification.

## Features
- Full-text ingestion with fallback to abstracts.
- Deterministic citation enforcement.
- Multi-step retrieval: the researcher can refine its query and search again when evidence is insufficient.
- Citation verification that checks cited PMIDs against the evidence retrieved for the current question.
- Robust error handling and retry logic.
- Model-directed tool selection between direct answering, initial search, refined search, and answer verification.
- Bounded execution with per-question tool traces and re-query metrics.
- Optional conversation memory for contextual follow-up questions while retrieving fresh evidence for each turn.
- Bounded multi-agent workflow with separate planner, researcher, and verification/synthesis roles.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Configure your `.env` file with the required API keys.

## Usage
Run the main script:
```bash
python src/main.py
```

### How to interact with the app
- When asked, choose `y` if you want to load sample PubMed articles.
- Enter a health topic (for example, `fatty liver`) when prompted.
- If you press Enter without typing a topic, the app uses `fatty liver` by default.
- After loading, ask your questions in plain language.
- Type `quit` anytime to exit.

## Benchmark
Run the benchmark script to measure:
- average end-to-end latency
- p95 latency
- citation coverage rate (answers containing PMID citations)
- inline citation rate (answers the model cited itself, without the deterministic fallback)
- citation validity (cited answers whose PMIDs all appear in the evidence retrieved for that question)

Use the default 20-question set:
```bash
python src/benchmark.py --questions-file benchmark_questions.txt --ingest-query "intermittent fasting" --max-articles 50 --output benchmark_results.json
```

The script writes metrics and per-question outputs to `benchmark_results.json`.
Per-question results also include the selected tool calls, retrieval-round count, and whether citation verification was called. The summary reports how often questions required a re-query and the verification call rate.

## Agent workflow

The primary execution path is a bounded multi-agent workflow. A planner classifies the question and defines the evidence need, a researcher uses LangChain tool calling to answer directly or call `document_search` and `search_more`, and a verification/synthesis agent reviews the draft and evidence before returning the answer. The researcher is limited to three searches per question, enforced in the tools, and to seven model calls through LangChain's `ModelCallLimitMiddleware`, which ends the run cleanly so the verifier can answer from the evidence gathered so far; deterministic citation enforcement remains as a final backstop.

Each role has a distinct handoff: the planner produces a research plan, the researcher produces a draft plus tool evidence, and the verifier produces the final response. Per-question traces record the participating roles, tool calls, retrieval rounds, and verification status.

Conversation memory is enabled by the interactive application. Previous questions and generated answers help interpret follow-ups such as `What about side effects?`, but each new question still performs its own retrieval against the loaded document collection.

## Tests

A few unit tests cover the citation helpers, the search budget, chunking and the benchmark maths. They don't call OpenAI or PubMed.

```bash
pip install pytest
pytest
```

## Contributing
Feel free to open issues or submit pull requests.
