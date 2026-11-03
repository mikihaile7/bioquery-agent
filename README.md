# BioQuery Agent

A small agentic AI app in Python. You ask a biology question in plain English; an LLM decides which public NCBI data sources to query, calls them as tools, and synthesizes one cited, human-readable answer.

## How it works

```
question ──> LLM (Claude, tool use) ──> picks tools ──> NCBI E-utilities
                  ^                                          |
                  └──────────── tool results (JSON) <────────┘
                  |
                  └──> final answer with PMID / GSE / Gene ID citations
```

The agent runs a loop: the model requests one or more tools, the app executes them and returns the results, and the loop repeats until the model has enough evidence to answer. A turn cap prevents runaway loops, and when it is hit the model is forced to answer from what it has gathered.

### Tools

| Tool | Source | Use |
| --- | --- | --- |
| `search_pubmed` | PubMed | Published evidence, with abstracts |
| `search_geo` | NCBI GEO | Public gene expression datasets (GSE series) |
| `get_gene_info` | NCBI Gene | Official gene record, location, function |

### Design choices

- **Grounded answers:** the system prompt requires every claim to cite an identifier from a tool result and forbids invented PMIDs or accessions.
- **Token efficiency:** abstracts and summaries are truncated, result counts are capped, and total tool output is size-limited before it goes back to the model.
- **Resilient tools:** failures are returned to the model as error messages instead of crashing, so it can retry or explain the gap. Requests respect NCBI rate limits and retry on HTTP 429.
- **Testable:** NCBI and the LLM are both mocked in the tests, so they run offline with no API key.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
# optional, raises the NCBI rate limit from 3 to 10 requests/second
export NCBI_API_KEY=your_ncbi_key
export NCBI_EMAIL=you@example.com
```

## Usage

```bash
python agent.py "What is known about IL-13 in asthma, and are there GEO datasets on airway expression?"
```

Tool calls are logged to stderr so you can watch the agent work; add `--quiet` to hide them. Options: `--max-turns`, `--model`.

## Tests

```bash
pytest
```

## Limitations

- Answers are only as good as the retrieved abstracts; the agent does not read full-text papers or download expression matrices.
- GEO search returns dataset metadata, not the expression data itself.
- This is a research aid, not a source of medical advice.

## Ideas for extension

- Add a retrieval step (embeddings over fetched abstracts) for RAG-style re-ranking.
- Add a code-execution agent that downloads a GEO series and runs a differential expression analysis.
- Add more sources (UniProt, Open Targets, ClinicalTrials.gov) as new tools.
