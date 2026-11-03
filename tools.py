"""NCBI E-utilities tools that the BioQuery agent can call.

Each tool takes simple arguments and returns plain dicts, so the output can be
serialized to JSON and handed back to the LLM as a tool result.
"""
from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TIMEOUT_SECONDS = 20
MAX_TEXT_CHARS = 700  # per abstract/summary, keeps token usage down
MAX_TOOL_OUTPUT_CHARS = 12000  # hard cap on what is sent back to the model


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------
def _pause() -> None:
    """Stay under NCBI's rate limit (3 req/s without a key, 10 with one)."""
    time.sleep(0.11 if os.getenv("NCBI_API_KEY") else 0.35)


def _params(extra: dict) -> dict:
    params = {"retmode": "json", "tool": "bioquery-agent"}
    if os.getenv("NCBI_EMAIL"):
        params["email"] = os.environ["NCBI_EMAIL"]
    if os.getenv("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    params.update(extra)
    return params


def _get(endpoint: str, **params) -> requests.Response:
    """GET an E-utilities endpoint, retrying politely if rate limited."""
    for attempt in range(3):
        _pause()
        resp = requests.get(
            f"{EUTILS}/{endpoint}.fcgi",
            params=_params(params),
            timeout=TIMEOUT_SECONDS,
        )
        if resp.status_code == 429:
            time.sleep(1.5 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError("NCBI rate limit exceeded; try again in a few seconds.")


def _clamp(n, low: int = 1, high: int = 10) -> int:
    try:
        return max(low, min(high, int(n)))
    except (TypeError, ValueError):
        return 5


def _truncate(text: str | None, limit: int = MAX_TEXT_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


# --------------------------------------------------------------------------
# Tool 1: PubMed literature search
# --------------------------------------------------------------------------
def _pubmed_abstracts(pmids: list[str]) -> dict[str, str]:
    """Fetch abstracts for a list of PMIDs. Returns {pmid: abstract}."""
    text = _get("efetch", db="pubmed", id=",".join(pmids), retmode="xml").text
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return {}
    abstracts = {}
    for article in root.iter("PubmedArticle"):
        pmid = article.findtext(".//PMID")
        parts = ["".join(node.itertext()) for node in article.findall(".//Abstract/AbstractText")]
        if pmid:
            abstracts[pmid] = _truncate(" ".join(parts))
    return abstracts


def search_pubmed(query: str, max_results: int = 5) -> dict:
    """Search PubMed and return titles, journals, dates, and abstracts."""
    max_results = _clamp(max_results)
    found = _get("esearch", db="pubmed", term=query, retmax=max_results, sort="relevance").json()
    pmids = found.get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return {"source": "PubMed", "query": query, "count": 0, "results": []}

    summaries = _get("esummary", db="pubmed", id=",".join(pmids)).json().get("result", {})
    abstracts = _pubmed_abstracts(pmids)

    results = []
    for pmid in pmids:
        item = summaries.get(pmid, {})
        results.append(
            {
                "pmid": pmid,
                "title": item.get("title"),
                "authors": [a.get("name") for a in item.get("authors", [])][:3],
                "journal": item.get("fulljournalname"),
                "pubdate": item.get("pubdate"),
                "abstract": abstracts.get(pmid, ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return {"source": "PubMed", "query": query, "count": len(results), "results": results}


# --------------------------------------------------------------------------
# Tool 2: GEO dataset search (gene expression series)
# --------------------------------------------------------------------------
def search_geo(query: str, max_results: int = 5) -> dict:
    """Search NCBI GEO for gene expression series (GSE accessions)."""
    max_results = _clamp(max_results)
    term = f"({query}) AND gse[ETYP]"
    found = _get("esearch", db="gds", term=term, retmax=max_results).json()
    uids = found.get("esearchresult", {}).get("idlist", [])
    if not uids:
        return {"source": "GEO", "query": query, "count": 0, "results": []}

    summaries = _get("esummary", db="gds", id=",".join(uids)).json().get("result", {})
    results = []
    for uid in uids:
        item = summaries.get(uid, {})
        accession = item.get("accession")
        results.append(
            {
                "accession": accession,
                "title": item.get("title"),
                "organism": item.get("taxon"),
                "study_type": item.get("gdstype"),
                "n_samples": item.get("n_samples"),
                "summary": _truncate(item.get("summary")),
                "url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
            }
        )
    return {"source": "GEO", "query": query, "count": len(results), "results": results}


# --------------------------------------------------------------------------
# Tool 3: NCBI Gene lookup
# --------------------------------------------------------------------------
def get_gene_info(symbol: str, organism: str = "Homo sapiens") -> dict:
    """Look up an official gene record by symbol."""
    term = f"{symbol}[Gene Name] AND {organism}[Organism]"
    found = _get("esearch", db="gene", term=term, retmax=1).json()
    ids = found.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return {"source": "NCBI Gene", "symbol": symbol, "found": False}

    gene_id = ids[0]
    item = _get("esummary", db="gene", id=gene_id).json().get("result", {}).get(gene_id, {})
    return {
        "source": "NCBI Gene",
        "found": True,
        "gene_id": gene_id,
        "symbol": item.get("name"),
        "description": item.get("description"),
        "organism": (item.get("organism") or {}).get("scientificname", organism),
        "chromosome": item.get("chromosome"),
        "map_location": item.get("maplocation"),
        "aliases": item.get("otheraliases"),
        "summary": _truncate(item.get("summary")),
        "url": f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}",
    }


# --------------------------------------------------------------------------
# Tool registry: schemas the LLM sees + dispatcher the agent calls
# --------------------------------------------------------------------------
TOOL_FUNCTIONS = {
    "search_pubmed": search_pubmed,
    "search_geo": search_geo,
    "get_gene_info": get_gene_info,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_pubmed",
            "description": (
                "Search PubMed biomedical literature. Use for mechanisms, disease "
                "associations, reviews, and clinical or experimental evidence. "
                "Returns PMIDs, titles, journals, dates, and abstracts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "PubMed query, e.g. 'IL13 asthma eosinophils'."},
                    "max_results": {"type": "integer", "description": "Number of papers, 1-10 (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_geo",
            "description": (
                "Search NCBI GEO for public gene expression datasets (GSE series). "
                "Use to find what expression data exists for a disease, tissue, or "
                "condition. Returns accessions, organisms, sample counts, and summaries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic, e.g. 'atopic dermatitis skin biopsy RNA-seq'."},
                    "max_results": {"type": "integer", "description": "Number of datasets, 1-10 (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gene_info",
            "description": (
                "Look up an official NCBI Gene record by gene symbol. Use to confirm "
                "what a gene is, where it sits in the genome, and its known function."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Official gene symbol, e.g. 'IL4R'."},
                    "organism": {"type": "string", "description": "Scientific name (default 'Homo sapiens')."},
                },
                "required": ["symbol"],
            },
        },
    },
]


def dispatch(name: str, tool_input: dict) -> str:
    """Run a tool by name and return a JSON string (errors included) for the LLM."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        output = json.dumps(func(**tool_input), ensure_ascii=False)
    except Exception as exc:  # report failures to the model so it can recover
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        output = output[:MAX_TOOL_OUTPUT_CHARS] + '..."[truncated]'
    return output