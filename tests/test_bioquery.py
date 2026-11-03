"""Offline tests: NCBI and the LLM are both mocked, so no network or API key is needed."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import agent  # noqa: E402
import tools  # noqa: E402


# ---------------------------------------------------------------- NCBI mocks
class FakeResponse:
    def __init__(self, payload=None, text="", status=200):
        self._payload, self.text, self.status_code = payload, text, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


PUBMED_XML = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
<PMID>111</PMID><Article><Abstract>
<AbstractText Label="BACKGROUND">IL-13 drives type 2 inflammation.</AbstractText>
<AbstractText>It is a therapeutic target.</AbstractText>
</Abstract></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(tools, "_pause", lambda: None)


def fake_get_factory(routes):
    def fake_get(url, params=None, timeout=None):
        key = (url.rsplit("/", 1)[-1].replace(".fcgi", ""), (params or {}).get("db"))
        return routes[key]

    return fake_get


def test_search_pubmed_parses_results(monkeypatch):
    routes = {
        ("esearch", "pubmed"): FakeResponse({"esearchresult": {"idlist": ["111"]}}),
        ("esummary", "pubmed"): FakeResponse(
            {"result": {"111": {"title": "IL-13 in asthma", "fulljournalname": "J Test",
                                "pubdate": "2024", "authors": [{"name": "Smith A"}]}}}
        ),
        ("efetch", "pubmed"): FakeResponse(text=PUBMED_XML),
    }
    monkeypatch.setattr(tools.requests, "get", fake_get_factory(routes))
    out = tools.search_pubmed("IL13 asthma", max_results=3)
    assert out["count"] == 1
    paper = out["results"][0]
    assert paper["pmid"] == "111" and paper["title"] == "IL-13 in asthma"
    assert "type 2 inflammation" in paper["abstract"] and "therapeutic target" in paper["abstract"]


def test_search_geo_builds_accession_url(monkeypatch):
    routes = {
        ("esearch", "gds"): FakeResponse({"esearchresult": {"idlist": ["200"]}}),
        ("esummary", "gds"): FakeResponse(
            {"result": {"200": {"accession": "GSE123", "title": "Asthma airway RNA-seq",
                                "taxon": "Homo sapiens", "gdstype": "Expression profiling by high throughput sequencing",
                                "n_samples": 24, "summary": "x" * 2000}}}
        ),
    }
    monkeypatch.setattr(tools.requests, "get", fake_get_factory(routes))
    out = tools.search_geo("asthma airway")
    item = out["results"][0]
    assert item["accession"] == "GSE123" and item["url"].endswith("acc=GSE123")
    assert len(item["summary"]) <= tools.MAX_TEXT_CHARS


def test_get_gene_info_not_found(monkeypatch):
    routes = {("esearch", "gene"): FakeResponse({"esearchresult": {"idlist": []}})}
    monkeypatch.setattr(tools.requests, "get", fake_get_factory(routes))
    assert tools.get_gene_info("NOTAGENE") == {"source": "NCBI Gene", "symbol": "NOTAGENE", "found": False}


def test_dispatch_reports_errors_instead_of_raising(monkeypatch):
    def boom(**kwargs):
        raise ValueError("bad input")

    monkeypatch.setitem(tools.TOOL_FUNCTIONS, "search_pubmed", boom)
    assert "bad input" in json.loads(tools.dispatch("search_pubmed", {"query": "x"}))["error"]
    assert "Unknown tool" in json.loads(tools.dispatch("nope", {}))["error"]


def test_clamp_handles_bad_values():
    assert tools._clamp("abc") == 5 and tools._clamp(99) == 10 and tools._clamp(0) == 1


# ------------------------------------------------------------ agent loop mock
def block_text(text):
    return SimpleNamespace(type="text", text=text)


def block_tool(name, args, tool_id="t1"):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=tool_id)


def response(content, stop_reason):
    return SimpleNamespace(content=content, stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20))


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def test_agent_calls_tools_then_synthesizes():
    client = FakeClient([
        response([block_tool("get_gene_info", {"symbol": "IL13"}, "a"),
                  block_tool("search_geo", {"query": "asthma"}, "b")], "tool_use"),
        response([block_text("IL13 is relevant (Gene ID 3596; GSE123).")], "end_turn"),
    ])
    seen = []
    result = agent.run_agent(
        "IL13 in asthma?", client=client,
        dispatch_fn=lambda name, args: seen.append(name) or json.dumps({"ok": True}),
    )
    assert seen == ["get_gene_info", "search_geo"]
    assert "GSE123" in result.answer
    assert result.turns == 2 and result.input_tokens == 200 and result.output_tokens == 40
    # tool results are sent back to the model on the second call
    # (the list is shared by reference, so the final assistant reply sits after it)
    last_user = client.calls[1]["messages"][-2]
    assert [r["tool_use_id"] for r in last_user["content"]] == ["a", "b"]


def test_agent_forces_answer_when_turn_budget_runs_out():
    looping = response([block_tool("search_pubmed", {"query": "x"})], "tool_use")
    client = FakeClient([looping, looping, response([block_text("Partial answer.")], "end_turn")])
    result = agent.run_agent("q", client=client, max_turns=2, dispatch_fn=lambda n, a: "{}")
    assert result.answer == "Partial answer."
    assert client.calls[-1]["tool_choice"] == {"type": "none"}
