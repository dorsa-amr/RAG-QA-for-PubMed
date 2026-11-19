from src.agent import (
    RetrievalTool,
    _ensure_citations,
    _extract_pmids,
    _normalize_inline_citations,
    build_research_tools,
)
from src.benchmark import percentile
from src.data_pipeline import DocumentProcessor


class FakeRetriever:
    def retrieve(self, query):
        return [{"content": "Fasting reduced weight.", "metadata": {"pmid": "111", "title": "A", "pub_date": "2025"}}]


def test_numbered_citations_become_pmids():
    assert _normalize_inline_citations("Weight fell [1].", ["111"]) == "Weight fell [PMID:111]."


def test_citation_numbers_out_of_range_are_dropped():
    assert _normalize_inline_citations("Claim [5] here.", ["111"]) == "Claim here."


def test_sources_line_added_when_answer_has_no_citations():
    assert _ensure_citations("Weight fell.", ["111", "222"]) == "Weight fell.\n\nSources: [PMID:111], [PMID:222]"


def test_answer_with_citations_is_left_alone():
    assert _ensure_citations("Weight fell [PMID:111].", ["111"]) == "Weight fell [PMID:111]."


def test_extract_pmids_removes_duplicates():
    results = [{"metadata": {"pmid": "1"}}, {"metadata": {"pmid": "1"}}, {"metadata": {"pmid": "N/A"}}]
    assert _extract_pmids(results) == ["1"]


def test_model_sees_the_whole_chunk():
    chunk = "x" * 900 + "END"
    text = RetrievalTool.format_results([{"content": chunk, "metadata": {"pmid": "1"}}])
    assert chunk in text


def test_fourth_search_is_refused():
    state = {"searches": 0}
    search, _, _ = build_research_tools(FakeRetriever(), set(), state)
    for _ in range(3):
        search.invoke({"query": "fasting"})
    assert "budget exhausted" in search.invoke({"query": "fasting"})
    assert state["searches"] == 3


def test_verify_answer_rejects_pmids_that_were_not_retrieved():
    search, _, verify = build_research_tools(FakeRetriever(), set(), {"searches": 0})
    search.invoke({"query": "fasting"})
    assert verify.invoke({"answer": "Weight fell [PMID:111]."}).startswith("PASS")
    assert verify.invoke({"answer": "Weight fell [PMID:999]."}).startswith("FAIL")


def article(abstract="", full_text=""):
    return {"pmid": "1", "title": "A", "abstract": abstract, "full_text": full_text, "pub_date": "2025"}


def test_full_text_is_used_before_abstract():
    docs = DocumentProcessor().process_articles([article(abstract="Short.", full_text="Long body.")])
    assert "Long body." in docs[0].page_content and "Short." not in docs[0].page_content


def test_articles_without_text_are_skipped():
    assert DocumentProcessor().process_articles([article()]) == []


def test_p95_latency():
    assert round(percentile(list(range(1, 21)), 0.95), 2) == 19.05


def test_search_keywords_drop_question_words():
    from src.main import extract_search_keywords
    question = "What evidence exists for fasting and inflammatory markers like CRP?"
    assert extract_search_keywords(question) == "fasting inflammatory markers crp"
