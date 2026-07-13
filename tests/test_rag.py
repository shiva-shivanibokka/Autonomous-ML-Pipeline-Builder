"""
tests.test_rag — retrieval quality for the ML-knowledge base.

Runs on the default TF-IDF backend (no network/keys), which is what tests and the
offline default use. Verifies retrieval actually surfaces the right doc — the
"evaluate retrieval quality" part of a real RAG system, not just a chat wrapper.
"""

from core.rag.retriever import _build, _KNOWLEDGE_DIR


def test_knowledge_base_loads_docs():
    kb = _build(_KNOWLEDGE_DIR)
    titles = {d["source"] for d in kb.docs}
    assert "data_leakage.md" in titles
    assert "class_imbalance.md" in titles
    assert kb.backend == "tfidf"  # offline default


def test_leakage_query_retrieves_leakage_doc():
    kb = _build(_KNOWLEDGE_DIR)
    hits = kb.retrieve("how do I avoid train test data leakage when preprocessing", k=1)
    assert hits, "expected at least one hit"
    assert hits[0]["source"] == "data_leakage.md"
    assert hits[0]["score"] > 0


def test_imbalance_query_retrieves_imbalance_doc():
    kb = _build(_KNOWLEDGE_DIR)
    hits = kb.retrieve("dataset is highly imbalanced, fraud detection, scale_pos_weight", k=1)
    assert hits[0]["source"] == "class_imbalance.md"


def test_retrieve_context_formats_and_is_bounded():
    from core.rag import retrieve_context

    ctx = retrieve_context("feature engineering datetime and encoding", k=2, max_chars=1200)
    assert ctx  # non-empty
    assert len(ctx) <= 1200 + 400  # roughly bounded (last block may slightly exceed budget check)
    assert "###" in ctx  # formatted with headers
