"""
core.rag.retriever — retrieval over a curated ML-knowledge base.

Grounds the feature-engineer and code-generator agents in vetted best-practices so
their generated code reflects real guidance (leakage-safety, imbalance handling,
serving) rather than only the model's parametric memory.

Pluggable backend, chosen at build time:
  - Default: TF-IDF sparse retrieval (scikit-learn) — offline, fast, zero extra deps.
  - If OPENAI_API_KEY is set: dense semantic embeddings (text-embedding-3-small),
    falling back to TF-IDF if the embedding call fails.

Docs live as .md files under knowledge/ (one focused topic per file = one chunk).
ponytail: one chunk per file keeps indexing trivial; add heading-level chunking only
if the docs grow long enough that whole-file retrieval gets noisy.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.config import settings

logger = logging.getLogger(__name__)

_KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"
_EMBED_MODEL = "text-embedding-3-small"


class KnowledgeBase:
    def __init__(self, docs: list[dict[str, str]], backend: str, matrix, vectorizer=None):
        self.docs = docs
        self.backend = backend  # "openai" | "tfidf"
        self._matrix = matrix  # doc vectors (dense ndarray or sparse tfidf matrix)
        self._vectorizer = vectorizer

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Return the top-k docs most relevant to `query`, each with a similarity score."""
        if not self.docs or not query.strip():
            return []
        q_vec = self._embed_query(query)
        sims = cosine_similarity(q_vec, self._matrix).ravel()
        top = np.argsort(sims)[::-1][:k]
        return [
            {**self.docs[i], "score": round(float(sims[i]), 4)}
            for i in top
            if sims[i] > 0
        ]

    def _embed_query(self, query: str):
        if self.backend == "openai":
            try:
                return np.array([_openai_embed([query])[0]])
            except Exception as exc:  # degrade to nothing rather than crash an agent
                logger.warning("OpenAI query embedding failed: %s", exc)
                return np.zeros((1, self._matrix.shape[1]))
        return self._vectorizer.transform([query])


def _openai_embed(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.embeddings.create(model=_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _load_docs(knowledge_dir: Path) -> list[dict[str, str]]:
    docs = []
    for md in sorted(knowledge_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8").strip()
        if not text:
            continue
        first = text.splitlines()[0].lstrip("# ").strip()
        docs.append({"title": first or md.stem, "source": md.name, "text": text})
    return docs


def _build(knowledge_dir: Path) -> KnowledgeBase:
    docs = _load_docs(knowledge_dir)
    if not docs:
        logger.warning("Knowledge base empty at %s", knowledge_dir)
        return KnowledgeBase([], "tfidf", np.zeros((0, 1)), TfidfVectorizer())

    corpus = [d["text"] for d in docs]

    # Prefer dense embeddings when an OpenAI key is available.
    if settings.openai_api_key.strip():
        try:
            matrix = np.array(_openai_embed(corpus))
            logger.info("Knowledge base built with OpenAI embeddings (%d docs)", len(docs))
            return KnowledgeBase(docs, "openai", matrix)
        except Exception as exc:
            logger.warning("Embedding backend unavailable, using TF-IDF: %s", exc)

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    logger.info("Knowledge base built with TF-IDF (%d docs)", len(docs))
    return KnowledgeBase(docs, "tfidf", matrix, vectorizer)


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    """Cached knowledge base singleton — indexed once per process."""
    return _build(_KNOWLEDGE_DIR)


def retrieve_context(query: str, k: int = 3, max_chars: int = 1800) -> str:
    """
    Retrieve grounding context for a prompt as a formatted string.

    Returns "" if nothing relevant is found, so callers can inject unconditionally.
    """
    try:
        hits = get_knowledge_base().retrieve(query, k=k)
    except Exception as exc:
        logger.warning("Retrieval failed (non-fatal): %s", exc)
        return ""
    if not hits:
        return ""
    blocks = []
    used = 0
    for h in hits:
        block = f"### {h['title']} (source: {h['source']})\n{h['text']}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)
