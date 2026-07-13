"""core.rag — retrieval-augmented grounding for the code-generating agents."""

from core.rag.retriever import KnowledgeBase, get_knowledge_base, retrieve_context

__all__ = ["KnowledgeBase", "get_knowledge_base", "retrieve_context"]
