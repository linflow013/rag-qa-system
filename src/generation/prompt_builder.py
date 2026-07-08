"""Prompt construction — system prompt, context formatting, history injection."""

from typing import List, Optional

from src.generation.llm_client import Message
from src.indexing.vector_store import SearchResult

SYSTEM_PROMPT_EN = """You are an enterprise QA assistant for AcmeTech Solutions.
Answer questions based ONLY on the provided context documents.
If the context does not contain sufficient information, say:
"I cannot answer this question based on the available documents. Please try rephrasing your question or consult the relevant department."

Rules:
- Do not fabricate or infer information beyond the context.
- Always cite the source document and page number for each claim: [Source: filename, Page: N].
- If multiple sources contribute, cite each one.
- For Chinese context documents, answer in Chinese. For English context, answer in English.
- Keep answers concise but complete.
"""

SYSTEM_PROMPT_CN = """你是阿科美科技的企业QA助手。
仅根据提供的上下文文档回答问题。
如果上下文中没有足够的信息，请说：
"根据现有文档，我无法回答此问题。请尝试重新表述问题或咨询相关部门。"

规则：
- 不要编造或推断超出上下文范围的信息。
- 每个观点必须引用源文档和页码：[来源：文件名，页码：N]。
- 如果有多个来源，请逐一引用。
- 保持回答简洁但完整。
"""


def build_system_prompt(language: str = "en") -> str:
    """Return the appropriate system prompt based on language."""
    return SYSTEM_PROMPT_CN if language == "cn" else SYSTEM_PROMPT_EN


def format_context(results: List[SearchResult]) -> str:
    """Format retrieved chunks as numbered context for the LLM."""
    parts = []
    for i, result in enumerate(results, start=1):
        source = result.metadata.get("source_file", "unknown")
        page = result.metadata.get("page_number", "?")
        parts.append(
            f"[Document {i}] Source: {source}, Page: {page}\n"
            f"{result.text}\n"
        )
    return "\n".join(parts)


def format_history(history: List[Message], max_turns: int = 5) -> str:
    """Format recent conversation history for the prompt."""
    if not history:
        return ""

    recent = history[-(max_turns * 2):]  # user + assistant pairs
    lines = []
    for msg in recent:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def build_prompt(
    question: str,
    context_results: List[SearchResult],
    history: Optional[List[Message]] = None,
) -> str:
    """Construct the full user prompt with context, history, and question."""
    context_text = format_context(context_results)
    history_text = format_history(history or [])

    prompt = f"""Context documents:
---
{context_text}
---

"""
    if history_text:
        prompt += f"""Conversation history:
---
{history_text}
---

"""

    prompt += f"""Question: {question}

Answer (with citations):"""

    return prompt
