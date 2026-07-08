"""Citation handling — parse, validate, and format citations in answers."""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


class Citation:
    def __init__(self, source_file: str, page_number: int, excerpt: str = ""):
        self.source_file = source_file
        self.page_number = page_number
        self.excerpt = excerpt

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "excerpt": self.excerpt,
        }

    def __repr__(self):
        return f"Citation({self.source_file}, p.{self.page_number})"


# Match patterns like: [Source: filename, Page: N] or [来源：文件名，页码：N]
CITATION_PATTERN = re.compile(
    r"\[(?:Source|来源)\s*:\s*(.+?)\s*,\s*(?:Page|页码)\s*:\s*(\d+)\]",
    re.IGNORECASE,
)


def extract_citations(text: str, retrieved_sources: List[str]) -> List[Citation]:
    """
    Extract citation markers from the LLM answer.
    Validates that cited sources exist in the retrieved documents.
    """
    citations = []
    seen = set()

    for match in CITATION_PATTERN.finditer(text):
        source_file = match.group(1).strip()
        page_number = int(match.group(2))

        # Validate the source
        source_basename = source_file.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

        # Check if source exists in retrieved documents
        is_valid = any(
            source_basename.lower() in s.lower() or s.lower() in source_basename.lower()
            for s in retrieved_sources
        )

        if not is_valid:
            logger.warning("Citation to untrusted source: %s", source_file)
            continue

        citation_key = f"{source_basename}:{page_number}"
        if citation_key not in seen:
            citations.append(Citation(
                source_file=source_basename,
                page_number=page_number,
            ))
            seen.add(citation_key)

    return citations


def clean_answer(text: str) -> str:
    """Clean up citation formatting in the final answer."""
    # Remove extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Ensure citations have consistent format
    return text.strip()
