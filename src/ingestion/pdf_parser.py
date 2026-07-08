"""PDF parser: text extraction with OCR fallback for scanned pages."""

import logging
from typing import List

import pdfplumber

logger = logging.getLogger(__name__)


class PageContent:
    def __init__(self, page_number: int, text: str, tables: list = None):
        self.page_number = page_number
        self.text = text.strip() if text else ""
        self.tables = tables or []

    @property
    def char_count(self) -> int:
        return len(self.text)


def parse_pdf_text(pdf_path: str) -> List[PageContent]:
    """Extract text and tables from a text-based PDF using pdfplumber."""
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                pages.append(PageContent(
                    page_number=i,
                    text=text,
                    tables=tables,
                ))
        logger.info("Parsed %d pages from %s", len(pages), pdf_path)
    except Exception as e:
        logger.error("Failed to parse PDF %s: %s", pdf_path, e)
        raise
    return pages


def parse_pdf_with_ocr_fallback(
    pdf_path: str, ocr_threshold: int = 50
) -> List[PageContent]:
    """
    Parse PDF, falling back to OCR for pages with insufficient text.
    Currently returns text-based extraction; OCR fallback is a stub.
    """
    pages = parse_pdf_text(pdf_path)
    low_text_pages = [p for p in pages if p.char_count < ocr_threshold]

    if low_text_pages:
        logger.info(
            "Pages with low text (<%d chars): %s — OCR fallback not yet applied",
            ocr_threshold,
            [p.page_number for p in low_text_pages],
        )
        # TODO: Integrate PaddleOCR/pytesseract for these pages
        # For the MVP, mark these pages as having limited extractable text.

    return pages
