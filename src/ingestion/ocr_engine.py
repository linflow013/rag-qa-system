"""OCR engine stub — PaddleOCR integration for scanned PDF pages."""

import logging

logger = logging.getLogger(__name__)


class OCREngine:
    """Wrapper for OCR. Falls back to a no-op if PaddleOCR is not installed."""

    def __init__(self, lang: str = "ch"):
        self.lang = lang
        self._ocr = None
        self._try_load()

    def _try_load(self):
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(lang=self.lang, use_angle_cls=True)
            logger.info("PaddleOCR loaded successfully (lang=%s)", self.lang)
        except ImportError:
            logger.warning(
                "PaddleOCR not installed. OCR for scanned PDFs is disabled. "
                "Install with: pip install paddleocr"
            )
        except Exception as e:
            logger.warning("Failed to load PaddleOCR: %s", e)

    def recognize(self, image) -> str:
        """Run OCR on a page image. Returns recognized text or empty string."""
        if self._ocr is None:
            return ""
        try:
            result = self._ocr.ocr(image, cls=True)
            if result and result[0]:
                lines = [line[1][0] for line in result[0] if line[1][0]]
                return "\n".join(lines)
        except Exception as e:
            logger.error("OCR failed: %s", e)
        return ""

    @property
    def available(self) -> bool:
        return self._ocr is not None
