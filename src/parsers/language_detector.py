"""Language detection utilities for resume parsing."""

from typing import Optional
import re

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class LanguageDetector:
    """Detect the language of resume text."""

    LANGUAGES = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'nl': 'Dutch',
        'pl': 'Polish',
        'ru': 'Russian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ar': 'Arabic',
    }

    @staticmethod
    def detect(text: str, default: str = 'en') -> str:
        """Detect language from text.

        Args:
            text: Text to analyze
            default: Default language code if detection fails

        Returns:
            Detected language code (e.g., 'en', 'es', 'fr')
        """
        if not text or not text.strip():
            return default

        # Clean text - remove numbers, emails, URLs, special characters
        cleaned_text = LanguageDetector._clean_text(text)

        if len(cleaned_text) < 100:
            # Too short for reliable detection, use heuristics
            return LanguageDetector._detect_by_heuristics(cleaned_text, default)

        if LANGDETECT_AVAILABLE:
            try:
                return detect(cleaned_text)
            except LangDetectException:
                pass

        return LanguageDetector._detect_by_heuristics(cleaned_text, default)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text for language detection."""
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove phone numbers
        text = re.sub(r'\+?\d[\d\s\-]{8,}\d', '', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _detect_by_heuristics(text: str, default: str) -> str:
        """Detect language using character-based heuristics."""
        # Check for specific character sets - Japanese first (Hiragana/Katakana are unique)
        if any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
            return 'ja'  # Japanese (Hiragana/Katakana)
        # Chinese characters (Hanzi/Kanji) - check after Japanese since Kanji overlaps
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return 'zh'  # Chinese
        if any('\uac00' <= char <= '\ud7af' for char in text):
            return 'ko'  # Korean
        if any('\u0600' <= char <= '\u06ff' for char in text):
            return 'ar'  # Arabic
        if any('\u0400' <= char <= '\u04ff' for char in text):
            return 'ru'  # Russian/Cyrillic

        # Check for accented characters common in European languages
        text_lower = text.lower()
        if any(char in text_lower for char in 'ñáéíóúü'):
            return 'es'  # Spanish
        if 'ç' in text_lower or 'œ' in text_lower or 'æ' in text_lower:
            return 'fr'  # French
        if 'ß' in text_lower or 'ä' in text_lower or 'ö' in text_lower or 'ü' in text_lower:
            return 'de'  # German

        return default

    @staticmethod
    def get_language_name(code: str) -> str:
        """Get full language name from code.

        Args:
            code: Language code (e.g., 'en', 'es')

        Returns:
            Full language name or the code if not found
        """
        return LanguageDetector.LANGUAGES.get(code, code.upper())
