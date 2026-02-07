"""Language detection utilities for resume parsing using Lingua."""

from typing import Optional, List
import re

try:
    from lingua import Language, LanguageDetectorBuilder
    LINGUA_AVAILABLE = True
except ImportError:
    LINGUA_AVAILABLE = False


class LanguageDetector:
    """Detect the language of resume text using Lingua (most accurate for short text)."""

    # Language code mapping from Lingua to our format
    LANGUAGE_CODES = {
        Language.ENGLISH: 'en',
        Language.FRENCH: 'fr',
        Language.SPANISH: 'es',
        Language.GERMAN: 'de',
        Language.ITALIAN: 'it',
        Language.PORTUGUESE: 'pt',
        Language.DUTCH: 'nl',
        Language.POLISH: 'pl',
        Language.RUSSIAN: 'ru',
        Language.CHINESE: 'zh',
        Language.JAPANESE: 'ja',
        Language.KOREAN: 'ko',
        Language.ARABIC: 'ar',
    }

    # Reverse mapping
    CODE_TO_LANGUAGE = {v: k for k, v in LANGUAGE_CODES.items()}

    # Full language names for display
    LANGUAGE_NAMES = {
        'en': 'English',
        'fr': 'French',
        'es': 'Spanish',
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

    _detector = None

    @classmethod
    def _get_detector(cls):
        """Get or create the Lingua detector (singleton pattern)."""
        if cls._detector is None and LINGUA_AVAILABLE:
            # Build detector with our supported languages
            languages = list(cls.LANGUAGE_CODES.keys())
            cls._detector = LanguageDetectorBuilder.from_languages(*languages).build()
        return cls._detector

    @staticmethod
    def detect(text: str, default: str = 'en') -> str:
        """Detect language from text using Lingua.

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

        if len(cleaned_text) < 20:
            # Too short for reliable detection, use heuristics
            return LanguageDetector._detect_by_heuristics(cleaned_text, default)

        if LINGUA_AVAILABLE:
            try:
                detector = LanguageDetector._get_detector()
                if detector:
                    detected_lang = detector.detect_language_of(cleaned_text)
                    if detected_lang:
                        return LanguageDetector.LANGUAGE_CODES.get(detected_lang, default)
            except Exception:
                pass

        return LanguageDetector._detect_by_heuristics(cleaned_text, default)

    @staticmethod
    def detect_with_confidence(text: str) -> tuple[str, float]:
        """Detect language with confidence score.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (language_code, confidence_score)
        """
        if not text or not text.strip():
            return 'en', 0.0

        cleaned_text = LanguageDetector._clean_text(text)

        if LINGUA_AVAILABLE:
            try:
                detector = LanguageDetector._get_detector()
                if detector:
                    confidence_values = detector.compute_language_confidence_values(cleaned_text)
                    if confidence_values:
                        top_result = confidence_values[0]
                        lang_code = LanguageDetector.LANGUAGE_CODES.get(top_result.language, 'en')
                        return lang_code, top_result.value
            except Exception:
                pass

        return LanguageDetector._detect_by_heuristics(cleaned_text, 'en'), 0.5

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
        """Detect language using character-based heuristics as fallback."""
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
        return LanguageDetector.LANGUAGE_NAMES.get(code, code.upper())

    @staticmethod
    def get_supported_languages() -> List[str]:
        """Get list of supported language codes."""
        return list(LanguageDetector.LANGUAGE_CODES.values())
