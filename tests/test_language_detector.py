"""Tests for language detector module."""

import sys
import unittest
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "parsers"))

from language_detector import LanguageDetector


class TestLanguageDetector(unittest.TestCase):
    """Test cases for LanguageDetector class."""

    def test_detect_english(self):
        """Test detecting English text."""
        text = "This is a resume in English language with professional experience and skills in software development."
        detected = LanguageDetector.detect(text)
        self.assertEqual(detected, "en")

    def test_detect_spanish(self):
        """Test detecting Spanish text."""
        text = "Experiencia profesional en desarrollo de software y gestión de proyectos con habilidades en análisis de datos."
        detected = LanguageDetector.detect(text)
        self.assertEqual(detected, "es")

    def test_detect_french(self):
        """Test detecting French text."""
        text = "Expérience professionnelle en développement de logiciels et gestion de projets avec compétences en analyse de données."
        detected = LanguageDetector.detect(text)
        self.assertEqual(detected, "fr")

    def test_detect_german(self):
        """Test detecting German text."""
        text = "Berufserfahrung in Softwareentwicklung und Projektmanagement mit Fähigkeiten in Datenanalyse."
        detected = LanguageDetector.detect(text)
        self.assertEqual(detected, "de")

    def test_heuristic_chinese(self):
        """Test heuristic detection of Chinese characters."""
        text = "这是一份中文简历，包含专业技能和工作经验。"
        detected = LanguageDetector.detect(text, default="en")
        self.assertEqual(detected, "zh")

    def test_heuristic_japanese(self):
        """Test heuristic detection of Japanese characters."""
        text = "これは日本語の履歴書で、専門スキルと職務経験が含まれています。"
        detected = LanguageDetector.detect(text, default="en")
        self.assertEqual(detected, "ja")

    def test_heuristic_korean(self):
        """Test heuristic detection of Korean characters."""
        text = "이것은 한국어 이력서로 전문 기술과 경험이 포함되어 있습니다."
        detected = LanguageDetector.detect(text, default="en")
        self.assertEqual(detected, "ko")

    def test_heuristic_arabic(self):
        """Test heuristic detection of Arabic characters."""
        text = "هذه سيرة ذاتية باللغة العربية تتضمن المهارات المهنية والخبرة."
        detected = LanguageDetector.detect(text, default="en")
        self.assertEqual(detected, "ar")

    def test_heuristic_russian(self):
        """Test heuristic detection of Russian/Cyrillic characters."""
        text = "Это резюме на русском языке с профессиональными навыками и опытом."
        detected = LanguageDetector.detect(text, default="en")
        self.assertEqual(detected, "ru")

    def test_short_text_default(self):
        """Test that short text returns default language."""
        text = "Short text"
        detected = LanguageDetector.detect(text, default="es")
        self.assertEqual(detected, "es")

    def test_empty_text_default(self):
        """Test that empty text returns default language."""
        text = ""
        detected = LanguageDetector.detect(text, default="fr")
        self.assertEqual(detected, "fr")

    def test_clean_text(self):
        """Test text cleaning removes emails and URLs."""
        text = "Contact: test@example.com Visit: https://example.com Experience in development."
        detected = LanguageDetector.detect(text)
        self.assertEqual(detected, "en")

    def test_get_language_name(self):
        """Test getting language name from code."""
        self.assertEqual(LanguageDetector.get_language_name("en"), "English")
        self.assertEqual(LanguageDetector.get_language_name("es"), "Spanish")
        self.assertEqual(LanguageDetector.get_language_name("fr"), "French")
        self.assertEqual(LanguageDetector.get_language_name("de"), "German")
        self.assertEqual(LanguageDetector.get_language_name("zh"), "Chinese")
        self.assertEqual(LanguageDetector.get_language_name("xx"), "XX")  # Unknown

    def test_mixed_content(self):
        """Test language detection with mixed content."""
        text = "Professional Experience - Worked on multiple projects implementing solutions. Technical skills include Python, JavaScript, SQL, and various cloud platforms."
        detected = LanguageDetector.detect(text)
        self.assertEqual(detected, "en")

    def test_detect_with_confidence(self):
        """Test language detection with confidence score."""
        text = "Professional software developer with 5 years of experience in Python and JavaScript."
        lang, conf = LanguageDetector.detect_with_confidence(text)
        self.assertEqual(lang, "en")
        self.assertGreater(conf, 0.5)  # Should have reasonable confidence

    def test_get_supported_languages(self):
        """Test getting list of supported languages."""
        languages = LanguageDetector.get_supported_languages()
        self.assertIn("en", languages)
        self.assertIn("fr", languages)
        self.assertIn("es", languages)


if __name__ == "__main__":
    unittest.main()
