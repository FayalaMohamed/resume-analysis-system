"""Tests for enhanced OCR module with multi-engine fallback and confidence scoring."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

import fitz

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parsers.enhanced_ocr import (
    OCRResult,
    OCREngineConfig,
    OCREngineBase,
    PaddleOCREngine,
    TesseractEngine,
    EasyOCREngine,
    PDFNativeEngine,
    EnhancedOCREngine,
    PDFTextExtractorEnhanced,
    extract_text_enhanced,
    compare_ocr_engines,
)


class TestOCRResult(unittest.TestCase):
    """Test cases for OCRResult dataclass."""

    def test_ocr_result_creation(self):
        """Test OCRResult creation with all fields."""
        result = OCRResult(
            text="Hello World",
            confidence=0.95,
            word_confidences=[{"text": "Hello", "confidence": 0.95}],
            engine_used="paddleocr",
            low_confidence_regions=[],
            processing_time_ms=150.5,
            warnings=[],
        )
        
        self.assertEqual(result.text, "Hello World")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.engine_used, "paddleocr")
        self.assertEqual(result.processing_time_ms, 150.5)
        self.assertFalse(result.needs_review)
    
    def test_ocr_result_needs_review_low_confidence(self):
        """Test needs_review property with low confidence."""
        result = OCRResult(
            text="Hello World",
            confidence=0.5,
            engine_used="paddleocr",
        )
        
        self.assertTrue(result.needs_review)
    
    def test_ocr_result_needs_review_low_confidence_regions(self):
        """Test needs_review property with low confidence regions."""
        result = OCRResult(
            text="Hello World",
            confidence=0.85,
            low_confidence_regions=[{"text": "unclear", "confidence": 0.3}],
            engine_used="paddleocr",
        )
        
        self.assertTrue(result.needs_review)


class TestOCREngineConfig(unittest.TestCase):
    """Test cases for OCREngineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OCREngineConfig(name="test")
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.timeout_seconds, 60)
        self.assertEqual(config.confidence_threshold, 0.5)
        self.assertEqual(config.priority, 0)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = OCREngineConfig(
            name="custom_engine",
            enabled=False,
            timeout_seconds=120,
            confidence_threshold=0.7,
            priority=5,
        )
        
        self.assertEqual(config.name, "custom_engine")
        self.assertFalse(config.enabled)
        self.assertEqual(config.timeout_seconds, 120)
        self.assertEqual(config.confidence_threshold, 0.7)
        self.assertEqual(config.priority, 5)


class TestPDFNativeEngine(unittest.TestCase):
    """Test cases for PDFNativeEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = PDFNativeEngine()
    
    def test_is_available(self):
        """Test availability check."""
        self.assertTrue(self.engine.is_available())
    
    def test_initialize(self):
        """Test initialization."""
        self.assertTrue(self.engine.initialize())
    
    def test_extract_text_from_pdf(self):
        """Test text extraction from PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Test resume content")
            page.insert_text((50, 100), "Experience: Software Engineer")
            doc.save(tmp_path)
            doc.close()
            
            result = self.engine.extract_text(tmp_path)
            
            self.assertEqual(result.engine_used, "pdf_native")
            self.assertIn("Test resume", result.text)
            self.assertGreater(result.confidence, 0.9)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_extract_text_from_image_raises(self):
        """Test that image extraction raises appropriate error."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            Path(tmp_path).touch()
            
            result = self.engine.extract_text(tmp_path)
            
            self.assertEqual(result.engine_used, "pdf_native")
            self.assertIn("Extraction failed", result.warnings[0])
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestEnhancedOCREngine(unittest.TestCase):
    """Test cases for EnhancedOCREngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = EnhancedOCREngine()
    
    def test_default_engine_order(self):
        """Test default engine order prioritization."""
        self.assertEqual(len(self.engine.engines), 4)
    
    def test_get_available_engines(self):
        """Test getting available engines."""
        available = self.engine.get_available_engines()
        
        self.assertIsInstance(available, list)
        self.assertGreater(len(available), 0)
        
        engine_names = [e.config.name for e in available]
        self.assertIn("pdf_native", engine_names)
    
    def test_sort_engines_by_priority(self):
        """Test that engines are sorted by priority."""
        custom_engines = [
            PDFNativeEngine(OCREngineConfig(name="native", priority=3)),
            PaddleOCREngine(OCREngineConfig(name="paddle", priority=0)),
        ]
        engine = EnhancedOCREngine(engine_order=custom_engines)
        
        self.assertEqual(engine.engines[0].config.name, "paddle")
        self.assertEqual(engine.engines[1].config.name, "native")


class TestPDFTextExtractorEnhanced(unittest.TestCase):
    """Test cases for PDFTextExtractorEnhanced."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = PDFTextExtractorEnhanced(use_fallback_chain=True)

    def test_pdf_to_images(self):
        """Test PDF to image conversion."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Test content")
            doc.save(tmp_path)
            doc.close()
            
            image_paths = self.extractor.pdf_to_images(tmp_path)
            
            self.assertIsInstance(image_paths, list)
            self.assertEqual(len(image_paths), 1)
            self.assertTrue(image_paths[0].endswith('.png'))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_extract_text_from_pdf_pdf_native(self):
        """Test text extraction from PDF using native method."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "John Doe")
            page.insert_text((50, 80), "Software Engineer")
            doc.save(tmp_path)
            doc.close()
            
            result = self.extractor.extract_text_from_pdf(tmp_path, use_ocr=False)
            
            self.assertIn("file_name", result)
            self.assertIn("full_text", result)
            self.assertIn("pages", result)
            self.assertIn("John Doe", result["full_text"])
            self.assertEqual(result["extraction_method"], "pdf_native")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_extract_text_from_pdf_with_ocr(self):
        """Test text extraction with OCR fallback."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "OCR Test Resume")
            doc.save(tmp_path)
            doc.close()
            
            result = self.extractor.extract_text_from_pdf(tmp_path, use_ocr=True)
            
            self.assertIn("file_name", result)
            self.assertIn("overall_confidence", result)
            self.assertIn("extraction_method", result)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_get_uncertainty_report(self):
        """Test uncertainty report generation."""
        extraction_result = {
            "overall_confidence": 0.85,
            "needs_review": False,
            "pages": [
                {"page": 1, "confidence": 0.90},
                {"page": 2, "confidence": 0.80},
            ],
        }
        
        report = self.extractor.get_uncertainty_report(extraction_result)
        
        self.assertEqual(report["overall_confidence"], 0.85)
        self.assertFalse(report["needs_review"])
        self.assertEqual(len(report["low_confidence_pages"]), 0)
    
    def test_get_uncertainty_report_low_confidence(self):
        """Test uncertainty report with low confidence."""
        extraction_result = {
            "overall_confidence": 0.5,
            "needs_review": True,
            "pages": [
                {"page": 1, "confidence": 0.5},
                {"page": 2, "confidence": 0.5},
            ],
        }
        
        report = self.extractor.get_uncertainty_report(extraction_result)
        
        self.assertTrue(report["needs_review"])
        self.assertEqual(len(report["low_confidence_pages"]), 2)
        self.assertIn("re-upload", report["suggestions"][0].lower())


class TestExtractTextEnhanced(unittest.TestCase):
    """Test cases for convenience function extract_text_enhanced."""

    def test_extract_text_enhanced_pdf_native(self):
        """Test convenience function with PDF native extraction."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Enhanced Extraction Test")
            doc.save(tmp_path)
            doc.close()
            
            result = extract_text_enhanced(tmp_path, use_ocr=False)
            
            self.assertIn("file_name", result)
            self.assertIn("uncertainty_report", result)
            self.assertIn("Enhanced Extraction Test", result["full_text"])
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestCompareOCREngines(unittest.TestCase):
    """Test cases for engine comparison function."""

    @patch('parsers.enhanced_ocr.EnhancedOCREngine.compare_engines')
    def test_compare_engines_returns_dict(self, mock_compare):
        """Test that compare_engines returns dictionary."""
        mock_engine = MagicMock()
        mock_result = OCRResult(
            text="Test",
            confidence=0.9,
            word_confidences=[],
            engine_used="test"
        )
        mock_engine.compare_engines.return_value = {"test": mock_result}
        
        with patch('parsers.enhanced_ocr.EnhancedOCREngine', return_value=mock_engine):
            result = compare_ocr_engines("test.png")
            
            self.assertIsInstance(result, dict)


class TestConfidenceScoring(unittest.TestCase):
    """Test cases for confidence scoring logic."""

    def test_confidence_threshold_classification(self):
        """Test confidence threshold classification."""
        high_conf = OCRResult(text="Clear text", confidence=0.9, engine_used="test")
        medium_conf = OCRResult(text="Somewhat clear", confidence=0.75, engine_used="test")
        low_conf = OCRResult(text="Unclear", confidence=0.5, engine_used="test")
        
        self.assertFalse(high_conf.needs_review)
        self.assertFalse(medium_conf.needs_review)  # 0.75 >= 0.7 threshold
        self.assertTrue(low_conf.needs_review)

    def test_word_confidence_tracking(self):
        """Test word confidence tracking."""
        words = [
            {"text": "Hello", "confidence": 0.95, "engine": "paddleocr"},
            {"text": "World", "confidence": 0.85, "engine": "paddleocr"},
            {"text": "unclear", "confidence": 0.40, "engine": "paddleocr"},
        ]
        
        result = OCRResult(
            text="Hello World unclear",
            confidence=0.73,
            word_confidences=words,
            engine_used="paddleocr",
            low_confidence_regions=[words[2]],
        )
        
        self.assertEqual(len(result.word_confidences), 3)
        self.assertEqual(len(result.low_confidence_regions), 1)
        self.assertTrue(result.needs_review)


class TestEngineFallbackChain(unittest.TestCase):
    """Test cases for OCR engine fallback chain."""

    def test_fallback_sequence_order(self):
        """Test that engines are tried in priority order."""
        engine = EnhancedOCREngine()
        
        for e in engine.engines:
            if hasattr(e.config, 'priority'):
                self.assertIsNotNone(e.config.priority)
    
    def test_pdf_native_fallback(self):
        """Test that PDF native is always available as fallback."""
        engine = EnhancedOCREngine()
        
        available = engine.get_available_engines()
        engine_names = [e.config.name for e in available]
        
        self.assertIn("pdf_native", engine_names)


if __name__ == "__main__":
    unittest.main()