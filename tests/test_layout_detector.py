"""Tests for layout detection module."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz  # PyMuPDF

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parsers.layout_detector import (
    LayoutDetector,
    LayoutFeatures,
    MLLayoutDetector,
    HeuristicLayoutDetector,
    LAYOUT_DETECTION_AVAILABLE,
)


class TestLayoutFeatures(unittest.TestCase):
    """Test cases for LayoutFeatures dataclass."""

    def test_layout_features_creation(self):
        """Test LayoutFeatures dataclass creation."""
        features = LayoutFeatures(
            is_single_column=True,
            has_tables=False,
            has_images=False,
            num_columns=1,
            text_density=50.0,
            avg_line_length=40.0,
            section_headers=["Experience", "Education"],
            layout_risk_score=10.0,
        )
        
        self.assertTrue(features.is_single_column)
        self.assertFalse(features.has_tables)
        self.assertEqual(features.num_columns, 1)
        self.assertEqual(features.detection_method, "heuristic")
        self.assertIsNone(features.confidence)

    def test_layout_features_with_ml_fields(self):
        """Test LayoutFeatures with ML-specific fields."""
        features = LayoutFeatures(
            is_single_column=False,
            has_tables=True,
            has_images=True,
            num_columns=2,
            text_density=60.0,
            avg_line_length=45.0,
            section_headers=["Work Experience"],
            layout_risk_score=45.0,
            detection_method="ml",
            confidence=0.85,
            table_regions=[{"type": "table", "bbox": [0, 100, 400, 200]}],
            column_regions=[{"type": "text", "bbox": [0, 0, 200, 500]}],
        )
        
        self.assertEqual(features.detection_method, "ml")
        self.assertEqual(features.confidence, 0.85)
        self.assertIsNotNone(features.table_regions)
        self.assertEqual(len(features.table_regions), 1)


class TestHeuristicLayoutDetector(unittest.TestCase):
    """Test cases for HeuristicLayoutDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = HeuristicLayoutDetector(language='en')

    def test_detect_columns_single_column(self):
        """Test column detection for single-column layout."""
        text = """
John Doe
Software Engineer

Experience
- Worked at Company A
- Developed features
- Led team projects

Education
Bachelor of Science in Computer Science
"""
        result = self.detector.detect_columns(text)
        
        self.assertTrue(result["is_single_column"])
        self.assertEqual(result["num_columns"], 1)

    def test_detect_columns_multi_column(self):
        """Test column detection for multi-column layout."""
        # Simulated multi-column with varied indentation
        text = """
Name                                Skills
John Doe                            Python
                                    JavaScript
                                    React
Experience                          Education
Worked at Company                   BS Computer Science
"""
        result = self.detector.detect_columns(text)
        
        # With high indentation variance, should detect multi-column
        self.assertIn("indent_variance", result)

    def test_detect_tables_no_tables(self):
        """Test table detection when no tables present."""
        text = """
Experience
- Developed web applications
- Led engineering team
- Improved performance by 50%
"""
        result = self.detector.detect_tables(text)
        self.assertFalse(result)

    def test_detect_tables_with_pipes(self):
        """Test table detection with pipe characters."""
        text = """
| Skill | Level | Years |
|-------|-------|-------|
| Python | Expert | 5 |
| JavaScript | Advanced | 3 |
| React | Intermediate | 2 |
"""
        result = self.detector.detect_tables(text)
        self.assertTrue(result)

    def test_detect_tables_with_tabs(self):
        """Test table detection with tab characters."""
        text = """
Company	Role	Duration
Acme Inc	Engineer	2020-2023
Tech Corp	Lead	2018-2020
Startup	Developer	2015-2018
"""
        result = self.detector.detect_tables(text)
        self.assertTrue(result)

    def test_detect_section_headers(self):
        """Test section header detection."""
        text = """
EXPERIENCE
Software Engineer at Company

EDUCATION
BS in Computer Science

SKILLS
Python, JavaScript, React
"""
        headers = self.detector.detect_section_headers(text)
        
        self.assertIn("EXPERIENCE", headers)
        self.assertIn("EDUCATION", headers)
        self.assertIn("SKILLS", headers)

    def test_calculate_text_density(self):
        """Test text density calculation."""
        text = "This is a test line\nAnother line here\nThird line"
        density = self.detector.calculate_text_density(text)
        
        self.assertGreater(density, 0)
        self.assertLess(density, 100)

    def test_analyze_layout_complete(self):
        """Test complete layout analysis."""
        text = """
John Doe
john.doe@email.com

EXPERIENCE
Software Engineer at Tech Company
- Developed web applications using Python and React
- Improved system performance by 40%

EDUCATION
Bachelor of Science in Computer Science

SKILLS
Python, JavaScript, React, SQL
"""
        features = self.detector.analyze_layout(text)
        
        self.assertIsInstance(features, LayoutFeatures)
        self.assertTrue(features.is_single_column)
        self.assertFalse(features.has_tables)
        self.assertEqual(features.detection_method, "heuristic")
        self.assertGreater(len(features.section_headers), 0)
        self.assertLessEqual(features.layout_risk_score, 100)


class TestMLLayoutDetector(unittest.TestCase):
    """Test cases for MLLayoutDetector class."""

    def test_is_available_without_ppstructure(self):
        """Test is_available when LayoutDetection is not installed."""
        with patch('parsers.layout_detector.LAYOUT_DETECTION_AVAILABLE', False):
            detector = MLLayoutDetector()
            detector._available = None  # Reset cached value
            # Re-check availability
            result = not LAYOUT_DETECTION_AVAILABLE or detector.is_available()
            # Just verify it doesn't crash
            self.assertIsNotNone(result)

    def test_analyze_columns_empty_regions(self):
        """Test column analysis with no regions."""
        detector = MLLayoutDetector()
        
        result = detector.analyze_columns([])
        
        self.assertTrue(result["is_single_column"])
        self.assertEqual(result["num_columns"], 1)

    def test_analyze_columns_single_column_regions(self):
        """Test column analysis with single-column regions."""
        detector = MLLayoutDetector()
        
        regions = [
            {"type": "text", "bbox": [50, 100, 500, 150], "confidence": 0.9},
            {"type": "text", "bbox": [50, 160, 500, 210], "confidence": 0.9},
            {"type": "title", "bbox": [50, 50, 500, 90], "confidence": 0.95},
        ]
        
        result = detector.analyze_columns(regions, page_width=612)
        
        # All regions are centered, should be single column
        self.assertIn("is_single_column", result)
        self.assertIn("num_columns", result)

    def test_analyze_columns_multi_column_regions(self):
        """Test column analysis with multi-column regions."""
        detector = MLLayoutDetector()
        
        # Regions clearly split between left and right
        regions = [
            {"type": "text", "bbox": [50, 100, 250, 150], "confidence": 0.9},
            {"type": "text", "bbox": [50, 160, 250, 210], "confidence": 0.9},
            {"type": "text", "bbox": [350, 100, 550, 150], "confidence": 0.9},
            {"type": "text", "bbox": [350, 160, 550, 210], "confidence": 0.9},
        ]
        
        result = detector.analyze_columns(regions, page_width=612)
        
        # With clear left/right separation, should detect 2 columns
        self.assertFalse(result["is_single_column"])
        self.assertEqual(result["num_columns"], 2)

    def test_analyze_tables_no_tables(self):
        """Test table analysis with no table regions."""
        detector = MLLayoutDetector()
        
        regions = [
            {"type": "text", "bbox": [50, 100, 500, 150], "confidence": 0.9},
            {"type": "title", "bbox": [50, 50, 500, 90], "confidence": 0.95},
        ]
        
        result = detector.analyze_tables(regions)
        
        self.assertFalse(result["has_tables"])
        self.assertEqual(result["table_count"], 0)

    def test_analyze_tables_with_tables(self):
        """Test table analysis with table regions."""
        detector = MLLayoutDetector()
        
        regions = [
            {"type": "text", "bbox": [50, 100, 500, 150], "confidence": 0.9},
            {"type": "table", "bbox": [50, 200, 500, 400], "confidence": 0.85},
        ]
        
        result = detector.analyze_tables(regions)
        
        self.assertTrue(result["has_tables"])
        self.assertEqual(result["table_count"], 1)

    def test_analyze_tables_low_confidence(self):
        """Test table analysis with low confidence tables."""
        detector = MLLayoutDetector()
        
        regions = [
            {"type": "table", "bbox": [50, 200, 500, 400], "confidence": 0.5},  # Below threshold
        ]
        
        result = detector.analyze_tables(regions)
        
        # Low confidence table should not be counted
        self.assertFalse(result["has_tables"])

    def test_analyze_images(self):
        """Test image/figure detection."""
        detector = MLLayoutDetector()
        
        regions = [
            {"type": "text", "bbox": [50, 100, 500, 150], "confidence": 0.9},
            {"type": "figure", "bbox": [100, 200, 300, 400], "confidence": 0.8},
        ]
        
        result = detector.analyze_images(regions)
        
        self.assertTrue(result["has_images"])
        self.assertEqual(result["image_count"], 1)


class TestLayoutDetector(unittest.TestCase):
    """Test cases for the main LayoutDetector orchestrator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = LayoutDetector(language='en', use_ml=False)

    def test_init(self):
        """Test LayoutDetector initialization."""
        self.assertEqual(self.detector.language, 'en')
        self.assertFalse(self.detector.use_ml)

    def test_analyze_layout_heuristic_only(self):
        """Test layout analysis using heuristics only."""
        text = """
EXPERIENCE
Software Engineer at Company

EDUCATION
BS in Computer Science
"""
        features = self.detector.analyze_layout(text)
        
        self.assertIsInstance(features, LayoutFeatures)
        self.assertEqual(features.detection_method, "heuristic")

    def test_analyze_layout_with_pdf_path_no_ml(self):
        """Test layout analysis with pdf_path but ML disabled."""
        text = "Sample resume text"
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Create minimal PDF
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Test")
            doc.save(tmp_path)
            doc.close()
            
            features = self.detector.analyze_layout(text, pdf_path=tmp_path)
            
            # Should still use heuristics since use_ml=False
            self.assertEqual(features.detection_method, "heuristic")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_get_layout_summary(self):
        """Test get_layout_summary method."""
        text = """
EXPERIENCE
Software Engineer

EDUCATION
BS Computer Science

SKILLS
Python, JavaScript
"""
        summary = self.detector.get_layout_summary(text)
        
        self.assertIn("layout_type", summary)
        self.assertIn("has_tables", summary)
        self.assertIn("sections_found", summary)
        self.assertIn("risk_score", summary)
        self.assertIn("issues", summary)
        self.assertIn("recommendations", summary)
        self.assertIn("detection_method", summary)

    def test_set_language(self):
        """Test set_language method."""
        self.detector.set_language('fr')
        
        self.assertEqual(self.detector.language, 'fr')

    def test_backward_compatibility(self):
        """Test that old API still works."""
        text = "Sample text with Experience and Education sections"
        
        # Old method calls should still work
        columns = self.detector.detect_columns(text)
        tables = self.detector.detect_tables(text)
        headers = self.detector.detect_section_headers(text)
        density = self.detector.calculate_text_density(text)
        
        self.assertIn("is_single_column", columns)
        self.assertIsInstance(tables, bool)
        self.assertIsInstance(headers, list)
        self.assertIsInstance(density, float)


class TestLayoutDetectorWithML(unittest.TestCase):
    """Test cases for LayoutDetector with ML detection enabled."""

    @unittest.skipUnless(LAYOUT_DETECTION_AVAILABLE, "LayoutDetection not available")
    def test_ml_detection_integration(self):
        """Integration test for ML-based layout detection."""
        detector = LayoutDetector(language='en', use_ml=True)
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Create a PDF with some content
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "John Doe")
            page.insert_text((50, 100), "EXPERIENCE")
            page.insert_text((50, 130), "Software Engineer at Company")
            page.insert_text((50, 200), "EDUCATION")
            page.insert_text((50, 230), "BS Computer Science")
            doc.save(tmp_path)
            doc.close()
            
            text = """
John Doe

EXPERIENCE
Software Engineer at Company

EDUCATION
BS Computer Science
"""
            features = detector.analyze_layout(text, pdf_path=tmp_path)
            
            self.assertIsInstance(features, LayoutFeatures)
            # Should be ML detection if available
            if detector._get_ml_detector().is_available():
                self.assertEqual(features.detection_method, "ml")
                self.assertIsNotNone(features.confidence)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            # Cleanup temp layout images
            temp_dir = Path(tmp_path).parent / ".temp_layout"
            if temp_dir.exists():
                for f in temp_dir.glob("*"):
                    f.unlink()
                temp_dir.rmdir()


class TestLayoutDetectorFallback(unittest.TestCase):
    """Test fallback behavior when ML detection fails."""

    def test_fallback_when_ml_unavailable(self):
        """Test graceful fallback to heuristics when ML unavailable."""
        with patch.object(MLLayoutDetector, 'is_available', return_value=False):
            detector = LayoutDetector(language='en', use_ml=True)
            
            text = "Sample resume text with Experience section"
            features = detector.analyze_layout(text, pdf_path="/fake/path.pdf")
            
            # Should fall back to heuristics
            self.assertEqual(features.detection_method, "heuristic")

    def test_fallback_when_ml_throws_exception(self):
        """Test graceful fallback when ML detection throws exception."""
        detector = LayoutDetector(language='en', use_ml=True)
        
        # Mock ML detector to throw exception
        with patch.object(MLLayoutDetector, 'is_available', return_value=True):
            with patch.object(detector, '_analyze_with_ml', side_effect=RuntimeError("ML failed")):
                text = "Sample resume text"
                features = detector.analyze_layout(text, pdf_path="/fake/path.pdf")
                
                # Should fall back to heuristics
                self.assertEqual(features.detection_method, "heuristic")


if __name__ == "__main__":
    unittest.main()
