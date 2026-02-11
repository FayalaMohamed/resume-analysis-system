"""Tests for OCR module."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import fitz  # PyMuPDF

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parsers import PDFTextExtractor, extract_text_from_resume


class TestPDFTextExtractor(unittest.TestCase):
    """Test cases for PDFTextExtractor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = PDFTextExtractor()

    def test_init(self):
        """Test PDFTextExtractor initialization."""
        self.assertEqual(self.extractor.lang, "en")
        self.assertTrue(self.extractor.auto_detect_lang)

    def test_pdf_to_images_file_not_found(self):
        """Test pdf_to_images with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.extractor.pdf_to_images("nonexistent.pdf")

    def test_pdf_to_images_success(self):
        """Test pdf_to_images with valid PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Test content")
            doc.save(tmp_path)
            doc.close()

            result = self.extractor.pdf_to_images(tmp_path)

            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0].endswith('.png'))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_extract_text_from_image_no_ocr(self):
        """Test extract_text_from_image when OCR is not available."""
        extractor = PDFTextExtractor(use_paddle=False)
        
        with self.assertRaises(RuntimeError) as context:
            extractor.extract_text_from_image("test.png")
        
        self.assertIn("not available", str(context.exception))

    def test_extract_text_from_pdf(self):
        """Test extract_text_from_pdf method."""
        # Create a proper mock PDF document
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create a minimal PDF with some content using fitz
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Test text content")
            doc.save(tmp_path)
            doc.close()

            extractor = PDFTextExtractor()
            result = extractor.extract_text_from_pdf(tmp_path)

            self.assertIn("file_name", result)
            self.assertIn("full_text", result)
            self.assertIn("pages", result)
            self.assertIn("num_pages", result)
            self.assertEqual(result["num_pages"], 1)
            self.assertIn("Test text", result["full_text"])
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_cleanup_temp_images(self):
        """Test cleanup_temp_images method."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            Path(tmp_path).touch()
            temp_dir = Path(tmp_path).parent / ".temp_ocr"
            temp_dir.mkdir(exist_ok=True)

            # Create a fake temp image
            fake_image = temp_dir / f"{Path(tmp_path).stem}_page_1.png"
            fake_image.touch()

            self.extractor.cleanup_temp_images(tmp_path)

            # Image should be deleted
            self.assertFalse(fake_image.exists())
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if temp_dir is not None and temp_dir.exists():
                # Clean up any remaining files
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)


class TestExtractTextFromResume(unittest.TestCase):
    """Test cases for extract_text_from_resume function."""

    @patch("parsers.ocr.PDFTextExtractor")
    def test_extract_text_from_resume(self, mock_extractor_class):
        """Test the convenience function."""
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor

        expected_result = {
            "file_name": "test.pdf",
            "full_text": "Extracted text",
            "num_pages": 1,
            "pages": [],
        }
        mock_extractor.extract_text_from_pdf.return_value = expected_result

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            Path(tmp_path).touch()
            result = extract_text_from_resume(tmp_path)

            self.assertEqual(result, expected_result)
            mock_extractor.extract_text_from_pdf.assert_called_once()
            mock_extractor.cleanup_temp_images.assert_called_once()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestIntegration(unittest.TestCase):
    """Integration tests that require actual PDF files."""

    def test_with_real_pdf(self):
        """Test OCR with a real PDF if available."""
        resumes_dir = Path("resumes")

        if not resumes_dir.exists():
            self.skipTest("resumes/ directory not found")

        pdf_files = list(resumes_dir.glob("*.pdf"))
        if not pdf_files:
            self.skipTest("No PDF files found in resumes/ directory")

        # This test requires actual PaddleOCR to be installed
        try:
            from paddleocr import PaddleOCR

            result = extract_text_from_resume(pdf_files[0], cleanup=True)

            self.assertIn("file_name", result)
            self.assertIn("full_text", result)
            self.assertIn("pages", result)
            self.assertIsInstance(result["full_text"], str)
            self.assertGreater(result["num_pages"], 0)
        except ImportError:
            self.skipTest("PaddleOCR not installed")


if __name__ == "__main__":
    unittest.main()
