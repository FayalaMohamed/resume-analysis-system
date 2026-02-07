"""OCR module for extracting text from PDF resumes using PaddleOCR."""

import os
from pathlib import Path
from typing import List, Tuple, Union, Optional

import fitz  # PyMuPDF

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False


class PDFTextExtractor:
    """Extract text from PDF files using PaddleOCR with PyMuPDF fallback."""

    def __init__(self, lang: str = "en", use_paddle: bool = True, auto_detect_lang: bool = True):
        """Initialize PDF text extractor.

        Args:
            lang: Language code for OCR (e.g., 'en', 'es', 'fr')
            use_paddle: Whether to use PaddleOCR for OCR
            auto_detect_lang: Whether to auto-detect language from text
        """
        self.use_paddle = use_paddle and PADDLEOCR_AVAILABLE
        self.auto_detect_lang = auto_detect_lang
        self.lang = lang
        self.ocr = None

        if self.use_paddle:
            try:
                self.ocr = PaddleOCR(
                    use_textline_orientation=True,
                    lang=lang,
                )
            except Exception as e:
                print(f"Warning: Could not initialize PaddleOCR: {e}")
                self.use_paddle = False

    def pdf_to_images(self, pdf_path: Union[str, Path]) -> List[str]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create temp directory for images
        temp_dir = pdf_path.parent / ".temp_ocr"
        temp_dir.mkdir(exist_ok=True)

        # Open PDF and convert pages to images
        doc = fitz.open(str(pdf_path))
        image_paths = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render page at 2x resolution for better OCR
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)

            image_path = temp_dir / f"{pdf_path.stem}_page_{page_num + 1}.png"
            pix.save(str(image_path))
            image_paths.append(str(image_path))

        doc.close()
        return image_paths

    def extract_text_from_image(self, image_path: Union[str, Path]) -> List[Tuple]:
        """Extract text from a single image using PaddleOCR.

        Args:
            image_path: Path to the image file

        Returns:
            List of OCR results (bounding boxes, text, confidence)
        """
        if not self.use_paddle or self.ocr is None:
            raise RuntimeError("PaddleOCR is not available")

        result = self.ocr.predict(str(image_path))
        
        # Handle PaddleOCR 3.3.0+ result format
        texts = []
        for res in result:
            if hasattr(res, 'rec_texts'):
                # New PaddleOCR 3.3.0+ format - extract texts from result object
                for i, text in enumerate(res.rec_texts):
                    confidence = res.rec_scores[i] if hasattr(res, 'rec_scores') and i < len(res.rec_scores) else 0.0
                    # Create compatible format: [bbox, (text, confidence)]
                    bbox = res.rec_boxes[i] if hasattr(res, 'rec_boxes') and i < len(res.rec_boxes) else []
                    texts.append([bbox, (text, confidence)])
            elif isinstance(res, list):
                # Old format - list of detection results
                texts.extend(res)
            elif isinstance(res, dict) and 'rec_texts' in res:
                # Dictionary format
                for i, text in enumerate(res['rec_texts']):
                    confidence = res.get('rec_scores', [0.0])[i] if 'rec_scores' in res else 0.0
                    bbox = res.get('rec_boxes', [[]])[i] if 'rec_boxes' in res else []
                    texts.append([bbox, (text, confidence)])
        
        return texts

    def extract_text_from_page(self, page: fitz.Page) -> str:
        return page.get_text()

    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> dict:
        """Extract text from a PDF file.

        Uses PyMuPDF for text extraction and PaddleOCR as fallback
        for image-based content.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with extracted text and metadata
        """
        pdf_path = Path(pdf_path)

        # Open PDF
        doc = fitz.open(str(pdf_path))

        all_text = []
        page_results = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text using PyMuPDF first
            page_text = self.extract_text_from_page(page)

            page_results.append({
                "page": page_num + 1,
                "text": page_text,
                "lines": len(page_text.split("\n")),
            })
            all_text.append(page_text)

        doc.close()

        full_text = "\n\n".join(all_text)

        # Auto-detect language if enabled
        detected_lang = self.lang
        if self.auto_detect_lang:
            detected_lang = self.detect_language(full_text)

        return {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "num_pages": len(page_results),
            "full_text": full_text,
            "pages": page_results,
            "detected_language": detected_lang,
        }

    def extract_text_from_pdf_with_ocr(self, pdf_path: Union[str, Path], detect_lang_first: bool = True) -> dict:
        """Extract text from a PDF file using OCR.

        This method converts PDF pages to images and uses PaddleOCR
        for text extraction. Use this for image-based PDFs.

        Args:
            pdf_path: Path to the PDF file
            detect_lang_first: Whether to detect language from PDF text first

        Returns:
            Dictionary with extracted text and metadata
        """
        if not self.use_paddle:
            raise RuntimeError(
                "PaddleOCR is not available. "
                "Install it with: pip install paddleocr paddlepaddle"
            )

        pdf_path = Path(pdf_path)

        # Auto-detect language from PDF first
        detected_lang = self.lang
        if detect_lang_first and self.auto_detect_lang:
            detected_lang, _ = self.get_detected_language(pdf_path)
            if detected_lang != self.lang:
                self.update_language(detected_lang)

        # Convert PDF to images
        image_paths = self.pdf_to_images(pdf_path)

        all_text = []
        page_results = []

        for image_path in image_paths:
            # Extract text from each page using OCR
            ocr_results = self.extract_text_from_image(image_path)

            page_text = []
            for line in ocr_results:
                if not line:
                    continue
                try:
                    # Handle different OCR result formats
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        # Standard format: [bounding_box, (text, confidence)]
                        if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                            text = line[1][0]
                            confidence = line[1][1]
                        else:
                            # Alternative format: [bounding_box, text]
                            text = line[1]
                    elif isinstance(line, str):
                        # Simple text format
                        text = line
                    else:
                        continue
                    
                    if text and isinstance(text, str):
                        page_text.append(text)
                except (IndexError, TypeError):
                    # Skip malformed results
                    continue

            page_text_str = "\n".join(page_text)
            page_results.append({
                "page": len(page_results) + 1,
                "text": page_text_str,
                "lines": len(ocr_results),
            })
            all_text.append(page_text_str)

        full_text = "\n\n".join(all_text)

        return {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "num_pages": len(image_paths),
            "full_text": full_text,
            "pages": page_results,
            "detected_language": detected_lang,
        }

    def cleanup_temp_images(self, pdf_path: Union[str, Path]) -> None:
        pdf_path = Path(pdf_path)
        temp_dir = pdf_path.parent / ".temp_ocr"

        if temp_dir.exists():
            for image_file in temp_dir.glob(f"{pdf_path.stem}_page_*.png"):
                image_file.unlink()

    def detect_language(self, text: str) -> str:
        try:
            from .language_detector import LanguageDetector
            return LanguageDetector.detect(text, default=self.lang)
        except Exception:
            return self.lang

    def update_language(self, lang: str) -> None:
        if self.lang != lang:
            self.lang = lang
            if self.use_paddle:
                try:
                    self.ocr = PaddleOCR(
                        use_textline_orientation=True,
                        lang=lang,
                    )
                except Exception as e:
                    print(f"Warning: Could not reinitialize PaddleOCR with language {lang}: {e}")

    def get_detected_language(self, pdf_path: Union[str, Path]) -> tuple[str, str]:
        if not self.auto_detect_lang:
            return self.lang, self.lang

        try:
            result = self.extract_text_from_pdf(pdf_path)
            text = result.get('full_text', '')
            lang_code = self.detect_language(text)
            from .language_detector import LanguageDetector
            lang_name = LanguageDetector.get_language_name(lang_code)
            return lang_code, lang_name
        except Exception:
            return self.lang, self.lang


def extract_text_from_resume(
    pdf_path: Union[str, Path],
    use_ocr: bool = False,
    cleanup: bool = True,
) -> dict:
    """Convenience function to extract text from a resume PDF.

    Args:
        pdf_path: Path to the PDF file
        use_ocr: Whether to use OCR (for image-based PDFs)
        cleanup: Whether to remove temporary images after processing

    Returns:
        Dictionary with extracted text and metadata
    """
    extractor = PDFTextExtractor(use_paddle=use_ocr)

    if use_ocr:
        result = extractor.extract_text_from_pdf_with_ocr(pdf_path)
    else:
        result = extractor.extract_text_from_pdf(pdf_path)

    if cleanup:
        extractor.cleanup_temp_images(pdf_path)

    return result
