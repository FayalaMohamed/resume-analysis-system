"""OCR module for extracting text from PDF resumes using PaddleOCR."""

import os
from pathlib import Path
from typing import List, Tuple, Union

import fitz  # PyMuPDF

# Try to import PaddleOCR, but make it optional
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False


class PDFTextExtractor:
    """Extract text from PDF files using PaddleOCR with PyMuPDF fallback."""

    def __init__(self, lang: str = "en", use_paddle: bool = True):
        """Initialize the PDF text extractor.

        Args:
            lang: Language code for OCR (default: 'en' for English)
            use_paddle: Whether to use PaddleOCR (if available)
        """
        self.lang = lang
        self.use_paddle = use_paddle and PADDLEOCR_AVAILABLE
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
        """Convert PDF pages to images.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of paths to the generated image files
        """
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
        return result[0] if result and result[0] else []

    def extract_text_from_page(self, page: fitz.Page) -> str:
        """Extract text from a PDF page using PyMuPDF.

        Args:
            page: PyMuPDF page object

        Returns:
            Extracted text from the page
        """
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

        return {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "num_pages": len(page_results),
            "full_text": full_text,
            "pages": page_results,
        }

    def extract_text_from_pdf_with_ocr(self, pdf_path: Union[str, Path]) -> dict:
        """Extract text from a PDF file using OCR.

        This method converts PDF pages to images and uses PaddleOCR
        for text extraction. Use this for image-based PDFs.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with extracted text and metadata
        """
        if not self.use_paddle:
            raise RuntimeError(
                "PaddleOCR is not available. "
                "Install it with: pip install paddleocr paddlepaddle"
            )

        pdf_path = Path(pdf_path)

        # Convert PDF to images
        image_paths = self.pdf_to_images(pdf_path)

        all_text = []
        page_results = []

        for image_path in image_paths:
            # Extract text from each page using OCR
            ocr_results = self.extract_text_from_image(image_path)

            page_text = []
            for line in ocr_results:
                if line:
                    # line format: [bounding_box, (text, confidence)]
                    text = line[1][0]
                    confidence = line[1][1]
                    page_text.append(text)

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
        }

    def cleanup_temp_images(self, pdf_path: Union[str, Path]) -> None:
        """Remove temporary images created during PDF processing.

        Args:
            pdf_path: Path to the PDF file
        """
        pdf_path = Path(pdf_path)
        temp_dir = pdf_path.parent / ".temp_ocr"

        if temp_dir.exists():
            for image_file in temp_dir.glob(f"{pdf_path.stem}_page_*.png"):
                image_file.unlink()


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


if __name__ == "__main__":
    # Simple test with first available PDF
    resumes_dir = Path("resumes")

    if resumes_dir.exists():
        pdf_files = list(resumes_dir.glob("*.pdf"))
        if pdf_files:
            test_pdf = pdf_files[0]
            print(f"Testing OCR with: {test_pdf.name}")
            print(f"PaddleOCR available: {PADDLEOCR_AVAILABLE}")

            result = extract_text_from_resume(test_pdf, use_ocr=False, cleanup=False)

            print(f"\nFile: {result['file_name']}")
            print(f"Pages: {result['num_pages']}")
            print(f"\n--- Extracted Text (first 500 chars) ---")
            print(result["full_text"][:500])
            print("\n--- ... ---")
        else:
            print("No PDF files found in resumes/ directory")
    else:
        print("resumes/ directory not found")
