"""Enhanced OCR module with multi-engine fallback chain and confidence scoring.

This module provides robust text extraction from PDF resumes using:
- PaddleOCR (primary)
- Tesseract (fallback)
- EasyOCR (tertiary)
- PDF native text (last resort)

Features:
- Automatic fallback when primary engine fails
- Per-word confidence scores
- Low-confidence text flagging
- Uncertainty reporting
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import shutil

import fitz

logger = logging.getLogger(__name__)


class OCROption(Enum):
    """Available OCR engines."""
    PADDLEOCR = "paddleocr"
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PDF_NATIVE = "pdf_native"


@dataclass
class OCRResult:
    """Result from OCR extraction with confidence scoring."""
    text: str
    confidence: float
    word_confidences: List[Dict] = field(default_factory=list)
    engine_used: str = ""
    low_confidence_regions: List[Dict] = field(default_factory=list)
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    @property
    def overall_confidence(self) -> float:
        """Get overall confidence score (0-1)."""
        return self.confidence
    
    @property
    def needs_review(self) -> bool:
        """Check if result needs human review."""
        return self.confidence < 0.7 or len(self.low_confidence_regions) > 0


@dataclass
class OCREngineConfig:
    """Configuration for an OCR engine."""
    name: str
    enabled: bool = True
    timeout_seconds: int = 60
    confidence_threshold: float = 0.5
    priority: int = 0


class OCREngineBase:
    """Base class for OCR engines."""
    
    def __init__(self, config: Optional[OCREngineConfig] = None):
        self.config = config or OCREngineConfig(name="base")
        self._instance = None
    
    def is_available(self) -> bool:
        """Check if engine is available."""
        raise NotImplementedError
    
    def initialize(self) -> bool:
        """Initialize the OCR engine."""
        raise NotImplementedError
    
    def extract_text(self, image_path: str) -> OCRResult:
        """Extract text from an image."""
        raise NotImplementedError
    
    def cleanup(self):
        """Cleanup resources."""
        pass


class PaddleOCREngine(OCREngineBase):
    """PaddleOCR engine wrapper."""
    
    def __init__(self, config: Optional[OCREngineConfig] = None):
        super().__init__(config)
        self._instance = None
    
    def is_available(self) -> bool:
        try:
            from paddleocr import PaddleOCR
            return True
        except ImportError:
            return False
    
    def initialize(self) -> bool:
        if not self.is_available():
            return False
        try:
            from paddleocr import PaddleOCR
            self._instance = PaddleOCR(
                use_textline_orientation=True,
                lang='en',
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            return False
    
    def extract_text(self, image_path: str) -> OCRResult:
        import time
        start_time = time.time()
        
        if self._instance is None:
            if not self.initialize():
                return OCRResult(
                    text="",
                    confidence=0.0,
                    engine_used="paddleocr",
                    warnings=["Engine not initialized"]
                )
        
        try:
            result = self._instance.predict(image_path)
            
            words_data = []
            full_text_parts = []
            confidences = []
            
            for res in result:
                if hasattr(res, 'rec_texts') and hasattr(res, 'rec_scores'):
                    for i, text in enumerate(res.rec_texts):
                        confidence = res.rec_scores[i] if i < len(res.rec_scores) else 0.0
                        words_data.append({
                            "text": text,
                            "confidence": confidence,
                            "engine": "paddleocr"
                        })
                        full_text_parts.append(text)
                        confidences.append(confidence)
                elif isinstance(res, list):
                    for item in res:
                        if len(item) >= 2:
                            text = item[1][0] if isinstance(item[1], tuple) else str(item[1])
                            confidence = item[1][1] if isinstance(item[1], tuple) else 0.5
                            words_data.append({
                                "text": text,
                                "confidence": confidence,
                                "engine": "paddleocr"
                            })
                            full_text_parts.append(text)
                            confidences.append(confidence)
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            processing_time = (time.time() - start_time) * 1000
            
            low_conf_regions = [
                w for w in words_data 
                if w.get("confidence", 1.0) < self.config.confidence_threshold
            ]
            
            return OCRResult(
                text="\n".join(full_text_parts),
                confidence=avg_confidence,
                word_confidences=words_data,
                engine_used="paddleocr",
                low_confidence_regions=low_conf_regions,
                processing_time_ms=processing_time,
            )
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                engine_used="paddleocr",
                warnings=[f"Extraction failed: {str(e)}"]
            )


class TesseractEngine(OCREngineBase):
    """Tesseract engine wrapper."""
    
    def __init__(self, config: Optional[OCREngineConfig] = None):
        super().__init__(config)
        self._instance = None
    
    def is_available(self) -> bool:
        try:
            from pytesseract import image_to_string
            return True
        except ImportError:
            return False
    
    def initialize(self) -> bool:
        if not self.is_available():
            return False
        try:
            from pytesseract import pytesseract
            self._instance = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {e}")
            return False
    
    def extract_text(self, image_path: str) -> OCRResult:
        import time
        start_time = time.time()
        
        if self._instance is None:
            if not self.initialize():
                return OCRResult(
                    text="",
                    confidence=0.0,
                    engine_used="tesseract",
                    warnings=["Engine not initialized"]
                )
        
        try:
            from pytesseract import image_to_string, Output
            
            img = fitz.open(image_path) if image_path.endswith('.pdf') else fitz.open(image_path)
            if isinstance(img, fitz.Document):
                page = img[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_path = tempfile.mktemp(suffix='.png')
                pix.save(img_path)
                img = fitz.open(img_path)
            
            img_arr = self._image_to_array(img)
            data = image_to_string(img_arr, output_type=Output.DICT)
            
            words = data.get('text', '').split('\n')
            confidences = []
            words_data = []
            
            try:
                conf_data = image_to_string(img_arr, output_type=Output.DATAFRAME)
                if 'conf' in conf_data.columns:
                    for idx, row in conf_data.iterrows():
                        if pd.notna(row.get('conf')):
                            confidences.append(row['conf'] / 100.0)
            except Exception:
                pass
            
            if not confidences:
                avg_conf = 0.7
                confidences = [avg_conf] * len(words)
            
            for i, word in enumerate(words):
                if word.strip():
                    words_data.append({
                        "text": word,
                        "confidence": confidences[i] if i < len(confidences) else 0.7,
                        "engine": "tesseract"
                    })
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            processing_time = (time.time() - start_time) * 1000
            
            low_conf_regions = [
                w for w in words_data 
                if w.get("confidence", 1.0) < self.config.confidence_threshold
            ]
            
            return OCRResult(
                text=data.get('text', ''),
                confidence=avg_confidence,
                word_confidences=words_data,
                engine_used="tesseract",
                low_confidence_regions=low_conf_regions,
                processing_time_ms=processing_time,
            )
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                engine_used="tesseract",
                warnings=[f"Extraction failed: {str(e)}"]
            )
    
    def _image_to_array(self, img):
        """Convert fitz image to numpy array."""
        import numpy as np
        if hasattr(img, 'tobytes'):
            return np.frombuffer(img.tobytes(), dtype=np.uint8).reshape(img.height, img.width, 4)
        return img


class EasyOCREngine(OCREngineBase):
    """EasyOCR engine wrapper."""
    
    def __init__(self, config: Optional[OCREngineConfig] = None):
        super().__init__(config)
        self._instance = None
    
    def is_available(self) -> bool:
        try:
            import easyocr
            return True
        except ImportError:
            return False
    
    def initialize(self) -> bool:
        if not self.is_available():
            return False
        try:
            import easyocr
            self._instance = easyocr.Reader(['en'], gpu=False)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            return False
    
    def extract_text(self, image_path: str) -> OCRResult:
        import time
        start_time = time.time()
        
        if self._instance is None:
            if not self.initialize():
                return OCRResult(
                    text="",
                    confidence=0.0,
                    engine_used="easyocr",
                    warnings=["Engine not initialized"]
                )
        
        try:
            result = self._instance.readtext(image_path)
            
            words_data = []
            full_text_parts = []
            confidences = []
            
            for detection in result:
                bbox, text, conf = detection
                if conf is not None:
                    confidences.append(conf)
                words_data.append({
                    "text": text,
                    "confidence": conf if conf else 0.7,
                    "engine": "easyocr"
                })
                full_text_parts.append(text)
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            processing_time = (time.time() - start_time) * 1000
            
            low_conf_regions = [
                w for w in words_data 
                if w.get("confidence", 1.0) < self.config.confidence_threshold
            ]
            
            return OCRResult(
                text="\n".join(full_text_parts),
                confidence=avg_confidence,
                word_confidences=words_data,
                engine_used="easyocr",
                low_confidence_regions=low_conf_regions,
                processing_time_ms=processing_time,
            )
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                engine_used="easyocr",
                warnings=[f"Extraction failed: {str(e)}"]
            )


class PDFNativeEngine(OCREngineBase):
    """PDF native text extraction fallback."""
    
    def __init__(self, config: Optional[OCREngineConfig] = None):
        super().__init__(config)
    
    def is_available(self) -> bool:
        return True
    
    def initialize(self) -> bool:
        return True
    
    def extract_text(self, image_path: str) -> OCRResult:
        import time
        start_time = time.time()
        
        try:
            if image_path.endswith('.pdf'):
                doc = fitz.open(image_path)
                full_text = []
                for page in doc:
                    full_text.append(page.get_text())
                doc.close()
                text = "\n\n".join(full_text)
            else:
                doc = fitz.open(image_path)
                text = doc[0].get_text()
                doc.close()
            
            processing_time = (time.time() - start_time) * 1000
            
            words = text.split()
            words_data = [{"text": w, "confidence": 0.99, "engine": "pdf_native"} for w in words if w.strip()]
            
            avg_confidence = 0.99 if text else 0.0
            
            return OCRResult(
                text=text,
                confidence=avg_confidence,
                word_confidences=words_data,
                engine_used="pdf_native",
                processing_time_ms=processing_time,
            )
        except Exception as e:
            logger.error(f"PDF native extraction failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                engine_used="pdf_native",
                warnings=[f"Extraction failed: {str(e)}"]
            )


class EnhancedOCREngine:
    """Multi-engine OCR engine with automatic fallback."""
    
    DEFAULT_ENGINE_ORDER = [
        PaddleOCREngine(OCREngineConfig(name="paddleocr", priority=0)),
        TesseractEngine(OCREngineConfig(name="tesseract", priority=1)),
        EasyOCREngine(OCREngineConfig(name="easyocr", priority=2)),
        PDFNativeEngine(OCREngineConfig(name="pdf_native", priority=3)),
    ]
    
    def __init__(self, engine_order: Optional[List[OCREngineBase]] = None):
        self.engines = engine_order or self.DEFAULT_ENGINE_ORDER.copy()
        self._sort_engines()
    
    def _sort_engines(self):
        """Sort engines by priority."""
        self.engines.sort(key=lambda e: getattr(e.config, 'priority', 0))
    
    def get_available_engines(self) -> List[OCREngineBase]:
        """Get list of available engines."""
        available = []
        for engine in self.engines:
            if engine.is_available():
                available.append(engine)
        return available
    
    def extract_with_fallback(self, image_path: str) -> OCRResult:
        """Extract text using engines in priority order with fallback."""
        available = self.get_available_engines()
        
        if not available:
            logger.warning("No OCR engines available")
            return OCRResult(
                text="",
                confidence=0.0,
                engine_used="none",
                warnings=["No OCR engines available"]
            )
        
        best_result = None
        used_engine = None
        
        for engine in available:
            try:
                logger.info(f"Trying {engine.config.name} OCR...")
                result = engine.extract_text(image_path)
                
                if result.text and result.confidence > 0:
                    if best_result is None or result.confidence > best_result.confidence:
                        best_result = result
                        used_engine = engine.config.name
                    
                    if result.confidence >= 0.7:
                        break
            except Exception as e:
                logger.warning(f"Engine {engine.config.name} failed: {e}")
                continue
        
        if best_result:
            best_result.engine_used = used_engine or best_result.engine_used
            return best_result
        
        return OCRResult(
            text="",
            confidence=0.0,
            engine_used="none",
            warnings=["All OCR engines failed"]
        )
    
    def compare_engines(self, image_path: str) -> Dict[str, OCRResult]:
        """Run all available engines and compare results."""
        results = {}
        available = self.get_available_engines()
        
        for engine in available:
            try:
                result = engine.extract_text(image_path)
                results[engine.config.name] = result
            except Exception as e:
                logger.error(f"Engine comparison failed for {engine.config.name}: {e}")
        
        return results


class PDFTextExtractorEnhanced:
    """Enhanced PDF text extractor with multi-engine OCR."""
    
    def __init__(self, use_fallback_chain: bool = True, min_confidence: float = 0.7):
        self.use_fallback_chain = use_fallback_chain
        self.min_confidence = min_confidence
        self.ocr_engine = EnhancedOCREngine()
        self.temp_dirs: List[Path] = []
    
    def pdf_to_images(self, pdf_path: Union[str, Path]) -> List[str]:
        """Convert PDF pages to images for OCR."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        temp_dir = pdf_path.parent / ".temp_ocr_enhanced"
        temp_dir.mkdir(exist_ok=True)
        self.temp_dirs.append(temp_dir)
        
        doc = fitz.open(str(pdf_path))
        image_paths = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            
            image_path = temp_dir / f"{pdf_path.stem}_page_{page_num + 1}.png"
            pix.save(str(image_path))
            image_paths.append(str(image_path))
        
        doc.close()
        return image_paths
    
    def extract_text_from_pdf(self, pdf_path: Union[str, Path], use_ocr: bool = False) -> Dict[str, Any]:
        """Extract text from PDF using best available method."""
        pdf_path = Path(pdf_path)
        
        doc = fitz.open(str(pdf_path))
        
        all_text = []
        page_results = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            if use_ocr:
                temp_img_path = self._page_to_image(page, pdf_path, page_num)
                ocr_result = self.ocr_engine.extract_with_fallback(temp_img_path)
                
                page_text = ocr_result.text
                confidence = ocr_result.confidence
                
                if Path(temp_img_path).exists():
                    try:
                        os.remove(temp_img_path)
                    except:
                        pass
            else:
                page_text = page.get_text()
                confidence = 0.99
            
            page_results.append({
                "page": page_num + 1,
                "text": page_text,
                "confidence": confidence,
                "needs_review": confidence < self.min_confidence,
            })
            all_text.append(page_text)
        
        doc.close()
        
        full_text = "\n\n".join(all_text)
        avg_confidence = sum(p.get("confidence", 0) for p in page_results) / len(page_results) if page_results else 0
        needs_review = any(p.get("needs_review", False) for p in page_results)
        
        return {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "num_pages": len(page_results),
            "full_text": full_text,
            "pages": page_results,
            "overall_confidence": avg_confidence,
            "needs_review": needs_review,
            "extraction_method": "ocr_with_fallback" if use_ocr else "pdf_native",
        }
    
    def _page_to_image(self, page: fitz.Page, pdf_path: Path, page_num: int) -> str:
        """Convert a single page to image."""
        temp_dir = pdf_path.parent / ".temp_ocr_enhanced"
        temp_dir.mkdir(exist_ok=True)
        self.temp_dirs.append(temp_dir)
        
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        
        image_path = temp_dir / f"{pdf_path.stem}_page_{page_num + 1}.png"
        pix.save(str(image_path))
        
        return str(image_path)
    
    def get_uncertainty_report(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate uncertainty report for extraction results."""
        report = {
            "overall_confidence": extraction_result.get("overall_confidence", 0),
            "needs_review": extraction_result.get("needs_review", False),
            "low_confidence_pages": [],
            "suggestions": [],
        }
        
        for page_result in extraction_result.get("pages", []):
            if page_result.get("confidence", 1.0) < self.min_confidence:
                report["low_confidence_pages"].append({
                    "page": page_result.get("page"),
                    "confidence": page_result.get("confidence"),
                })
        
        if report["needs_review"]:
            report["suggestions"].append("Consider re-uploading a higher quality PDF")
        
        if len(report["low_confidence_pages"]) > len(extraction_result.get("pages", [])) * 0.5:
            report["suggestions"].append("Multiple pages have low confidence - file may be scanned image")
        
        return report
    
    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")
        self.temp_dirs = []


def extract_text_enhanced(
    pdf_path: Union[str, Path],
    use_ocr: bool = True,
    min_confidence: float = 0.7,
) -> Dict[str, Any]:
    """Convenience function for enhanced text extraction.
    
    Args:
        pdf_path: Path to PDF file
        use_ocr: Whether to use OCR (for image-based PDFs)
        min_confidence: Minimum confidence threshold (default 0.7)
    
    Returns:
        Dictionary with extraction results and confidence scores
    """
    extractor = PDFTextExtractorEnhanced(use_fallback_chain=True, min_confidence=min_confidence)
    
    try:
        result = extractor.extract_text_from_pdf(pdf_path, use_ocr=use_ocr)
        result["uncertainty_report"] = extractor.get_uncertainty_report(result)
        return result
    finally:
        extractor.cleanup()


def compare_ocr_engines(image_path: str) -> Dict[str, Dict[str, Any]]:
    """Compare results from all available OCR engines.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dictionary with results from each engine
    """
    engine = EnhancedOCREngine()
    results = engine.compare_engines(image_path)
    
    comparison = {}
    for engine_name, result in results.items():
        comparison[engine_name] = {
            "text_length": len(result.text),
            "confidence": result.confidence,
            "word_count": len(result.word_confidences),
            "low_confidence_words": len(result.low_confidence_regions),
            "processing_time_ms": result.processing_time_ms,
            "warnings": result.warnings,
        }
    
    return comparison