"""Layout detection module for analyzing resume structure.

This module provides both ML-based and heuristic-based layout detection
for resume documents. The ML detector is preferred when a PDF path is
available, with graceful fallback to heuristics.
"""

import re
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import MULTILINGUAL_SECTIONS, LANGUAGE_INDICES, EXPECTED_SECTIONS

# Import from submodules
from .layout_types import LayoutFeatures
from .ml_layout_detector import MLLayoutDetector, LAYOUT_DETECTION_AVAILABLE
from .heuristic_layout_detector import HeuristicLayoutDetector

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    'LayoutDetector',
    'LayoutFeatures',
    'MLLayoutDetector',
    'HeuristicLayoutDetector',
    'LAYOUT_DETECTION_AVAILABLE',
]


class LayoutDetector:
    """Main layout detector that orchestrates ML and heuristic detection.
    
    Uses LayoutDetection (ML) when a PDF path is provided and available,
    with graceful fallback to heuristic-based detection.
    """

    MULTILINGUAL_SECTIONS = MULTILINGUAL_SECTIONS
    LANGUAGE_INDICES = LANGUAGE_INDICES

    def __init__(self, language: str = 'auto', use_ml: bool = True):
        """Initialize LayoutDetector.

        Args:
            language: Language code ('auto' for automatic detection, or specific like 'en', 'fr', etc.)
            use_ml: Whether to attempt ML-based detection when pdf_path is provided
        """
        self.language = language
        self.use_ml = use_ml
        self._ml_detector = None
        self._heuristic_detector = HeuristicLayoutDetector(language)
        self._compiled_patterns = None
        self._detected_language = None
    
    def _get_ml_detector(self) -> MLLayoutDetector:
        """Get or create the ML detector (lazy loading)."""
        if self._ml_detector is None:
            lang = 'en' if self.language == 'auto' else self.language
            self._ml_detector = MLLayoutDetector(lang=lang)
        return self._ml_detector

    def _get_patterns_for_language(self, lang_code: str) -> List[re.Pattern]:
        """Get compiled patterns for a specific language."""
        return self._heuristic_detector._get_patterns_for_language(lang_code)

    def set_language(self, lang_code: str) -> None:
        """Set the language for section detection.
        
        Args:
            lang_code: Language code (e.g., 'en', 'fr', 'es', 'de', 'it', 'pt', 'auto')
        """
        self.language = lang_code
        self._heuristic_detector = HeuristicLayoutDetector(lang_code)
        self._ml_detector = None  # Reset to recreate with new language

    def detect_columns(self, text: str, page_width: float = 612) -> Dict[str, Any]:
        """Detect if resume has single or multi-column layout (heuristic)."""
        return self._heuristic_detector.detect_columns(text)
    
    def detect_tables(self, text: str) -> bool:
        """Detect if resume contains table structures (heuristic)."""
        return self._heuristic_detector.detect_tables(text)
    
    def detect_section_headers(self, text: str, lang_code: Optional[str] = None) -> List[str]:
        """Identify section headers in the resume."""
        return self._heuristic_detector.detect_section_headers(text, lang_code)
    
    def calculate_text_density(self, text: str) -> float:
        """Calculate text density (characters per line on average)."""
        return self._heuristic_detector.calculate_text_density(text)
    
    def _convert_pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to images for ML detection."""
        import fitz  # PyMuPDF
        
        pdf_path = Path(pdf_path)
        temp_dir = pdf_path.parent / ".temp_layout"
        temp_dir.mkdir(exist_ok=True)
        
        doc = fitz.open(str(pdf_path))
        image_paths = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render at 150 DPI for layout detection (lower than OCR for speed)
            mat = fitz.Matrix(150/72, 150/72)
            pix = page.get_pixmap(matrix=mat)
            
            image_path = temp_dir / f"{pdf_path.stem}_layout_page_{page_num + 1}.png"
            pix.save(str(image_path))
            image_paths.append(str(image_path))
        
        doc.close()
        return image_paths
    
    def _cleanup_temp_images(self, pdf_path: str) -> None:
        """Remove temporary layout detection images."""
        pdf_path = Path(pdf_path)
        temp_dir = pdf_path.parent / ".temp_layout"
        
        if temp_dir.exists():
            for image_file in temp_dir.glob(f"{pdf_path.stem}_layout_page_*.png"):
                try:
                    image_file.unlink()
                except Exception:
                    pass
    
    def _analyze_with_ml(self, text: str, pdf_path: str, lang_code: Optional[str] = None) -> LayoutFeatures:
        """Perform layout analysis using ML (LayoutDetection).
        
        Args:
            text: Extracted text from resume (for section headers and text metrics)
            pdf_path: Path to the PDF file
            lang_code: Optional language code
            
        Returns:
            LayoutFeatures with ML-based detection results
        """
        ml_detector = self._get_ml_detector()
        
        # Convert PDF to images
        image_paths = self._convert_pdf_to_images(pdf_path)
        
        try:
            all_regions = []
            page_width = 612  # Default US Letter width
            
            # Process each page
            for image_path in image_paths:
                try:
                    regions = ml_detector.detect_layout(image_path)
                    all_regions.extend(regions)
                    
                    # Get page width from first image
                    if page_width == 612:
                        try:
                            from PIL import Image
                            with Image.open(image_path) as img:
                                page_width = img.width
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"ML detection failed for page {image_path}: {e}")
                    continue
            
            # Analyze columns from all regions
            column_result = ml_detector.analyze_columns(all_regions, page_width)
            
            # Analyze tables from all regions
            table_result = ml_detector.analyze_tables(all_regions)
            
            # Analyze images/figures
            image_result = ml_detector.analyze_images(all_regions)
            
            # Get section headers using heuristics (more reliable for text patterns)
            section_headers = self._heuristic_detector.detect_section_headers(text, lang_code)
            
            # Calculate text metrics
            text_density = self._heuristic_detector.calculate_text_density(text)
            lines = [line for line in text.split('\n') if line.strip()]
            avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
            
            # Calculate risk score
            risk_score = 0
            if not column_result['is_single_column']:
                risk_score += 25
            if table_result['has_tables']:
                risk_score += 20
            if image_result['has_images']:
                risk_score += 15
            
            has_standard = any(
                any(std in h.lower() for std in EXPECTED_SECTIONS)
                for h in section_headers
            )
            if not has_standard:
                risk_score += 15
            
            # Calculate overall confidence
            confidences = [
                column_result.get('confidence', 0.5),
                table_result.get('confidence', 0.5) if table_result['has_tables'] else 0.9,
            ]
            avg_confidence = sum(confidences) / len(confidences)
            
            return LayoutFeatures(
                is_single_column=column_result['is_single_column'],
                has_tables=table_result['has_tables'],
                has_images=image_result['has_images'],
                num_columns=column_result['num_columns'],
                text_density=text_density,
                avg_line_length=avg_line_length,
                section_headers=section_headers,
                layout_risk_score=min(risk_score, 100),
                detection_method="ml",
                confidence=avg_confidence,
                table_regions=table_result.get('table_regions'),
                column_regions=column_result.get('column_regions'),
            )
            
        finally:
            # Cleanup temp images
            self._cleanup_temp_images(pdf_path)
    
    def analyze_layout(self, text: str, pdf_path: Optional[str] = None, 
                       lang_code: Optional[str] = None) -> LayoutFeatures:
        """Perform complete layout analysis on resume.
        
        Uses ML detection if pdf_path is provided and LayoutDetection is available,
        otherwise falls back to heuristic detection.
        
        Args:
            text: Extracted text from resume
            pdf_path: Optional path to PDF file for ML-based detection
            lang_code: Optional language code to use for section detection
            
        Returns:
            LayoutFeatures with analysis results
        """
        # Try ML detection if enabled and pdf_path provided
        if self.use_ml and pdf_path and self._get_ml_detector().is_available():
            try:
                logger.info("Using ML-based layout detection (LayoutDetection)")
                return self._analyze_with_ml(text, pdf_path, lang_code)
            except Exception as e:
                logger.warning(f"ML layout detection failed, falling back to heuristics: {e}")
        
        # Fall back to heuristic detection
        logger.debug("Using heuristic-based layout detection")
        return self._heuristic_detector.analyze_layout(text, lang_code)
    
    def get_layout_summary(self, text: str, pdf_path: Optional[str] = None) -> Dict[str, Any]:
        """Get a summary of layout analysis results.
        
        Args:
            text: Extracted text from resume
            pdf_path: Optional path to PDF file for ML-based detection
            
        Returns:
            Dictionary with layout summary, issues, and recommendations
        """
        features = self.analyze_layout(text, pdf_path)
        
        issues = []
        if not features.is_single_column:
            issues.append("Multi-column layout detected - may cause ATS parsing issues")
        if features.has_tables:
            issues.append("Table structures detected - ATS may not read correctly")
        if features.has_images:
            issues.append("Images/figures detected - content may not be parsed")
        if features.layout_risk_score > 50:
            issues.append("High layout complexity - recommend simplification")
        
        summary = {
            "layout_type": "Single-column" if features.is_single_column else f"{features.num_columns}-column",
            "has_tables": features.has_tables,
            "has_images": features.has_images,
            "sections_found": features.section_headers,
            "risk_score": features.layout_risk_score,
            "detection_method": features.detection_method,
            "issues": issues,
            "recommendations": self._generate_recommendations(features),
        }
        
        if features.confidence is not None:
            summary["confidence"] = features.confidence
        
        return summary
    
    def _generate_recommendations(self, features: LayoutFeatures) -> List[str]:
        """Generate recommendations based on detected layout issues."""
        recommendations = []
        
        if not features.is_single_column:
            recommendations.append("Convert to single-column layout for better ATS compatibility")
        
        if features.has_tables:
            recommendations.append("Replace tables with simple bullet points or plain text")
        
        if features.has_images:
            recommendations.append("Remove decorative images or replace with text descriptions")
        
        if len(features.section_headers) < 3:
            recommendations.append("Add clear section headers (Experience, Education, Skills)")
        
        if features.text_density > 100:
            recommendations.append("Consider breaking up dense text blocks")
        
        return recommendations
