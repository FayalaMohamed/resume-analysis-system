"""Layout detection module for analyzing resume structure."""

import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import MULTILINGUAL_SECTIONS, LANGUAGE_INDICES, EXPECTED_SECTIONS


@dataclass
class LayoutFeatures:
    """Features extracted from resume layout analysis."""
    is_single_column: bool
    has_tables: bool
    has_images: bool
    num_columns: int
    text_density: float
    avg_line_length: float
    section_headers: List[str]
    layout_risk_score: float


class LayoutDetector:
    """Detect layout patterns and potential ATS issues in resumes."""

    MULTILINGUAL_SECTIONS = MULTILINGUAL_SECTIONS
    LANGUAGE_INDICES = LANGUAGE_INDICES

    def __init__(self, language: str = 'auto'):
        """Initialize LayoutDetector.

        Args:
            language: Language code ('auto' for automatic detection, or specific like 'en', 'fr', etc.)
        """
        self.language = language
        self._compiled_patterns = None  # Lazy loading
        self._detected_language = None

    def _get_patterns_for_language(self, lang_code: str) -> List[re.Pattern]:
        """Get compiled patterns for a specific language.
        
        Returns patterns for the specified language plus English as fallback.
        
        Args:
            lang_code: Language code (e.g., 'en', 'fr', 'es')
            
        Returns:
            List of compiled regex patterns
        """
        if self._compiled_patterns is not None and self._detected_language == lang_code:
            return self._compiled_patterns

        patterns_to_use = []

        for section_patterns in self.MULTILINGUAL_SECTIONS.values():
            if lang_code == 'en':
                # For English, just use English patterns
                patterns_to_use.append(section_patterns[0])  # English is always first
            elif lang_code == 'auto':
                # Use all patterns when in auto mode
                patterns_to_use.extend(section_patterns)
            else:
                # For other languages, add English first, then the specific language
                patterns_to_use.append(section_patterns[0])  # English

                # Add patterns for the detected language based on position
                if lang_code in self.LANGUAGE_INDICES and self.LANGUAGE_INDICES[lang_code] < len(section_patterns):
                    patterns_to_use.append(section_patterns[self.LANGUAGE_INDICES[lang_code]])

        self._compiled_patterns = [re.compile(p) for p in patterns_to_use]
        self._detected_language = lang_code
        return self._compiled_patterns

    def set_language(self, lang_code: str) -> None:
        """Set the language for section detection.
        
        Args:
            lang_code: Language code (e.g., 'en', 'fr', 'es', 'de', 'it', 'pt', 'auto')
        """
        self.language = lang_code
        self._compiled_patterns = None  # Reset cached patterns
        self._detected_language = None

    def detect_columns(self, text: str, page_width: float = 612) -> Dict[str, Any]:
        """Detect if resume has single or multi-column layout.
        
        Args:
            text: Extracted text from resume
            page_width: Width of page in points (default: 612 for US Letter)
            
        Returns:
            Dictionary with column detection results
        """
        lines = text.split('\n')
        
        # Analyze line positions to detect columns
        # This is a heuristic based on indentation patterns
        left_positions = []
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                left_positions.append(indent)
        
        if not left_positions:
            return {"num_columns": 1, "is_single_column": True}
        
        # Check for bimodal distribution (indicates two columns)
        from statistics import stdev
        
        # If significant variation in left margins, might be multi-column
        std_indent = 0.0
        try:
            std_indent = stdev(left_positions)
            is_multi_column = std_indent > 10  # Threshold for variation
        except:
            is_multi_column = False
        
        num_columns = 2 if is_multi_column else 1
        
        return {
            "num_columns": num_columns,
            "is_single_column": num_columns == 1,
            "indent_variance": std_indent,
        }
    
    def detect_tables(self, text: str) -> bool:
        """Detect if resume contains table structures.
        
        Args:
            text: Extracted text content
            
        Returns:
            True if tables detected
        """
        # Table detection heuristics
        lines = text.split('\n')
        
        # Look for patterns that suggest tables:
        # 1. Multiple spaces/tabs in consistent positions
        # 2. Pipe characters or box drawing characters
        # 3. Consistent columnar data patterns
        
        table_indicators = 0
        
        for line in lines:
            # Check for pipe characters (markdown tables)
            if '|' in line:
                table_indicators += 2
            
            # Check for multiple consecutive spaces (table-like formatting)
            if re.search(r'\s{3,}', line):
                table_indicators += 1
            
            # Check for tab characters
            if '\t' in line:
                table_indicators += 1
        
        # If more than 5% of lines have table indicators
        threshold = max(3, len(lines) * 0.05)
        return table_indicators >= threshold
    
    def detect_section_headers(self, text: str, lang_code: Optional[str] = None) -> List[str]:
        """Identify section headers in the resume.
        
        Args:
            text: Extracted text content
            lang_code: Language code to use for detection (overrides self.language)
            
        Returns:
            List of detected section headers
        """
        if lang_code:
            patterns = self._get_patterns_for_language(lang_code)
        elif self.language == 'auto':
            patterns = self._get_patterns_for_language('auto')
        else:
            patterns = self._get_patterns_for_language(self.language)

        lines = text.split('\n')
        headers = []

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check if line matches any section pattern
            for pattern in patterns:
                if pattern.match(line_stripped):
                    headers.append(line_stripped)
                    break

            # Also detect by formatting (all caps, followed by separator)
            if line_stripped.isupper() and len(line_stripped) < 50:
                if line_stripped not in headers:
                    headers.append(line_stripped)

        return headers
    
    def calculate_text_density(self, text: str) -> float:
        """Calculate text density (characters per line on average).
        
        Args:
            text: Extracted text content
            
        Returns:
            Text density score
        """
        lines = [line for line in text.split('\n') if line.strip()]
        if not lines:
            return 0.0
        
        total_chars = sum(len(line) for line in lines)
        return total_chars / len(lines)
    
    def analyze_layout(self, text: str, lang_code: Optional[str] = None) -> LayoutFeatures:
        """Perform complete layout analysis on resume text.
        
        Args:
            text: Extracted text from resume
            lang_code: Optional language code to use for section detection
            
        Returns:
            LayoutFeatures with analysis results
        """
        column_info = self.detect_columns(text)
        
        has_tables = self.detect_tables(text)
        
        section_headers = self.detect_section_headers(text, lang_code)
        
        text_density = self.calculate_text_density(text)
        
        lines = [line for line in text.split('\n') if line.strip()]
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        
        risk_score = 0
        
        if not column_info["is_single_column"]:
            risk_score += 25
        
        if has_tables:
            risk_score += 20
        
        standard_sections = EXPECTED_SECTIONS
        has_standard = any(
            any(std in h.lower() for std in standard_sections)
            for h in section_headers
        )
        if not has_standard:
            risk_score += 15
        
        return LayoutFeatures(
            is_single_column=column_info["is_single_column"],
            has_tables=has_tables,
            has_images=False,  # Would need image analysis
            num_columns=column_info["num_columns"],
            text_density=text_density,
            avg_line_length=avg_line_length,
            section_headers=section_headers,
            layout_risk_score=min(risk_score, 100),
        )
    
    def get_layout_summary(self, text: str) -> Dict[str, Any]:
        features = self.analyze_layout(text)
        
        issues = []
        if not features.is_single_column:
            issues.append("Multi-column layout detected - may cause ATS parsing issues")
        if features.has_tables:
            issues.append("Table structures detected - ATS may not read correctly")
        if features.layout_risk_score > 50:
            issues.append("High layout complexity - recommend simplification")
        
        return {
            "layout_type": "Single-column" if features.is_single_column else f"{features.num_columns}-column",
            "has_tables": features.has_tables,
            "sections_found": features.section_headers,
            "risk_score": features.layout_risk_score,
            "issues": issues,
            "recommendations": self._generate_recommendations(features),
        }
    
    def _generate_recommendations(self, features: LayoutFeatures) -> List[str]:
        recommendations = []
        
        if not features.is_single_column:
            recommendations.append("Convert to single-column layout for better ATS compatibility")
        
        if features.has_tables:
            recommendations.append("Replace tables with simple bullet points or plain text")
        
        if len(features.section_headers) < 3:
            recommendations.append("Add clear section headers (Experience, Education, Skills)")
        
        if features.text_density > 100:
            recommendations.append("Consider breaking up dense text blocks")
        
        return recommendations
