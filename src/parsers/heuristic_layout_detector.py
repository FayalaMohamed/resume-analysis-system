"""Heuristic-based layout detection using text pattern analysis."""

import re
from typing import Dict, List, Any, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import MULTILINGUAL_SECTIONS, LANGUAGE_INDICES, EXPECTED_SECTIONS

from .layout_types import LayoutFeatures


class HeuristicLayoutDetector:
    """Heuristic-based layout detection using text pattern analysis."""
    
    MULTILINGUAL_SECTIONS = MULTILINGUAL_SECTIONS
    LANGUAGE_INDICES = LANGUAGE_INDICES
    
    def __init__(self, language: str = 'auto'):
        """Initialize heuristic layout detector.
        
        Args:
            language: Language code for section detection
        """
        self.language = language
        self._compiled_patterns = None
        self._detected_language = None
    
    def _get_patterns_for_language(self, lang_code: str) -> List[re.Pattern]:
        """Get compiled patterns for a specific language."""
        if self._compiled_patterns is not None and self._detected_language == lang_code:
            return self._compiled_patterns

        patterns_to_use = []

        for section_patterns in self.MULTILINGUAL_SECTIONS.values():
            if lang_code == 'en':
                patterns_to_use.append(section_patterns[0])
            elif lang_code == 'auto':
                patterns_to_use.extend(section_patterns)
            else:
                patterns_to_use.append(section_patterns[0])
                if lang_code in self.LANGUAGE_INDICES and self.LANGUAGE_INDICES[lang_code] < len(section_patterns):
                    patterns_to_use.append(section_patterns[self.LANGUAGE_INDICES[lang_code]])

        self._compiled_patterns = [re.compile(p) for p in patterns_to_use]
        self._detected_language = lang_code
        return self._compiled_patterns
    
    def detect_columns(self, text: str) -> Dict[str, Any]:
        """Detect column layout using indentation variance."""
        lines = text.split('\n')
        left_positions = []
        
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                left_positions.append(indent)
        
        if not left_positions:
            return {"num_columns": 1, "is_single_column": True, "indent_variance": 0.0}
        
        from statistics import stdev
        
        std_indent = 0.0
        try:
            std_indent = stdev(left_positions)
            is_multi_column = std_indent > 10
        except:
            is_multi_column = False
        
        return {
            "num_columns": 2 if is_multi_column else 1,
            "is_single_column": not is_multi_column,
            "indent_variance": std_indent,
        }
    
    def detect_tables(self, text: str) -> bool:
        """Detect tables using pattern matching."""
        lines = text.split('\n')
        table_indicators = 0
        
        for line in lines:
            if '|' in line:
                table_indicators += 2
            if re.search(r'\s{3,}', line):
                table_indicators += 1
            if '\t' in line:
                table_indicators += 1
        
        threshold = max(3, len(lines) * 0.05)
        return table_indicators >= threshold
    
    def detect_section_headers(self, text: str, lang_code: Optional[str] = None) -> List[str]:
        """Detect section headers using regex patterns."""
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

            for pattern in patterns:
                if pattern.match(line_stripped):
                    headers.append(line_stripped)
                    break

            if line_stripped.isupper() and len(line_stripped) < 50:
                if line_stripped not in headers:
                    headers.append(line_stripped)

        return headers
    
    def calculate_text_density(self, text: str) -> float:
        """Calculate text density (characters per line)."""
        lines = [line for line in text.split('\n') if line.strip()]
        if not lines:
            return 0.0
        
        total_chars = sum(len(line) for line in lines)
        return total_chars / len(lines)
    
    def analyze_layout(self, text: str, lang_code: Optional[str] = None) -> LayoutFeatures:
        """Perform complete layout analysis using heuristics."""
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
        
        has_standard = any(
            any(std in h.lower() for std in EXPECTED_SECTIONS)
            for h in section_headers
        )
        if not has_standard:
            risk_score += 15
        
        return LayoutFeatures(
            is_single_column=column_info["is_single_column"],
            has_tables=has_tables,
            has_images=False,
            num_columns=column_info["num_columns"],
            text_density=text_density,
            avg_line_length=avg_line_length,
            section_headers=section_headers,
            layout_risk_score=min(risk_score, 100),
            detection_method="heuristic",
            confidence=None,
            table_regions=None,
            column_regions=None,
        )
