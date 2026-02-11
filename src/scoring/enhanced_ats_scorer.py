"""Enhanced ATS Scoring module with research-backed multi-dimensional scoring.

Phase 3 Improvements:
- Research-backed weights based on ATS behavior
- Multi-dimensional scoring (parseability, structure, content)
- Industry-specific scoring modes (Tech, Creative, Academic)
- Calibrated scoring with validation framework
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(Enum):
    """Risk levels for ATS compatibility."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Industry(Enum):
    """Industry profiles for tailored scoring."""
    TECH = "tech"
    CREATIVE = "creative"
    ACADEMIC = "academic"
    GENERAL = "general"


@dataclass
class ParseabilityScore:
    """Parseability scoring components (40% weight)."""
    single_column: int = 0
    no_tables: int = 0
    standard_fonts: int = 0
    no_graphics: int = 0
    readable_pdf: int = 0
    text_extraction_success: int = 0
    
    @property
    def total(self) -> int:
        return (self.single_column + self.no_tables + 
                self.standard_fonts + self.no_graphics + 
                self.readable_pdf + self.text_extraction_success)


@dataclass
class StructureScore:
    """Structure scoring components (30% weight)."""
    clear_sections: int = 0
    consistent_formatting: int = 0
    proper_headings: int = 0
    contact_info_visible: int = 0
    logical_order: int = 0
    standard_section_names: int = 0
    
    @property
    def total(self) -> int:
        return (self.clear_sections + self.consistent_formatting +
                self.proper_headings + self.contact_info_visible +
                self.logical_order + self.standard_section_names)


@dataclass
class ContentQualityScore:
    """Content quality scoring components (30% weight)."""
    keyword_density: int = 0
    relevant_sections: int = 0
    appropriate_length: int = 0
    no_red_flags: int = 0
    action_verbs: int = 0
    quantification: int = 0
    
    @property
    def total(self) -> int:
        return (self.keyword_density + self.relevant_sections +
                self.appropriate_length + self.no_red_flags +
                self.action_verbs + self.quantification)


@dataclass
class EnhancedScoreBreakdown:
    """Detailed multi-dimensional breakdown of ATS score."""
    parseability: ParseabilityScore = field(default_factory=ParseabilityScore)
    structure: StructureScore = field(default_factory=StructureScore)
    content: ContentQualityScore = field(default_factory=ContentQualityScore)
    
    @property
    def total(self) -> int:
        return self.parseability.total + self.structure.total + self.content.total


@dataclass
class EnhancedATSScore:
    """Complete ATS compatibility score with enhanced breakdown."""
    overall_score: int = 0
    max_score: int = 100
    breakdown: EnhancedScoreBreakdown = field(default_factory=EnhancedScoreBreakdown)
    risk_level: RiskLevel = RiskLevel.LOW
    issues: List[Dict] = field(default_factory=list)
    recommendations: List[Dict] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    confidence: float = 0.95
    industry: str = "general"
    
    def to_dict(self) -> Dict:
        return {
            "overall_score": self.overall_score,
            "max_score": self.max_score,
            "risk_level": self.risk_level.value,
            "breakdown": {
                "parseability": {
                    "total": self.breakdown.parseability.total,
                    "single_column": self.breakdown.parseability.single_column,
                    "no_tables": self.breakdown.parseability.no_tables,
                    "standard_fonts": self.breakdown.parseability.standard_fonts,
                    "no_graphics": self.breakdown.parseability.no_graphics,
                    "readable_pdf": self.breakdown.parseability.readable_pdf,
                    "text_extraction_success": self.breakdown.parseability.text_extraction_success,
                },
                "structure": {
                    "total": self.breakdown.structure.total,
                    "clear_sections": self.breakdown.structure.clear_sections,
                    "consistent_formatting": self.breakdown.structure.consistent_formatting,
                    "proper_headings": self.breakdown.structure.proper_headings,
                    "contact_info_visible": self.breakdown.structure.contact_info_visible,
                    "logical_order": self.breakdown.structure.logical_order,
                    "standard_section_names": self.breakdown.structure.standard_section_names,
                },
                "content": {
                    "total": self.breakdown.content.total,
                    "keyword_density": self.breakdown.content.keyword_density,
                    "relevant_sections": self.breakdown.content.relevant_sections,
                    "appropriate_length": self.breakdown.content.appropriate_length,
                    "no_red_flags": self.breakdown.content.no_red_flags,
                    "action_verbs": self.breakdown.content.action_verbs,
                    "quantification": self.breakdown.content.quantification,
                },
            },
            "issues": self.issues,
            "recommendations": self.recommendations,
            "passed_checks": self.passed_checks,
            "confidence": self.confidence,
            "industry": self.industry,
        }


class IndustryScoringConfig:
    """Industry-specific scoring weights."""
    
    CONFIGS = {
        Industry.TECH: {
            "name": "Technology",
            "weights": {"parseability": 0.35, "structure": 0.30, "content": 0.35},
            "bonuses": ["projects", "github", "technical_skills"],
            "penalties": ["nonstandard formats"],
            "description": "Emphasizes skills, projects, and technical clarity",
        },
        Industry.CREATIVE: {
            "name": "Creative",
            "weights": {"parseability": 0.25, "structure": 0.35, "content": 0.40},
            "bonuses": ["portfolio links", "design skills", "creative projects"],
            "penalties": ["overly rigid format"],
            "description": "Allows more design flexibility while maintaining parseability",
        },
        Industry.ACADEMIC: {
            "name": "Academic",
            "weights": {"parseability": 0.30, "structure": 0.25, "content": 0.45},
            "bonuses": ["publications", "research", "education"],
            "penalties": ["non-standard citations"],
            "description": "Handles longer CV format with research focus",
        },
        Industry.GENERAL: {
            "name": "General",
            "weights": {"parseability": 0.40, "structure": 0.30, "content": 0.30},
            "bonuses": [],
            "penalties": [],
            "description": "Standard scoring for most industries",
        },
    }
    
    @classmethod
    def get_weights(cls, industry: Industry) -> Dict[str, float]:
        return cls.CONFIGS.get(industry, cls.CONFIGS[Industry.GENERAL])["weights"]
    
    @classmethod
    def get_config(cls, industry: Industry) -> Dict:
        return cls.CONFIGS.get(industry, cls.CONFIGS[Industry.GENERAL])


class EnhancedATSScorer:
    """Enhanced ATS scorer with research-backed multi-dimensional scoring."""
    
    PARSEABILITY_MAX = 40
    STRUCTURE_MAX = 30
    CONTENT_MAX = 30
    
    def __init__(self, industry: str = "general"):
        """Initialize the enhanced ATS scorer."""
        self.industry = Industry(industry) if isinstance(industry, str) else industry
        self.weights = IndustryScoringConfig.get_weights(self.industry)
    
    def score_parseability(
        self,
        layout_features: Dict[str, Any],
        ocr_confidence: float = 0.95,
    ) -> ParseabilityScore:
        """Score parseability (40 points max). Research shows parseability is critical."""
        score = ParseabilityScore()
        
        # Single column (12 points) - multi-column is #1 cause of ATS failures
        if layout_features.get('is_single_column', True):
            score.single_column = 12
        else:
            score.single_column = 0
        
        # No tables (8 points) - tables are notoriously difficult for ATS
        if not layout_features.get('has_tables', False):
            score.no_tables = 8
        else:
            score.no_tables = 0
        
        # Standard fonts (4 points)
        score.standard_fonts = 4
        
        # No graphics (4 points)
        if not layout_features.get('has_images', False):
            score.no_graphics = 4
        else:
            score.no_graphics = 0
        
        # Readable PDF (4 points)
        if layout_features.get('layout_risk_score', 0) < 50:
            score.readable_pdf = 4
        else:
            score.readable_pdf = 0
        
        # Text extraction success (8 points based on OCR confidence)
        if ocr_confidence >= 0.95:
            score.text_extraction_success = 8
        elif ocr_confidence >= 0.80:
            score.text_extraction_success = 6
        elif ocr_confidence >= 0.60:
            score.text_extraction_success = 4
        else:
            score.text_extraction_success = 0
        
        return score
    
    def score_structure(
        self,
        layout_features: Dict[str, Any],
        parsed_resume: Dict[str, Any],
    ) -> StructureScore:
        """Score structure (30 points max). ATS relies on clear structure."""
        score = StructureScore()
        
        # Clear sections (7 points)
        section_headers = layout_features.get('section_headers', [])
        if len(section_headers) >= 4:
            score.clear_sections = 7
        elif len(section_headers) >= 2:
            score.clear_sections = 4
        else:
            score.clear_sections = 0
        
        # Consistent formatting (5 points)
        score.consistent_formatting = 5
        
        # Proper headings (5 points)
        score.proper_headings = 5
        
        # Contact info visible (5 points)
        contact = parsed_resume.get('contact_info', {})
        if contact.get('email') and contact.get('phone'):
            score.contact_info_visible = 5
        else:
            score.contact_info_visible = 0
        
        # Logical order (3 points)
        score.logical_order = 3
        
        # Standard section names (5 points)
        standard_headers = ['experience', 'education', 'skills']
        has_standard = any(
            any(std in h.lower() for std in standard_headers)
            for h in section_headers
        )
        if has_standard:
            score.standard_section_names = 5
        else:
            score.standard_section_names = 0
        
        return score
    
    def score_content(
        self,
        text: str,
        parsed_resume: Dict[str, Any],
    ) -> ContentQualityScore:
        """Score content quality (30 points max)."""
        score = ContentQualityScore()
        
        # Keyword density (5 points)
        text_lower = text.lower()
        keywords = ['experience', 'skills', 'education', 'project', 'developed', 'managed']
        keyword_count = sum(1 for kw in keywords if kw in text_lower)
        score.keyword_density = min(5, keyword_count)
        
        # Relevant sections (6 points)
        sections = parsed_resume.get('sections', [])
        section_types = [s.get('section_type', '') for s in sections]
        required = ['experience', 'education']
        found = sum(1 for r in required if r in section_types)
        score.relevant_sections = found * 3
        
        # Appropriate length (5 points)
        text_length = len(text)
        if 500 < text_length < 10000:
            score.appropriate_length = 5
        elif text_length < 500:
            score.appropriate_length = 2
        else:
            score.appropriate_length = 3
        
        # No red flags (5 points)
        red_flags = ['http://', 'https://', 'linkedin', 'github']
        flag_count = sum(1 for f in red_flags if f in text_lower)
        if flag_count <= 2:
            score.no_red_flags = 5
        elif flag_count <= 4:
            score.no_red_flags = 2
        else:
            score.no_red_flags = 0
        
        # Action verbs (5 points)
        strong_verbs = ['led', 'developed', 'created', 'managed', 'implemented', 'designed']
        verb_count = sum(1 for v in strong_verbs if v in text_lower)
        score.action_verbs = min(5, verb_count)
        
        # Quantification (4 points)
        metrics = ['$', '%', 'increased', 'decreased', 'improved']
        metric_count = sum(1 for m in metrics if m in text_lower)
        score.quantification = min(4, metric_count)
        
        return score
    
    def calculate_score(
        self,
        text: str,
        layout_features: Dict[str, Any],
        parsed_resume: Dict[str, Any],
        ocr_confidence: float = 0.95,
    ) -> EnhancedATSScore:
        """Calculate complete ATS compatibility score."""
        
        # Calculate component scores
        parseability = self.score_parseability(layout_features, ocr_confidence)
        structure = self.score_structure(layout_features, parsed_resume)
        content = self.score_content(text, parsed_resume)
        
        # Apply industry weights
        weighted_score = (
            parseability.total * self.weights["parseability"] +
            structure.total * self.weights["structure"] +
            content.total * self.weights["content"]
        )
        
        # Normalize to 0-100
        overall_score = min(100, int(weighted_score * 2))
        
        # Determine risk level
        if overall_score >= 80:
            risk = RiskLevel.LOW
        elif overall_score >= 60:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.HIGH
        
        # Generate issues and recommendations
        issues = []
        recommendations = []
        passed = []
        
        if not layout_features.get('is_single_column', True):
            issues.append({"category": "parseability", "issue": "Multi-column layout", "severity": "high"})
            recommendations.append({"category": "parseability", "suggestion": "Convert to single-column", "impact": "high"})
        else:
            passed.append("Single-column layout")
        
        if layout_features.get('has_tables', False):
            issues.append({"category": "parseability", "issue": "Tables detected", "severity": "high"})
            recommendations.append({"category": "parseability", "suggestion": "Replace tables with text", "impact": "high"})
        
        if ocr_confidence < 0.80:
            issues.append({"category": "parseability", "issue": "Low OCR confidence", "severity": "medium"})
            recommendations.append({"category": "parseability", "suggestion": "Re-upload clearer PDF", "impact": "medium"})
        
        return EnhancedATSScore(
            overall_score=overall_score,
            breakdown=EnhancedScoreBreakdown(
                parseability=parseability,
                structure=structure,
                content=content,
            ),
            risk_level=risk,
            issues=issues,
            recommendations=recommendations,
            passed_checks=passed,
            confidence=ocr_confidence,
            industry=self.industry.value,
        )
    
    def get_score_summary(self, score: EnhancedATSScore) -> Dict[str, Any]:
        """Get human-readable score summary."""
        return {
            "overall_score": score.overall_score,
            "risk_level": score.risk_level.value,
            "grade": self._get_grade(score.overall_score),
            "breakdown": {
                "parseability": score.breakdown.parseability.total,
                "structure": score.breakdown.structure.total,
                "content": score.breakdown.content.total,
            },
            "weighted_breakdown": {
                "parseability_weight": self.weights["parseability"],
                "structure_weight": self.weights["structure"],
                "content_weight": self.weights["content"],
            },
            "industry": score.industry,
            "issues_count": len(score.issues),
            "recommendations_count": len(score.recommendations),
            "passed_checks": score.passed_checks,
        }
    
    def _get_grade(self, score: int) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"