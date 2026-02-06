"""ATS Scoring module for evaluating resume compatibility."""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(Enum):
    """Risk levels for ATS compatibility."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of ATS score components."""
    layout_score: int = 0
    format_score: int = 0
    content_score: int = 0
    structure_score: int = 0
    
    @property
    def total(self) -> int:
        return self.layout_score + self.format_score + self.content_score + self.structure_score


@dataclass
class ATSScore:
    """Complete ATS compatibility score."""
    overall_score: int = 0
    max_score: int = 100
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    risk_level: RiskLevel = RiskLevel.LOW
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)


class ATSScorer:
    """Score resume ATS compatibility based on various criteria."""
    
    def __init__(self):
        """Initialize the ATS scorer."""
        pass
    
    def score_layout(self, layout_features: Dict[str, Any]) -> tuple:
        """Score resume layout (25 points max).
        
        Args:
            layout_features: Layout analysis results
            
        Returns:
            Tuple of (score, issues, recommendations)
        """
        score = 25
        issues = []
        recommendations = []
        passed = []
        
        # Single column bonus
        if layout_features.get('is_single_column', True):
            passed.append("Single-column layout")
        else:
            score -= 25
            issues.append("Multi-column layout detected")
            recommendations.append("Convert to single-column format")
        
        # Table penalty
        if layout_features.get('has_tables', False):
            score -= 20
            issues.append("Tables detected")
            recommendations.append("Remove tables, use simple text instead")
        else:
            passed.append("No tables detected")
        
        return max(score, 0), issues, recommendations, passed
    
    def score_format(self, text: str, layout_features: Dict[str, Any]) -> tuple:
        """Score resume format (25 points max).
        
        Args:
            text: Resume text
            layout_features: Layout analysis results
            
        Returns:
            Tuple of (score, issues, recommendations, passed)
        """
        score = 25
        issues = []
        recommendations = []
        passed = []
        
        # Check for standard fonts (basic check)
        # In a real implementation, we'd analyze the PDF font metadata
        
        # Check text length
        text_length = len(text)
        if 500 < text_length < 10000:
            passed.append("Appropriate resume length")
        elif text_length < 500:
            score -= 10
            issues.append("Resume appears too short")
            recommendations.append("Add more detail to your experience")
        else:
            score -= 5
            issues.append("Resume is quite long")
            recommendations.append("Consider condensing to 1-2 pages")
        
        # Check for images/graphics
        if layout_features.get('has_images', False):
            score -= 15
            issues.append("Images or graphics detected")
            recommendations.append("Remove images, use text only")
        else:
            passed.append("Text-based content")
        
        return max(score, 0), issues, recommendations, passed
    
    def score_content(self, parsed_resume: Dict[str, Any]) -> tuple:
        """Score resume content quality (25 points max).
        
        Args:
            parsed_resume: Parsed resume data
            
        Returns:
            Tuple of (score, issues, recommendations, passed)
        """
        score = 25
        issues = []
        recommendations = []
        passed = []
        
        # Check for contact info
        contact = parsed_resume.get('contact_info', {})
        if contact.get('email') and contact.get('phone'):
            passed.append("Contact information present")
        else:
            score -= 5
            if not contact.get('email'):
                issues.append("Email not found")
            if not contact.get('phone'):
                issues.append("Phone not found")
            recommendations.append("Add complete contact information")
        
        # Check for sections
        sections = parsed_resume.get('sections', [])
        section_types = [s.get('section_type', 'unknown') for s in sections]
        
        required_sections = ['experience', 'education']
        for req in required_sections:
            if req in section_types:
                passed.append(f"{req.title()} section present")
            else:
                score -= 10
                issues.append(f"Missing {req} section")
                recommendations.append(f"Add a {req} section")
        
        # Optional but recommended
        if 'skills' in section_types:
            passed.append("Skills section present")
            # Check if skills have content
            skills = parsed_resume.get('skills', [])
            if len(skills) < 3:
                score -= 3
                issues.append("Skills section appears sparse")
                recommendations.append("Add more skills")
        else:
            score -= 5
            recommendations.append("Consider adding a skills section")
        
        return max(score, 0), issues, recommendations, passed
    
    def score_structure(self, layout_features: Dict[str, Any]) -> tuple:
        """Score resume structure (25 points max).
        
        Args:
            layout_features: Layout analysis results
            
        Returns:
            Tuple of (score, issues, recommendations, passed)
        """
        score = 25
        issues = []
        recommendations = []
        passed = []
        
        # Check section headers
        headers = layout_features.get('section_headers', [])
        if len(headers) >= 3:
            passed.append("Clear section headers present")
        else:
            score -= 5
            issues.append("Few section headers detected")
            recommendations.append("Add clear section headers")
        
        # Check text density
        density = layout_features.get('text_density', 0)
        if 30 < density < 150:
            passed.append("Good text density")
        elif density > 150:
            score -= 5
            issues.append("Text too dense")
            recommendations.append("Add more white space")
        
        # Standard section naming
        standard_headers = ['experience', 'education', 'skills']
        has_standard = any(
            any(std in h.lower() for std in standard_headers)
            for h in headers
        )
        if has_standard:
            passed.append("Standard section names used")
        else:
            score -= 3
            recommendations.append("Use standard section names")
        
        return max(score, 0), issues, recommendations, passed
    
    def calculate_score(
        self,
        text: str,
        layout_features: Dict[str, Any],
        parsed_resume: Dict[str, Any]
    ) -> ATSScore:
        """Calculate complete ATS compatibility score.
        
        Args:
            text: Raw resume text
            layout_features: Layout analysis results
            parsed_resume: Parsed resume structure
            
        Returns:
            ATSScore object with detailed breakdown
        """
        # Calculate component scores
        layout_s, layout_issues, layout_rec, layout_passed = self.score_layout(layout_features)
        format_s, format_issues, format_rec, format_passed = self.score_format(text, layout_features)
        content_s, content_issues, content_rec, content_passed = self.score_content(parsed_resume)
        structure_s, structure_issues, structure_rec, structure_passed = self.score_structure(layout_features)
        
        # Normalize to 25 points each
        breakdown = ScoreBreakdown(
            layout_score=min(layout_s, 25),
            format_score=min(format_s, 25),
            content_score=min(content_s, 25),
            structure_score=min(structure_s, 25),
        )
        
        total_score = breakdown.total
        
        # Determine risk level
        if total_score >= 80:
            risk = RiskLevel.LOW
        elif total_score >= 60:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.HIGH
        
        # Compile all feedback
        all_issues = layout_issues + format_issues + content_issues + structure_issues
        all_recommendations = layout_rec + format_rec + content_rec + structure_rec
        all_passed = layout_passed + format_passed + content_passed + structure_passed
        
        return ATSScore(
            overall_score=total_score,
            breakdown=breakdown,
            risk_level=risk,
            issues=all_issues,
            recommendations=all_recommendations,
            passed_checks=all_passed,
        )
    
    def get_score_summary(self, score: ATSScore) -> Dict[str, Any]:
        """Get human-readable score summary.
        
        Args:
            score: ATSScore object
            
        Returns:
            Dictionary with summary information
        """
        return {
            "overall_score": score.overall_score,
            "risk_level": score.risk_level.value,
            "grade": self._get_grade(score.overall_score),
            "breakdown": {
                "layout": score.breakdown.layout_score,
                "format": score.breakdown.format_score,
                "content": score.breakdown.content_score,
                "structure": score.breakdown.structure_score,
            },
            "passed": score.passed_checks,
            "issues": score.issues,
            "recommendations": score.recommendations,
        }
    
    def _get_grade(self, score: int) -> str:
        """Convert numeric score to letter grade.
        
        Args:
            score: Numeric score (0-100)
            
        Returns:
            Letter grade
        """
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
