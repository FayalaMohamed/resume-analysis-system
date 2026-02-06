"""Recommendation engine for resume improvements."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class Priority(Enum):
    """Recommendation priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Recommendation:
    """Single recommendation item."""
    category: str
    priority: Priority
    issue: str
    suggestion: str
    example: Optional[str] = None
    estimated_impact: str = ""


class RecommendationEngine:
    """Generate prioritized resume improvement recommendations."""
    
    def __init__(self, llm_client: Optional[Any] = None):
        """Initialize recommendation engine.
        
        Args:
            llm_client: Optional LLM client for AI-powered suggestions
        """
        self.llm_client = llm_client
    
    def generate_recommendations(
        self,
        ats_score: Dict[str, Any],
        content_score: Dict[str, Any],
        layout_analysis: Dict[str, Any],
        job_match: Optional[Dict[str, Any]] = None
    ) -> List[Recommendation]:
        """Generate comprehensive recommendations.
        
        Args:
            ats_score: ATS scoring results
            content_score: Content quality analysis
            layout_analysis: Layout analysis results
            job_match: Optional job matching results
            
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        
        # Layout recommendations
        recommendations.extend(self._layout_recommendations(layout_analysis))
        
        # Content recommendations
        recommendations.extend(self._content_recommendations(content_score))
        
        # ATS recommendations
        recommendations.extend(self._ats_recommendations(ats_score))
        
        # Job match recommendations
        if job_match:
            recommendations.extend(self._job_match_recommendations(job_match))
        
        # Sort by priority
        priority_order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
        recommendations.sort(key=lambda r: priority_order[r.priority])
        
        return recommendations
    
    def _layout_recommendations(self, layout: Dict[str, Any]) -> List[Recommendation]:
        """Generate layout-based recommendations."""
        recommendations = []
        
        if not layout.get('is_single_column', True):
            recommendations.append(Recommendation(
                category="Layout",
                priority=Priority.HIGH,
                issue="Multi-column layout detected",
                suggestion="Convert to single-column format. ATS systems read left-to-right, top-to-bottom and may parse multi-column layouts incorrectly.",
                example="Use a clean, single-column layout with clear section headers",
                estimated_impact="+15-25 ATS points"
            ))
        
        if layout.get('has_tables', False):
            recommendations.append(Recommendation(
                category="Layout",
                priority=Priority.HIGH,
                issue="Tables detected in resume",
                suggestion="Replace tables with simple bullet points or plain text. ATS systems often cannot read table content correctly.",
                example="Instead of a skills table, use: • Python • JavaScript • SQL",
                estimated_impact="+10-20 ATS points"
            ))
        
        if len(layout.get('section_headers', [])) < 3:
            recommendations.append(Recommendation(
                category="Structure",
                priority=Priority.MEDIUM,
                issue="Missing standard section headers",
                suggestion="Add clear section headers: Experience, Education, and Skills. This helps ATS identify and categorize your information.",
                example="EXPERIENCE\nSoftware Engineer, Company Name\n• Bullet point 1\n• Bullet point 2",
                estimated_impact="+5-10 ATS points"
            ))
        
        return recommendations
    
    def _content_recommendations(self, content: Dict[str, Any]) -> List[Recommendation]:
        """Generate content-based recommendations."""
        recommendations = []
        
        # Action verb recommendations
        action_verb_score = content.get('action_verb_score', 25)
        weak_verbs = content.get('weak_verbs_found', [])
        
        if action_verb_score < 15:
            recommendations.append(Recommendation(
                category="Content",
                priority=Priority.HIGH,
                issue=f"Weak action verbs detected ({len(weak_verbs)} found)",
                suggestion="Replace weak verbs with strong action verbs. Start bullet points with impactful verbs that demonstrate achievement.",
                example="❌ 'Was responsible for managing team'\n✅ 'Led 5-person team to deliver $2M project'",
                estimated_impact="+10-15 content points"
            ))
        
        # Quantification recommendations
        quant_score = content.get('quantification_score', 25)
        
        if quant_score < 15:
            recommendations.append(Recommendation(
                category="Content",
                priority=Priority.HIGH,
                issue="Lack of quantified achievements",
                suggestion="Add metrics to your achievements. Numbers make accomplishments concrete and memorable.",
                example="❌ 'Improved process efficiency'\n✅ 'Reduced processing time by 40%, saving 20 hours weekly'",
                estimated_impact="+10-20 content points"
            ))
        
        # Bullet structure recommendations
        bullet_score = content.get('bullet_structure_score', 25)
        
        if bullet_score < 15:
            recommendations.append(Recommendation(
                category="Format",
                priority=Priority.MEDIUM,
                issue="Bullet point issues",
                suggestion="Keep bullets 15-25 words. Be specific and lead with outcomes. Avoid paragraphs in experience sections.",
                example="Ideal: 'Developed Python automation script reducing data processing time by 60%' (12 words)",
                estimated_impact="+5-10 content points"
            ))
        
        # Conciseness recommendations
        conciseness_score = content.get('conciseness_score', 25)
        
        if conciseness_score < 15:
            recommendations.append(Recommendation(
                category="Writing",
                priority=Priority.LOW,
                issue="Wordy or verbose language",
                suggestion="Remove filler words and phrases. Be direct and concise.",
                example="❌ 'In order to improve efficiency'\n✅ 'To improve efficiency'\n❌ 'Due to the fact that'\n✅ 'Because'",
                estimated_impact="+3-5 content points"
            ))
        
        return recommendations
    
    def _ats_recommendations(self, ats_score: Dict[str, Any]) -> List[Recommendation]:
        """Generate ATS-specific recommendations."""
        recommendations = []
        
        breakdown = ats_score.get('breakdown', {})
        issues = ats_score.get('issues', [])
        
        # Content issues
        if breakdown.get('content', 25) < 20:
            recommendations.append(Recommendation(
                category="ATS",
                priority=Priority.HIGH,
                issue="Missing key information",
                suggestion="Ensure your resume includes: phone, email, work experience with dates, education, and skills section.",
                example="Contact: Name, Phone, Email, LinkedIn\nExperience: Job Title, Company, Dates, 3-5 bullets",
                estimated_impact="+10-15 ATS points"
            ))
        
        # Format issues
        if breakdown.get('format', 25) < 20:
            recommendations.append(Recommendation(
                category="ATS",
                priority=Priority.MEDIUM,
                issue="Format issues detected",
                suggestion="Use standard fonts (Arial, Calibri, Times New Roman). Avoid graphics, images, or unusual formatting.",
                estimated_impact="+5-10 ATS points"
            ))
        
        return recommendations
    
    def _job_match_recommendations(self, job_match: Dict[str, Any]) -> List[Recommendation]:
        """Generate job match recommendations."""
        recommendations = []
        
        overall_match = job_match.get('overall_match', 0)
        missing_skills = job_match.get('missing_skills', [])
        missing_keywords = job_match.get('missing_keywords', [])
        
        if overall_match < 0.5:
            recommendations.append(Recommendation(
                category="Job Match",
                priority=Priority.HIGH,
                issue=f"Low job match score ({overall_match:.0%})",
                suggestion="Your resume doesn't align well with this job. Consider tailoring it more specifically to the requirements.",
                estimated_impact="+20-30 match points"
            ))
        
        if missing_skills:
            top_skills = missing_skills[:3]
            recommendations.append(Recommendation(
                category="Skills",
                priority=Priority.HIGH if len(missing_skills) > 2 else Priority.MEDIUM,
                issue=f"Missing {len(missing_skills)} required skills",
                suggestion=f"Add these skills to your resume if you have them: {', '.join(top_skills)}",
                example=f"Skills section: ... • {top_skills[0]} • {top_skills[1] if len(top_skills) > 1 else 'Other skill'}",
                estimated_impact=f"+{len(missing_skills) * 5} match points"
            ))
        
        if missing_keywords:
            top_keywords = missing_keywords[:5]
            recommendations.append(Recommendation(
                category="Keywords",
                priority=Priority.MEDIUM,
                issue=f"Missing {len(missing_keywords)} job keywords",
                suggestion=f"Incorporate these keywords naturally: {', '.join(top_keywords)}",
                estimated_impact=f"+{len(missing_keywords) * 2} match points"
            ))
        
        return recommendations
    
    def get_priority_summary(self, recommendations: List[Recommendation]) -> Dict[str, int]:
        """Get count of recommendations by priority.
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Dictionary with priority counts
        """
        counts = {Priority.HIGH: 0, Priority.MEDIUM: 0, Priority.LOW: 0}
        for rec in recommendations:
            counts[rec.priority] = counts.get(rec.priority, 0) + 1
        return {
            'high': counts[Priority.HIGH],
            'medium': counts[Priority.MEDIUM],
            'low': counts[Priority.LOW],
            'total': len(recommendations),
        }
    
    def format_recommendations(
        self,
        recommendations: List[Recommendation],
        max_items: int = 10
    ) -> List[Dict[str, str]]:
        """Format recommendations for display.
        
        Args:
            recommendations: List of recommendations
            max_items: Maximum number to return
            
        Returns:
            List of formatted recommendation dictionaries
        """
        formatted = []
        
        for rec in recommendations[:max_items]:
            formatted.append({
                'category': rec.category,
                'priority': rec.priority.value.upper(),
                'issue': rec.issue,
                'suggestion': rec.suggestion,
                'example': rec.example or '',
                'impact': rec.estimated_impact,
            })
        
        return formatted
