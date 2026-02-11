"""Tests for content understanding module."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.content_understanding import (
    SectionDetector,
    ContentQualityAnalyzer,
    RedFlagDetector,
    ContentEnricher,
    ContentUnderstandingEngine,
    analyze_resume_content,
    SectionType,
    Severity,
)


class TestSectionDetector(unittest.TestCase):
    """Test cases for section detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = SectionDetector()

    def test_detect_standard_sections(self):
        """Test detection of standard sections."""
        text = """
PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years of experience.

WORK EXPERIENCE
Senior Developer at Tech Corp
2020 - Present

EDUCATION
Bachelor's in Computer Science

SKILLS
Python, JavaScript, React
"""
        
        sections = self.detector.detect_sections(text)
        section_types = [s.section_type for s in sections]
        
        self.assertIn(SectionType.SUMMARY, section_types)
        self.assertIn(SectionType.EXPERIENCE, section_types)
        self.assertIn(SectionType.EDUCATION, section_types)
        self.assertIn(SectionType.SKILLS, section_types)

    def test_detect_alternative_section_names(self):
        """Test detection of alternative section names."""
        text = """
CAREER OBJECTIVE
Looking for challenging role.

EMPLOYMENT HISTORY
Various positions that I held over the years.
"""
        
        sections = self.detector.detect_sections(text)
        section_types = [s.section_type for s in sections]
        
        self.assertIn(SectionType.SUMMARY, section_types)

    def test_get_missing_sections(self):
        """Test detection of missing critical sections."""
        text = """
SUMMARY
Some content
"""
        
        sections = self.detector.detect_sections(text)
        missing = self.detector.get_missing_sections(sections)
        
        self.assertIn("experience", missing)
        self.assertIn("education", missing)


class TestContentQualityAnalyzer(unittest.TestCase):
    """Test cases for content quality analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ContentQualityAnalyzer()

    def test_strong_verb_detection(self):
        """Test detection of strong action verbs."""
        text = "Led development team and achieved 40% improvement"
        
        metrics = self.analyzer.analyze(text)
        
        self.assertIn("led", [v.lower() for v in metrics.strong_verbs])
        self.assertIn("achieved", [v.lower() for v in metrics.strong_verbs])

    def test_weak_verb_detection(self):
        """Test detection of weak verbs."""
        text = "Responsible for tasks and helped with work"
        
        metrics = self.analyzer.analyze(text)
        
        self.assertTrue(len(metrics.weak_verbs) > 0)

    def test_quantification_detection(self):
        """Test detection of quantified achievements."""
        text = "Increased revenue by 40%, saved $100k, managed 5-person team"
        
        metrics = self.analyzer.analyze(text)
        
        self.assertGreater(len(metrics.quantified_achievements), 0)

    def test_action_verb_score(self):
        """Test action verb scoring."""
        text = """
Led the team and achieved remarkable results
Created innovative solutions that improved performance by 50%
"""
        
        metrics = self.analyzer.analyze(text)
        
        # "Led" and "achieved" are strong verbs
        self.assertGreater(metrics.action_verb_score, 40)

    def test_quantification_score(self):
        """Test quantification scoring."""
        text_with_metrics = "Achieved 40% growth and $1M revenue"
        text_without = "Did work and had good results at the company"
        
        metrics_with = self.analyzer.analyze(text_with_metrics)
        metrics_without = self.analyzer.analyze(text_without)
        
        # With metrics should have higher score than without
        self.assertGreaterEqual(metrics_with.quantification_score, metrics_without.quantification_score)

    def test_bullet_quality_analysis(self):
        """Test bullet point quality analysis."""
        text = """
• Responsible for daily tasks
• Led cross-functional team and delivered project on time
• A
"""
        
        metrics = self.analyzer.analyze(text)
        
        self.assertGreater(len(metrics.bullet_issues), 0)

    def test_overall_score_calculation(self):
        """Test overall score is calculated."""
        text = "Led development team that achieved 40% improvement."
        
        metrics = self.analyzer.analyze(text)
        
        self.assertGreater(metrics.overall_score, 0)
        self.assertLessEqual(metrics.overall_score, 100)


class TestRedFlagDetector(unittest.TestCase):
    """Test cases for red flag detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = RedFlagDetector()

    def test_missing_email_detection(self):
        """Test detection of missing email."""
        text = "No email here, just text without any contact information"
        
        flags = self.detector.detect(text, [])
        
        # Check for contact-related flags
        contact_flags = [f for f in flags if f.category == "contact"]
        self.assertTrue(len(contact_flags) > 0)

    def test_missing_phone_detection(self):
        """Test detection of missing phone."""
        text = "test@test.com but no phone"
        
        flags = self.detector.detect(text, [])
        
        phone_flags = [f for f in flags if "phone" in f.category or "contact" in f.category]
        self.assertTrue(len(phone_flags) > 0)

    def test_outdated_technology_detection(self):
        """Test detection of outdated technologies."""
        text = "Experienced with Visual Basic 6 and Perl"
        
        flags = self.detector.detect(text, [])
        
        relevance_flags = [f for f in flags if f.category == "relevance"]
        self.assertTrue(len(relevance_flags) > 0)

    def test_severity_levels(self):
        """Test that severity levels are assigned."""
        text = "No email at all"
        
        flags = self.detector.detect(text, [])
        
        for flag in flags:
            self.assertIsInstance(flag.severity, Severity)


class TestContentEnricher(unittest.TestCase):
    """Test cases for content enrichment."""

    def setUp(self):
        """Set up test fixtures."""
        self.enricher = ContentEnricher()

    def test_seniority_estimation(self):
        """Test seniority level estimation."""
        text = "Senior Software Engineer with 10 years of experience leading teams"
        
        sections = []
        enrichment = self.enricher.enrich(text, sections)
        
        self.assertIn(enrichment.estimated_seniority, ['senior', 'mid-level'])

    def test_transferable_skills_identification(self):
        """Test identification of transferable skills."""
        text = "Strong communication and leadership skills with excellent problem solving abilities"
        
        enrichment = self.enricher.enrich(text, [])
        
        transferable = [s.lower() for s in enrichment.transferable_skills]
        self.assertIn("communication", transferable)
        self.assertIn("leadership", transferable)
        self.assertIn("problem solving", transferable)

    def test_skill_experience_estimation(self):
        """Test estimation of experience per skill."""
        text = "I have 5 years of experience with Python and JavaScript for 3 years"
        
        enrichment = self.enricher.enrich(text, [])
        
        # Check that skill experience was found
        found_skills = list(enrichment.skill_experience_years.keys())
        self.assertTrue(len(found_skills) > 0)

    def test_total_experience_calculation(self):
        """Test calculation of total experience years."""
        text = """Worked at Company A from 2018 to 2020.
Then at Company B from 2020 to 2024."""
        
        enrichment = self.enricher.enrich(text, [])
        
        # Value should be calculated (0 or greater)
        self.assertIsNotNone(enrichment.total_experience_years)

    def test_theme_extraction(self):
        """Test extraction of key themes."""
        text = "Python developer with AWS experience leading team in cloud computing"
        
        enrichment = self.enricher.enrich(text, [])
        
        self.assertTrue(len(enrichment.key_themes) > 0)


class TestContentUnderstandingEngine(unittest.TestCase):
    """Test cases for the main content understanding engine."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = ContentUnderstandingEngine()

    def test_complete_analysis(self):
        """Test complete content understanding analysis."""
        text = """
PROFESSIONAL SUMMARY
Experienced software engineer with expertise in Python and cloud technologies.

WORK EXPERIENCE
Senior Developer at Tech Corp
2020 - Present
- Led development of cloud-native applications
- Improved system performance by 40%

Software Engineer at StartUp Inc
2018 - 2020
- Developed REST APIs using Python
- Collaborated with cross-functional team

EDUCATION
Bachelor's in Computer Science, 2018

SKILLS
Python, JavaScript, AWS, Docker

CONTACT
test@email.com
555-123-4567
"""
        
        result = self.engine.analyze(text)
        
        self.assertIsInstance(result, object)
        self.assertTrue(len(result.sections) > 0)
        self.assertGreater(result.quality_metrics.overall_score, 0)
        self.assertIn(result.enrichment.estimated_seniority, ['senior', 'mid-level'])

    def test_confidence_score(self):
        """Test that confidence score is calculated."""
        text = "Test resume content"
        
        result = self.engine.analyze(text)
        
        self.assertGreater(result.overall_confidence, 0)
        self.assertLessEqual(result.overall_confidence, 1.0)


class TestAnalyzeResumeContent(unittest.TestCase):
    """Test cases for convenience function."""

    def test_analyze_resume_content(self):
        """Test the convenience function."""
        text = """
SUMMARY
Professional with experience.

EXPERIENCE
Job at Company

EDUCATION
Degree
"""
        
        result = analyze_resume_content(text)
        
        self.assertIn("sections", result)
        self.assertIn("quality_metrics", result)
        self.assertIn("red_flags", result)
        self.assertIn("enrichment", result)

    def test_result_to_dict(self):
        """Test that result converts to dictionary."""
        text = "Some resume text"
        
        engine = ContentUnderstandingEngine()
        result = engine.analyze(text)
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertIn("sections", result_dict)
        self.assertIn("quality_metrics", result_dict)


if __name__ == "__main__":
    unittest.main()