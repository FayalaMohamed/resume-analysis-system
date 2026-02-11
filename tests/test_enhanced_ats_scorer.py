"""Tests for enhanced ATS scoring module."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring.enhanced_ats_scorer import (
    EnhancedATSScorer,
    EnhancedATSScore,
    EnhancedScoreBreakdown,
    ParseabilityScore,
    StructureScore,
    ContentQualityScore,
    Industry,
    IndustryScoringConfig,
    RiskLevel,
)


class TestIndustryScoringConfig(unittest.TestCase):
    """Test cases for industry scoring configurations."""

    def test_get_weights_general(self):
        """Test general industry weights."""
        weights = IndustryScoringConfig.get_weights(Industry.GENERAL)
        
        self.assertIn("parseability", weights)
        self.assertIn("structure", weights)
        self.assertIn("content", weights)
        self.assertEqual(weights["parseability"], 0.40)
        self.assertEqual(weights["structure"], 0.30)
        self.assertEqual(weights["content"], 0.30)

    def test_get_weights_tech(self):
        """Test tech industry weights."""
        weights = IndustryScoringConfig.get_weights(Industry.TECH)
        
        self.assertEqual(weights["content"], 0.35)
        self.assertEqual(weights["parseability"], 0.35)

    def test_get_weights_creative(self):
        """Test creative industry weights."""
        weights = IndustryScoringConfig.get_weights(Industry.CREATIVE)
        
        self.assertEqual(weights["content"], 0.40)
        self.assertEqual(weights["parseability"], 0.25)

    def test_get_weights_academic(self):
        """Test academic industry weights."""
        weights = IndustryScoringConfig.get_weights(Industry.ACADEMIC)
        
        self.assertEqual(weights["content"], 0.45)
        self.assertEqual(weights["parseability"], 0.30)

    def test_get_config(self):
        """Test getting full config."""
        config = IndustryScoringConfig.get_config(Industry.TECH)
        
        self.assertEqual(config["name"], "Technology")
        self.assertIn("bonuses", config)
        self.assertIn("penalties", config)


class TestParseabilityScore(unittest.TestCase):
    """Test cases for parseability score."""

    def test_total_calculation(self):
        """Test that total is calculated correctly."""
        score = ParseabilityScore(
            single_column=15,
            no_tables=10,
            standard_fonts=5,
            no_graphics=5,
            readable_pdf=5,
            text_extraction_success=10,
        )
        
        self.assertEqual(score.total, 50)


class TestStructureScore(unittest.TestCase):
    """Test cases for structure score."""

    def test_total_calculation(self):
        """Test that total is calculated correctly."""
        score = StructureScore(
            clear_sections=8,
            consistent_formatting=5,
            proper_headings=5,
            contact_info_visible=5,
            logical_order=3,
            standard_section_names=4,
        )
        
        self.assertEqual(score.total, 30)


class TestContentQualityScore(unittest.TestCase):
    """Test cases for content quality score."""

    def test_total_calculation(self):
        """Test that total is calculated correctly."""
        score = ContentQualityScore(
            keyword_density=6,
            relevant_sections=6,
            appropriate_length=5,
            no_red_flags=6,
            action_verbs=4,
            quantification=3,
        )
        
        self.assertEqual(score.total, 30)


class TestEnhancedScoreBreakdown(unittest.TestCase):
    """Test cases for enhanced score breakdown."""

    def test_total(self):
        """Test combined total."""
        breakdown = EnhancedScoreBreakdown(
            parseability=ParseabilityScore(single_column=15, no_tables=10),
            structure=StructureScore(clear_sections=8),
            content=ContentQualityScore(keyword_density=6),
        )
        
        self.assertEqual(breakdown.total, 15 + 10 + 8 + 6)


class TestEnhancedATSScorer(unittest.TestCase):
    """Test cases for enhanced ATS scorer."""

    def setUp(self):
        """Set up test fixtures."""
        self.scorer_general = EnhancedATSScorer(industry="general")
        self.scorer_tech = EnhancedATSScorer(industry="tech")

    def test_init_general(self):
        """Test initialization with general industry."""
        self.assertEqual(self.scorer_general.industry, Industry.GENERAL)

    def test_init_tech(self):
        """Test initialization with tech industry."""
        self.assertEqual(self.scorer_tech.industry, Industry.TECH)

    def test_score_parseability_ideal(self):
        """Test parseability scoring with ideal layout."""
        layout = {
            "is_single_column": True,
            "has_tables": False,
            "has_images": False,
            "layout_risk_score": 20,
        }
        
        score = self.scorer_general.score_parseability(layout, ocr_confidence=0.98)
        
        self.assertEqual(score.single_column, 12)
        self.assertEqual(score.no_tables, 8)
        self.assertEqual(score.no_graphics, 4)
        self.assertEqual(score.text_extraction_success, 8)
        self.assertEqual(score.total, 40)  # 12+8+4+4+4+8=40

    def test_score_parseability_multi_column(self):
        """Test parseability scoring with multi-column layout."""
        layout = {
            "is_single_column": False,
            "has_tables": True,
            "has_images": False,
            "layout_risk_score": 60,
        }
        
        score = self.scorer_general.score_parseability(layout, ocr_confidence=0.85)
        
        self.assertEqual(score.single_column, 0)
        self.assertEqual(score.no_tables, 0)
        # readable_pdf: risk_score 60 >= 50, so gets 0
        self.assertEqual(score.readable_pdf, 0)
        # standard_fonts=4, no_graphics=4, text_extraction_success=6
        self.assertEqual(score.total, 14)  # 0+0+4+4+0+6=14

    def test_score_parseability_low_ocr_confidence(self):
        """Test parseability scoring with low OCR confidence."""
        layout = {
            "is_single_column": True,
            "has_tables": False,
            "has_images": False,
            "layout_risk_score": 10,
        }
        
        score = self.scorer_general.score_parseability(layout, ocr_confidence=0.55)
        
        # 0.55 < 0.60, so text_extraction_success = 0
        self.assertEqual(score.text_extraction_success, 0)

    def test_score_structure(self):
        """Test structure scoring."""
        layout = {
            "section_headers": ["EXPERIENCE", "EDUCATION", "SKILLS", "PROJECTS"],
        }
        parsed = {
            "contact_info": {"email": "test@test.com", "phone": "123-456-7890"},
            "sections": [
                {"section_type": "experience"},
                {"section_type": "education"},
                {"section_type": "skills"},
            ],
        }
        
        score = self.scorer_general.score_structure(layout, parsed)
        
        self.assertEqual(score.clear_sections, 7)
        self.assertEqual(score.contact_info_visible, 5)
        self.assertEqual(score.standard_section_names, 5)
        self.assertEqual(score.consistent_formatting, 5)
        self.assertEqual(score.proper_headings, 5)
        self.assertEqual(score.logical_order, 3)
        # Total: 7+5+5+5+5+5+3 = 30

    def test_score_content(self):
        """Test content quality scoring."""
        text = """
        I developed a web application using Python and React.
        I led a team of 5 developers and increased efficiency by 40%.
        Experience includes managing projects and implementing solutions.
        """
        parsed = {
            "sections": [
                {"section_type": "experience"},
                {"section_type": "education"},
            ],
        }
        
        score = self.scorer_general.score_content(text, parsed)
        
        self.assertGreater(score.keyword_density, 0)
        self.assertGreater(score.relevant_sections, 0)
        self.assertGreater(score.action_verbs, 0)

    def test_calculate_score(self):
        """Test full score calculation."""
        layout = {
            "is_single_column": True,
            "has_tables": False,
            "has_images": False,
            "layout_risk_score": 10,
            "section_headers": ["EXPERIENCE", "EDUCATION", "SKILLS"],
        }
        parsed = {
            "contact_info": {"email": "test@test.com", "phone": "123-456-7890"},
            "sections": [
                {"section_type": "experience"},
                {"section_type": "education"},
            ],
        }
        text = "Test resume text with experience and education sections."
        
        result = self.scorer_general.calculate_score(text, layout, parsed, ocr_confidence=0.95)
        
        self.assertIsInstance(result, EnhancedATSScore)
        self.assertGreater(result.overall_score, 50)
        self.assertIn(result.risk_level, [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH])

    def test_calculate_score_tech_industry(self):
        """Test score calculation with tech industry."""
        layout = {"is_single_column": True, "has_tables": False}
        parsed = {"contact_info": {}, "sections": []}
        text = "Developed Python applications with React and AWS."
        
        result = self.scorer_tech.calculate_score(text, layout, parsed, ocr_confidence=0.95)
        
        self.assertEqual(result.industry, "tech")
        self.assertIsInstance(result, EnhancedATSScore)

    def test_calculate_score_with_issues(self):
        """Test that issues are generated correctly."""
        layout = {
            "is_single_column": False,
            "has_tables": True,
            "has_images": False,
            "section_headers": [],
        }
        parsed = {"contact_info": {}, "sections": []}
        text = "Short"
        
        result = self.scorer_general.calculate_score(text, layout, parsed, ocr_confidence=0.95)
        
        self.assertGreater(len(result.issues), 0)
        issue_categories = [i["category"] for i in result.issues]
        self.assertIn("parseability", issue_categories)

    def test_get_score_summary(self):
        """Test getting human-readable summary."""
        layout = {"is_single_column": True, "has_tables": False, "section_headers": []}
        parsed = {"contact_info": {"email": "test@test.com"}, "sections": []}
        text = "Test text with experience and education."
        
        result = self.scorer_general.calculate_score(text, layout, parsed)
        summary = self.scorer_general.get_score_summary(result)
        
        self.assertIn("overall_score", summary)
        self.assertIn("grade", summary)
        self.assertIn("breakdown", summary)
        self.assertIn("risk_level", summary)

    def test_grade_conversion(self):
        """Test that grades are correct."""
        test_cases = [
            (95, "A"),
            (85, "B"),
            (75, "C"),
            (65, "D"),
            (50, "F"),
        ]
        
        for score, expected_grade in test_cases:
            with self.subTest(score=score):
                grade = self.scorer_general._get_grade(score)
                self.assertEqual(grade, expected_grade)


class TestEnhancedATSScore(unittest.TestCase):
    """Test cases for EnhancedATSScore dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        breakdown = EnhancedScoreBreakdown(
            parseability=ParseabilityScore(single_column=15),
            structure=StructureScore(clear_sections=8),
            content=ContentQualityScore(keyword_density=6),
        )
        
        score = EnhancedATSScore(
            overall_score=75,
            breakdown=breakdown,
            risk_level=RiskLevel.MEDIUM,
            industry="tech",
        )
        
        result = score.to_dict()
        
        self.assertEqual(result["overall_score"], 75)
        self.assertEqual(result["risk_level"], "medium")
        self.assertEqual(result["industry"], "tech")
        self.assertIn("parseability", result["breakdown"])
        self.assertIn("structure", result["breakdown"])
        self.assertIn("content", result["breakdown"])


if __name__ == "__main__":
    unittest.main()