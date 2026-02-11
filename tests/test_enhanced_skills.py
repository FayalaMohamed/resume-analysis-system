"""Tests for enhanced skills extraction and taxonomy module."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.enhanced_skills import (
    EnhancedSkillTaxonomy,
    EnhancedSkillExtractor,
    SkillGapAnalyzer,
    extract_skills_from_resume,
    analyze_skill_gaps,
    SkillCategory,
    ProficiencyLevel,
    SkillInfo,
    SkillGapResult,
)


class TestEnhancedSkillTaxonomy(unittest.TestCase):
    """Test cases for skill taxonomy."""

    def test_get_category_programming(self):
        """Test programming language categorization."""
        category = EnhancedSkillTaxonomy.get_category("python")
        self.assertEqual(category, SkillCategory.PROGRAMMING_LANGUAGE)

    def test_get_category_framework(self):
        """Test framework categorization."""
        category = EnhancedSkillTaxonomy.get_category("react")
        self.assertEqual(category, SkillCategory.FRAMEWORK)

    def test_get_category_database(self):
        """Test database categorization."""
        category = EnhancedSkillTaxonomy.get_category("postgresql")
        self.assertEqual(category, SkillCategory.DATABASE)

    def test_get_category_cloud(self):
        """Test cloud/devops categorization."""
        category = EnhancedSkillTaxonomy.get_category("aws")
        self.assertEqual(category, SkillCategory.CLOUD_DEVOPS)

    def test_get_category_unknown(self):
        """Test unknown skill categorization."""
        category = EnhancedSkillTaxonomy.get_category("unknownskill")
        self.assertEqual(category, SkillCategory.UNKNOWN)

    def test_normalize_skill_alias(self):
        """Test skill normalization with aliases."""
        self.assertEqual(EnhancedSkillTaxonomy.normalize_skill("js"), "javascript")
        self.assertEqual(EnhancedSkillTaxonomy.normalize_skill("ts"), "typescript")
        self.assertEqual(EnhancedSkillTaxonomy.normalize_skill("py"), "python")

    def test_normalize_skill_canonical(self):
        """Test skill normalization with canonical names."""
        self.assertEqual(EnhancedSkillTaxonomy.normalize_skill("python"), "python")
        self.assertEqual(EnhancedSkillTaxonomy.normalize_skill("react"), "react")

    def test_get_related(self):
        """Test getting related skills."""
        related = EnhancedSkillTaxonomy.get_related("react")
        self.assertIn("vue", related)
        self.assertIn("javascript", related)

    def test_all_categories(self):
        """Test getting all categories."""
        categories = EnhancedSkillTaxonomy.get_all_categories()
        self.assertGreater(len(categories), 10)
        self.assertIn(SkillCategory.PROGRAMMING_LANGUAGE, categories)
        self.assertIn(SkillCategory.FRAMEWORK, categories)


class TestEnhancedSkillExtractor(unittest.TestCase):
    """Test cases for skill extractor."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = EnhancedSkillExtractor()

    def test_extract_from_skills_section(self):
        """Test extraction from skills section."""
        text = "Python, JavaScript, React, Docker, AWS, Kubernetes"
        skills = self.extractor._extract_from_skills_section(text)
        
        self.assertEqual(len(skills), 6)
        skill_names = [s.canonical_name for s in skills]
        self.assertIn("python", skill_names)
        self.assertIn("javascript", skill_names)
        self.assertIn("react", skill_names)
        self.assertIn("docker", skill_names)
        self.assertIn("amazon web services", skill_names)
        self.assertIn("kubernetes", skill_names)

    def test_extract_proficiency_expert(self):
        """Test proficiency detection for expert level."""
        proficiency = self.extractor._detect_proficiency("Expert in Python")
        self.assertEqual(proficiency, ProficiencyLevel.EXPERT)

    def test_extract_proficiency_advanced(self):
        """Test proficiency detection for advanced level."""
        proficiency = self.extractor._detect_proficiency("Senior Developer")
        self.assertEqual(proficiency, ProficiencyLevel.ADVANCED)

    def test_extract_proficiency_intermediate(self):
        """Test proficiency detection for intermediate level."""
        proficiency = self.extractor._detect_proficiency("Proficient in Java")
        self.assertEqual(proficiency, ProficiencyLevel.INTERMEDIATE)

    def test_extract_proficiency_beginner(self):
        """Test proficiency detection for beginner level."""
        proficiency = self.extractor._detect_proficiency("Basic knowledge of SQL")
        self.assertEqual(proficiency, ProficiencyLevel.BEGINNER)

    def test_extract_skills_with_sections(self):
        """Test extraction from sections."""
        sections = [
            {"section_type": "skills", "raw_text": "Python, React, Node.js"},
            {"section_type": "experience", "raw_text": "Worked with Docker and AWS"},
        ]
        skills = self.extractor.extract_skills(sections=sections)
        
        self.assertGreater(len(skills), 0)


class TestSkillGapAnalyzer(unittest.TestCase):
    """Test cases for skill gap analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SkillGapAnalyzer()

    def test_analyze_perfect_match(self):
        """Test gap analysis with perfect match."""
        resume_skills = [
            SkillInfo("Python", "python", SkillCategory.PROGRAMMING_LANGUAGE, 0.95),
            SkillInfo("React", "react", SkillCategory.FRAMEWORK, 0.95),
            SkillInfo("AWS", "amazon web services", SkillCategory.CLOUD_DEVOPS, 0.95),
        ]
        job_skills = ["Python", "React", "AWS"]
        
        result = self.analyzer.analyze(resume_skills, job_skills)
        
        self.assertEqual(len(result.matched_skills), 3)
        self.assertEqual(len(result.missing_skills), 0)
        self.assertEqual(result.gap_score, 0.0)

    def test_analyze_partial_match(self):
        """Test gap analysis with partial match."""
        resume_skills = [
            SkillInfo("Python", "python", SkillCategory.PROGRAMMING_LANGUAGE, 0.95),
        ]
        job_skills = ["Python", "React", "AWS", "Docker"]
        
        result = self.analyzer.analyze(resume_skills, job_skills)
        
        self.assertEqual(len(result.matched_skills), 1)
        self.assertEqual(len(result.missing_skills), 3)
        self.assertGreater(result.gap_score, 0)

    def test_analyze_with_related_skills(self):
        """Test gap analysis recognizing related skills."""
        resume_skills = [
            SkillInfo("Vue", "vue", SkillCategory.FRAMEWORK, 0.95),
        ]
        job_skills = ["React", "Angular"]
        
        result = self.analyzer.analyze(resume_skills, job_skills)
        
        # Vue is related to React and Angular
        self.assertEqual(len(result.matched_skills), 0)
        self.assertEqual(len(result.partial_matches), 2)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""

    def test_extract_skills_from_resume(self):
        """Test convenience function for skill extraction."""
        skills_text = "Python, JavaScript, Docker, Kubernetes"
        
        result = extract_skills_from_resume(skills_text=skills_text)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        # Check structure
        for skill in result:
            self.assertIn("name", skill)
            self.assertIn("canonical_name", skill)
            self.assertIn("category", skill)
            self.assertIn("confidence", skill)

    def test_analyze_skill_gaps(self):
        """Test convenience function for gap analysis."""
        resume_skills = [
            {"name": "Python", "canonical_name": "python", "category": "programming_language", "confidence": 0.95},
        ]
        job_skills = ["Python", "Java", "React"]
        
        result = analyze_skill_gaps(resume_skills, job_skills)
        
        self.assertIsInstance(result, dict)
        self.assertIn("matched_skills", result)
        self.assertIn("missing_skills", result)
        self.assertIn("gap_score", result)
        self.assertIn("recommendations", result)


class TestSkillInfo(unittest.TestCase):
    """Test cases for SkillInfo dataclass."""

    def test_to_dict(self):
        """Test SkillInfo to_dict conversion."""
        skill = SkillInfo(
            name="Python",
            canonical_name="python",
            category=SkillCategory.PROGRAMMING_LANGUAGE,
            confidence=0.95,
            proficiency=ProficiencyLevel.ADVANCED,
            is_explicit=True,
            source_section="skills",
            related_skills=["django", "pandas"],
            synonyms=["py"],
        )
        
        result = skill.to_dict()
        
        self.assertEqual(result["name"], "Python")
        self.assertEqual(result["canonical_name"], "python")
        self.assertEqual(result["category"], "programming_language")
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(result["proficiency"], "advanced")
        self.assertTrue(result["is_explicit"])


if __name__ == "__main__":
    unittest.main()