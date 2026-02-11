"""Tests for job matching and JD parsing module."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.job_matcher import JobDescriptionParser, JobMatchResult
from src.analysis.advanced_job_matcher import (
    AdvancedJobMatcher,
    AdvancedJobMatchResult,
    SkillTaxonomy,
    match_resume_to_job_advanced,
)


class TestJobDescriptionParser(unittest.TestCase):
    """Test cases for JD parser."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = JobDescriptionParser()

    def test_parse_basic_jd(self):
        """Test parsing basic job description."""
        jd_text = """
        We are looking for a Python Developer with experience in Django and PostgreSQL.
        Must have 5+ years of experience with Python.
        Preferred: Experience with AWS and Docker.
        """
        result = self.parser.parse(jd_text)
        
        self.assertIn("skills", result)
        self.assertIn("required", result["skills"])
        self.assertIn("preferred", result["skills"])
        self.assertIn("keywords", result)
        self.assertIn("summary", result)

    def test_extract_required_skills(self):
        """Test extraction of required skills."""
        jd_text = "Must have Python, Java, and Docker experience."
        result = self.parser.parse(jd_text)
        
        skills = result["skills"]
        required_skills = [s.lower() for s in skills.get("required", [])]
        
        self.assertTrue(len(required_skills) > 0)
        self.assertIn("python", required_skills)
        self.assertIn("java", required_skills)

    def test_extract_preferred_skills(self):
        """Test extraction of preferred skills."""
        jd_text = "Preferred: Experience with Kubernetes and Terraform."
        result = self.parser.parse(jd_text)
        
        skills = result["skills"]
        preferred_skills = [s.lower() for s in skills.get("preferred", [])]
        
        self.assertTrue(len(preferred_skills) > 0)

    def test_extract_keywords(self):
        """Test keyword extraction."""
        jd_text = """
        Join our dynamic team as a Senior Software Engineer.
        Work with cutting-edge technologies.
        Strong communication skills required.
        """
        result = self.parser.parse(jd_text)
        
        keywords = result.get("keywords", [])
        self.assertIsInstance(keywords, list)

    def test_parse_linkedin_format(self):
        """Test parsing LinkedIn-style JD format."""
        jd_text = """
        About the job:
        We are seeking a talented Software Engineer to join our team.
        
        Responsibilities:
        - Design and develop scalable applications
        - Collaborate with cross-functional teams
        - Write clean, well-documented code
        
        Requirements:
        • Bachelor's degree in Computer Science or equivalent
        • 3+ years of experience in software development
        • Strong proficiency in Python and JavaScript
        • Experience with React and Node.js
        
        Nice to have:
        • Experience with machine learning
        • AWS certification
        """
        result = self.parser.parse(jd_text)
        
        self.assertIn("skills", result)
        self.assertIn("summary", result)


class TestSkillTaxonomy(unittest.TestCase):
    """Test cases for skill taxonomy."""

    def test_get_canonical_name(self):
        """Test canonical name conversion."""
        self.assertEqual(SkillTaxonomy.get_canonical_name("js"), "javascript")
        self.assertEqual(SkillTaxonomy.get_canonical_name("TS"), "typescript")
        self.assertEqual(SkillTaxonomy.get_canonical_name("React Native"), "react native")

    def test_get_all_variations(self):
        """Test getting all skill variations."""
        variations = SkillTaxonomy.get_all_variations("python")
        self.assertIn("python", variations)
        self.assertIn("py", variations)

    def test_get_related_skills(self):
        """Test getting related skills."""
        related = SkillTaxonomy.get_related_skills("react")
        self.assertIn("vue", related)
        self.assertIn("angular", related)


class TestAdvancedJobMatcher(unittest.TestCase):
    """Test cases for advanced job matcher."""

    def setUp(self):
        """Set up test fixtures."""
        self.matcher = AdvancedJobMatcher(use_embeddings=False)

    def test_fuzzy_match(self):
        """Test fuzzy string matching."""
        score = self.matcher.fuzzy_match("python", "pyton")
        self.assertGreater(score, 0.8)

    def test_match_skill_with_confidence(self):
        """Test skill matching with confidence."""
        match = self.matcher.match_skill_with_confidence(
            "javascript",
            ["Python", "JavaScript", "Java"]
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, "exact")


class TestResumeJobMatching(unittest.TestCase):
    """Test resume-job matching scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.matcher = AdvancedJobMatcher(use_embeddings=False)

    def test_perfect_match(self):
        """Test scenario with perfect match."""
        resume_text = "Experienced Python developer with Django and PostgreSQL skills."
        resume_skills = ["python", "django", "postgresql"]
        jd_text = "Looking for Python Django PostgreSQL developer."
        
        result = match_resume_to_job_advanced(resume_text, resume_skills, jd_text)
        
        self.assertIsInstance(result, AdvancedJobMatchResult)
        self.assertGreater(result.overall_match, 0.5)

    def test_partial_match(self):
        """Test scenario with partial match."""
        resume_text = "Java developer with Spring experience."
        resume_skills = ["java", "spring"]
        jd_text = "Need Python developer with Django experience."
        
        result = match_resume_to_job_advanced(resume_text, resume_skills, jd_text)
        
        self.assertIsInstance(result, AdvancedJobMatchResult)
        self.assertLess(result.skill_match, 0.5)

    def test_related_skills_match(self):
        """Test matching with related skills."""
        resume_skills = ["vue", "angular"]
        jd_text = "Looking for React developer."
        
        result = match_resume_to_job_advanced("", resume_skills, jd_text)
        
        self.assertIsInstance(result, AdvancedJobMatchResult)
        self.assertGreater(len(result.related_skills), 0)

    def test_recommendations_generation(self):
        """Test that recommendations are generated."""
        resume_text = "Basic resume with few keywords."
        resume_skills = ["python"]
        jd_text = """
        Senior Java Developer with Kubernetes, Docker, AWS required.
        Must have 5+ years experience.
        """
        
        result = match_resume_to_job_advanced(resume_text, resume_skills, jd_text)
        
        self.assertIsInstance(result.recommendations, list)
        self.assertTrue(len(result.recommendations) > 0)

    def test_experience_extraction(self):
        """Test experience requirement extraction."""
        jd_text = "Must have 5+ years of experience with Python."
        
        requirements = self.matcher.extract_experience_requirements(jd_text)
        
        # The pattern extracts skill-specific experience when a skill is mentioned nearby
        self.assertTrue(len(requirements) > 0)
        self.assertTrue("_general" in requirements or "python" in requirements)

    def test_match_types_breakdown(self):
        """Test that match types are properly counted."""
        resume_text = "Skilled in Python, JavaScript, and React."
        resume_skills = ["python", "javascript", "react"]
        jd_text = "Need Python and JavaScript expert. React is a plus."
        
        result = match_resume_to_job_advanced(resume_text, resume_skills, jd_text)
        
        self.assertGreater(result.exact_matches, 0)
        self.assertIsInstance(result.synonym_matches, int)


if __name__ == "__main__":
    unittest.main()