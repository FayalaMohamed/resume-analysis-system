"""Tests for recommendation engine."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.recommendation_engine import (
    RecommendationEngine,
    Recommendation,
    Priority,
)


class TestRecommendationEngine(unittest.TestCase):
    """Test cases for recommendation engine."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = RecommendationEngine()

    def test_generate_recommendations_basic(self):
        """Test basic recommendation generation."""
        ats_score = {
            'total_score': 50,
            'breakdown': {
                'layout': 10,
                'format': 10,
                'content': 15,
                'structure': 15
            },
            'issues': []
        }
        content_score = {
            'action_verb_score': 10,
            'quantification_score': 10,
            'bullet_structure_score': 10,
            'conciseness_score': 20,
            'weak_verbs_found': ['was', 'did'],
            'bullets': []
        }
        layout_analysis = {
            'is_single_column': True,
            'has_tables': False,
            'section_headers': ['Experience', 'Education'],
            'word_count': 500
        }

        recommendations = self.engine.generate_recommendations(
            ats_score, content_score, layout_analysis
        )

        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) > 0)

    def test_layout_recommendations_multi_column(self):
        """Test layout recommendations for multi-column layout."""
        layout = {
            'is_single_column': False,
            'has_tables': False,
            'section_headers': [],
            'word_count': 500
        }

        recommendations = self.engine._layout_recommendations(layout)

        # Should have at least one high-priority recommendation
        high_priority = [r for r in recommendations if r.priority == Priority.HIGH]
        self.assertTrue(len(high_priority) > 0)

        # Check that the recommendation has proper fields
        for rec in recommendations:
            self.assertIsInstance(rec.category, str)
            self.assertIsInstance(rec.issue, str)
            self.assertIsInstance(rec.suggestion, str)
            self.assertIsNotNone(rec.example)

    def test_layout_recommendations_tables(self):
        """Test layout recommendations for tables."""
        layout = {
            'is_single_column': True,
            'has_tables': True,
            'section_headers': [],
            'word_count': 500
        }

        recommendations = self.engine._layout_recommendations(layout)

        table_recs = [r for r in recommendations if 'table' in r.issue.lower()]
        self.assertTrue(len(table_recs) > 0)

    def test_content_recommendations_weak_verbs(self):
        """Test content recommendations for weak verbs."""
        content = {
            'action_verb_score': 10,
            'quantification_score': 25,
            'bullet_structure_score': 25,
            'conciseness_score': 25,
            'weak_verbs_found': ['was', 'were', 'did'],
            'bullets': []
        }
        ats_score = {'total_score': 50, 'breakdown': {}}

        recommendations = self.engine._content_recommendations(content, ats_score)

        # With low action_verb_score, should have content-related recommendations
        self.assertTrue(len(recommendations) > 0)

        # Check that at least one recommendation is about content improvement
        content_recs = [r for r in recommendations if 'Content' in r.category or 'verb' in r.issue.lower() or 'action' in r.issue.lower()]
        self.assertTrue(len(content_recs) > 0)

    def test_content_recommendations_quantification(self):
        """Test content recommendations for quantification."""
        content = {
            'action_verb_score': 25,
            'quantification_score': 10,
            'bullet_structure_score': 25,
            'conciseness_score': 25,
            'weak_verbs_found': [],
            'bullets': []
        }
        ats_score = {'total_score': 50, 'breakdown': {}}

        recommendations = self.engine._content_recommendations(content, ats_score)

        quant_recs = [r for r in recommendations if 'quant' in r.issue.lower() or 'metric' in r.issue.lower()]
        self.assertTrue(len(quant_recs) > 0)

    def test_content_recommendations_bullet_structure(self):
        """Test content recommendations for bullet structure."""
        content = {
            'action_verb_score': 25,
            'quantification_score': 25,
            'bullet_structure_score': 10,
            'conciseness_score': 25,
            'weak_verbs_found': [],
            'bullets': []
        }
        ats_score = {'total_score': 50, 'breakdown': {}}

        recommendations = self.engine._content_recommendations(content, ats_score)

        bullet_recs = [r for r in recommendations if 'bullet' in r.issue.lower() or 'format' in r.category.lower()]
        self.assertTrue(len(bullet_recs) > 0)

    def test_content_recommendations_conciseness(self):
        """Test content recommendations for conciseness."""
        content = {
            'action_verb_score': 25,
            'quantification_score': 25,
            'bullet_structure_score': 25,
            'conciseness_score': 10,
            'weak_verbs_found': [],
            'bullets': []
        }
        ats_score = {'total_score': 50, 'breakdown': {}}

        recommendations = self.engine._content_recommendations(content, ats_score)

        concise_recs = [r for r in recommendations if 'concis' in r.category.lower() or 'verbos' in r.issue.lower()]
        self.assertTrue(len(concise_recs) > 0)

    def test_ats_recommendations_content(self):
        """Test ATS recommendations for content issues."""
        ats_score = {
            'total_score': 50,
            'breakdown': {
                'content': 10,
                'format': 25,
                'layout': 25,
                'structure': 25
            },
            'issues': []
        }

        recommendations = self.engine._ats_recommendations(ats_score)

        ats_recs = [r for r in recommendations if r.category == 'ATS']
        self.assertTrue(len(ats_recs) > 0)

    def test_ats_recommendations_format(self):
        """Test ATS recommendations for format issues."""
        ats_score = {
            'total_score': 50,
            'breakdown': {
                'content': 25,
                'format': 10,
                'layout': 25,
                'structure': 25
            },
            'issues': []
        }

        recommendations = self.engine._ats_recommendations(ats_score)

        ats_recs = [r for r in recommendations if r.category == 'ATS']
        self.assertTrue(len(ats_recs) > 0)

    def test_job_match_recommendations_low_match(self):
        """Test job match recommendations for low match."""
        job_match = {
            'overall_match': 0.3,
            'missing_skills': ['python', 'aws', 'docker'],
            'missing_keywords': ['senior', 'leadership']
        }

        recommendations = self.engine._job_match_recommendations(job_match)

        # Should have high priority recommendation for low match
        high_priority = [r for r in recommendations if r.priority == Priority.HIGH]
        self.assertTrue(len(high_priority) > 0)

        # Check missing skills recommendation
        skill_recs = [r for r in recommendations if 'skill' in r.category.lower() or 'missing' in r.issue.lower()]
        self.assertTrue(len(skill_recs) > 0)

    def test_job_match_recommendations_missing_keywords(self):
        """Test job match recommendations for missing keywords."""
        job_match = {
            'overall_match': 0.7,
            'missing_skills': [],
            'missing_keywords': ['agile', 'scrum', 'jira']
        }

        recommendations = self.engine._job_match_recommendations(job_match)

        keyword_recs = [r for r in recommendations if 'keyword' in r.category.lower()]
        self.assertTrue(len(keyword_recs) > 0)

    def test_priority_sorting(self):
        """Test that recommendations are properly sorted by priority."""
        ats_score = {
            'total_score': 30,
            'breakdown': {
                'content': 5,
                'format': 5,
                'layout': 5,
                'structure': 5
            },
            'issues': []
        }
        content_score = {
            'action_verb_score': 5,
            'quantification_score': 5,
            'bullet_structure_score': 5,
            'conciseness_score': 5,
            'weak_verbs_found': ['was', 'were', 'did'],
            'bullets': []
        }
        layout_analysis = {
            'is_single_column': False,
            'has_tables': True,
            'section_headers': [],
            'word_count': 500
        }

        recommendations = self.engine.generate_recommendations(
            ats_score, content_score, layout_analysis
        )

        # Should have HIGH priority recommendations first
        if len(recommendations) > 1:
            priority_order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
            for i in range(len(recommendations) - 1):
                self.assertLessEqual(
                    priority_order[recommendations[i].priority],
                    priority_order[recommendations[i + 1].priority]
                )

    def test_get_priority_summary(self):
        """Test getting priority summary."""
        ats_score = {
            'total_score': 50,
            'breakdown': {
                'layout': 10,
                'format': 10,
                'content': 15,
                'structure': 15
            },
            'issues': []
        }
        content_score = {
            'action_verb_score': 10,
            'quantification_score': 10,
            'bullet_structure_score': 10,
            'conciseness_score': 20,
            'weak_verbs_found': ['was', 'did'],
            'bullets': []
        }
        layout_analysis = {
            'is_single_column': True,
            'has_tables': False,
            'section_headers': ['Experience', 'Education'],
            'word_count': 500
        }

        recommendations = self.engine.generate_recommendations(
            ats_score, content_score, layout_analysis
        )

        summary = self.engine.get_priority_summary(recommendations)

        self.assertIn('high', summary)
        self.assertIn('medium', summary)
        self.assertIn('low', summary)
        self.assertGreaterEqual(summary['high'], 0)

    def test_format_recommendations(self):
        """Test format recommendations."""
        layout = {
            'is_single_column': True,
            'has_tables': False,
            'section_headers': [],
            'word_count': 500
        }

        recommendations = self.engine._layout_recommendations(layout)

        # Should have structure recommendations when headers are minimal
        structure_recs = [r for r in recommendations if 'structure' in r.category.lower() or 'header' in r.issue.lower()]
        self.assertTrue(len(structure_recs) > 0)

    def test_enhancement_recommendations(self):
        """Test enhancement recommendations for good resumes."""
        content = {
            'action_verb_score': 20,
            'quantification_score': 20,
            'bullet_structure_score': 25,
            'conciseness_score': 25,
            'weak_verbs_found': [],
            'bullets': ['Achievement 1', 'Achievement 2']
        }
        ats_score = {
            'total_score': 85,
            'breakdown': {
                'layout': 25,
                'format': 20,
                'content': 20,
                'structure': 20
            },
            'issues': []
        }

        recommendations = self.engine._content_recommendations(content, ats_score)

        # Should have enhancement recommendations for scores 75-95
        enhancement_recs = [r for r in recommendations if r.category == 'Enhancement']
        self.assertTrue(len(enhancement_recs) > 0)


class TestRecommendation(unittest.TestCase):
    """Test cases for Recommendation dataclass."""

    def test_create_recommendation(self):
        """Test creating a recommendation."""
        rec = Recommendation(
            category="Content",
            priority=Priority.HIGH,
            issue="Weak verbs",
            suggestion="Use strong action verbs",
            example="❌ 'Was responsible for'\n✅ 'Led team of 5'",
            estimated_impact="+10 points"
        )

        self.assertEqual(rec.category, "Content")
        self.assertEqual(rec.priority, Priority.HIGH)
        self.assertEqual(rec.issue, "Weak verbs")
        self.assertIn("Led", rec.example)

    def test_recommendation_default_values(self):
        """Test recommendation default values."""
        rec = Recommendation(
            category="Test",
            priority=Priority.LOW,
            issue="Test issue",
            suggestion="Test suggestion"
        )

        self.assertIsNone(rec.example)
        self.assertEqual(rec.estimated_impact, "")


if __name__ == "__main__":
    unittest.main()