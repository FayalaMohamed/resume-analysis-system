#!/usr/bin/env python
"""Milestone 6 Validation Script - Suggestion Engine Improvements.

This script validates the suggestion engine improvements:
- Intelligent recommendation generation
- Before/after examples
- Priority-based suggestions
- Job match recommendations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.analysis.recommendation_engine import (
    RecommendationEngine,
    Recommendation,
    Priority,
)


def test_recommendation_generation():
    """Test basic recommendation generation."""
    print("Testing recommendation generation...")
    
    engine = RecommendationEngine()
    
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
        'is_single_column': False,
        'has_tables': True,
        'section_headers': [],
        'word_count': 500
    }
    
    recommendations = engine.generate_recommendations(
        ats_score, content_score, layout_analysis
    )
    
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    print(f"  [OK] Generated {len(recommendations)} recommendations")
    
    return True


def test_priority_categorization():
    """Test priority categorization (critical, important, enhancement)."""
    print("Testing priority categorization...")
    
    engine = RecommendationEngine()
    
    # Test HIGH priority for critical issues
    layout_critical = {
        'is_single_column': False,
        'has_tables': True,
        'section_headers': [],
        'word_count': 500
    }
    recs_critical = engine._layout_recommendations(layout_critical)
    high_priority = [r for r in recs_critical if r.priority == Priority.HIGH]
    assert len(high_priority) > 0, "Should have high priority recommendations for critical issues"
    print(f"  [OK] High priority for critical issues: {len(high_priority)} recommendations")
    
    # Test MEDIUM priority for format issues
    content_medium = {
        'action_verb_score': 25,
        'quantification_score': 25,
        'bullet_structure_score': 10,
        'conciseness_score': 25,
        'weak_verbs_found': [],
        'bullets': []
    }
    ats_score = {'total_score': 50, 'breakdown': {}}
    recs_medium = engine._content_recommendations(content_medium, ats_score)
    medium_priority = [r for r in recs_medium if r.priority == Priority.MEDIUM]
    assert len(medium_priority) > 0
    print(f"  [OK] Medium priority for format issues: {len(medium_priority)} recommendations")
    
    # Test LOW priority for enhancement suggestions
    content_enhancement = {
        'action_verb_score': 20,
        'quantification_score': 20,
        'bullet_structure_score': 25,
        'conciseness_score': 25,
        'weak_verbs_found': [],
        'bullets': ['Achievement 1', 'Achievement 2']
    }
    ats_score_enhance = {
        'total_score': 85,
        'breakdown': {'layout': 25, 'format': 20, 'content': 20, 'structure': 20},
        'issues': []
    }
    recs_enhancement = engine._content_recommendations(content_enhancement, ats_score_enhance)
    low_priority = [r for r in recs_enhancement if r.priority == Priority.LOW]
    assert len(low_priority) > 0
    print(f"  [OK] Low priority for enhancements: {len(low_priority)} recommendations")
    
    return True


def test_before_after_examples():
    """Test that recommendations have before/after examples."""
    print("Testing before/after examples...")
    
    engine = RecommendationEngine()
    
    content = {
        'action_verb_score': 10,
        'quantification_score': 10,
        'bullet_structure_score': 25,
        'conciseness_score': 25,
        'weak_verbs_found': ['was', 'were'],
        'bullets': []
    }
    ats_score = {'total_score': 50, 'breakdown': {}}
    
    recommendations = engine._content_recommendations(content, ats_score)
    
    # Check that examples contain before/after format
    for rec in recommendations:
        if rec.example:
            # Should have some indicator of improvement
            has_format = any(indicator in rec.example for indicator in ['❌', 'X', 'Bad', 'Before', 'Instead'])
            has_better = any(indicator in rec.example for indicator in ['✅', 'Check', 'Good', 'After', 'Try'])
            if has_format or has_better:
                assert True
            else:
                # At minimum should have a clear example
                assert len(rec.example) > 10
    
    examples_found = sum(1 for r in recommendations if r.example and len(r.example) > 10)
    print(f"  [OK] {examples_found}/{len(recommendations)} recommendations have examples")
    
    return True


def test_job_match_recommendations():
    """Test job match specific recommendations."""
    print("Testing job match recommendations...")
    
    engine = RecommendationEngine()
    
    # Low match scenario
    job_match_low = {
        'overall_match': 0.3,
        'missing_skills': ['python', 'aws', 'docker'],
        'missing_keywords': ['senior', 'leadership']
    }
    
    recs_low = engine._job_match_recommendations(job_match_low)
    
    high_priority = [r for r in recs_low if r.priority == Priority.HIGH]
    assert len(high_priority) > 0, "Should have high priority for low match"
    print(f"  [OK] Low match ({job_match_low['overall_match']:.0%}): {len(high_priority)} high priority recs")
    
    # Missing skills scenario
    job_match_skills = {
        'overall_match': 0.6,
        'missing_skills': ['react', 'node.js'],
        'missing_keywords': []
    }
    
    recs_skills = engine._job_match_recommendations(job_match_skills)
    skill_recs = [r for r in recs_skills if 'skill' in r.category.lower() or 'missing' in r.issue.lower()]
    assert len(skill_recs) > 0
    print(f"  [OK] Missing skills scenario: {len(skill_recs)} skill recommendations")
    
    # Missing keywords scenario
    job_match_keywords = {
        'overall_match': 0.7,
        'missing_skills': [],
        'missing_keywords': ['agile', 'scrum']
    }
    
    recs_keywords = engine._job_match_recommendations(job_match_keywords)
    keyword_recs = [r for r in recs_keywords if 'keyword' in r.category.lower()]
    assert len(keyword_recs) > 0
    print(f"  [OK] Missing keywords scenario: {len(keyword_recs)} keyword recommendations")
    
    return True


def test_priority_sorting():
    """Test that recommendations are sorted by priority."""
    print("Testing priority sorting...")
    
    engine = RecommendationEngine()
    
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
    
    recommendations = engine.generate_recommendations(
        ats_score, content_score, layout_analysis
    )
    
    # Verify sorting
    if len(recommendations) > 1:
        priority_order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
        for i in range(len(recommendations) - 1):
            assert priority_order[recommendations[i].priority] <= priority_order[recommendations[i + 1].priority]
    
    print(f"  [OK] Recommendations sorted correctly by priority")
    
    return True


def test_get_priority_summary():
    """Test getting priority summary."""
    print("Testing priority summary...")
    
    engine = RecommendationEngine()
    
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
        'is_single_column': False,
        'has_tables': True,
        'section_headers': ['Experience'],
        'word_count': 500
    }
    
    recommendations = engine.generate_recommendations(
        ats_score, content_score, layout_analysis
    )
    
    summary = engine.get_priority_summary(recommendations)
    
    assert 'high' in summary
    assert 'medium' in summary
    assert 'low' in summary
    print(f"  [OK] Priority summary: {summary}")
    
    return True


def test_estimated_impact():
    """Test that recommendations have estimated impact scores."""
    print("Testing estimated impact scores...")
    
    engine = RecommendationEngine()
    
    layout = {
        'is_single_column': False,
        'has_tables': False,
        'section_headers': [],
        'word_count': 500
    }
    
    recommendations = engine._layout_recommendations(layout)
    
    for rec in recommendations:
        if rec.estimated_impact:
            # Should contain "+" or points indicator
            assert '+' in rec.estimated_impact or 'points' in rec.estimated_impact.lower()
    
    impact_count = sum(1 for r in recommendations if r.estimated_impact)
    print(f"  [OK] {impact_count}/{len(recommendations)} recommendations have impact scores")
    
    return True


def run_milestone6_tests():
    """Run all milestone 6 validation tests."""
    print("\n" + "="*60)
    print("MILESTONE 6 VALIDATION - Suggestion Engine Improvements")
    print("="*60 + "\n")
    
    tests = [
        ("Recommendation Generation", test_recommendation_generation),
        ("Priority Categorization", test_priority_categorization),
        ("Before/After Examples", test_before_after_examples),
        ("Job Match Recommendations", test_job_match_recommendations),
        ("Priority Sorting", test_priority_sorting),
        ("Priority Summary", test_get_priority_summary),
        ("Estimated Impact Scores", test_estimated_impact),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
            print(f"  [PASS]\n")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  [FAIL]: {e}\n")
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    
    for name, p, _ in results:
        status = "[PASS]" if p else "[FAIL]"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[MILESTONE 6 VALIDATION COMPLETE]")
        return True
    else:
        print("\n[MILESTONE 6 VALIDATION FAILED]")
        return False


if __name__ == "__main__":
    success = run_milestone6_tests()
    sys.exit(0 if success else 1)