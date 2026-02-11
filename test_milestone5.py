#!/usr/bin/env python
"""Milestone 5 Validation Script - Job Matching System."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.analysis.job_matcher import JobDescriptionParser, JobMatchResult
from src.analysis.advanced_job_matcher import (
    AdvancedJobMatcher,
    AdvancedJobMatchResult,
    SkillTaxonomy,
    match_resume_to_job_advanced,
)


def test_jd_parsing():
    print("Testing JD parsing...")
    parser = JobDescriptionParser()
    
    jd_text = "We are looking for a Python Developer with Django experience. Required: Python, Django."
    result = parser.parse(jd_text)
    
    assert "skills" in result
    assert "required" in result["skills"]
    print(f"  [OK] JD parsing works: {len(result['skills'].get('required', []))} required skills")
    return True


def test_multi_layer_matching():
    print("Testing multi-layer matching...")
    matcher = AdvancedJobMatcher(use_embeddings=False)
    
    keyword_score, matched, missing = matcher.calculate_keyword_match(
        "python developer django",
        ["python", "django", "react"]
    )
    print(f"  [OK] Keyword matching: {keyword_score:.0%}")
    
    return True


def test_skill_taxonomy():
    print("Testing skill taxonomy...")
    assert SkillTaxonomy.get_canonical_name("js") == "javascript"
    assert SkillTaxonomy.get_canonical_name("TS") == "typescript"
    
    variations = SkillTaxonomy.get_all_variations("python")
    assert "python" in variations
    assert "py" in variations
    print(f"  [OK] Skill taxonomy works")
    return True


def test_gap_analysis():
    print("Testing gap analysis...")
    resume_skills = ["python", "django"]
    jd_text = "Need Python, Django, AWS developer."
    
    result = match_resume_to_job_advanced("", resume_skills, jd_text)
    
    print(f"  [OK] Overall match: {result.overall_match:.0%}")
    print(f"  [OK] Matched skills: {len(result.matched_skills)}")
    print(f"  [OK] Missing skills: {len(result.missing_skills)}")
    
    assert isinstance(result.recommendations, list)
    print(f"  [OK] Recommendations generated")
    return True


def test_related_skills():
    print("Testing related skills...")
    resume_skills = ["vue", "angular"]
    jd_text = "Looking for React developer."
    
    result = match_resume_to_job_advanced("", resume_skills, jd_text)
    
    assert len(result.related_skills) > 0
    print(f"  [OK] Related skills recognized: {len(result.related_skills)}")
    return True


def test_match_types():
    print("Testing match type counting...")
    resume_skills = ["python", "javascript", "react"]
    jd_text = "Need Python, JavaScript, React developer."
    
    result = match_resume_to_job_advanced("", resume_skills, jd_text)
    
    assert result.exact_matches >= 3
    print(f"  [OK] Exact matches: {result.exact_matches}")
    return True


def test_convenience_function():
    print("Testing convenience function...")
    result = match_resume_to_job_advanced("Python developer", ["python"], "Looking for Python developer")
    assert result.overall_match > 0
    print(f"  [OK] Convenience function works")
    return True


def run_milestone5_tests():
    print("\n" + "="*60)
    print("MILESTONE 5 VALIDATION - Job Matching System")
    print("="*60 + "\n")
    
    tests = [
        ("JD Parsing", test_jd_parsing),
        ("Multi-Layer Matching", test_multi_layer_matching),
        ("Skill Taxonomy", test_skill_taxonomy),
        ("Gap Analysis", test_gap_analysis),
        ("Related Skills", test_related_skills),
        ("Match Types", test_match_types),
        ("Convenience Function", test_convenience_function),
    ]
    
    passed = 0
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"  [PASS]\n")
        except Exception as e:
            print(f"  [FAIL]: {e}\n")
    
    print("="*60)
    print(f"Total: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n[MILESTONE 5 VALIDATION COMPLETE]")
        return True
    else:
        print("\n[MILESTONE 5 VALIDATION FAILED]")
        return False


if __name__ == "__main__":
    success = run_milestone5_tests()
    sys.exit(0 if success else 1)