#!/usr/bin/env python
"""Milestone 4 Validation Script - Skills Extraction & Taxonomy."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.analysis.enhanced_skills import (
    EnhancedSkillTaxonomy,
    EnhancedSkillExtractor,
    SkillGapAnalyzer,
    extract_skills_from_resume,
    analyze_skill_gaps,
    SkillCategory,
    ProficiencyLevel,
    SkillInfo,
)


def test_taxonomy_coverage():
    print("Testing taxonomy coverage...")
    categories = EnhancedSkillTaxonomy.get_all_categories()
    required = [SkillCategory.PROGRAMMING_LANGUAGE, SkillCategory.FRAMEWORK, SkillCategory.DATABASE, SkillCategory.CLOUD_DEVOPS]
    for cat in required:
        assert cat in categories
    print(f"  [OK] {len(categories)} categories available")
    return True


def test_skill_normalization():
    print("Testing skill normalization...")
    assert EnhancedSkillTaxonomy.normalize_skill("js") == "javascript"
    assert EnhancedSkillTaxonomy.normalize_skill("aws") == "amazon web services"
    print(f"  [OK] Normalization works correctly")
    return True


def test_related_skills():
    print("Testing related skills...")
    related = EnhancedSkillTaxonomy.get_related("react")
    assert "vue" in related
    print(f"  [OK] Related skills: {len(related)} skills")
    return True


def test_skill_extraction():
    print("Testing skill extraction...")
    extractor = EnhancedSkillExtractor()
    skills = extractor._extract_from_skills_section("Python, JavaScript, React, Docker, AWS, Kubernetes")
    skill_names = [s.canonical_name for s in skills]
    assert "amazon web services" in skill_names
    print(f"  [OK] Extracted {len(skills)} skills")
    return True


def test_proficiency_detection():
    print("Testing proficiency detection...")
    extractor = EnhancedSkillExtractor()
    assert extractor._detect_proficiency("Expert in Python") == ProficiencyLevel.EXPERT
    assert extractor._detect_proficiency("Senior Developer") == ProficiencyLevel.ADVANCED
    print(f"  [OK] Proficiency detection works")
    return True


def test_skill_gap_analysis():
    print("Testing skill gap analysis...")
    analyzer = SkillGapAnalyzer()
    resume_skills = [SkillInfo("Python", "python", SkillCategory.PROGRAMMING_LANGUAGE, 0.95)]
    result = analyzer.analyze(resume_skills, ["Python", "React"])
    assert len(result.matched_skills) == 1
    assert len(result.missing_skills) == 1
    print(f"  [OK] Gap analysis works")
    return True


def test_convenience_functions():
    print("Testing convenience functions...")
    skills = extract_skills_from_resume(skills_text="Python, JavaScript")
    assert len(skills) >= 2
    print(f"  [OK] Convenience functions work")
    return True


def run_milestone4_tests():
    print("\n" + "="*60)
    print("MILESTONE 4 VALIDATION - Skills Extraction & Taxonomy")
    print("="*60 + "\n")
    
    tests = [
        ("Taxonomy Coverage", test_taxonomy_coverage),
        ("Skill Normalization", test_skill_normalization),
        ("Related Skills", test_related_skills),
        ("Skill Extraction", test_skill_extraction),
        ("Proficiency Detection", test_proficiency_detection),
        ("Skill Gap Analysis", test_skill_gap_analysis),
        ("Convenience Functions", test_convenience_functions),
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
        print("\n[MILESTONE 4 VALIDATION COMPLETE]")
        return True
    else:
        print("\n[MILESTONE 4 VALIDATION FAILED]")
        return False


if __name__ == "__main__":
    success = run_milestone4_tests()
    sys.exit(0 if success else 1)