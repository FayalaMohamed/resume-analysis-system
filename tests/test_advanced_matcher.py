"""Test the advanced job matcher."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.advanced_job_matcher import (
    AdvancedJobMatcher,
    SkillTaxonomy,
    match_resume_to_job_advanced
)


def test_skill_taxonomy():
    """Test skill taxonomy functionality."""
    print("Testing Skill Taxonomy...")
    
    # Test synonym resolution
    assert SkillTaxonomy.get_canonical_name('reactjs') == 'react'
    assert SkillTaxonomy.get_canonical_name('gcp') == 'google cloud platform'
    assert SkillTaxonomy.get_canonical_name('js') == 'javascript'
    print("✓ Synonym resolution works")
    
    # Test getting all variations
    variations = SkillTaxonomy.get_all_variations('react')
    assert 'reactjs' in variations
    assert 'react.js' in variations
    print("✓ Variations extraction works")
    
    # Test related skills
    related = SkillTaxonomy.get_related_skills('react')
    assert 'vue' in related
    assert 'angular' in related
    print("✓ Related skills work")


def test_fuzzy_matching():
    """Test fuzzy string matching."""
    print("\nTesting Fuzzy Matching...")
    
    matcher = AdvancedJobMatcher(fuzzy_threshold=0.85)
    
    # Test similar strings
    score1 = matcher.fuzzy_match('javascript', 'javascript')
    assert score1 == 1.0
    print(f"✓ Exact match: {score1}")
    
    score2 = matcher.fuzzy_match('Javascript', 'javascript')
    assert score2 == 1.0  # Case insensitive
    print(f"✓ Case insensitive match: {score2}")
    
    score3 = matcher.fuzzy_match('javascript', 'javscript')  # Typo
    assert score3 > 0.9
    print(f"✓ Typo tolerance: {score3:.2f}")


def test_skill_matching():
    """Test advanced skill matching."""
    print("\nTesting Advanced Skill Matching...")
    
    matcher = AdvancedJobMatcher()
    
    # Test exact match
    resume_skills = ['python', 'react', 'aws']
    result = matcher.match_skill_with_confidence('python', resume_skills)
    assert result is not None
    assert result.match_type == 'exact'
    assert result.confidence == 1.0
    print("✓ Exact skill matching works")
    
    # Test synonym match
    result = matcher.match_skill_with_confidence('reactjs', resume_skills)
    assert result is not None
    assert result.match_type == 'synonym'
    print("✓ Synonym matching works")
    
    # Test no match
    result = matcher.match_skill_with_confidence('java', resume_skills)
    assert result is None
    print("✓ No false positives")
    
    # Test related skills
    related = matcher.find_related_skill_matches('vue', resume_skills)
    assert len(related) > 0
    print(f"✓ Related skills found: {[r.skill_name for r in related]}")


def test_experience_extraction():
    """Test experience requirement extraction."""
    print("\nTesting Experience Extraction...")
    
    matcher = AdvancedJobMatcher()
    
    text1 = "We need 5+ years of Python experience"
    reqs1 = matcher.extract_experience_requirements(text1)
    print(f"✓ Extracted from '{text1}': {reqs1}")
    
    text2 = "Minimum 3 years experience with AWS"
    reqs2 = matcher.extract_experience_requirements(text2)
    print(f"✓ Extracted from '{text2}': {reqs2}")


def test_full_matching():
    """Test full job matching workflow."""
    print("\nTesting Full Job Matching Workflow...")
    
    resume_text = """
    Senior Software Engineer with 5+ years of experience
    Skills: Python, React.js, AWS, Docker, Kubernetes, PostgreSQL
    Experience with machine learning and data science
    """
    
    resume_skills = ['python', 'react', 'aws', 'docker', 'kubernetes', 'postgresql']
    
    jd_text = """
    Software Engineer Position
    
    Required Skills:
    - 3+ years of Python experience
    - React.js or similar frontend framework
    - AWS or cloud platforms
    
    Preferred:
    - GCP
    - MongoDB
    
    Keywords: machine learning, software development, api
    """
    
    result = match_resume_to_job_advanced(resume_text, resume_skills, jd_text)
    
    print(f"\nMatch Results:")
    print(f"  Overall Match: {result.overall_match:.1%}")
    print(f"  Skills Match: {result.skill_match:.1%}")
    print(f"  Keywords Match: {result.keyword_match:.1%}")
    print(f"  Experience Match: {result.experience_match:.1%}")
    
    print(f"\nDetailed Stats:")
    print(f"  Exact Matches: {result.exact_matches}")
    print(f"  Synonym Matches: {result.synonym_matches}")
    print(f"  Related Skills: {result.related_matches}")
    
    print(f"\nMatched Skills:")
    for skill in result.matched_skills:
        exp_info = f" ({skill.experience_years:.0f} yrs)" if skill.experience_years else ""
        print(f"  ✓ {skill.skill_name} ({skill.match_type}, {skill.confidence:.0%})" + exp_info)
    
    if result.related_skills:
        print(f"\nRelated Skills (Partial Credit):")
        for skill in result.related_skills:
            print(f"  → {skill.skill_name} ({skill.confidence:.0%}) - {skill.context}")
    
    if result.missing_skills:
        print(f"\nMissing Skills:")
        for skill in result.missing_skills:
            print(f"  ✗ {skill}")
    
    print("\n✓ Full matching workflow completed successfully")


if __name__ == "__main__":
    print("=" * 60)
    print("Advanced Job Matcher Tests")
    print("=" * 60)
    
    try:
        test_skill_taxonomy()
        test_fuzzy_matching()
        test_skill_matching()
        test_experience_extraction()
        test_full_matching()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
