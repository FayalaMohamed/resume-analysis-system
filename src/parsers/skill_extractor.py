#!/usr/bin/env python3
"""
Intelligent Skill Extractor

Parses skill sections from resumes to extract individual tools, technologies,
and competencies. Works with multiple languages and various formatting styles.

Usage:
    from skill_extractor import SkillExtractor
    
    extractor = SkillExtractor()
    skills = extractor.extract_from_section(raw_text, section_type='skills')
"""

import re
from typing import List, Set, Dict, Any, Optional
from pathlib import Path
import sys

# Import skill taxonomy
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.advanced_job_matcher import SkillTaxonomy


class SkillExtractor:
    """Extract individual skills from resume sections."""
    
    # Common skill separators
    SEPARATORS = r'[,;|•\-·\n\r]+'
    
    # Patterns to clean up skill text
    CLEANUP_PATTERNS = [
        (r'^\s*[\-\•·]\s*', ''),  # Leading bullets
        (r'\s+', ' '),  # Multiple spaces
        (r'^\s+|\s+$', ''),  # Leading/trailing spaces
    ]
    
    # Language-specific programming terms
    PROGRAMMING_TERMS = {
        'en': ['programming', 'development', 'software', 'tools', 'technologies'],
        'fr': ['programmation', 'développement', 'logiciels', 'outils', 'technologies', 'informatique'],
        'es': ['programación', 'desarrollo', 'software', 'herramientas'],
        'de': ['programmierung', 'entwicklung', 'software', 'tools'],
    }
    
    # Common prefixes to remove
    SKILL_PREFIXES = [
        r'(?i)^(?:programming|programmation|développement|development)\s+(?:in|en|with|avec)?\s*',
        r'(?i)^(?:software|logiciels?|outils?)\s+(?:de|d\'|for|pour)?\s*',
        r'(?i)^(?:pack|suite|microsoft)\s+',
        r'(?i)^(?:logiciels?\s+de\s+|software\s+for\s+)',
    ]
    
    def __init__(self):
        """Initialize the skill extractor."""
        self.taxonomy = SkillTaxonomy()
        self.known_skills = set(self.taxonomy.SKILL_SYNONYMS.keys())
        
        # Add common variations
        for skill, synonyms in self.taxonomy.SKILL_SYNONYMS.items():
            self.known_skills.add(skill)
            self.known_skills.update(synonyms)
    
    def extract_from_section(self, raw_text: str, section_type: str = 'skills', 
                            language: str = 'en') -> List[str]:
        """
        Extract skills from a section's raw text.
        
        Args:
            raw_text: The raw text from the skills section
            section_type: Type of section ('skills', 'software', etc.)
            language: Detected language code
            
        Returns:
            List of extracted individual skills
        """
        if not raw_text or not raw_text.strip():
            return []
        
        # Normalize text
        text = self._normalize_text(raw_text)
        
        # Remove section headers and noise
        text = self._clean_section_text(text, language)
        
        # Split by common separators
        raw_items = re.split(self.SEPARATORS, text, flags=re.IGNORECASE)
        
        # Process each item
        skills = []
        for item in raw_items:
            item = item.strip()
            if not item or len(item) < 2:
                continue
                
            # Skip if it's just punctuation
            if re.match(r'^[\s,;|•\-·]+$', item):
                continue
            
            # Extract skills from item (handles parenthetical content)
            extracted = self._extract_from_item(item)
            skills.extend(extracted)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower not in seen and len(skill) > 1:
                seen.add(skill_lower)
                unique_skills.append(skill)
        
        return unique_skills
    
    def extract_from_description(self, description: str, language: str = 'en') -> List[str]:
        """
        Extract skills from a description string.
        
        This is useful when skills are embedded in descriptions like:
        "Logiciels de CAO et de simulation dynamique Fusion 360, ADAMS"
        """
        if not description:
            return []
        
        # First, try to extract known skills from the text
        found_skills = []
        text_lower = description.lower()
        
        # Check for known skills in the text
        for skill in self.known_skills:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
        
        # Also split by separators and check each part
        parts = re.split(self.SEPARATORS, description)
        for part in parts:
            part = part.strip()
            if len(part) > 1 and not re.match(r'^[\s,;|•\-·]+$', part):
                # Clean up the part
                cleaned = self._cleanup_text(part)
                if cleaned and len(cleaned) > 1:
                    found_skills.append(cleaned)
        
        # Remove duplicates
        seen = set()
        unique_skills = []
        for skill in found_skills:
            skill_lower = skill.lower()
            if skill_lower not in seen:
                seen.add(skill_lower)
                unique_skills.append(skill)
        
        return unique_skills
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing."""
        # Replace common unicode characters
        replacements = {
            '�': '-',  # Common encoding issue
            '\xa0': ' ',  # Non-breaking space
            '\u2022': '•',  # Bullet
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def _clean_section_text(self, text: str, language: str) -> str:
        """Remove section headers and noise from text."""
        # Remove common headers
        headers_to_remove = [
            r'(?i)^\s*skills?\s*:?\s*',
            r'(?i)^\s*compétences\s*:?\s*',
            r'(?i)^\s*technical skills\s*:?\s*',
            r'(?i)^\s*programming\s*:?\s*',
            r'(?i)^\s*programmation\s*:?\s*',
            r'(?i)^\s*software\s*:?\s*',
            r'(?i)^\s*logiciels\s*:?\s*',
            r'(?i)^\s*tools\s*:?\s*',
            r'(?i)^\s*outils\s*:?\s*',
            r'(?i)^\s*technologies\s*:?\s*',
            r'(?i)^\s*languages?\s*:?\s*',
            r'(?i)^\s*langues?\s*:?\s*',
        ]
        
        for pattern in headers_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove email addresses and phone numbers
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\(?\d{2,4}\)?[\s.-]?\d{2,4}[\s.-]?\d{2,4}', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _extract_from_item(self, item: str) -> List[str]:
        """Extract skills from a single item."""
        skills = []

        # Remove prefixes like "Programmation", "Logiciels de", "Pack"
        for prefix in self.SKILL_PREFIXES:
            item = re.sub(prefix, '', item, flags=re.IGNORECASE)

        # Check for parenthetical content
        paren_match = re.match(r'^(.+?)\s*\((.+?)\)', item)
        if paren_match:
            main_skill = self._cleanup_text(paren_match.group(1).strip())
            paren_content = paren_match.group(2).strip()

            if main_skill and len(main_skill.split()) <= 4:
                skills.append(main_skill)

            # Process specializations in parentheses
            sub_items = re.split(self.SEPARATORS, paren_content)
            for sub in sub_items:
                sub = re.sub(r'[\(\)]', '', sub).strip()
                cleaned = self._cleanup_text(sub)
                if cleaned and len(cleaned) <= 4:
                    skills.append(cleaned)
        else:
            # Clean and add
            cleaned = self._cleanup_text(item)
            if cleaned and 1 < len(cleaned) <= 30:
                skills.append(cleaned)

        return skills
    
    def _cleanup_text(self, text: str) -> str:
        """Clean up skill text."""
        for pattern, replacement in self.CLEANUP_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Remove trailing punctuation
        text = re.sub(r'[,;:|\-·]+$', '', text)
        text = re.sub(r'^[,;:|\-·]+', '', text)
        
        return text.strip()
    
    def get_skill_confidence(self, skill: str) -> float:
        """
        Get confidence score for a skill (0.0 to 1.0).
        
        Higher confidence if skill is in known taxonomy.
        """
        skill_lower = skill.lower()
        
        # Exact match in taxonomy
        if skill_lower in self.known_skills:
            return 1.0
        
        # Check canonical name
        canonical = self.taxonomy.get_canonical_name(skill)
        if canonical != skill_lower:
            return 0.9
        
        # Check if it looks like a programming language or tool
        if re.match(r'^[a-zA-Z0-9\+\#\.]+$', skill):
            return 0.6
        
        return 0.5


# Convenience function for pipeline integration
def extract_skills_from_resume(unified_result, language: str = 'en') -> List[str]:
    """
    Extract all skills from a unified extraction result.

    Args:
        unified_result: Result from UnifiedResumeExtractor (or dict with sections)
        language: Detected language code

    Returns:
        List of extracted skills
    """
    extractor = SkillExtractor()
    all_skills = []

    # Handle both object and dict formats
    sections = unified_result.sections if hasattr(unified_result, 'sections') else unified_result

    for section in sections:
        # Get section_type (handle both object and dict)
        section_type = section.section_type if hasattr(section, 'section_type') else section.get('section_type', '')

        # Check if this is a skills section (also check for common variations)
        if section_type == 'skills' or 'skill' in str(section_type).lower():
            # Get raw_text (handle both object and dict)
            raw_text = section.raw_text if hasattr(section, 'raw_text') else section.get('raw_text', '')

            # Extract from raw text
            skills = extractor.extract_from_section(
                raw_text,
                section_type='skills',
                language=language
            )
            all_skills.extend(skills)

        # Also check all sections for skill-related content (catch non-classified sections)
        raw_text = section.raw_text if hasattr(section, 'raw_text') else section.get('raw_text', '')

        # Check if section title contains skill-related keywords
        section_title = section.title if hasattr(section, 'title') else section.get('title', '')
        title_lower = str(section_title).lower()

        skill_keywords = ['skill', 'competenc', 'technolog', 'logiciel', 'software', 'outil', 'tool', 'programmation']
        if any(kw in title_lower for kw in skill_keywords):
            skills = extractor.extract_from_section(
                raw_text,
                section_type='skills',
                language=language
            )
            all_skills.extend(skills)

    # Remove duplicates while preserving order
    seen = set()
    unique_skills = []
    for skill in all_skills:
        skill_lower = skill.lower()
        if skill_lower not in seen and len(skill) > 1:
            seen.add(skill_lower)
            unique_skills.append(skill)

    return unique_skills


if __name__ == '__main__':
    # Test the extractor
    extractor = SkillExtractor()

    # Test cases
    test_cases = [
        ("Development Python (Data Science, Web), Java, C/C++, PHP", 'en'),
        ("Logiciels de CAO et de simulation dynamique Fusion 360, ADAMS", 'fr'),
        ("VS Code, Git, Jira, Notion, Slack, Microsoft Office", 'en'),
        ("Programmation Matlab, Pack Microsoft Office Word, Excel", 'fr'),
        ("AWS, Docker, Kubernetes, Terraform", 'en'),
    ]

    print("Skill Extractor Tests:")
    print("=" * 60)
    for text, lang in test_cases:
        skills = extractor.extract_from_description(text, lang)
        print(f"\nInput: {text}")
        print(f"Language: {lang}")
        print(f"Extracted: {skills}")
