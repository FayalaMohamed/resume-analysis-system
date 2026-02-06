"""Section parser for extracting structured data from resume text."""

import re
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field

# Import constants
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import MULTILINGUAL_SECTIONS, LANGUAGE_INDICES


@dataclass
class ParsedResume:
    """Structured representation of a parsed resume."""
    contact_info: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    experience: List[Dict[str, Any]] = field(default_factory=list)
    education: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    projects: List[Dict[str, Any]] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    languages: List[Dict[str, str]] = field(default_factory=list)
    awards: List[str] = field(default_factory=list)
    publications: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    references: List[Dict[str, str]] = field(default_factory=list)
    volunteer_work: List[Dict[str, Any]] = field(default_factory=list)
    professional_affiliations: List[str] = field(default_factory=list)
    speaking_engagements: List[str] = field(default_factory=list)
    patents: List[str] = field(default_factory=list)
    workshops: List[str] = field(default_factory=list)
    activities: List[str] = field(default_factory=list)
    online_presence: Dict[str, str] = field(default_factory=dict)
    research: List[str] = field(default_factory=list)
    exhibitions: List[str] = field(default_factory=list)
    productions: List[str] = field(default_factory=list)
    teaching: List[str] = field(default_factory=list)
    clinical_experience: List[Dict[str, Any]] = field(default_factory=list)
    technical_skills: List[str] = field(default_factory=list)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""


class SectionParser:
    """Parse resume text into structured sections."""

    def __init__(self, language: str = 'auto'):
        """Initialize the section parser.

        Args:
            language: Language code ('auto' for automatic detection, or specific like 'en', 'fr', etc.)
        """
        self.language = language
        self._section_patterns = None
        self._initialize_patterns()

    def _initialize_patterns(self) -> None:
        def clean_pattern(pattern: str) -> str:
            """Remove (?i) flag from pattern since we use re.IGNORECASE."""
            return pattern.replace('(?i)', '')

        if self.language in LANGUAGE_INDICES:
            # Use only English + specific language patterns
            lang_idx = LANGUAGE_INDICES[self.language]
            patterns_dict = {}
            for section_name, pattern_list in MULTILINGUAL_SECTIONS.items():
                patterns_to_use = [pattern_list[0]]  # English is always index 0
                if lang_idx < len(pattern_list):
                    patterns_to_use.append(pattern_list[lang_idx])
                combined_pattern = '|'.join(f'({clean_pattern(p)})' for p in patterns_to_use)
                patterns_dict[section_name] = re.compile(combined_pattern, re.IGNORECASE)
            self._section_patterns = patterns_dict
        else:
            # Use all language patterns
            patterns_dict = {}
            for section_name, pattern_list in MULTILINGUAL_SECTIONS.items():
                combined_pattern = '|'.join(f'({clean_pattern(p)})' for p in pattern_list)
                patterns_dict[section_name] = re.compile(combined_pattern, re.IGNORECASE)
            self._section_patterns = patterns_dict

    def set_language(self, language: str) -> None:
        """Set the language for section detection.

        Args:
            language: Language code (e.g., 'en', 'fr', 'es', 'de', 'it', 'pt', 'auto')
        """
        self.language = language
        self._initialize_patterns()

    def identify_section_type(self, header: str) -> str:
        """Identify the type of section based on header text."""
        header_lower = header.lower().strip()

        for section_type, pattern in self._section_patterns.items():
            if pattern.search(header_lower):
                return section_type

        return 'unknown'

    def split_into_sections(self, text: str) -> List[Dict[str, Any]]:
        """Split resume text into sections."""
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []
        start_line = 0

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            section_type = self.identify_section_type(line_stripped)
            is_header = section_type != 'unknown' or self._is_likely_header(line_stripped)

            if is_header:
                if current_section:
                    sections.append({
                        'name': current_section,
                        'content': '\n'.join(current_content),
                        'section_type': self.identify_section_type(current_section)
                    })

                current_section = line_stripped
                current_content = []
            else:
                if current_section:
                    current_content.append(line_stripped)

        if current_section:
            sections.append({
                'name': current_section,
                'content': '\n'.join(current_content),
                'section_type': self.identify_section_type(current_section)
            })

        return sections

    def _is_likely_header(self, line: str) -> bool:
        """Heuristic to determine if a line is likely a section header."""
        if line.isupper() and len(line) < 50 and len(line) > 3:
            return True
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*$', line):
            return True
        return False

    def parse_list_section(self, content: str) -> List[str]:
        """Parse a section that contains a list of items."""
        items = []
        for delimiter in [',', '·', '-', '\n', '|', '/']:
            if delimiter in content:
                parts = content.split(delimiter)
                items = [s.strip() for s in parts if s.strip()]
                break

        if not items:
            items = [line.strip() for line in content.split('\n') if line.strip()]

        return items

    def parse_skills(self, content: str) -> List[str]:
        """Parse skills section into list of skills."""
        return self.parse_list_section(content)

    def parse_languages(self, content: str) -> List[Dict[str, str]]:
        """Parse languages section into structured format."""
        languages = []
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        for line in lines:
            # Look for language and proficiency level
            # Examples: "English - Native", "French (Fluent)", "Spanish: Intermediate"
            match = re.match(r'^(\w+(?:\s+\w+)?)\s*(?:[-:()]\s*)?(\w+)?', line)
            if match:
                lang = match.group(1)
                level = match.group(2) if match.group(2) else ''
                languages.append({'language': lang, 'level': level})
            else:
                languages.append({'language': line, 'level': ''})

        return languages

    def parse_contact_info(self, text: str) -> Dict[str, Any]:
        """Extract contact information from resume text."""
        contact = {
            'name': '',
            'email': '',
            'phone': '',
            'linkedin': '',
            'website': '',
        }

        lines = text.split('\n')[:30]

        for line in lines:
            line = line.strip()

            # Email
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', line)
            if email_match and not contact['email']:
                contact['email'] = email_match.group(0)

            # Phone
            phone_match = re.search(r'[\+]?[\d\s\-\(\)]{7,20}', line)
            if phone_match and not contact['phone']:
                candidate = phone_match.group(0)
                if len(re.sub(r'\D', '', candidate)) >= 7:
                    contact['phone'] = candidate

            # LinkedIn
            linkedin_match = re.search(r'linkedin\.com/in/[\w\-]+', line, re.IGNORECASE)
            if linkedin_match:
                contact['linkedin'] = linkedin_match.group(0)

            # Website
            website_match = re.search(r'(?:https?://)?[\w\.-]+\.(?:com|io|dev|fr|net|org)', line, re.IGNORECASE)
            if website_match and 'linkedin' not in line.lower():
                contact['website'] = website_match.group(0)

        # Try to extract name
        for line in text.split('\n')[:5]:
            if line.strip() and not any(x in line.lower() for x in ['@', 'http', 'phone', 'tel']):
                if len(line.strip().split()) <= 4:
                    contact['name'] = line.strip()
                    break

        return contact

    def parse(self, text: str) -> ParsedResume:
        """Parse complete resume text into structured format."""
        parsed = ParsedResume(raw_text=text)

        # Extract contact info
        parsed.contact_info = self.parse_contact_info(text)

        # Split into sections
        sections = self.split_into_sections(text)
        parsed.sections = sections

        # Parse each section
        for section in sections:
            section_type = section['section_type']
            content = section['content']

            if section_type == 'summary':
                parsed.summary = content
            elif section_type == 'skills':
                parsed.skills = self.parse_skills(content)
            elif section_type == 'languages':
                parsed.languages = self.parse_languages(content)
            elif section_type == 'projects':
                parsed.projects = self._parse_experience_like(content)
            elif section_type == 'certifications':
                parsed.certifications = self.parse_list_section(content)
            elif section_type == 'awards':
                parsed.awards = self.parse_list_section(content)
            elif section_type == 'publications':
                parsed.publications = self.parse_list_section(content)
            elif section_type == 'interests':
                parsed.interests = self.parse_list_section(content)
            elif section_type == 'references':
                parsed.references = self._parse_references(content)
            elif section_type == 'volunteer_work':
                parsed.volunteer_work = self._parse_experience_like(content)
            elif section_type == 'professional_affiliations':
                parsed.professional_affiliations = self.parse_list_section(content)
            elif section_type == 'speaking_engagements':
                parsed.speaking_engagements = self.parse_list_section(content)
            elif section_type == 'patents':
                parsed.patents = self.parse_list_section(content)
            elif section_type == 'workshops':
                parsed.workshops = self.parse_list_section(content)
            elif section_type == 'activities':
                parsed.activities = self.parse_list_section(content)
            elif section_type == 'online_presence':
                parsed.online_presence = self._parse_online_presence(content)
            elif section_type == 'research':
                parsed.research = self.parse_list_section(content)
            elif section_type == 'exhibitions':
                parsed.exhibitions = self.parse_list_section(content)
            elif section_type == 'productions':
                parsed.productions = self.parse_list_section(content)
            elif section_type == 'teaching':
                parsed.teaching = self.parse_list_section(content)
            elif section_type == 'clinical_experience':
                parsed.clinical_experience = self._parse_experience_like(content)
            elif section_type == 'technical_skills':
                parsed.technical_skills = self.parse_list_section(content)

        return parsed

    def _parse_experience_like(self, content: str) -> List[Dict[str, Any]]:
        """Parse experience-like sections (experience, volunteer work, clinical experience, etc.)."""
        entries = []
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        current_entry = None
        current_description = []

        for line in lines:
            # Check if this is a new entry (typically starts with company/organization or date)
            if re.match(r'^[A-Z][\w\s&]+(?:,|\s\-|\.|-)', line) or \
               re.match(r'^(?:19|20)\d{2}', line) or \
               (line.isupper() and len(line) < 50):
                # Save previous entry
                if current_entry:
                    current_entry['description'] = '\n'.join(current_description)
                    entries.append(current_entry)

                # Start new entry
                current_entry = {
                    'title': '',
                    'organization': line,
                    'date': '',
                    'description': ''
                }
                current_description = []
            elif current_entry:
                # This is part of the current entry's description
                if re.match(r'^(?:19|20)\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', line, re.IGNORECASE):
                    current_entry['date'] = line
                elif not current_entry['title']:
                    current_entry['title'] = line
                else:
                    current_description.append(line)

        # Don't forget the last entry
        if current_entry:
            current_entry['description'] = '\n'.join(current_description)
            entries.append(current_entry)

        return entries

    def _parse_references(self, content: str) -> List[Dict[str, str]]:
        """Parse references section."""
        references = []
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        current_ref = {}
        for line in lines:
            if line.isupper() and len(line) < 50:
                # This looks like a name
                if current_ref:
                    references.append(current_ref)
                current_ref = {'name': line, 'title': '', 'company': '', 'contact': ''}
            elif '@' in line or 'phone' in line.lower():
                current_ref['contact'] = line
            elif current_ref:
                if not current_ref['title']:
                    current_ref['title'] = line
                elif not current_ref['company']:
                    current_ref['company'] = line

        if current_ref:
            references.append(current_ref)

        return references

    def _parse_online_presence(self, content: str) -> Dict[str, str]:
        """Parse online presence section."""
        presence = {}
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        for line in lines:
            # Look for URL patterns
            url_match = re.search(r'(https?://[^\s]+)', line)
            if url_match:
                url = url_match.group(1)
                # Categorize based on domain
                if 'linkedin' in url.lower():
                    presence['linkedin'] = url
                elif 'github' in url.lower():
                    presence['github'] = url
                elif 'twitter' in url.lower() or 'x.com' in url.lower():
                    presence['twitter'] = url
                elif 'portfolio' in url.lower() or 'behance' in url.lower() or 'dribbble' in url.lower():
                    presence['portfolio'] = url
                else:
                    presence[f'website_{len(presence)}'] = url
            else:
                # Try to extract platform and handle
                parts = line.split(':', 1)
                if len(parts) == 2:
                    presence[parts[0].strip().lower()] = parts[1].strip()

        return presence
