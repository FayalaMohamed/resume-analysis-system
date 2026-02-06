"""ATS simulation module - show what ATS systems see."""

import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ATSSimulationResult:
    """Results from ATS simulation."""
    plain_text: str = ""
    extracted_sections: Dict[str, str] = field(default_factory=dict)
    detected_skills: List[str] = field(default_factory=list)
    detected_contact: Dict[str, str] = field(default_factory=dict)
    
    lost_content: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    readability_score: float = 0.0
    parsing_confidence: float = 0.0


class ATSSimulator:
    """Simulate how ATS systems parse resumes."""
    
    def __init__(self):
        """Initialize ATS simulator."""
        pass
    
    def simulate_parsing(self, text: str, layout_info: Optional[Dict[str, Any]] = None) -> ATSSimulationResult:
        """Simulate ATS parsing of resume.
        
        Args:
            text: Resume text
            layout_info: Optional layout analysis info
            
        Returns:
            ATSSimulationResult with simulation results
        """
        # Convert to plain text (ATS systems typically do this)
        plain_text = self._extract_plain_text(text)
        
        # Extract sections
        sections = self._extract_sections(plain_text)
        
        # Detect skills
        skills = self._detect_skills(plain_text)
        
        # Extract contact info
        contact = self._extract_contact(plain_text)
        
        # Identify potentially lost content
        lost_content = self._identify_lost_content(text, layout_info)
        
        # Generate warnings
        warnings = self._generate_warnings(text, layout_info, sections)
        
        # Calculate scores
        readability = self._calculate_readability(plain_text)
        confidence = self._calculate_parsing_confidence(sections, contact, layout_info)
        
        return ATSSimulationResult(
            plain_text=plain_text[:2000] + "..." if len(plain_text) > 2000 else plain_text,
            extracted_sections=sections,
            detected_skills=skills,
            detected_contact=contact,
            lost_content=lost_content,
            warnings=warnings,
            readability_score=readability,
            parsing_confidence=confidence,
        )
    
    def _extract_plain_text(self, text: str) -> str:
        """Extract plain text (simulate ATS stripping formatting)."""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove special characters but keep structure
        text = re.sub(r'[^\w\s\n\-\.\@\+]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections as ATS would."""
        sections = {}
        
        # Common section headers
        section_patterns = {
            'experience': r'(?:experience|work experience|employment|professional experience)',
            'education': r'(?:education|academic|qualifications)',
            'skills': r'(?:skills|technical skills|competencies)',
            'summary': r'(?:summary|objective|profile)',
            'contact': r'(?:contact|personal information)',
        }
        
        lines = text.split('\n')
        current_section = None
        section_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            for section_name, pattern in section_patterns.items():
                if re.search(pattern, line_lower) and len(line) < 50:
                    # Save previous section
                    if current_section:
                        sections[current_section] = '\n'.join(section_content)
                    
                    current_section = section_name
                    section_content = []
                    break
            else:
                if current_section and line.strip():
                    section_content.append(line)
        
        # Save last section
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content)
        
        return sections
    
    def _detect_skills(self, text: str) -> List[str]:
        """Detect skills as ATS would."""
        # Common technical skills
        skill_patterns = [
            r'\bpython\b', r'\bjava\b', r'\bjavascript\b', r'\bjs\b', r'\btypescript\b',
            r'\breact\b', r'\bangular\b', r'\bvue\b', r'\bnode\.?js\b', r'\bdjango\b',
            r'\bsql\b', r'\bmysql\b', r'\bpostgresql\b', r'\bmongodb\b', r'\bredis\b',
            r'\baws\b', r'\bazure\b', r'\bgcp\b', r'\bdocker\b', r'\bkubernetes\b',
            r'\bgit\b', r'\bgithub\b', r'\bjenkins\b', r'\bci/cd\b',
            r'\bmachine learning\b', r'\bdata science\b', r'\banalysis\b',
            r'\bagile\b', r'\bscrum\b', r'\bproject management\b',
            r'\bleadership\b', r'\bcommunication\b', r'\bteamwork\b',
        ]
        
        text_lower = text.lower()
        found_skills = []
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                # Clean up the skill name
                skill = pattern.replace(r'\b', '').replace('\\.', '.')
                found_skills.append(skill)
        
        return list(set(found_skills))
    
    def _extract_contact(self, text: str) -> Dict[str, str]:
        """Extract contact information."""
        contact = {
            'name': '',
            'email': '',
            'phone': '',
            'location': '',
        }
        
        lines = text.split('\n')[:20]  # Check first 20 lines
        
        for line in lines:
            line = line.strip()
            
            # Email
            if not contact['email']:
                email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', line)
                if email_match:
                    contact['email'] = email_match.group(0)
            
            # Phone
            if not contact['phone']:
                phone_match = re.search(r'[\+]?[\d\s\-\(\)]{7,20}', line)
                if phone_match:
                    contact['phone'] = phone_match.group(0)
            
            # Potential name (first non-email, non-url line with 2-4 words)
            if not contact['name']:
                words = line.split()
                if 2 <= len(words) <= 4:
                    if '@' not in line and 'http' not in line and not any(w.isdigit() for w in words):
                        contact['name'] = line
        
        return contact
    
    def _identify_lost_content(
        self,
        original_text: str,
        layout_info: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Identify content that may be lost to ATS."""
        lost = []
        
        if layout_info:
            if layout_info.get('has_tables', False):
                lost.append("Table content may not be parsed correctly")
            
            if not layout_info.get('is_single_column', True):
                lost.append("Multi-column content may be read out of order")
        
        # Check for image references
        if re.search(r'\.(jpg|jpeg|png|gif)\b', original_text, re.IGNORECASE):
            lost.append("Images cannot be parsed by ATS")
        
        # Check for special formatting
        if '│' in original_text or '|' in original_text:
            lost.append("Vertical separators may confuse parsing")
        
        return lost
    
    def _generate_warnings(
        self,
        text: str,
        layout_info: Optional[Dict[str, Any]],
        sections: Dict[str, str]
    ) -> List[str]:
        """Generate ATS parsing warnings."""
        warnings = []
        
        # Layout warnings
        if layout_info:
            if not layout_info.get('is_single_column', True):
                warnings.append("Multi-column layout may cause parsing errors")
            
            if layout_info.get('has_tables', False):
                warnings.append("Tables may not be readable by ATS")
        
        # Section warnings
        if 'experience' not in sections:
            warnings.append("Experience section not clearly identified")
        
        if 'education' not in sections:
            warnings.append("Education section not clearly identified")
        
        if 'skills' not in sections:
            warnings.append("Skills section not clearly identified")
        
        # Content warnings
        if len(text) < 500:
            warnings.append("Resume is very short - may lack sufficient content")
        
        if len(text) > 10000:
            warnings.append("Resume is very long - may be truncated by some ATS")
        
        # Check for unusual characters
        unusual_chars = set(text) - set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t.,;:!?@#$%&*()-_=+[]{}|\'"<>/')
        if unusual_chars:
            warnings.append(f"Unusual characters detected: {''.join(list(unusual_chars)[:5])}")
        
        return warnings
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score."""
        # Simple Flesch Reading Ease approximation
        sentences = max(len(re.split(r'[.!?]+', text)), 1)
        words = len(text.split())
        syllables = len(re.findall(r'[aeiouAEIOU]+', text))
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Flesch Reading Ease formula (simplified)
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words if words > 0 else 0
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1
        normalized = (score - 0) / 100
        return max(0.0, min(1.0, normalized))
    
    def _calculate_parsing_confidence(
        self,
        sections: Dict[str, str],
        contact: Dict[str, str],
        layout_info: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in ATS parsing."""
        confidence = 1.0
        
        # Penalize missing sections
        if not sections:
            confidence -= 0.3
        
        required_sections = ['experience', 'education', 'skills']
        for section in required_sections:
            if section not in sections:
                confidence -= 0.1
        
        # Penalize missing contact info
        if not contact['email']:
            confidence -= 0.1
        
        # Penalize layout issues
        if layout_info:
            if not layout_info.get('is_single_column', True):
                confidence -= 0.2
            
            if layout_info.get('has_tables', False):
                confidence -= 0.15
        
        return max(0.0, confidence)
    
    def compare_versions(
        self,
        original_text: str,
        ats_text: str
    ) -> Dict[str, Any]:
        """Compare original to ATS-parsed version.
        
        Args:
            original_text: Original resume text
            ats_text: ATS-parsed text
            
        Returns:
            Comparison results
        """
        original_words = set(original_text.lower().split())
        ats_words = set(ats_text.lower().split())
        
        lost_words = original_words - ats_words
        preserved_words = original_words & ats_words
        
        word_retention = len(preserved_words) / len(original_words) if original_words else 0
        
        return {
            'original_word_count': len(original_text.split()),
            'ats_word_count': len(ats_text.split()),
            'words_preserved': len(preserved_words),
            'words_lost': len(lost_words),
            'retention_rate': word_retention,
            'lost_keywords': list(lost_words)[:20],
        }


# Convenience function
def simulate_ats_parsing(text: str, layout_info: Optional[Dict[str, Any]] = None) -> ATSSimulationResult:
    """Simulate ATS parsing.
    
    Args:
        text: Resume text
        layout_info: Optional layout analysis
        
    Returns:
        ATSSimulationResult
    """
    simulator = ATSSimulator()
    return simulator.simulate_parsing(text, layout_info)
