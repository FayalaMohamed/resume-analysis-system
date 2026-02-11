"""Resume Content Understanding module.

Phase 3 improvements for deep content analysis:
- Section detection and classification
- Content quality analysis (action verbs, quantification, bullets)
- Red flag detection with severity scoring
- Content enrichment insights

This module provides comprehensive understanding of resume content
beyond simple text extraction.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import Counter


class SectionType(Enum):
    """Standard section types for classification."""
    CONTACT = "contact"
    SUMMARY = "summary"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    SKILLS = "skills"
    PROJECTS = "projects"
    CERTIFICATIONS = "certifications"
    AWARDS = "awards"
    PUBLICATIONS = "publications"
    LANGUAGES = "languages"
    INTERESTS = "interests"
    REFERENCES = "references"
    UNKNOWN = "unknown"


class Severity(Enum):
    """Severity levels for red flags."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SectionInfo:
    """Information about a detected section."""
    section_type: SectionType
    title: str
    confidence: float
    raw_text: str = ""
    items: List[Dict] = field(default_factory=list)
    line_range: Tuple[int, int] = (0, 0)


@dataclass
class ContentQualityMetrics:
    """Metrics for content quality assessment."""
    action_verb_score: float = 0.0
    quantification_score: float = 0.0
    bullet_quality_score: float = 0.0
    conciseness_score: float = 0.0
    overall_score: float = 0.0
    
    strong_verbs: List[str] = field(default_factory=list)
    weak_verbs: List[str] = field(default_factory=list)
    quantified_achievements: List[str] = field(default_factory=list)
    bullet_issues: List[Dict] = field(default_factory=list)


@dataclass
class RedFlag:
    """A detected red flag in the resume."""
    category: str
    description: str
    severity: Severity
    evidence: str = ""
    suggestion: str = ""


@dataclass
class ContentEnrichment:
    """Enriched content insights."""
    estimated_seniority: str = ""
    career_trajectory: List[Dict] = field(default_factory=list)
    transferable_skills: List[str] = field(default_factory=list)
    skill_experience_years: Dict[str, float] = field(default_factory=dict)
    total_experience_years: float = 0.0
    key_themes: List[str] = field(default_factory=list)


@dataclass
class ContentUnderstandingResult:
    """Complete content understanding result."""
    sections: List[SectionInfo] = field(default_factory=list)
    missing_critical_sections: List[str] = field(default_factory=list)
    quality_metrics: ContentQualityMetrics = field(default_factory=ContentQualityMetrics)
    red_flags: List[RedFlag] = field(default_factory=list)
    enrichment: ContentEnrichment = field(default_factory=ContentEnrichment)
    overall_confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "sections": [
                {
                    "type": s.section_type.value,
                    "title": s.title,
                    "confidence": s.confidence,
                    "item_count": len(s.items),
                }
                for s in self.sections
            ],
            "missing_critical_sections": self.missing_critical_sections,
            "quality_metrics": {
                "action_verb_score": self.quality_metrics.action_verb_score,
                "quantification_score": self.quality_metrics.quantification_score,
                "bullet_quality_score": self.quality_metrics.bullet_quality_score,
                "conciseness_score": self.quality_metrics.conciseness_score,
                "overall_score": self.quality_metrics.overall_score,
                "strong_verbs": self.quality_metrics.strong_verbs,
                "weak_verbs": self.quality_metrics.weak_verbs,
                "quantified_achievements": self.quality_metrics.quantified_achievements,
            },
            "red_flags": [
                {
                    "category": f.category,
                    "description": f.description,
                    "severity": f.severity.value,
                    "evidence": f.evidence,
                    "suggestion": f.suggestion,
                }
                for f in self.red_flags
            ],
            "enrichment": {
                "estimated_seniority": self.enrichment.estimated_seniority,
                "career_trajectory": self.enrichment.career_trajectory,
                "transferable_skills": self.enrichment.transferable_skills,
                "total_experience_years": self.enrichment.total_experience_years,
                "key_themes": self.enrichment.key_themes,
            },
            "overall_confidence": self.overall_confidence,
        }


class SectionDetector:
    """Detect and classify resume sections."""
    
    SECTION_PATTERNS = {
        SectionType.CONTACT: [
            r'^contact\s*information?$',
            r'^contact\s*info$',
            r'^personal\s*information?$',
        ],
        SectionType.SUMMARY: [
            r'^professional\s*summary$',
            r'^career\s*objective$',
            r'^summary$',
            r'^objective$',
            r'^profile$',
            r'^about\s*me$',
        ],
        SectionType.EXPERIENCE: [
            r'^experience$',
            r'^work\s*experience$',
            r'^employment$',
            r'^professional\s*experience$',
            r'^work\s*history$',
            r'^career\s*history$',
            r'^relevant\s*experience$',
        ],
        SectionType.EDUCATION: [
            r'^education$',
            r'^academic\s*background$',
            r'^qualifications$',
            r'^educational\s*qualification',
        ],
        SectionType.SKILLS: [
            r'^skills$',
            r'^technical\s*skills$',
            r'^core\s*competencies$',
            r'^key\s*skills$',
            r'^proficiencies$',
            r'^areas\s*of\s*expertise$',
        ],
        SectionType.PROJECTS: [
            r'^projects$',
            r'^personal\s*projects$',
            r'^key\s*projects$',
            r'^project\s*experience$',
        ],
        SectionType.CERTIFICATIONS: [
            r'^certifications$',
            r'^certificates$',
            r'^professional\s*certification',
        ],
        SectionType.AWARDS: [
            r'^awards$',
            r'^honors$',
            r'^achievements$',
            r'^recognition',
        ],
        SectionType.PUBLICATIONS: [
            r'^publications$',
            r'^papers$',
            r'^research',
        ],
        SectionType.LANGUAGES: [
            r'^languages$',
            r'^language\s*proficiency',
        ],
        SectionType.INTERESTS: [
            r'^interests$',
            r'^hobbies$',
            r'^personal\s*interests$',
        ],
        SectionType.REFERENCES: [
            r'^references$',
        ],
    }
    
    CRITICAL_SECTIONS = [SectionType.EXPERIENCE, SectionType.EDUCATION]
    
    def __init__(self):
        """Initialize section detector."""
        pass
    
    def detect_sections(self, text: str) -> List[SectionInfo]:
        """Detect and classify sections in resume text.
        
        Args:
            text: Full resume text
            
        Returns:
            List of detected sections with metadata
        """
        lines = text.split('\n')
        sections = []
        current_section = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if line is a section header
            section_type = self._classify_line(line)
            if section_type:
                # Get section content
                content_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    next_type = self._classify_line(next_line)
                    if next_type and len(next_line) < 50:  # Short line likely a header
                        break
                    content_lines.append(next_line)
                    j += 1
                
                # Create section info
                section = SectionInfo(
                    section_type=section_type,
                    title=line,
                    confidence=self._calculate_confidence(line, section_type),
                    raw_text='\n'.join(content_lines),
                    line_range=(i, j),
                )
                sections.append(section)
                i = j
            else:
                i += 1
        
        return sections
    
    def _classify_line(self, line: str) -> Optional[SectionType]:
        """Classify a line as a section header.
        
        Args:
            line: Line text to classify
            
        Returns:
            SectionType if classified, None otherwise
        """
        line_lower = line.lower().strip()
        
        if not line_lower:
            return None
        
        # Check if line is a header (short, all caps or title case)
        if len(line) > 60 or len(line) < 3:
            return None
        
        # Check if line is all caps (likely a header)
        is_all_caps = line.isupper() and len(line) > 4
        
        # Check against patterns
        for section_type, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, line_lower, re.IGNORECASE):
                    return section_type
        
        # If all caps and short, might be a header
        if is_all_caps:
            return SectionType.UNKNOWN
        
        return None
    
    def _calculate_confidence(self, title: str, section_type: SectionType) -> float:
        """Calculate confidence score for section detection."""
        title_lower = title.lower()
        patterns = self.SECTION_PATTERNS.get(section_type, [])
        
        for pattern in patterns:
            if re.search(pattern, title_lower, re.IGNORECASE):
                return 1.0
        
        return 0.5
    
    def get_missing_sections(self, sections: List[SectionInfo]) -> List[str]:
        """Check for missing critical sections."""
        found_types = {s.section_type for s in sections}
        missing = []
        
        for critical in self.CRITICAL_SECTIONS:
            if critical not in found_types:
                missing.append(critical.value)
        
        return missing


class ContentQualityAnalyzer:
    """Analyze content quality metrics."""
    
    STRONG_VERBS = [
        'achieved', 'advanced', 'analyzed', 'built', 'chaired', 'changed',
        'collaborated', 'communicated', 'completed', 'conducted', 'created',
        'designed', 'developed', 'directed', 'earned', 'enhanced', 'established',
        'executed', 'facilitated', 'formed', 'generated', 'implemented', 'improved',
        'initiated', 'innovated', 'led', 'managed', 'mentored', 'negotiated',
        'optimized', 'organized', 'performed', 'planned', 'prepared', 'presented',
        'produced', 'programmed', 'projected', 'proposed', 'resolved', 'reviewed',
        'scheduled', 'secured', 'simplified', 'solved', 'spearheaded', 'structured',
        'trained', 'transformed', 'wrote',
    ]
    
    WEAK_VERBS = [
        'assisted', 'attempted', 'began', 'believed', 'collaborated', 'contributed',
        'dealt', 'done', 'experienced', 'familiar', 'good', 'got', 'had',
        'helped', 'involved', 'know', 'made', 'managed', 'participated',
        'performed', 'responsible', 'saw', 'served', 'supported', 'took',
        'tried', 'used', 'utilized', 'worked',
    ]
    
    QUANTITY_PATTERNS = [
        r'\$[\d,]+',  # Dollar amounts
        r'\d+%\s',    # Percentages
        r'\d+\s*(million|billion|thousand)',  # Large numbers
        r'\d+\s*(years?|yrs?)',  # Years
        r'\d+\s*(months?|mos?)',  # Months
        r'\d+\s*(users?|customers?|clients?|employees?|team\s*members?)',  # People
        r'(increased|decreased|improved|reduced)\s*by\s*\d+',  # Change metrics
    ]
    
    BULLET_PREFIXES = ['•', '-', '*', '·', '○', '●', '▸', '▪']
    
    def __init__(self):
        """Initialize content quality analyzer."""
        pass
    
    def analyze(self, text: str) -> ContentQualityMetrics:
        """Analyze content quality of resume.
        
        Args:
            text: Resume text to analyze
            
        Returns:
            ContentQualityMetrics with all scores
        """
        metrics = ContentQualityMetrics()
        
        # Extract bullet points
        bullets = self._extract_bullets(text)
        
        # Analyze action verbs
        metrics.strong_verbs = self._find_strong_verbs(text)
        metrics.weak_verbs = self._find_weak_verbs(text)
        metrics.action_verb_score = self._score_action_verbs(
            metrics.strong_verbs, metrics.weak_verbs, bullets
        )
        
        # Analyze quantification
        metrics.quantified_achievements = self._find_quantified(text)
        metrics.quantification_score = self._score_quantification(
            metrics.quantified_achievements, bullets
        )
        
        # Analyze bullet quality
        metrics.bullet_issues = self._analyze_bullet_quality(bullets)
        metrics.bullet_quality_score = self._score_bullet_quality(
            metrics.bullet_issues, bullets
        )
        
        # Analyze conciseness
        metrics.conciseness_score = self._score_conciseness(text, bullets)
        
        # Calculate overall score
        metrics.overall_score = self._calculate_overall(metrics)
        
        return metrics
    
    def _extract_bullets(self, text: str) -> List[str]:
        """Extract bullet points from text."""
        bullets = []
        for line in text.split('\n'):
            line = line.strip()
            if any(line.startswith(p) for p in self.BULLET_PREFIXES):
                bullets.append(line)
            elif re.match(r'^[\-\*]\s', line):
                bullets.append(line)
        return bullets
    
    def _find_strong_verbs(self, text: str) -> List[str]:
        """Find strong action verbs in text."""
        found = []
        text_lower = text.lower()
        for verb in self.STRONG_VERBS:
            if verb in text_lower:
                found.append(verb)
        return list(set(found))
    
    def _find_weak_verbs(self, text: str) -> List[str]:
        """Find weak or overused verbs in text."""
        found = []
        text_lower = text.lower()
        for verb in self.WEAK_VERBS:
            if verb in text_lower:
                found.append(verb)
        return list(set(found))
    
    def _score_action_verbs(self, strong: List[str], weak: List[str], bullets: List[str]) -> float:
        """Score action verb usage."""
        if not bullets:
            return 50.0
        
        total = len(bullets)
        if total == 0:
            return 50.0
        
        # Count strong vs weak
        text = ' '.join(bullets).lower()
        strong_count = sum(1 for v in strong if v in text)
        weak_count = sum(1 for v in weak if v in text)
        
        # Score: bonus for strong, penalty for weak
        strong_bonus = min(50, strong_count * 12)
        weak_penalty = min(20, weak_count * 8)
        score = 40.0 + strong_bonus - weak_penalty
        
        return max(0, min(100, score))
    
    def _find_quantified(self, text: str) -> List[str]:
        """Find quantified achievements."""
        found = []
        
        # Dollar amounts
        dollar_matches = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
        found.extend(dollar_matches)
        
        # Percentages
        pct_matches = re.findall(r'\d+(?:\.\d+)?%', text)
        found.extend(pct_matches)
        
        # Large numbers
        large_matches = re.findall(r'\d+\s*(million|billion|thousand)', text, re.IGNORECASE)
        found.extend([f"{m[0]} {m[1]}" for m in large_matches])
        
        # Years
        year_matches = re.findall(r'\d+\s*(years?|yrs?)', text, re.IGNORECASE)
        found.extend(year_matches)
        
        # People
        people_matches = re.findall(r'\d+\s*(users?|customers?|clients?|employees?|team\s*members?)', text, re.IGNORECASE)
        found.extend(people_matches)
        
        return list(set(found))
    
    def _score_quantification(self, quantified: List[str], bullets: List[str]) -> float:
        """Score quantification density."""
        if not bullets:
            return 50.0
        
        total = len(bullets)
        quantified_count = len(quantified)
        
        # Ideal: 30-50% of bullets have metrics
        ratio = quantified_count / total if total > 0 else 0
        
        if ratio >= 0.3 and ratio <= 0.5:
            return 100.0
        elif ratio > 0.5:
            return 70.0  # Maybe too many
        elif ratio >= 0.1:
            return 60.0
        else:
            return 30.0  # Needs more quantification
    
    def _analyze_bullet_quality(self, bullets: List[str]) -> List[Dict]:
        """Analyze individual bullet points for quality issues."""
        issues = []
        
        for i, bullet in enumerate(bullets):
            bullet_issues = []
            
            # Check length
            word_count = len(bullet.split())
            if word_count < 3:
                bullet_issues.append("too_short")
            elif word_count > 40:
                bullet_issues.append("too_long")
            
            # Check for duty vs achievement
            duty_words = ['responsible', 'duty', 'task', 'work']
            achievement_words = ['achieved', 'increased', 'improved', 'reduced', 'saved']
            
            has_duty = any(w in bullet.lower() for w in duty_words)
            has_achievement = any(w in bullet.lower() for w in achievement_words)
            
            if has_duty and not has_achievement:
                bullet_issues.append("duty_focused")
            
            if bullet_issues:
                issues.append({
                    "bullet": bullet[:50] + "..." if len(bullet) > 50 else bullet,
                    "issues": bullet_issues,
                })
        
        return issues
    
    def _score_bullet_quality(self, issues: List[Dict], bullets: List[str]) -> float:
        """Score overall bullet quality."""
        if not bullets:
            return 50.0
        
        issue_count = len(issues)
        issue_ratio = issue_count / len(bullets)
        
        return max(0, 100 - (issue_ratio * 100))
    
    def _score_conciseness(self, text: str, bullets: List[str]) -> float:
        """Score conciseness of content."""
        word_count = len(text.split())
        
        # Ideal: 300-800 words
        if 300 <= word_count <= 800:
            return 100.0
        elif word_count < 300:
            return 50.0 + (word_count / 300 * 50)
        elif word_count < 1200:
            return 100.0 - ((word_count - 800) / 400 * 30)
        else:
            return max(30, 70 - ((word_count - 1200) / 800 * 40))
    
    def _calculate_overall(self, metrics: ContentQualityMetrics) -> float:
        """Calculate overall content quality score."""
        weights = {
            'action_verb': 0.25,
            'quantification': 0.30,
            'bullet_quality': 0.25,
            'conciseness': 0.20,
        }
        
        return (
            metrics.action_verb_score * weights['action_verb'] +
            metrics.quantification_score * weights['quantification'] +
            metrics.bullet_quality_score * weights['bullet_quality'] +
            metrics.conciseness_score * weights['conciseness']
        )


class RedFlagDetector:
    """Detect potential red flags in resume."""
    
    def __init__(self):
        """Initialize red flag detector."""
        pass
    
    def detect(self, text: str, sections: List[SectionInfo]) -> List[RedFlag]:
        """Detect red flags in resume.
        
        Args:
            text: Resume text
            sections: Detected sections
            
        Returns:
            List of detected red flags
        """
        flags = []
        
        # Check contact info
        contact_flags = self._check_contact_info(text, sections)
        flags.extend(contact_flags)
        
        # Check employment gaps
        gap_flags = self._check_employment_gaps(text, sections)
        flags.extend(gap_flags)
        
        # Check job hopping
        hopping_flags = self._check_job_hopping(text, sections)
        flags.extend(hopping_flags)
        
        # Check outdated content
        outdated_flags = self._check_outdated_content(text, sections)
        flags.extend(outdated_flags)
        
        # Check for unprofessional elements
        unprofessional_flags = self._check_unprofessional(text)
        flags.extend(unprofessional_flags)
        
        return flags
    
    def _check_contact_info(self, text: str, sections: List[SectionInfo]) -> List[RedFlag]:
        """Check for missing contact information."""
        flags = []
        
        has_email = bool(re.search(r'\S+@\S+\.\S+', text))
        has_phone = bool(re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text))
        
        # Also check for common email patterns
        email_patterns = [
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            r'\b\w+@\w+\.\w+\b',
        ]
        has_email = any(re.search(p, text) for p in email_patterns) or has_email
        
        if not has_email:
            flags.append(RedFlag(
                category="contact",
                description="No email address found",
                severity=Severity.HIGH if not has_phone else Severity.MEDIUM,
                suggestion="Add a professional email address",
            ))
        
        if not has_phone:
            flags.append(RedFlag(
                category="contact",
                description="No phone number found",
                severity=Severity.MEDIUM,
                suggestion="Add a phone number",
            ))
        
        return flags
    
    def _check_employment_gaps(self, text: str, sections: List[SectionInfo]) -> List[RedFlag]:
        """Check for employment gaps."""
        flags = []
        
        # Extract dates
        date_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        
        # This is simplified - would need more sophisticated date parsing
        # to accurately detect gaps
        
        return flags
    
    def _check_job_hopping(self, text: str, sections: List[SectionInfo]) -> List[RedFlag]:
        """Check for job hopping pattern."""
        flags = []
        
        # Count distinct companies/titles
        company_pattern = r'(?:at|for|@)\s+([A-Z][a-zA-Z\s&]+?)(?:\s+\d{4}|$)'
        companies = re.findall(company_pattern, text)
        
        if len(companies) > 5:
            flags.append(RedFlag(
                category="stability",
                description=f"Multiple job changes detected ({len(companies)} positions)",
                severity=Severity.LOW,
                evidence=f"Found {len(companies)} different positions",
                suggestion="Consider emphasizing achievements and stability factors",
            ))
        
        return flags
    
    def _check_outdated_content(self, text: str, sections: List[SectionInfo]) -> List[RedFlag]:
        """Check for outdated skills or experience."""
        flags = []
        
        # Check for outdated technologies
        outdated_tech = [
            'visual basic 6', 'vb6', 'perl', 'cobol', 'fortran', 'pascal',
            'sharepoint 2010', 'windows 98', 'windows xp',
        ]
        
        text_lower = text.lower()
        found_outdated = [t for t in outdated_tech if t in text_lower]
        
        if found_outdated:
            flags.append(RedFlag(
                category="relevance",
                description="Potentially outdated technologies found",
                severity=Severity.LOW,
                evidence=", ".join(found_outdated),
                suggestion="Update to current technologies or remove if no longer used",
            ))
        
        return flags
    
    def _check_unprofessional(self, text: str) -> List[RedFlag]:
        """Check for unprofessional content."""
        flags = []
        
        # Check for unprofessional email
        email_pattern = r'\b(\w+@(gmail|yahoo|hotmail|outlook)\.com)\b'
        matches = re.findall(email_pattern, text, re.IGNORECASE)
        
        # Check for unusual domains
        unusual = re.findall(r'@\s*([a-z]{5,}\.(com|net|org))', text, re.IGNORECASE)
        
        if unusual:
            flags.append(RedFlag(
                category="professionalism",
                description="Unusual email domain detected",
                severity=Severity.LOW,
                suggestion="Consider using a more professional email",
            ))
        
        return flags


class ContentEnricher:
    """Enrich resume content with insights."""
    
    SENIORITY_INDICATORS = {
        'junior': ['junior', 'jr', 'entry', 'intern', 'trainee', 'associate'],
        'mid': ['senior', 'sr', 'specialist', 'analyst', 'engineer', 'developer'],
        'senior': ['lead', 'principal', 'staff', 'senior', 'manager', 'director', 'head', 'chief', 'vp'],
    }
    
    TRAJECTORY_KEYWORDS = {
        'promotion': ['promoted', 'promotion', 'advanced', 'elevated'],
        'expansion': ['expanded', 'grew', 'increased', 'scaled'],
        'leadership': ['led', 'managed', 'directed', 'headed', 'overseen'],
    }
    
    def __init__(self):
        """Initialize content enricher."""
        pass
    
    def enrich(self, text: str, sections: List[SectionInfo]) -> ContentEnrichment:
        """Enrich resume content with insights.
        
        Args:
            text: Resume text
            sections: Detected sections
            
        Returns:
            ContentEnrichment with insights
        """
        enrichment = ContentEnrichment()
        
        # Estimate seniority
        enrichment.estimated_seniority = self._estimate_seniority(text, sections)
        
        # Detect career trajectory
        enrichment.career_trajectory = self._detect_trajectory(text, sections)
        
        # Identify transferable skills
        enrichment.transferable_skills = self._identify_transferable(text)
        
        # Estimate experience years per skill
        enrichment.skill_experience_years = self._estimate_skill_experience(text)
        
        # Calculate total experience
        enrichment.total_experience_years = self._calculate_total_experience(text, sections)
        
        # Extract key themes
        enrichment.key_themes = self._extract_themes(text, sections)
        
        return enrichment
    
    def _estimate_seniority(self, text: str, sections: List[SectionInfo]) -> str:
        """Estimate seniority level from content."""
        text_lower = text.lower()
        
        scores = {'junior': 0, 'mid': 0, 'senior': 0}
        
        for level, keywords in self.SENIORITY_INDICATORS.items():
            for keyword in keywords:
                count = text_lower.count(keyword)
                scores[level] += count
        
        if scores['senior'] > scores['mid'] and scores['senior'] > scores['junior']:
            return 'senior'
        elif scores['mid'] > scores['junior']:
            return 'mid-level'
        else:
            return 'junior'
    
    def _detect_trajectory(self, text: str, sections: List[SectionInfo]) -> List[Dict]:
        """Detect career trajectory patterns."""
        trajectory = []
        
        text_lower = text.lower()
        
        for pattern_type, keywords in self.TRAJECTORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    trajectory.append({
                        'type': pattern_type,
                        'keyword': keyword,
                        'evidence': True,
                    })
        
        return trajectory
    
    def _identify_transferable(self, text: str) -> List[str]:
        """Identify transferable skills."""
        transferable = []
        
        transferables = [
            'communication', 'leadership', 'teamwork', 'problem solving',
            'project management', 'analytical skills', 'presentation',
            'mentoring', 'collaboration', 'negotiation',
        ]
        
        text_lower = text.lower()
        for skill in transferables:
            if skill in text_lower:
                transferable.append(skill)
        
        return transferable
    
    def _estimate_skill_experience(self, text: str) -> Dict[str, float]:
        """Estimate years of experience per skill."""
        experience = {}
        
        # Pattern: X years of experience with Y skill
        pattern1 = r'(\d+)\+?\s*years?\s*(?:of\s+)?(?:experience\s+)?(?:in|with|using)\s+([a-zA-Z]+)'
        matches1 = re.findall(pattern1, text, re.IGNORECASE)
        for match in matches1:
            if len(match) == 2:
                years = float(match[0])
                skill = match[1].lower()
                experience[skill] = max(experience.get(skill, 0), years)
        
        # Pattern: Y skill for X years
        pattern2 = r'([a-zA-Z]+)\s+(?:for\s+)?(\d+)\+?\s*years?'
        matches2 = re.findall(pattern2, text, re.IGNORECASE)
        for match in matches2:
            if len(match) == 2:
                skill = match[0].lower()
                years = float(match[1])
                experience[skill] = max(experience.get(skill, 0), years)
        
        return experience
    
    def _calculate_total_experience(self, text: str, sections: List[SectionInfo]) -> float:
        """Calculate total years of experience."""
        # Look for date patterns
        year_pattern = r'(19|20)\d{2}'
        years = re.findall(year_pattern, text)
        
        if len(years) >= 2:
            years_int = [int(y) for y in years]
            min_year = min(years_int)
            max_year = max(years_int)
            return max_year - min_year
        
        return 0.0
    
    def _extract_themes(self, text: str, sections: List[SectionInfo]) -> List[str]:
        """Extract key themes/topics from resume."""
        themes = []
        
        # Common resume themes
        tech_themes = ['python', 'javascript', 'react', 'aws', 'cloud', 'data', 'api']
        management_themes = ['lead', 'manage', 'team', 'project', 'stakeholder']
        soft_themes = ['communication', 'collaboration', 'problem', 'solution']
        
        text_lower = text.lower()
        
        for theme in tech_themes:
            if theme in text_lower:
                themes.append(f"Tech: {theme}")
        
        for theme in management_themes:
            if theme in text_lower:
                themes.append(f"Leadership: {theme}")
        
        for theme in soft_themes:
            if theme in text_lower:
                themes.append(f"Soft Skills: {theme}")
        
        return themes[:5]  # Limit to top 5 themes


class ContentUnderstandingEngine:
    """Main engine for resume content understanding."""
    
    def __init__(self):
        """Initialize content understanding engine."""
        self.section_detector = SectionDetector()
        self.quality_analyzer = ContentQualityAnalyzer()
        self.red_flag_detector = RedFlagDetector()
        self.content_enricher = ContentEnricher()
    
    def analyze(self, text: str) -> ContentUnderstandingResult:
        """Perform complete content understanding analysis.
        
        Args:
            text: Resume text to analyze
            
        Returns:
            ContentUnderstandingResult with all insights
        """
        result = ContentUnderstandingResult()
        
        # Detect sections
        sections = self.section_detector.detect_sections(text)
        result.sections = sections
        
        # Check for missing critical sections
        result.missing_critical_sections = self.section_detector.get_missing_sections(sections)
        
        # Analyze content quality
        result.quality_metrics = self.quality_analyzer.analyze(text)
        
        # Detect red flags
        result.red_flags = self.red_flag_detector.detect(text, sections)
        
        # Enrich content
        result.enrichment = self.content_enricher.enrich(text, sections)
        
        # Calculate overall confidence
        result.overall_confidence = self._calculate_confidence(result)
        
        return result
    
    def _calculate_confidence(self, result: ContentUnderstandingResult) -> float:
        """Calculate overall confidence score."""
        base = 0.9
        
        # Penalize for missing sections
        if result.missing_critical_sections:
            base -= len(result.missing_critical_sections) * 0.05
        
        # Adjust for red flags
        high_severity = sum(1 for f in result.red_flags if f.severity == Severity.HIGH)
        if high_severity > 0:
            base -= high_severity * 0.02
        
        return max(0.5, min(1.0, base))


def analyze_resume_content(text: str) -> Dict[str, Any]:
    """Convenience function for resume content analysis.
    
    Args:
        text: Resume text to analyze
        
    Returns:
        Dictionary with all analysis results
    """
    engine = ContentUnderstandingEngine()
    result = engine.analyze(text)
    return result.to_dict()