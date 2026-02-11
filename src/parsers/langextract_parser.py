#!/usr/bin/env python3
"""
LangExtract Resume Parser

High-quality LLM-based resume extraction using Google's LangExtract library.
Provides detailed structured information including granular experience items,
categorized skills, and source grounding.

Usage:
    from parsers.langextract_parser import LangExtractResumeParser
    
    parser = LangExtractResumeParser(model_id="gemini-2.5-flash")
    result = parser.extract_from_pdf("resume.pdf")
    
    # Or use with text directly
    result = parser.extract_from_text(resume_text)
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import traceback

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Check for langextract availability
try:
    import langextract as lx
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    print("[WARN] langextract not installed. Install with: pip install langextract")

# Import constants (prompts and examples)
try:
    from .langextract_constants import (
        RESUME_EXTRACTION_PROMPT,
        create_resume_examples,
        DEFAULT_CONFIG,
    )
except ImportError:
    # Fallback if constants file is not available
    RESUME_EXTRACTION_PROMPT = ""
    create_resume_examples = lambda: []
    DEFAULT_CONFIG = {}


@dataclass
class ExperienceItem:
    """A work experience entry."""
    job_title: str = ""
    company: str = ""
    date_range: str = ""
    location: str = ""
    employment_type: str = ""  # Full-time, Part-time, Contract, Internship
    description: str = ""
    bullet_points: List[Dict] = field(default_factory=list)  # Each with text and metrics
    technologies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EducationItem:
    """An education entry."""
    degree: str = ""
    field_of_study: str = ""
    institution: str = ""
    date_range: str = ""
    gpa: str = ""
    coursework: List[str] = field(default_factory=list)
    honors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Skill:
    """A skill with categorization."""
    name: str = ""
    category: str = ""  # programming_language, framework, cloud_platform, etc.
    parent_category: str = ""  # Programming, Frameworks, Cloud, etc.
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Certification:
    """A certification entry."""
    name: str = ""
    provider: str = ""
    date_obtained: str = ""
    expiration_date: str = ""
    level: str = ""  # Professional, Associate, etc.
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Project:
    """A project entry."""
    name: str = ""
    description: str = ""
    technologies: List[str] = field(default_factory=list)
    url: str = ""
    bullet_points: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ContactInfo:
    """Contact information."""
    name: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    github: str = ""
    website: str = ""
    location: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LangExtractResult:
    """Complete structured resume extraction result."""
    success: bool = False
    error_message: str = ""
    duration_ms: float = 0.0
    
    # Contact
    contact: ContactInfo = field(default_factory=ContactInfo)
    
    # Summary
    summary: str = ""
    objective: str = ""
    
    # Experience
    experience: List[ExperienceItem] = field(default_factory=list)
    
    # Education
    education: List[EducationItem] = field(default_factory=list)
    
    # Skills
    skills: List[Skill] = field(default_factory=list)
    
    # Certifications
    certifications: List[Certification] = field(default_factory=list)
    
    # Projects
    projects: List[Project] = field(default_factory=list)
    
    # Languages
    languages: List[Dict] = field(default_factory=list)  # {language, proficiency}
    
    # Additional
    awards: List[Dict] = field(default_factory=list)
    publications: List[Dict] = field(default_factory=list)
    volunteer: List[Dict] = field(default_factory=list)
    
    # Raw extractions for debugging
    raw_extractions: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'error_message': self.error_message,
            'duration_ms': self.duration_ms,
            'contact': self.contact.to_dict(),
            'summary': self.summary,
            'objective': self.objective,
            'experience': [e.to_dict() for e in self.experience],
            'education': [e.to_dict() for e in self.education],
            'skills': [s.to_dict() for s in self.skills],
            'certifications': [c.to_dict() for c in self.certifications],
            'projects': [p.to_dict() for p in self.projects],
            'languages': self.languages,
            'awards': self.awards,
            'publications': self.publications,
            'volunteer': self.volunteer,
            'raw_extractions_count': len(self.raw_extractions),
        }


class LangExtractResumeParser:
    """
    High-quality resume parser using Google's LangExtract library.
    
    Provides detailed extraction with:
    - Granular experience items with bullet points
    - Categorized skills (programming, cloud, database, etc.)
    - Source grounding for verification
    - Certifications, projects, languages, and more
    """
    
    def __init__(self, model_id: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initialize the LangExtract parser.
        
        Args:
            model_id: Model to use (gemini-2.5-flash, gemini-2.5-pro, etc.)
            api_key: API key (if None, uses LANGEXTRACT_API_KEY from env)
        """
        if not LANGEXTRACT_AVAILABLE:
            raise ImportError("langextract not installed. Run: pip install langextract")
        
        self.model_id = model_id
        self.api_key = api_key or os.environ.get('LANGEXTRACT_API_KEY')
        
        if not self.api_key:
            print("[WARN] LANGEXTRACT_API_KEY not set. Set it in .env file or environment.")
        
        # Use constants from langextract_constants module
        self.examples = create_resume_examples() if LANGEXTRACT_AVAILABLE else []
        self.prompt = RESUME_EXTRACTION_PROMPT
    
    def is_available(self) -> bool:
        """Check if LangExtract is available and configured."""
        return LANGEXTRACT_AVAILABLE and bool(self.api_key)
    
    def extract_from_text(self, text: str, extraction_passes: int = 1) -> LangExtractResult:
        """
        Extract structured information from resume text.
        
        Args:
            text: Resume text content
            extraction_passes: Number of extraction passes (1-3, higher = more thorough but slower)
            
        Returns:
            LangExtractResult with structured data
        """
        result = LangExtractResult()
        
        if not self.is_available():
            result.error_message = "LangExtract not available or API key not configured"
            return result
        
        if not text or not text.strip():
            result.error_message = "Empty text provided"
            return result
        
        try:
            import time
            start_time = time.time()
            
            # Build extraction kwargs
            kwargs = {
                'text_or_documents': text,
                'prompt_description': self.prompt,
                'examples': self.examples,
                'model_id': self.model_id,
                'extraction_passes': extraction_passes,
                'max_workers': 2,
                'max_char_buffer': 4000,
            }
            
            if self.api_key:
                kwargs['api_key'] = self.api_key
            
            # Run extraction
            extraction_result = lx.extract(**kwargs)
            
            result.duration_ms = (time.time() - start_time) * 1000
            result.success = True
            
            # Process extractions
            self._process_extractions(result, extraction_result)
            
        except Exception as e:
            result.error_message = str(e)
            print(f"[ERROR] LangExtract extraction failed: {e}")
            traceback.print_exc()
        
        return result
    
    def extract_from_pdf(self, pdf_path: Union[str, Path], extraction_passes: int = 1) -> LangExtractResult:
        """
        Extract structured information from a PDF resume.
        
        Args:
            pdf_path: Path to PDF file
            extraction_passes: Number of extraction passes (1-3)
            
        Returns:
            LangExtractResult with structured data
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            result = LangExtractResult()
            result.error_message = f"File not found: {pdf_path}"
            return result
        
        # Extract text from PDF using PyMuPDF
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            if not text.strip():
                result = LangExtractResult()
                result.error_message = "No text extracted from PDF"
                return result
            
            return self.extract_from_text(text, extraction_passes)
            
        except Exception as e:
            result = LangExtractResult()
            result.error_message = f"PDF extraction failed: {e}"
            return result
    
    def _process_extractions(self, result: LangExtractResult, extraction_result):
        """Process raw extractions into structured result."""
        
        # Track current items for parent-child relationships
        current_experience = None
        current_project = None
        
        for extraction in extraction_result.extractions:
            ext_class = extraction.extraction_class.lower()
            text = extraction.extraction_text
            attrs = extraction.attributes if hasattr(extraction, 'attributes') else {}
            
            # Store raw extraction
            result.raw_extractions.append({
                'class': extraction.extraction_class,
                'text': text,
                'attributes': attrs,
            })
            
            # Contact Information
            if 'contact' in ext_class:
                if 'name' in ext_class:
                    result.contact.name = text
                elif 'email' in ext_class:
                    result.contact.email = text
                elif 'phone' in ext_class:
                    result.contact.phone = text
                elif 'linkedin' in ext_class:
                    result.contact.linkedin = text
                elif 'github' in ext_class:
                    result.contact.github = text
                elif 'website' in ext_class:
                    result.contact.website = text
                elif 'location' in ext_class:
                    result.contact.location = text
            
            # Summary
            elif 'summary' in ext_class:
                result.summary = text
            elif 'objective' in ext_class:
                result.objective = text
            
            # Experience (but not education)
            elif ext_class == 'experience':
                exp = ExperienceItem(
                    job_title=attrs.get('job_title', ''),
                    company=attrs.get('company', ''),
                    date_range=attrs.get('date_range', ''),
                    location=attrs.get('location', ''),
                    employment_type=attrs.get('employment_type', ''),
                )
                result.experience.append(exp)
                current_experience = exp
                current_project = None
            
            # Education
            elif ext_class == 'education':
                edu = EducationItem(
                    degree=attrs.get('degree', ''),
                    field=attrs.get('field', ''),
                    institution=attrs.get('institution', ''),
                    date_range=attrs.get('date_range', ''),
                    gpa=attrs.get('gpa', ''),
                )
                result.education.append(edu)
            
            # Skills
            elif ext_class == 'skill':
                skill = Skill(
                    name=text,
                    category=attrs.get('category', ''),
                    parent_category=attrs.get('parent_category', ''),
                )
                result.skills.append(skill)
            
            # Certifications
            elif ext_class == 'certification':
                cert = Certification(
                    name=attrs.get('name', text),
                    provider=attrs.get('provider', ''),
                    date_obtained=attrs.get('date_obtained', ''),
                    expiration_date=attrs.get('expiration_date', ''),
                    level=attrs.get('level', ''),
                )
                result.certifications.append(cert)
            
            # Projects
            elif ext_class == 'project':
                proj = Project(
                    name=attrs.get('name', text),
                    description=attrs.get('description', ''),
                    technologies=attrs.get('technologies', '').split(', ') if attrs.get('technologies') else [],
                    url=attrs.get('url', ''),
                )
                result.projects.append(proj)
                current_project = proj
                current_experience = None
            
            # Bullet points (achievements/responsibilities)
            elif 'bullet' in ext_class:
                bullet = {
                    'text': text,
                    'has_metric': attrs.get('has_metric', False),
                    'metric': attrs.get('metric', ''),
                    'parent': attrs.get('parent', ''),
                }
                
                # Try to associate with current experience or project
                if current_experience:
                    current_experience.bullet_points.append(bullet)
                elif current_project:
                    current_project.bullet_points.append(text)
            
            # Languages
            elif ext_class == 'language':
                result.languages.append({
                    'language': attrs.get('language', text.split('(')[0].strip()),
                    'proficiency': attrs.get('proficiency', ''),
                })
            
            # Awards
            elif ext_class == 'award':
                result.awards.append({
                    'name': text,
                    'attributes': attrs,
                })
            
            # Publications
            elif ext_class == 'publication':
                result.publications.append({
                    'name': text,
                    'attributes': attrs,
                })
            
            # Volunteer
            elif ext_class == 'volunteer':
                result.volunteer.append({
                    'name': text,
                    'attributes': attrs,
                })


def extract_with_langextract(pdf_path: str, model_id: str = "gemini-2.5-flash", 
                              extraction_passes: int = 1) -> Dict:
    """
    Convenience function to extract resume data using LangExtract.
    
    Args:
        pdf_path: Path to PDF file
        model_id: Model to use
        extraction_passes: Number of extraction passes
        
    Returns:
        Dictionary with extraction results
    """
    parser = LangExtractResumeParser(model_id=model_id)
    result = parser.extract_from_pdf(pdf_path, extraction_passes)
    return result.to_dict()


# For testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python langextract_parser.py <resume.pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    print(f"Testing LangExtract parser on: {pdf_path}")
    print("="*70)
    
    parser = LangExtractResumeParser()
    
    if not parser.is_available():
        print("[ERROR] LangExtract not available. Check installation and API key.")
        sys.exit(1)
    
    result = parser.extract_from_pdf(pdf_path)
    
    if result.success:
        print(f"✓ Extraction successful in {result.duration_ms:.0f}ms")
        print(f"\nContact: {result.contact.name}")
        print(f"Email: {result.contact.email}")
        print(f"\nExperience entries: {len(result.experience)}")
        print(f"Education entries: {len(result.education)}")
        print(f"Skills extracted: {len(result.skills)}")
        print(f"Certifications: {len(result.certifications)}")
        print(f"Projects: {len(result.projects)}")
        
        if result.skills:
            print(f"\nSkills by category:")
            categories = {}
            for skill in result.skills:
                cat = skill.category or 'uncategorized'
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(skill.name)
            
            for cat, skills in categories.items():
                print(f"  {cat}: {', '.join(skills[:5])}{'...' if len(skills) > 5 else ''}")
        
        # Save to JSON
        output_path = Path(pdf_path).stem + "_langextract.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_path}")
    else:
        print(f"✗ Extraction failed: {result.error_message}")
