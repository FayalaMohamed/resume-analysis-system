#!/usr/bin/env python3
"""
LangExtract vs Existing System Comparison Script

This script compares Google's LangExtract library with the existing ATS extraction system.
It tests both approaches on the same resume samples and provides detailed metrics.

Usage:
    python test_langextract_comparison.py [--resume <path>] [--output <dir>]
    
Requirements:
    pip install langextract
    
API Key Setup:
    Option 1: Create a .env file in the project root with:
        LANGEXTRACT_API_KEY=your-api-key-here
    
    Option 2: Set environment variable:
        export LANGEXTRACT_API_KEY=your-api-key-here
    
    Get your API key from: https://aistudio.google.com/app/apikey
"""

import os
import sys
import json
import time
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import traceback

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add src to path for existing system
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Check for langextract
try:
    import langextract as lx
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    print("WARNING: langextract not installed. Install with: pip install langextract")

# Import existing system
from parsers.unified_extractor import UnifiedResumeExtractor, extract_unified
from parsers.skill_extractor import extract_skills_from_resume


@dataclass
class ExtractionMetrics:
    """Metrics for a single extraction run."""
    system: str  # 'langextract' or 'existing'
    duration_ms: float = 0.0
    success: bool = False
    error_message: str = ""
    
    # Resume data metrics
    name_found: bool = False
    email_found: bool = False
    phone_found: bool = False
    num_sections: int = 0
    num_experience_items: int = 0
    num_education_items: int = 0
    num_skills: int = 0
    
    # Quality metrics
    has_source_grounding: bool = False
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ComparisonResult:
    """Complete comparison result for a resume."""
    resume_path: str
    timestamp: str
    text_length: int
    
    existing_metrics: ExtractionMetrics = None
    langextract_metrics: ExtractionMetrics = None
    
    existing_result: Dict = field(default_factory=dict)
    langextract_result: Dict = field(default_factory=dict)
    
    # Detailed comparison
    section_overlap: float = 0.0
    skills_overlap: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'resume_path': self.resume_path,
            'timestamp': self.timestamp,
            'text_length': self.text_length,
            'existing_metrics': self.existing_metrics.to_dict() if self.existing_metrics else None,
            'langextract_metrics': self.langextract_metrics.to_dict() if self.langextract_metrics else None,
            'existing_result': self.existing_result,
            'langextract_result': self.langextract_result,
            'section_overlap': self.section_overlap,
            'skills_overlap': self.skills_overlap,
        }


class ExistingSystemTester:
    """Test wrapper for the existing extraction system."""
    
    def __init__(self):
        self.extractor = UnifiedResumeExtractor()
    
    def test_extraction(self, pdf_path: Path, full_text: str) -> Tuple[Dict, ExtractionMetrics]:
        """Run existing system extraction and return result + metrics."""
        metrics = ExtractionMetrics(system='existing')
        result = {}
        
        try:
            start_time = time.time()
            structured = self.extractor.extract(pdf_path)
            duration_ms = (time.time() - start_time) * 1000
            
            metrics.duration_ms = duration_ms
            metrics.success = True
            
            # Count extracted elements
            metrics.name_found = bool(structured.name)
            metrics.email_found = bool(structured.contact_info.get('email'))
            metrics.phone_found = bool(structured.contact_info.get('phone'))
            metrics.num_sections = len(structured.sections)
            
            # Count experience and education items
            for section in structured.sections:
                if section.section_type == 'experience':
                    metrics.num_experience_items = len(section.items)
                elif section.section_type == 'education':
                    metrics.num_education_items = len(section.items)
            
            # Extract skills
            skills = extract_skills_from_resume(structured, language='en')
            metrics.num_skills = len(skills)
            
            # Convert to dict for storage
            result = structured.to_dict()
            result['extracted_skills'] = skills
            
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            print(f"  [FAIL] Existing system error: {e}")
        
        return result, metrics


class LangExtractTester:
    """Test wrapper for LangExtract."""
    
    def __init__(self, model_id: str = "gemini-2.5-flash"):
        self.model_id = model_id
        self.api_key = os.environ.get('LANGEXTRACT_API_KEY')
        
        if not LANGEXTRACT_AVAILABLE:
            raise ImportError("langextract not installed")
        
        if not self.api_key:
            print("[WARN]  Warning: LANGEXTRACT_API_KEY not set. Set it to use cloud models.")
    
    def _create_resume_prompt(self) -> str:
        """Create the extraction prompt for resume data."""
        return textwrap.dedent("""\
            Extract structured information from this resume text with MAXIMUM detail and coverage.
            
            Extract ALL of the following entity types:
            
            1. CONTACT INFORMATION:
               - Full name (contact_name)
               - Email address (contact_email)
               - Phone number (contact_phone)
               - LinkedIn URL (contact_linkedin)
               - GitHub URL (contact_github)
               - Portfolio/website URL (contact_website)
               - Location/address (contact_location)
            
            2. PROFESSIONAL SUMMARY/OBJECTIVE:
               - Summary text (summary)
               - Career objective (objective)
            
            3. WORK EXPERIENCE:
               - Job title and company (experience)
               - Employment dates (date_range)
               - Work location (location)
               - Employment type (Full-time, Part-time, Contract, Internship)
               - Individual bullet points/achievements (bullet_point)
               - Technologies used in each role
            
            4. EDUCATION:
               - Degree and field of study (education)
               - Institution name (institution)
               - Attendance dates (date_range)
               - GPA if mentioned (gpa)
               - Relevant coursework (education_coursework)
               - Honors/awards (education_awards)
            
            5. SKILLS - Extract EACH skill individually with category:
               - Programming languages (skill: category="programming_language")
               - Frameworks/libraries (skill: category="framework")
               - Cloud platforms (skill: category="cloud_platform")
               - Databases (skill: category="database")
               - DevOps tools (skill: category="devops_tool")
               - Soft skills (skill: category="soft_skill")
               - Languages (skill: category="language_skill")
               - Other tools/technologies (skill: category="tool")
            
            6. CERTIFICATIONS:
               - Certification name (certification)
               - Issuing organization
               - Date obtained
               - Expiration date if mentioned
            
            7. PROJECTS:
               - Project name (project)
               - Project description (project_bullet)
               - Technologies used
               - Project URL if available
            
            8. ADDITIONAL SECTIONS:
               - Awards/honors (award)
               - Publications (publication)
               - Volunteer work (volunteer)
               - Professional affiliations (affiliation)
               - Languages spoken (language)
               - Interests/hobbies (interest)
            
            EXTRACTION RULES:
            - Use EXACT text from the resume (no paraphrasing or summarizing)
            - Include ALL relevant attributes for each extraction
            - Preserve the order of appearance in the document
            - Extract bullet points as separate entities linked to their parent
            - Identify and extract metrics/numbers in bullet points
            - Categorize skills with appropriate category attributes
            - For work experience, extract each position separately
            - For skills, extract each skill individually even if listed together""")
    
    def _create_examples(self) -> List:
        """Create comprehensive few-shot examples for resume extraction."""
        examples = [
            lx.data.ExampleData(
                text="""John Smith
john.smith@email.com
(555) 123-4567
linkedin.com/in/johnsmith
github.com/johnsmith
San Francisco, CA

PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years in full-stack development, specializing in cloud infrastructure and distributed systems.

EXPERIENCE
Software Engineer, Google
January 2020 - Present
San Francisco, CA
• Led development of cloud infrastructure serving 1M+ users
• Improved system performance by 40% through optimization
• Mentored 3 junior engineers and conducted 50+ code reviews

Junior Developer, Microsoft
June 2017 - December 2019
Seattle, WA
• Developed REST APIs using Python and Django
• Reduced database query time by 60%

EDUCATION
Bachelor of Science in Computer Science
Stanford University
2016 - 2020
GPA: 3.8/4.0

SKILLS
Programming: Python, Java, JavaScript, C++, Go
Frameworks: Django, React, Node.js, FastAPI
Cloud: AWS (EC2, S3, Lambda), GCP, Azure
Tools: Docker, Kubernetes, Terraform, Git, Jenkins
Databases: PostgreSQL, MongoDB, Redis

CERTIFICATIONS
AWS Solutions Architect - Professional
Google Cloud Professional Data Engineer

PROJECTS
Personal Portfolio Website
• Built with React and Node.js
• Implements CI/CD with GitHub Actions
• Deployed on AWS with auto-scaling

LANGUAGES
English (Native), Spanish (Conversational)""",
                extractions=[
                    # Contact Information
                    lx.data.Extraction(
                        extraction_class="contact_name",
                        extraction_text="John Smith",
                        attributes={"type": "full_name"}
                    ),
                    lx.data.Extraction(
                        extraction_class="contact_email",
                        extraction_text="john.smith@email.com",
                        attributes={"type": "email"}
                    ),
                    lx.data.Extraction(
                        extraction_class="contact_phone",
                        extraction_text="(555) 123-4567",
                        attributes={"type": "phone"}
                    ),
                    lx.data.Extraction(
                        extraction_class="contact_linkedin",
                        extraction_text="linkedin.com/in/johnsmith",
                        attributes={"type": "linkedin"}
                    ),
                    lx.data.Extraction(
                        extraction_class="contact_github",
                        extraction_text="github.com/johnsmith",
                        attributes={"type": "github"}
                    ),
                    lx.data.Extraction(
                        extraction_class="contact_location",
                        extraction_text="San Francisco, CA",
                        attributes={"type": "location"}
                    ),
                    # Summary
                    lx.data.Extraction(
                        extraction_class="summary",
                        extraction_text="Experienced software engineer with 5+ years in full-stack development, specializing in cloud infrastructure and distributed systems.",
                        attributes={"type": "professional_summary"}
                    ),
                    # Experience
                    lx.data.Extraction(
                        extraction_class="experience",
                        extraction_text="Software Engineer, Google",
                        attributes={
                            "job_title": "Software Engineer",
                            "company": "Google",
                            "date_range": "January 2020 - Present",
                            "location": "San Francisco, CA",
                            "duration_years": "5+"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="bullet_point",
                        extraction_text="Led development of cloud infrastructure serving 1M+ users",
                        attributes={
                            "parent": "Software Engineer, Google",
                            "has_metric": True,
                            "metric": "1M+ users"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="bullet_point",
                        extraction_text="Improved system performance by 40% through optimization",
                        attributes={
                            "parent": "Software Engineer, Google",
                            "has_metric": True,
                            "metric": "40%"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="bullet_point",
                        extraction_text="Mentored 3 junior engineers and conducted 50+ code reviews",
                        attributes={
                            "parent": "Software Engineer, Google",
                            "has_metric": True,
                            "metric": "3 junior engineers, 50+ code reviews"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="experience",
                        extraction_text="Junior Developer, Microsoft",
                        attributes={
                            "job_title": "Junior Developer",
                            "company": "Microsoft",
                            "date_range": "June 2017 - December 2019",
                            "location": "Seattle, WA",
                            "duration_years": "2.5"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="bullet_point",
                        extraction_text="Developed REST APIs using Python and Django",
                        attributes={"parent": "Junior Developer, Microsoft"}
                    ),
                    lx.data.Extraction(
                        extraction_class="bullet_point",
                        extraction_text="Reduced database query time by 60%",
                        attributes={
                            "parent": "Junior Developer, Microsoft",
                            "has_metric": True,
                            "metric": "60%"
                        }
                    ),
                    # Education
                    lx.data.Extraction(
                        extraction_class="education",
                        extraction_text="Bachelor of Science in Computer Science",
                        attributes={
                            "degree": "Bachelor of Science",
                            "field": "Computer Science",
                            "institution": "Stanford University",
                            "date_range": "2016 - 2020",
                            "gpa": "3.8/4.0"
                        }
                    ),
                    # Skills by category
                    lx.data.Extraction(
                        extraction_class="skill",
                        extraction_text="Python",
                        attributes={"category": "programming_language", "parent_category": "Programming"}
                    ),
                    lx.data.Extraction(
                        extraction_class="skill",
                        extraction_text="Java",
                        attributes={"category": "programming_language", "parent_category": "Programming"}
                    ),
                    lx.data.Extraction(
                        extraction_class="skill",
                        extraction_text="JavaScript",
                        attributes={"category": "programming_language", "parent_category": "Programming"}
                    ),
                    lx.data.Extraction(
                        extraction_class="skill",
                        extraction_text="Django",
                        attributes={"category": "framework", "parent_category": "Frameworks"}
                    ),
                    lx.data.Extraction(
                        extraction_class="skill",
                        extraction_text="React",
                        attributes={"category": "framework", "parent_category": "Frameworks"}
                    ),
                    lx.data.Extraction(
                        extraction_class="skill",
                        extraction_text="AWS",
                        attributes={"category": "cloud_platform", "parent_category": "Cloud"}
                    ),
                    lx.data.Extraction(
                        extraction_class="skill",
                        extraction_text="Docker",
                        attributes={"category": "devops_tool", "parent_category": "Tools"}
                    ),
                    lx.data.Extraction(
                        extraction_class="skill",
                        extraction_text="Kubernetes",
                        attributes={"category": "devops_tool", "parent_category": "Tools"}
                    ),
                    lx.data.Extraction(
                        extraction_class="skill",
                        extraction_text="PostgreSQL",
                        attributes={"category": "database", "parent_category": "Databases"}
                    ),
                    lx.data.Extraction(
                        extraction_class="skill",
                        extraction_text="MongoDB",
                        attributes={"category": "database", "parent_category": "Databases"}
                    ),
                    # Certifications
                    lx.data.Extraction(
                        extraction_class="certification",
                        extraction_text="AWS Solutions Architect - Professional",
                        attributes={
                            "name": "AWS Solutions Architect - Professional",
                            "provider": "AWS",
                            "level": "Professional"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="certification",
                        extraction_text="Google Cloud Professional Data Engineer",
                        attributes={
                            "name": "Google Cloud Professional Data Engineer",
                            "provider": "Google Cloud",
                            "level": "Professional"
                        }
                    ),
                    # Projects
                    lx.data.Extraction(
                        extraction_class="project",
                        extraction_text="Personal Portfolio Website",
                        attributes={
                            "name": "Personal Portfolio Website",
                            "technologies": "React, Node.js, AWS"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="project_bullet",
                        extraction_text="Built with React and Node.js",
                        attributes={"parent": "Personal Portfolio Website"}
                    ),
                    lx.data.Extraction(
                        extraction_class="project_bullet",
                        extraction_text="Implements CI/CD with GitHub Actions",
                        attributes={"parent": "Personal Portfolio Website"}
                    ),
                    lx.data.Extraction(
                        extraction_class="project_bullet",
                        extraction_text="Deployed on AWS with auto-scaling",
                        attributes={"parent": "Personal Portfolio Website"}
                    ),
                    # Languages
                    lx.data.Extraction(
                        extraction_class="language",
                        extraction_text="English (Native)",
                        attributes={
                            "language": "English",
                            "proficiency": "Native"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="language",
                        extraction_text="Spanish (Conversational)",
                        attributes={
                            "language": "Spanish",
                            "proficiency": "Conversational"
                        }
                    ),
                ]
            ),
            lx.data.ExampleData(
                text="""EDUCATION
Master of Science in Data Science
MIT - Massachusetts Institute of Technology
September 2018 - May 2020
GPA: 3.9/4.0
Relevant Coursework: Machine Learning, Deep Learning, Statistics, Big Data Analytics

EXPERIENCE
Data Scientist Intern, Amazon
June 2019 - August 2019
• Built recommendation system increasing CTR by 25%
• Analyzed 10TB+ of customer data using Spark
• Created predictive models with 95% accuracy""",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="education",
                        extraction_text="Master of Science in Data Science",
                        attributes={
                            "degree": "Master of Science",
                            "field": "Data Science",
                            "institution": "MIT - Massachusetts Institute of Technology",
                            "date_range": "September 2018 - May 2020",
                            "gpa": "3.9/4.0"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="education_coursework",
                        extraction_text="Machine Learning, Deep Learning, Statistics, Big Data Analytics",
                        attributes={
                            "parent": "Master of Science in Data Science",
                            "courses": "Machine Learning, Deep Learning, Statistics, Big Data Analytics"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="experience",
                        extraction_text="Data Scientist Intern, Amazon",
                        attributes={
                            "job_title": "Data Scientist Intern",
                            "company": "Amazon",
                            "date_range": "June 2019 - August 2019",
                            "employment_type": "Internship"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="bullet_point",
                        extraction_text="Built recommendation system increasing CTR by 25%",
                        attributes={
                            "parent": "Data Scientist Intern, Amazon",
                            "has_metric": True,
                            "metric": "25% CTR increase"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="bullet_point",
                        extraction_text="Analyzed 10TB+ of customer data using Spark",
                        attributes={
                            "parent": "Data Scientist Intern, Amazon",
                            "has_metric": True,
                            "metric": "10TB+ data",
                            "technology": "Spark"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="bullet_point",
                        extraction_text="Created predictive models with 95% accuracy",
                        attributes={
                            "parent": "Data Scientist Intern, Amazon",
                            "has_metric": True,
                            "metric": "95% accuracy"
                        }
                    ),
                ]
            )
        ]
        return examples
    
    def test_extraction(self, pdf_path: Path, full_text: str) -> Tuple[Dict, ExtractionMetrics]:
        """Run LangExtract extraction and return result + metrics."""
        metrics = ExtractionMetrics(system='langextract')
        metrics.has_source_grounding = True
        result = {}
        
        try:
            prompt = self._create_resume_prompt()
            examples = self._create_examples()
            
            # Build extraction kwargs with ALL optimization options
            # For maximum extraction with reasonable speed
            kwargs = {
                'text_or_documents': full_text,
                'prompt_description': prompt,
                'examples': examples,
                'model_id': self.model_id,
                # Extraction optimization options for maximum coverage
                'extraction_passes': 1,      # Single pass for speed (increase to 2-3 for higher recall)
                'max_workers': 2,            # Parallel processing (keep low for API rate limits)
                'max_char_buffer': 4000,     # Buffer size for chunking
            }
            
            if self.api_key:
                kwargs['api_key'] = self.api_key
            
            print(f"  [LANGEXTRACT] Configuration (Maximum Extraction Mode):")
            print(f"    - Model: {self.model_id}")
            print(f"    - Extraction passes: 1 (set to 2-3 for higher recall)")
            print(f"    - Max workers: 2 (parallel processing)")
            print(f"    - Buffer size: 4000 chars")
            print(f"    - Comprehensive examples: 2 detailed resumes")
            print(f"    - Enhanced prompt: 8 entity types with detailed instructions")
            
            print(f"  [LANGEXTRACT] Running LangExtract with {self.model_id}...")
            start_time = time.time()
            
            extraction_result = lx.extract(**kwargs)
            
            duration_ms = (time.time() - start_time) * 1000
            metrics.duration_ms = duration_ms
            metrics.success = True
            
            # Convert to our format
            result = self._convert_to_dict(extraction_result)
            
            # Calculate metrics from result
            metrics.name_found = bool(result.get('contact', {}).get('name'))
            metrics.email_found = bool(result.get('contact', {}).get('email'))
            metrics.phone_found = bool(result.get('contact', {}).get('phone'))
            metrics.num_sections = len(result.get('sections', []))
            metrics.num_experience_items = len(result.get('experience', []))
            metrics.num_education_items = len(result.get('education', []))
            metrics.num_skills = len(result.get('skills', []))
            
            # Calculate confidence (based on extraction coverage)
            total_extractions = len(extraction_result.extractions)
            expected_extractions = 10  # rough estimate
            metrics.confidence_score = min(1.0, total_extractions / expected_extractions)
            
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            print(f"  [FAIL] LangExtract error: {e}")
            traceback.print_exc()
        
        return result, metrics
    
    def _convert_to_dict(self, extraction_result) -> Dict:
        """Convert LangExtract result to comparable dict format."""
        result = {
            'contact': {},
            'summary': '',
            'sections': [],
            'experience': [],
            'education': [],
            'skills': [],
            'certifications': [],
            'projects': [],
            'languages': [],
            'awards': [],
            'publications': [],
            'volunteer': [],
            'bullet_points': [],
            'raw_extractions': []
        }
        
        for extraction in extraction_result.extractions:
            extraction_dict = {
                'class': extraction.extraction_class,
                'text': extraction.extraction_text,
                'attributes': extraction.attributes if hasattr(extraction, 'attributes') else {},
                'start_char': extraction.start_char if hasattr(extraction, 'start_char') else None,
                'end_char': extraction.end_char if hasattr(extraction, 'end_char') else None,
            }
            result['raw_extractions'].append(extraction_dict)
            
            # Categorize by extraction class
            ext_class = extraction.extraction_class.lower()
            
            # Contact information
            if 'contact' in ext_class:
                if 'name' in ext_class:
                    result['contact']['name'] = extraction.extraction_text
                elif 'email' in ext_class:
                    result['contact']['email'] = extraction.extraction_text
                elif 'phone' in ext_class:
                    result['contact']['phone'] = extraction.extraction_text
                elif 'linkedin' in ext_class:
                    result['contact']['linkedin'] = extraction.extraction_text
                elif 'github' in ext_class:
                    result['contact']['github'] = extraction.extraction_text
                elif 'website' in ext_class:
                    result['contact']['website'] = extraction.extraction_text
                elif 'location' in ext_class:
                    result['contact']['location'] = extraction.extraction_text
            
            # Summary
            elif 'summary' in ext_class or 'objective' in ext_class:
                if not result['summary']:
                    result['summary'] = extraction.extraction_text
                else:
                    result['summary'] += ' ' + extraction.extraction_text
            
            # Work experience
            elif 'experience' in ext_class and 'education' not in ext_class:
                result['experience'].append(extraction_dict)
            
            # Education
            elif 'education' in ext_class:
                result['education'].append(extraction_dict)
            
            # Skills
            elif 'skill' in ext_class:
                result['skills'].append(extraction.extraction_text)
            
            # Certifications
            elif 'certification' in ext_class:
                result['certifications'].append(extraction_dict)
            
            # Projects
            elif 'project' in ext_class:
                result['projects'].append(extraction_dict)
            
            # Languages
            elif 'language' in ext_class:
                result['languages'].append(extraction_dict)
            
            # Awards
            elif 'award' in ext_class:
                result['awards'].append(extraction_dict)
            
            # Publications
            elif 'publication' in ext_class:
                result['publications'].append(extraction_dict)
            
            # Volunteer work
            elif 'volunteer' in ext_class:
                result['volunteer'].append(extraction_dict)
            
            # Bullet points (achievements, responsibilities)
            elif 'bullet' in ext_class:
                result['bullet_points'].append(extraction_dict)
        
        return result


class ComparisonAnalyzer:
    """Analyze and compare extraction results."""
    
    def calculate_overlap(self, list1: List[str], list2: List[str]) -> float:
        """Calculate Jaccard similarity between two lists."""
        if not list1 and not list2:
            return 1.0
        if not list1 or not list2:
            return 0.0
        
        set1 = set(s.lower().strip() for s in list1)
        set2 = set(s.lower().strip() for s in list2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def compare_results(self, existing: Dict, langextract: Dict) -> Tuple[float, float]:
        """Compare two extraction results and return overlap scores."""
        # Compare sections
        existing_sections = existing.get('sections', [])
        langextract_sections = langextract.get('sections', [])
        
        existing_section_names = [s.get('title', '') for s in existing_sections]
        langextract_section_names = [s.get('title', '') for s in langextract_sections]
        
        section_overlap = self.calculate_overlap(existing_section_names, langextract_section_names)
        
        # Compare skills
        existing_skills = existing.get('extracted_skills', existing.get('skills', []))
        langextract_skills = langextract.get('skills', [])
        
        skills_overlap = self.calculate_overlap(existing_skills, langextract_skills)
        
        return section_overlap, skills_overlap


class ComparisonReport:
    """Generate comparison reports."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ComparisonResult] = []
    
    def add_result(self, result: ComparisonResult):
        self.results.append(result)
    
    def save_visualization(self, result: ComparisonResult, index: int):
        """Save LangExtract visualization if available."""
        if LANGEXTRACT_AVAILABLE and result.langextract_result.get('raw_extractions'):
            try:
                # Create a minimal document for visualization
                doc = lx.data.AnnotatedDocument(
                    text=result.langextract_result.get('full_text', ''),
                    extractions=[
                        lx.data.Extraction(
                            extraction_class=e['class'],
                            extraction_text=e['text'],
                            attributes=e.get('attributes', {})
                        )
                        for e in result.langextract_result['raw_extractions']
                    ]
                )
                
                output_name = f"langextract_viz_{index}.html"
                html_content = lx.visualize([doc])
                
                viz_path = self.output_dir / output_name
                with open(viz_path, 'w', encoding='utf-8') as f:
                    if hasattr(html_content, 'data'):
                        f.write(html_content.data)
                    else:
                        f.write(html_content)
                
                print(f"  [STATS] Saved LangExtract visualization: {viz_path}")
                
            except Exception as e:
                print(f"  [WARN]  Could not save visualization: {e}")
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics."""
        if not self.results:
            return {
                'total_resumes_tested': 0,
                'existing_system': {'success_rate': 0, 'avg_duration_ms': 0, 'total_success': 0},
                'langextract': {'success_rate': 0, 'avg_duration_ms': 0, 'total_success': 0},
                'comparison': {'avg_section_overlap': 0, 'avg_skills_overlap': 0},
            }
        
        total = len(self.results)
        
        existing_success = sum(1 for r in self.results if r.existing_metrics and r.existing_metrics.success)
        langextract_success = sum(1 for r in self.results if r.langextract_metrics and r.langextract_metrics.success)
        
        existing_times = [r.existing_metrics.duration_ms for r in self.results if r.existing_metrics and r.existing_metrics.success]
        langextract_times = [r.langextract_metrics.duration_ms for r in self.results if r.langextract_metrics and r.langextract_metrics.success]
        
        summary = {
            'total_resumes_tested': total,
            'existing_system': {
                'success_rate': existing_success / total,
                'avg_duration_ms': sum(existing_times) / len(existing_times) if existing_times else 0,
                'total_success': existing_success,
            },
            'langextract': {
                'success_rate': langextract_success / total,
                'avg_duration_ms': sum(langextract_times) / len(langextract_times) if langextract_times else 0,
                'total_success': langextract_success,
            },
            'comparison': {
                'avg_section_overlap': sum(r.section_overlap for r in self.results) / total,
                'avg_skills_overlap': sum(r.skills_overlap for r in self.results) / total,
            }
        }
        
        return summary
    
    def print_summary(self):
        """Print summary to console."""
        summary = self.generate_summary()
        
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        print(f"\n[STATS] Total Resumes Tested: {summary['total_resumes_tested']}")
        
        print("\n[EXISTING] Existing System:")
        print(f"   Success Rate: {summary['existing_system']['success_rate']:.1%}")
        print(f"   Avg Duration: {summary['existing_system']['avg_duration_ms']:.0f}ms")
        
        print("\n[LANGEXTRACT] LangExtract:")
        print(f"   Success Rate: {summary['langextract']['success_rate']:.1%}")
        print(f"   Avg Duration: {summary['langextract']['avg_duration_ms']:.0f}ms")
        
        print("\n[COMPARE] Overlap Analysis:")
        print(f"   Avg Section Overlap: {summary['comparison']['avg_section_overlap']:.1%}")
        print(f"   Avg Skills Overlap: {summary['comparison']['avg_skills_overlap']:.1%}")
        
        # Recommendations
        print("\n[TIP] Recommendations:")
        if summary['langextract']['success_rate'] < summary['existing_system']['success_rate']:
            print("   - LangExtract has lower success rate than existing system")
        if summary['langextract']['avg_duration_ms'] > summary['existing_system']['avg_duration_ms'] * 2:
            print("   - LangExtract is significantly slower (API latency)")
        if summary['comparison']['avg_skills_overlap'] < 0.5:
            print("   - Low skills overlap suggests different extraction approaches")
        
        print("\n" + "="*80)
    
    def save_report(self):
        """Save complete report to JSON."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.generate_summary(),
            'results': [r.to_dict() for r in self.results]
        }
        
        report_path = self.output_dir / 'comparison_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n[FILE] Full report saved to: {report_path}")


def find_resumes(resume_dir: Path, limit: int = 5) -> List[Path]:
    """Find resume PDFs in directory."""
    resumes = []
    
    if resume_dir.exists():
        for pdf in resume_dir.glob('*.pdf'):
            resumes.append(pdf)
            if len(resumes) >= limit:
                break
    
    # Also check test_results for previously analyzed resumes
    test_results = Path('test_results')
    if test_results.exists():
        for json_file in test_results.glob('*_analysis.json'):
            # Try to find corresponding PDF
            pdf_name = json_file.stem.replace('_analysis', '') + '.pdf'
            pdf_path = Path('resumes') / pdf_name
            if pdf_path.exists() and pdf_path not in resumes:
                resumes.append(pdf_path)
                if len(resumes) >= limit:
                    break
    
    return resumes


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract full text from PDF using PyMuPDF."""
    # Check if file exists first
    if not pdf_path.exists():
        print(f"  [FAIL] File not found: {pdf_path}")
        # Try to suggest where to look
        print(f"     Looking in: {pdf_path.absolute()}")
        # Check if resumes directory exists
        resumes_dir = Path('resumes')
        if resumes_dir.exists():
            pdfs = list(resumes_dir.glob('*.pdf'))
            if pdfs:
                print(f"     Found {len(pdfs)} PDF(s) in 'resumes/' directory:")
                for p in pdfs[:5]:  # Show first 5
                    print(f"       - {p.name}")
        return ""
    
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"  [FAIL] Error extracting text from {pdf_path}: {e}")
        return ""


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare LangExtract with existing ATS extraction system"
    )
    parser.add_argument(
        '--resume', '-r',
        help='Path to a specific resume PDF to test'
    )
    parser.add_argument(
        '--resume-dir', '-d',
        default='resumes',
        help='Directory containing resume PDFs (default: resumes)'
    )
    parser.add_argument(
        '--output', '-o',
        default='langextract_comparison_results',
        help='Output directory for results (default: langextract_comparison_results)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=5,
        help='Maximum number of resumes to test (default: 5)'
    )
    parser.add_argument(
        '--model',
        default='gemini-2.5-flash',
        help='LangExtract model ID (default: gemini-2.5-flash)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip testing existing system (test only LangExtract)'
    )
    parser.add_argument(
        '--skip-langextract',
        action='store_true',
        help='Skip testing LangExtract (test only existing system)'
    )
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    report = ComparisonReport(output_dir)
    analyzer = ComparisonAnalyzer()
    
    print("="*80)
    print("LangExtract vs Existing System Comparison")
    print("="*80)
    
    # Check langextract availability
    if not LANGEXTRACT_AVAILABLE:
        print("\n[WARN]  langextract not installed. Install with:")
        print("   pip install langextract")
        print("\n   Continuing with existing system only...")
        args.skip_langextract = True
    
    # Find resumes to test
    if args.resume:
        resumes = [Path(args.resume)]
    else:
        resumes = find_resumes(Path(args.resume_dir), limit=args.limit)
    
    if not resumes:
        print("\n[FAIL] No resume PDFs found!")
        print(f"   Looked in: {args.resume_dir}")
        print("\n   Usage: python test_langextract_comparison.py --resume path/to/resume.pdf")
        sys.exit(1)
    
    print(f"\n[FILE] Found {len(resumes)} resume(s) to test\n")
    
    # Initialize testers
    existing_tester = None if args.skip_existing else ExistingSystemTester()
    langextract_tester = None if args.skip_langextract else LangExtractTester(model_id=args.model)
    
    # Run tests
    for i, resume_path in enumerate(resumes, 1):
        print(f"\n{'='*80}")
        print(f"Testing Resume {i}/{len(resumes)}: {resume_path.name}")
        print(f"{'='*80}")
        
        # Extract text first
        full_text = extract_text_from_pdf(resume_path)
        if not full_text:
            print(f"  [FAIL] Could not extract text from {resume_path}")
            continue
        
        print(f"  Text length: {len(full_text)} characters")
        
        result = ComparisonResult(
            resume_path=str(resume_path),
            timestamp=datetime.now().isoformat(),
            text_length=len(full_text)
        )
        
        # Test existing system
        if existing_tester:
            print("\n  [TEST] Testing Existing System...")
            existing_result, existing_metrics = existing_tester.test_extraction(resume_path, full_text)
            result.existing_result = existing_result
            result.existing_metrics = existing_metrics
            
            if existing_metrics.success:
                print(f"    [OK] Success in {existing_metrics.duration_ms:.0f}ms")
                print(f"    [STATS] Found: {existing_metrics.num_sections} sections, "
                      f"{existing_metrics.num_experience_items} experience items, "
                      f"{existing_metrics.num_skills} skills")
            else:
                print(f"    [FAIL] Failed: {existing_metrics.error_message}")
        
        # Test LangExtract
        if langextract_tester:
            print("\n  [LANGEXTRACT] Testing LangExtract...")
            langextract_result, langextract_metrics = langextract_tester.test_extraction(resume_path, full_text)
            result.langextract_result = langextract_result
            result.langextract_metrics = langextract_metrics
            
            if langextract_metrics.success:
                print(f"    [OK] Success in {langextract_metrics.duration_ms:.0f}ms")
                print(f"    [STATS] Found: {langextract_metrics.num_sections} sections, "
                      f"{langextract_metrics.num_experience_items} experience items, "
                      f"{langextract_metrics.num_skills} skills")
                print(f"    [CONFIDENCE] Confidence: {langextract_metrics.confidence_score:.2f}")
            else:
                print(f"    [FAIL] Failed: {langextract_metrics.error_message}")
        
        # Compare results
        if result.existing_metrics and result.existing_metrics.success and \
           result.langextract_metrics and result.langextract_metrics.success:
            section_overlap, skills_overlap = analyzer.compare_results(
                result.existing_result,
                result.langextract_result
            )
            result.section_overlap = section_overlap
            result.skills_overlap = skills_overlap
            print(f"\n  [COMPARE] Overlap with Existing System:")
            print(f"    Sections: {section_overlap:.1%}")
            print(f"    Skills: {skills_overlap:.1%}")
        
        report.add_result(result)
        
        # Save visualization
        if langextract_tester and not args.skip_langextract:
            report.save_visualization(result, i)
    
    # Generate reports
    print("\n")
    report.print_summary()
    report.save_report()
    
    print(f"\n[SUCCESS] Comparison complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
