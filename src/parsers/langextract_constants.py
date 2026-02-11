"""
LangExtract Resume Parser Constants

This module contains prompts and examples for LangExtract resume parsing.
Separating these from the main parser keeps the code clean and maintainable.
"""

import textwrap

# =============================================================================
# EXTRACTION PROMPT
# =============================================================================

RESUME_EXTRACTION_PROMPT = textwrap.dedent("""\
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
       - Languages spoken (language)
    
    EXTRACTION RULES:
    - Use EXACT text from the resume (no paraphrasing or summarizing)
    - Include ALL relevant attributes for each extraction
    - Preserve the order of appearance in the document
    - Extract bullet points as separate entities linked to their parent
    - Identify and extract metrics/numbers in bullet points
    - Categorize skills with appropriate category attributes
    - For work experience, extract each position separately
    - For skills, extract each skill individually even if listed together""")


# =============================================================================
# FEW-SHOT EXAMPLES
# =============================================================================

def create_resume_examples():
    """
    Create comprehensive few-shot examples for resume extraction.
    
    Returns:
        List of ExampleData objects for LangExtract
    """
    # Import here to avoid circular imports
    try:
        import langextract as lx
    except ImportError:
        return []
    
    examples = [
        # Example 1: Complete software engineer resume
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
• Improved system performance by 40%

Junior Developer, Microsoft
June 2017 - December 2019
Seattle, WA
• Developed REST APIs using Python and Django

EDUCATION
Bachelor of Science in Computer Science
Stanford University
2016 - 2020
GPA: 3.8/4.0

SKILLS
Programming: Python, Java, JavaScript
Cloud: AWS, GCP
Tools: Docker, Kubernetes

CERTIFICATIONS
AWS Solutions Architect - Professional""",
            extractions=[
                # Contact
                lx.data.Extraction(extraction_class="contact_name", extraction_text="John Smith"),
                lx.data.Extraction(extraction_class="contact_email", extraction_text="john.smith@email.com"),
                lx.data.Extraction(extraction_class="contact_phone", extraction_text="(555) 123-4567"),
                lx.data.Extraction(extraction_class="contact_linkedin", extraction_text="linkedin.com/in/johnsmith"),
                lx.data.Extraction(extraction_class="contact_github", extraction_text="github.com/johnsmith"),
                lx.data.Extraction(extraction_class="contact_location", extraction_text="San Francisco, CA"),
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
                        "location": "San Francisco, CA"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="bullet_point",
                    extraction_text="Led development of cloud infrastructure serving 1M+ users",
                    attributes={"parent": "Software Engineer, Google", "has_metric": True}
                ),
                lx.data.Extraction(
                    extraction_class="bullet_point",
                    extraction_text="Improved system performance by 40%",
                    attributes={"parent": "Software Engineer, Google", "has_metric": True}
                ),
                lx.data.Extraction(
                    extraction_class="experience",
                    extraction_text="Junior Developer, Microsoft",
                    attributes={
                        "job_title": "Junior Developer",
                        "company": "Microsoft",
                        "date_range": "June 2017 - December 2019",
                        "location": "Seattle, WA"
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
                # Skills
                lx.data.Extraction(extraction_class="skill", extraction_text="Python", attributes={"category": "programming_language"}),
                lx.data.Extraction(extraction_class="skill", extraction_text="Java", attributes={"category": "programming_language"}),
                lx.data.Extraction(extraction_class="skill", extraction_text="AWS", attributes={"category": "cloud_platform"}),
                lx.data.Extraction(extraction_class="skill", extraction_text="Docker", attributes={"category": "devops_tool"}),
                # Certification
                lx.data.Extraction(
                    extraction_class="certification",
                    extraction_text="AWS Solutions Architect - Professional",
                    attributes={"name": "AWS Solutions Architect - Professional", "provider": "AWS"}
                ),
            ]
        ),
        
        # Example 2: Data scientist with detailed education
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
                    attributes={"parent": "Data Scientist Intern, Amazon", "has_metric": True, "metric": "25% CTR increase"}
                ),
                lx.data.Extraction(
                    extraction_class="bullet_point",
                    extraction_text="Analyzed 10TB+ of customer data using Spark",
                    attributes={"parent": "Data Scientist Intern, Amazon", "has_metric": True, "metric": "10TB+ data", "technology": "Spark"}
                ),
                lx.data.Extraction(
                    extraction_class="bullet_point",
                    extraction_text="Created predictive models with 95% accuracy",
                    attributes={"parent": "Data Scientist Intern, Amazon", "has_metric": True, "metric": "95% accuracy"}
                ),
            ]
        ),
        
        # Example 3: Multi-lingual candidate with projects
        lx.data.ExampleData(
            text="""PROJECTS
Personal Portfolio Website
• Built with React and Node.js
• Implements CI/CD with GitHub Actions
• Deployed on AWS with auto-scaling

Machine Learning Image Classifier
• Achieved 98% accuracy on test dataset
• Used Python, TensorFlow, and Keras
• Trained on 50,000 images

LANGUAGES
English (Native)
Spanish (Conversational)
French (Basic)""",
            extractions=[
                lx.data.Extraction(
                    extraction_class="project",
                    extraction_text="Personal Portfolio Website",
                    attributes={"name": "Personal Portfolio Website", "technologies": "React, Node.js, AWS"}
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
                lx.data.Extraction(
                    extraction_class="project",
                    extraction_text="Machine Learning Image Classifier",
                    attributes={"name": "Machine Learning Image Classifier", "technologies": "Python, TensorFlow, Keras"}
                ),
                lx.data.Extraction(
                    extraction_class="project_bullet",
                    extraction_text="Achieved 98% accuracy on test dataset",
                    attributes={"parent": "Machine Learning Image Classifier", "has_metric": True, "metric": "98% accuracy"}
                ),
                lx.data.Extraction(
                    extraction_class="project_bullet",
                    extraction_text="Trained on 50,000 images",
                    attributes={"parent": "Machine Learning Image Classifier", "has_metric": True, "metric": "50,000 images"}
                ),
                lx.data.Extraction(
                    extraction_class="language",
                    extraction_text="English (Native)",
                    attributes={"language": "English", "proficiency": "Native"}
                ),
                lx.data.Extraction(
                    extraction_class="language",
                    extraction_text="Spanish (Conversational)",
                    attributes={"language": "Spanish", "proficiency": "Conversational"}
                ),
                lx.data.Extraction(
                    extraction_class="language",
                    extraction_text="French (Basic)",
                    attributes={"language": "French", "proficiency": "Basic"}
                ),
            ]
        ),
    ]
    
    return examples


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'model_id': 'gemini-2.5-flash',
    'extraction_passes': 1,        # 1-3, higher = more thorough but slower
    'max_workers': 2,              # Parallel processing (respect rate limits)
    'max_char_buffer': 4000,       # Text chunking size
}

# Skill categories for classification
SKILL_CATEGORIES = {
    'programming_language': ['python', 'java', 'javascript', 'c++', 'go', 'rust', 'typescript', 'ruby', 'php', 'swift', 'kotlin'],
    'framework': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'fastapi', 'rails', 'laravel'],
    'cloud_platform': ['aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean'],
    'database': ['postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'dynamodb', 'sqlite'],
    'devops_tool': ['docker', 'kubernetes', 'terraform', 'jenkins', 'gitlab ci', 'github actions', 'ansible', 'puppet'],
    'soft_skill': ['leadership', 'communication', 'teamwork', 'problem solving', 'critical thinking'],
}
