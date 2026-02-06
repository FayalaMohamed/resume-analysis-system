"""Job description parser and resume matching system."""

import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
import math


@dataclass
class JobMatchResult:
    """Results from job-resume matching."""
    overall_match: float = 0.0
    keyword_match: float = 0.0
    semantic_similarity: float = 0.0
    skill_match: float = 0.0
    
    matched_keywords: List[str] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    
    jd_summary: str = ""
    recommendations: List[str] = field(default_factory=list)


class JobDescriptionParser:
    """Parse and analyze job descriptions."""
    
    # Common skill keywords (tech-focused, can be expanded)
    SKILL_KEYWORDS = {
        'programming': [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
            'ruby', 'php', 'swift', 'kotlin', 'scala', 'perl', 'r', 'matlab',
            'sql', 'nosql', 'html', 'css', 'sass', 'less'
        ],
        'frameworks': [
            'react', 'angular', 'vue', 'django', 'flask', 'fastapi', 'spring',
            'express', 'next.js', 'nuxt', 'rails', 'laravel', 'dotnet', 'nodejs'
        ],
        'databases': [
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'dynamodb',
            'cassandra', 'neo4j', 'sqlite', 'oracle', 'mssql', 'firebase'
        ],
        'cloud': [
            'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'terraform',
            'jenkins', 'gitlab ci', 'github actions', 'circleci', 'travis ci'
        ],
        'ml_ai': [
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
            'scikit-learn', 'pandas', 'numpy', 'jupyter', 'nlp', 'computer vision'
        ],
        'data': [
            'data analysis', 'data science', 'etl', 'data warehouse', 'hadoop',
            'spark', 'kafka', 'airflow', 'dbt', 'tableau', 'power bi', 'looker'
        ],
        'soft_skills': [
            'leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
            'project management', 'agile', 'scrum', 'collaboration', 'mentoring'
        ],
        'tools': [
            'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'slack',
            'vscode', 'intellij', 'postman', 'figma', 'sketch', 'photoshop'
        ]
    }
    
    # Flatten all skills
    ALL_SKILLS = [skill for skills in SKILL_KEYWORDS.values() for skill in skills]
    
    # Required vs preferred indicators
    REQUIRED_INDICATORS = [
        'required', 'must have', 'essential', 'mandatory', 'need to have',
        'prerequisites', 'qualifications required'
    ]
    
    PREFERRED_INDICATORS = [
        'preferred', 'nice to have', 'desired', 'optional', 'bonus',
        'plus', 'advantageous', 'beneficial'
    ]
    
    def __init__(self):
        """Initialize the JD parser."""
        self.skill_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(skill) for skill in self.ALL_SKILLS) + r')\b',
            re.IGNORECASE
        )
    
    def parse(self, jd_text: str) -> Dict[str, Any]:
        """Parse job description into structured components.
        
        Args:
            jd_text: Raw job description text
            
        Returns:
            Dictionary with parsed components
        """
        # Extract skills
        skills = self._extract_skills(jd_text)
        
        # Categorize as required vs preferred
        required_skills, preferred_skills = self._categorize_skills(jd_text, skills)
        
        # Extract keywords (general)
        keywords = self._extract_keywords(jd_text)
        
        # Extract experience requirements
        experience = self._extract_experience(jd_text)
        
        # Generate summary
        summary = self._generate_summary(jd_text)
        
        return {
            'raw_text': jd_text,
            'summary': summary,
            'skills': {
                'all': skills,
                'required': required_skills,
                'preferred': preferred_skills,
            },
            'keywords': keywords,
            'experience_required': experience,
        }
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from job description."""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.ALL_SKILLS:
            # Use word boundary matching
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
        
        return list(set(found_skills))
    
    def _categorize_skills(self, text: str, skills: List[str]) -> Tuple[List[str], List[str]]:
        """Categorize skills as required or preferred."""
        text_lower = text.lower()
        
        required = []
        preferred = []
        
        # Split text into sections
        sections = text_lower.split('\n\n')
        
        for skill in skills:
            skill_lower = skill.lower()
            skill_context = ""
            
            # Find context around skill mention
            for section in sections:
                if skill_lower in section:
                    skill_context = section
                    break
            
            # Check if in required section
            is_required = any(indicator in skill_context for indicator in self.REQUIRED_INDICATORS)
            is_preferred = any(indicator in skill_context for indicator in self.PREFERRED_INDICATORS)
            
            if is_required or (not is_preferred and 'required' in skill_context):
                required.append(skill)
            else:
                preferred.append(skill)
        
        return required, preferred
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords using TF-IDF-like approach."""
        # Simple keyword extraction based on frequency and importance
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'about', 'above', 'after', 'again', 'against', 'all', 'also', 'and', 'any',
            'are', 'because', 'been', 'before', 'being', 'below', 'between', 'both',
            'but', 'can', 'could', 'did', 'does', 'doing', 'down', 'during', 'each',
            'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'here',
            'how', 'into', 'its', 'itself', 'let', 'more', 'most', 'much', 'must',
            'myself', 'nor', 'not', 'now', 'off', 'once', 'only', 'other', 'ought',
            'our', 'ours', 'out', 'over', 'own', 'same', 'shall', 'should', 'some',
            'such', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these',
            'they', 'this', 'those', 'through', 'too', 'under', 'until', 'very',
            'was', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
            'why', 'with', 'would', 'you', 'your', 'will', 'work', 'team', 'role',
            'position', 'job', 'company', 'looking', 'seeking', 'candidate'
        }
        
        # Filter words
        filtered_words = [w for w in words if w not in stop_words and len(w) > 4]
        
        # Get top keywords by frequency
        word_freq = Counter(filtered_words)
        top_keywords = [word for word, count in word_freq.most_common(20)]
        
        return top_keywords
    
    def _extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract experience requirements."""
        text_lower = text.lower()
        
        # Look for years of experience patterns
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
            r'(\d+)\+?\s*years?\s*(?:of)?\s*relevant\s*experience',
            r'minimum\s*(?:of)?\s*(\d+)\s*years?',
            r'at\s*least\s*(\d+)\s*years?',
        ]
        
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            years.extend([int(m) for m in matches])
        
        min_years = min(years) if years else 0
        max_years = max(years) if years else 0
        
        return {
            'min_years': min_years,
            'max_years': max_years,
            'all_mentions': years,
        }
    
    def _generate_summary(self, text: str) -> str:
        """Generate a brief summary of the job description."""
        # Extract first paragraph or first 300 characters
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if paragraphs:
            summary = paragraphs[0][:300]
            if len(paragraphs[0]) > 300:
                summary += "..."
            return summary
        
        return text[:300] + "..." if len(text) > 300 else text


class ResumeJobMatcher:
    """Match resume to job description."""
    
    def __init__(self, use_embeddings: bool = False):
        """Initialize the matcher.
        
        Args:
            use_embeddings: Whether to use semantic embeddings (requires sentence-transformers)
        """
        self.use_embeddings = use_embeddings
        self.embedder = None
        
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("Warning: sentence-transformers not installed. Disabling embeddings.")
                self.use_embeddings = False
    
    def calculate_keyword_match(
        self, 
        resume_text: str, 
        jd_keywords: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """Calculate keyword match score.
        
        Args:
            resume_text: Resume text
            jd_keywords: Keywords from job description
            
        Returns:
            Tuple of (score, matched_keywords, missing_keywords)
        """
        resume_lower = resume_text.lower()
        
        matched = []
        missing = []
        
        for keyword in jd_keywords:
            keyword_lower = keyword.lower()
            # Check for exact match or word boundary match
            if keyword_lower in resume_lower:
                matched.append(keyword)
            else:
                missing.append(keyword)
        
        # Calculate score
        if jd_keywords:
            score = len(matched) / len(jd_keywords)
        else:
            score = 0.0
        
        return score, matched, missing
    
    def calculate_skill_match(
        self,
        resume_skills: List[str],
        jd_required_skills: List[str],
        jd_preferred_skills: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """Calculate skill match score.
        
        Args:
            resume_skills: Skills from resume
            jd_required_skills: Required skills from JD
            jd_preferred_skills: Preferred skills from JD
            
        Returns:
            Tuple of (score, matched_skills, missing_skills)
        """
        resume_skills_lower = [s.lower() for s in resume_skills]
        
        matched = []
        missing = []
        
        # Check required skills (weighted more heavily)
        for skill in jd_required_skills:
            if skill.lower() in resume_skills_lower:
                matched.append(skill)
            else:
                missing.append(skill)
        
        # Check preferred skills (bonus points)
        preferred_matched = []
        for skill in jd_preferred_skills:
            if skill.lower() in resume_skills_lower:
                preferred_matched.append(skill)
        
        # Calculate weighted score
        total_skills = len(jd_required_skills) + len(jd_preferred_skills) * 0.5
        if total_skills > 0:
            score = (len(matched) + len(preferred_matched) * 0.5) / total_skills
        else:
            score = 0.0
        
        return min(score, 1.0), matched + preferred_matched, missing
    
    def calculate_semantic_similarity(
        self,
        resume_text: str,
        jd_text: str
    ) -> float:
        """Calculate semantic similarity using embeddings.
        
        Args:
            resume_text: Resume text
            jd_text: Job description text
            
        Returns:
            Similarity score (0-1)
        """
        if not self.use_embeddings or self.embedder is None:
            return 0.0
        
        try:
            # Encode texts
            resume_embedding = self.embedder.encode(resume_text[:1000])  # Limit length
            jd_embedding = self.embedder.encode(jd_text[:1000])
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(
                resume_embedding.reshape(1, -1),
                jd_embedding.reshape(1, -1)
            )[0][0]
            
            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def match(
        self,
        resume_text: str,
        resume_skills: List[str],
        jd_data: Dict[str, Any]
    ) -> JobMatchResult:
        """Perform complete job matching.
        
        Args:
            resume_text: Full resume text
            resume_skills: Extracted skills from resume
            jd_data: Parsed job description data
            
        Returns:
            JobMatchResult with all match metrics
        """
        # Calculate keyword match
        keyword_score, matched_keywords, missing_keywords = self.calculate_keyword_match(
            resume_text,
            jd_data['keywords']
        )
        
        # Calculate skill match
        skill_score, matched_skills, missing_skills = self.calculate_skill_match(
            resume_skills,
            jd_data['skills']['required'],
            jd_data['skills']['preferred']
        )
        
        # Calculate semantic similarity (if enabled)
        semantic_score = 0.0
        if self.use_embeddings:
            semantic_score = self.calculate_semantic_similarity(
                resume_text,
                jd_data['raw_text']
            )
        
        # Calculate overall match
        # Weights: Skills 40%, Keywords 30%, Semantic 30%
        if self.use_embeddings:
            overall = (
                skill_score * 0.40 +
                keyword_score * 0.30 +
                semantic_score * 0.30
            )
        else:
            # Without embeddings: Skills 60%, Keywords 40%
            overall = skill_score * 0.60 + keyword_score * 0.40
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            missing_keywords,
            missing_skills,
            jd_data['skills']['required']
        )
        
        return JobMatchResult(
            overall_match=overall,
            keyword_match=keyword_score,
            semantic_similarity=semantic_score,
            skill_match=skill_score,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            jd_summary=jd_data['summary'],
            recommendations=recommendations,
        )
    
    def _generate_recommendations(
        self,
        missing_keywords: List[str],
        missing_skills: List[str],
        required_skills: List[str]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Skill recommendations
        if missing_skills:
            if len(missing_skills) <= 3:
                recommendations.append(
                    f"Add these required skills: {', '.join(missing_skills)}"
                )
            else:
                recommendations.append(
                    f"Missing {len(missing_skills)} required skills. "
                    f"Top priorities: {', '.join(missing_skills[:3])}"
                )
        
        # Keyword recommendations
        if missing_keywords:
            top_keywords = missing_keywords[:5]
            recommendations.append(
                f"Include these keywords: {', '.join(top_keywords)}"
            )
        
        # Generic recommendations
        if not missing_skills and not missing_keywords:
            recommendations.append("Great match! Your resume aligns well with this job.")
        
        return recommendations


# Convenience functions
def parse_job_description(jd_text: str) -> Dict[str, Any]:
    """Parse job description text.
    
    Args:
        jd_text: Job description text
        
    Returns:
        Parsed JD data
    """
    parser = JobDescriptionParser()
    return parser.parse(jd_text)


def match_resume_to_job(
    resume_text: str,
    resume_skills: List[str],
    jd_text: str,
    use_embeddings: bool = False
) -> JobMatchResult:
    """Match resume to job description.
    
    Args:
        resume_text: Resume text
        resume_skills: Skills from resume
        jd_text: Job description text
        use_embeddings: Whether to use semantic similarity
        
    Returns:
        JobMatchResult with match scores
    """
    # Parse JD
    parser = JobDescriptionParser()
    jd_data = parser.parse(jd_text)
    
    # Match
    matcher = ResumeJobMatcher(use_embeddings=use_embeddings)
    return matcher.match(resume_text, resume_skills, jd_data)
