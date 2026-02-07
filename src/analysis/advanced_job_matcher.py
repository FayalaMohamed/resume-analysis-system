"""Advanced job matching with semantic understanding, fuzzy matching, and skill relationships."""

import re
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import math
from difflib import SequenceMatcher


@dataclass
class SkillMatch:
    """Detailed skill match result."""
    skill_name: str
    confidence: float  # 0.0 to 1.0
    match_type: str  # 'exact', 'synonym', 'fuzzy', 'related'
    experience_years: Optional[float] = None
    context: Optional[str] = None


@dataclass
class AdvancedJobMatchResult:
    """Enhanced results from job-resume matching."""
    overall_match: float = 0.0
    keyword_match: float = 0.0
    semantic_similarity: float = 0.0
    skill_match: float = 0.0
    experience_match: float = 0.0
    
    matched_keywords: List[str] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)
    matched_skills: List[SkillMatch] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    related_skills: List[SkillMatch] = field(default_factory=list)  # Partial credit
    experience_gaps: List[Dict[str, Any]] = field(default_factory=list)
    
    jd_summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    # Detailed metrics
    exact_matches: int = 0
    synonym_matches: int = 0
    fuzzy_matches: int = 0
    related_matches: int = 0


class SkillTaxonomy:
    """Comprehensive skill taxonomy with synonyms and relationships."""
    
    # Skill aliases and synonyms
    SKILL_SYNONYMS = {
        # Programming Languages
        'javascript': ['js', 'ecmascript', 'es6', 'es2015'],
        'typescript': ['ts'],
        'python': ['py'],
        'c++': ['cpp', 'c plus plus'],
        'c#': ['csharp', 'cs', 'c sharp'],
        'go': ['golang'],
        'ruby': ['rb'],
        'php': ['php7', 'php8'],
        'objective-c': ['objc'],
        
        # Frameworks & Libraries
        'react': ['reactjs', 'react.js'],
        'react native': ['react-native', 'rn'],
        'vue': ['vuejs', 'vue.js'],
        'angular': ['angularjs', 'angular.js'],
        'node.js': ['nodejs', 'node'],
        'next.js': ['nextjs', 'next'],
        'express': ['expressjs', 'express.js'],
        'jquery': ['jQuery'],
        'bootstrap': ['bootstrap4', 'bootstrap5', 'bs4', 'bs5'],
        'tensorflow': ['tf'],
        'pytorch': ['torch'],
        
        # Databases
        'postgresql': ['postgres', 'psql', 'pg'],
        'mongodb': ['mongo'],
        'mysql': ['sql'],
        'elasticsearch': ['es'],
        'microsoft sql server': ['mssql', 'sql server'],
        'redis': ['redis cache'],
        
        # Cloud & DevOps
        'amazon web services': ['aws', 'amazon cloud'],
        'google cloud platform': ['gcp', 'google cloud'],
        'microsoft azure': ['azure', 'ms azure'],
        'docker': ['containerization', 'containers'],
        'kubernetes': ['k8s'],
        'gitlab ci/cd': ['gitlab ci', 'gitlab-cicd'],
        'github actions': ['gh actions'],
        'jenkins': ['jenkins ci'],
        
        # ML/AI
        'machine learning': ['ml', 'machine-learning'],
        'deep learning': ['dl', 'deep-learning'],
        'natural language processing': ['nlp'],
        'computer vision': ['cv'],
        'artificial intelligence': ['ai'],
        'scikit-learn': ['sklearn'],
        
        # Data
        'data science': ['data-science', 'datascience'],
        'data analysis': ['data-analysis', 'data analytics'],
        'extract transform load': ['etl'],
        'business intelligence': ['bi'],
        
        # Tools
        'visual studio code': ['vscode', 'vs code'],
        'github': ['gh'],
        'microsoft excel': ['excel', 'ms excel'],
        'powerpoint': ['ppt'],
        
        # Soft Skills
        'project management': ['pm', 'project-management'],
        'user experience': ['ux'],
        'user interface': ['ui'],
    }
    
    # Related skills (for partial credit)
    RELATED_SKILLS = {
        'react': ['vue', 'angular', 'svelte', 'next.js', 'javascript'],
        'vue': ['react', 'angular', 'nuxt', 'javascript'],
        'angular': ['react', 'vue', 'typescript'],
        'python': ['r', 'julia', 'matlab', 'data science'],
        'tensorflow': ['pytorch', 'keras', 'jax', 'deep learning'],
        'pytorch': ['tensorflow', 'jax', 'deep learning'],
        'aws': ['azure', 'gcp', 'docker', 'kubernetes'],
        'azure': ['aws', 'gcp', 'microsoft'],
        'gcp': ['aws', 'azure', 'google'],
        'docker': ['kubernetes', 'containerization', 'devops'],
        'kubernetes': ['docker', 'helm', 'devops'],
        'postgresql': ['mysql', 'mongodb', 'sql'],
        'mysql': ['postgresql', 'mariadb', 'sql'],
        'mongodb': ['dynamodb', 'cassandra', 'nosql'],
        'javascript': ['typescript', 'node.js', 'react', 'vue'],
        'typescript': ['javascript', 'angular'],
        'java': ['kotlin', 'scala', 'spring'],
        'spring': ['java', 'spring boot', 'microservices'],
        'django': ['flask', 'fastapi', 'python'],
        'flask': ['django', 'fastapi', 'python'],
        'git': ['github', 'gitlab', 'version control'],
        'github': ['git', 'gitlab', 'bitbucket'],
    }
    
    # Skill categories for context matching
    SKILL_CATEGORIES = {
        'frontend': ['react', 'vue', 'angular', 'javascript', 'typescript', 'css', 'html', 'webpack', 'sass'],
        'backend': ['python', 'java', 'node.js', 'django', 'flask', 'spring', 'sql', 'api'],
        'data_science': ['python', 'r', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'sql', 'machine learning'],
        'devops': ['docker', 'kubernetes', 'aws', 'azure', 'jenkins', 'terraform', 'linux'],
        'mobile': ['react native', 'swift', 'kotlin', 'flutter', 'ios', 'android'],
    }
    
    @classmethod
    def get_canonical_name(cls, skill: str) -> str:
        """Get canonical name for a skill."""
        skill_lower = skill.lower().strip()
        
        # Check if it's already canonical
        if skill_lower in cls.SKILL_SYNONYMS:
            return skill_lower
        
        # Check synonyms
        for canonical, synonyms in cls.SKILL_SYNONYMS.items():
            if skill_lower in synonyms or skill_lower == canonical:
                return canonical
        
        return skill_lower
    
    @classmethod
    def get_all_variations(cls, skill: str) -> Set[str]:
        """Get all variations of a skill name."""
        canonical = cls.get_canonical_name(skill)
        variations = {canonical}
        
        if canonical in cls.SKILL_SYNONYMS:
            variations.update(cls.SKILL_SYNONYMS[canonical])
        
        return variations
    
    @classmethod
    def get_related_skills(cls, skill: str) -> List[str]:
        """Get related skills for partial credit."""
        canonical = cls.get_canonical_name(skill)
        return cls.RELATED_SKILLS.get(canonical, [])


class AdvancedJobMatcher:
    """Advanced job matcher with semantic understanding."""
    
    # Experience extraction patterns
    EXPERIENCE_PATTERNS = [
        r'(?:(\d+)\+?)\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience\s+)?(?:with\s+|in\s+)?',
        r'(?:minimum|at least|min\.?)\s+(?:(\d+)\+?)\s*(?:years?|yrs?)',
        r'(?:(\d+)\+?)\s*(?:years?|yrs?)\s+(?:of\s+)?(?:professional\s+)?(?:work\s+)?experience',
    ]
    
    def __init__(self, use_embeddings: bool = False, fuzzy_threshold: float = 0.85):
        """
        Initialize advanced job matcher.
        
        Args:
            use_embeddings: Whether to use semantic embeddings
            fuzzy_threshold: Threshold for fuzzy string matching (0.0-1.0)
        """
        self.use_embeddings = use_embeddings
        self.fuzzy_threshold = fuzzy_threshold
        self.taxonomy = SkillTaxonomy()
        
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("Warning: sentence-transformers not installed. Disabling embeddings.")
                self.use_embeddings = False
    
    def fuzzy_match(self, str1: str, str2: str) -> float:
        """Calculate fuzzy string match score."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def extract_experience_requirements(self, text: str) -> Dict[str, float]:
        """Extract experience requirements from text."""
        requirements = {}
        
        for pattern in self.EXPERIENCE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                years = float(match.group(1))
                # Look for skill context nearby
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end]
                
                # Try to identify which skill this experience is for
                for skill in self.taxonomy.SKILL_SYNONYMS.keys():
                    if skill in context.lower():
                        requirements[skill] = years
                        break
        
        return requirements
    
    def match_skill_with_confidence(
        self, 
        target_skill: str, 
        candidate_skills: List[str]
    ) -> Optional[SkillMatch]:
        """
        Match a skill with confidence scoring.
        
        Returns SkillMatch if found, None otherwise.
        """
        target_lower = target_skill.lower()
        target_canonical = self.taxonomy.get_canonical_name(target_skill)
        
        for candidate in candidate_skills:
            candidate_lower = candidate.lower()
            candidate_canonical = self.taxonomy.get_canonical_name(candidate)
            
            # Exact match
            if target_lower == candidate_lower:
                return SkillMatch(
                    skill_name=candidate,
                    confidence=1.0,
                    match_type='exact'
                )
            
            # Canonical match (synonyms)
            if target_canonical == candidate_canonical:
                return SkillMatch(
                    skill_name=candidate,
                    confidence=0.95,
                    match_type='synonym'
                )
            
            # Check all variations
            target_variations = self.taxonomy.get_all_variations(target_skill)
            candidate_variations = self.taxonomy.get_all_variations(candidate)
            
            if target_variations & candidate_variations:  # Intersection
                return SkillMatch(
                    skill_name=candidate,
                    confidence=0.9,
                    match_type='synonym'
                )
            
            # Fuzzy match
            fuzzy_score = self.fuzzy_match(target_skill, candidate)
            if fuzzy_score >= self.fuzzy_threshold:
                return SkillMatch(
                    skill_name=candidate,
                    confidence=fuzzy_score * 0.8,  # Slightly lower confidence for fuzzy
                    match_type='fuzzy'
                )
        
        return None
    
    def find_related_skill_matches(
        self,
        target_skill: str,
        candidate_skills: List[str]
    ) -> List[SkillMatch]:
        """Find related skill matches for partial credit."""
        related = []
        target_canonical = self.taxonomy.get_canonical_name(target_skill)
        
        for related_skill in self.taxonomy.get_related_skills(target_skill):
            match = self.match_skill_with_confidence(related_skill, candidate_skills)
            if match:
                related.append(SkillMatch(
                    skill_name=match.skill_name,
                    confidence=match.confidence * 0.5,  # 50% credit for related skills
                    match_type='related',
                    context=f"Related to {target_skill}"
                ))
        
        return related
    
    def calculate_advanced_skill_match(
        self,
        resume_skills: List[str],
        jd_required_skills: List[str],
        jd_preferred_skills: List[str],
        jd_text: str = ""
    ) -> Tuple[float, List[SkillMatch], List[str], List[SkillMatch]]:
        """
        Calculate advanced skill match with confidence scoring.
        
        Returns:
            Tuple of (score, matched_skills, missing_skills, related_matches)
        """
        matched = []
        missing = []
        related = []
        total_weight = 0.0
        achieved_weight = 0.0
        
        # Extract experience requirements if JD text provided
        experience_reqs = self.extract_experience_requirements(jd_text) if jd_text else {}
        
        # Process required skills (weight = 2.0)
        for skill in jd_required_skills:
            total_weight += 2.0
            
            # Try direct match
            match = self.match_skill_with_confidence(skill, resume_skills)
            if match:
                # Check experience requirements
                canonical = self.taxonomy.get_canonical_name(skill)
                if canonical in experience_reqs:
                    match.experience_years = experience_reqs[canonical]
                
                matched.append(match)
                achieved_weight += 2.0 * match.confidence
            else:
                # Try related skills
                related_matches = self.find_related_skill_matches(skill, resume_skills)
                if related_matches:
                    # Take the best related match
                    best_related = max(related_matches, key=lambda x: x.confidence)
                    related.append(best_related)
                    achieved_weight += 2.0 * best_related.confidence * 0.5  # 50% bonus
                    missing.append(skill)  # Still counts as missing but with partial credit
                else:
                    missing.append(skill)
        
        # Process preferred skills (weight = 1.0)
        for skill in jd_preferred_skills:
            total_weight += 1.0
            
            match = self.match_skill_with_confidence(skill, resume_skills)
            if match:
                matched.append(match)
                achieved_weight += 1.0 * match.confidence
            else:
                related_matches = self.find_related_skill_matches(skill, resume_skills)
                if related_matches:
                    best_related = max(related_matches, key=lambda x: x.confidence)
                    related.append(best_related)
                    achieved_weight += 1.0 * best_related.confidence * 0.5
        
        # Calculate final score
        if total_weight > 0:
            score = min(achieved_weight / total_weight, 1.0)
        else:
            score = 0.0
        
        return score, matched, missing, related
    
    def calculate_keyword_match(
        self,
        resume_text: str,
        jd_keywords: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """Enhanced keyword matching with fuzzy support."""
        resume_lower = resume_text.lower()
        matched = []
        missing = []
        
        for keyword in jd_keywords:
            keyword_lower = keyword.lower()
            
            # Check for exact match
            if keyword_lower in resume_lower:
                matched.append(keyword)
            else:
                # Try fuzzy matching on individual words
                words = resume_lower.split()
                best_match = max(
                    (self.fuzzy_match(keyword_lower, word) for word in words),
                    default=0
                )
                
                if best_match >= self.fuzzy_threshold:
                    matched.append(f"{keyword} (~{int(best_match*100)}% match)")
                else:
                    missing.append(keyword)
        
        score = len(matched) / len(jd_keywords) if jd_keywords else 0.0
        return score, matched, missing
    
    def match(
        self,
        resume_text: str,
        resume_skills: List[str],
        jd_data: Dict[str, Any]
    ) -> AdvancedJobMatchResult:
        """
        Perform advanced matching between resume and job description.
        
        Args:
            resume_text: Full resume text
            resume_skills: Extracted skills from resume
            jd_data: Parsed job description data
            
        Returns:
            AdvancedJobMatchResult with detailed matching information
        """
        # Calculate keyword match
        keyword_score, matched_keywords, missing_keywords = self.calculate_keyword_match(
            resume_text,
            jd_data.get('keywords', [])
        )
        
        # Calculate advanced skill match
        jd_text = jd_data.get('raw_text', '')
        skill_score, matched_skills, missing_skills, related_skills = self.calculate_advanced_skill_match(
            resume_skills,
            jd_data.get('skills', {}).get('required', []),
            jd_data.get('skills', {}).get('preferred', []),
            jd_text
        )
        
        # Calculate semantic similarity if enabled
        semantic_score = 0.0
        if self.use_embeddings:
            semantic_score = self._calculate_semantic_similarity(resume_text, jd_text)
        
        # Calculate experience match
        experience_score = self._calculate_experience_match(
            resume_text,
            jd_data.get('experience_requirements', {})
        )
        
        # Calculate overall match with weights
        # Skills: 35%, Keywords: 25%, Experience: 20%, Semantic: 20%
        if self.use_embeddings:
            overall = (
                skill_score * 0.35 +
                keyword_score * 0.25 +
                experience_score * 0.20 +
                semantic_score * 0.20
            )
        else:
            # Without embeddings: Skills 45%, Keywords 35%, Experience 20%
            overall = (
                skill_score * 0.45 +
                keyword_score * 0.35 +
                experience_score * 0.20
            )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall, skill_score, missing_skills, missing_keywords, related_skills
        )
        
        # Count match types
        exact_matches = sum(1 for m in matched_skills if m.match_type == 'exact')
        synonym_matches = sum(1 for m in matched_skills if m.match_type == 'synonym')
        fuzzy_matches = sum(1 for m in matched_skills if m.match_type == 'fuzzy')
        related_matches = len(related_skills)
        
        return AdvancedJobMatchResult(
            overall_match=overall,
            keyword_match=keyword_score,
            semantic_similarity=semantic_score,
            skill_match=skill_score,
            experience_match=experience_score,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            related_skills=related_skills,
            recommendations=recommendations,
            exact_matches=exact_matches,
            synonym_matches=synonym_matches,
            fuzzy_matches=fuzzy_matches,
            related_matches=related_matches
        )
    
    def _calculate_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        """Calculate semantic similarity using embeddings."""
        if not self.use_embeddings or not hasattr(self, 'embedder'):
            return 0.0
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            resume_embedding = self.embedder.encode(resume_text[:2000])  # Limit length
            jd_embedding = self.embedder.encode(jd_text[:2000])
            
            similarity = cosine_similarity(
                resume_embedding.reshape(1, -1),
                jd_embedding.reshape(1, -1)
            )
            
            return float(similarity[0][0])
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _calculate_experience_match(
        self,
        resume_text: str,
        requirements: Dict[str, float]
    ) -> float:
        """Calculate experience match score."""
        if not requirements:
            return 1.0  # No requirements = perfect match
        
        # Extract experience from resume
        resume_experience = self.extract_experience_requirements(resume_text)
        
        total_reqs = len(requirements)
        matched = 0
        
        for skill, required_years in requirements.items():
            canonical_skill = self.taxonomy.get_canonical_name(skill)
            
            # Check if candidate has experience with this skill
            for res_skill, res_years in resume_experience.items():
                res_canonical = self.taxonomy.get_canonical_name(res_skill)
                
                if canonical_skill == res_canonical or \
                   canonical_skill in self.taxonomy.get_related_skills(res_skill):
                    if res_years >= required_years:
                        matched += 1
                    else:
                        matched += 0.5  # Partial credit for less experience
                    break
        
        return matched / total_reqs if total_reqs > 0 else 1.0
    
    def _generate_recommendations(
        self,
        overall: float,
        skill_score: float,
        missing_skills: List[str],
        missing_keywords: List[str],
        related_skills: List[SkillMatch]
    ) -> List[str]:
        """Generate intelligent recommendations."""
        recommendations = []
        
        if overall < 0.3:
            recommendations.append(
                "Low overall match. Consider targeting different roles or significantly updating your resume."
            )
        elif overall < 0.6:
            recommendations.append(
                "Moderate match. Focus on adding missing skills and keywords to improve your chances."
            )
        
        if missing_skills:
            top_missing = missing_skills[:5]
            recommendations.append(
                f"Missing key skills: {', '.join(top_missing)}. "
                f"Consider adding these if you have experience with them."
            )
        
        if related_skills:
            related_names = [f"{s.skill_name} (related to {s.context or 'required skill'})" 
                           for s in related_skills[:3]]
            recommendations.append(
                f"You have related skills that may help: {', '.join(related_names)}. "
                f"Consider highlighting the connection more explicitly."
            )
        
        if missing_keywords:
            top_keywords = missing_keywords[:5]
            recommendations.append(
                f"Add these keywords naturally: {', '.join(top_keywords)}"
            )
        
        return recommendations


def match_resume_to_job_advanced(
    resume_text: str,
    resume_skills: List[str],
    jd_text: str,
    use_embeddings: bool = False
) -> AdvancedJobMatchResult:
    """
    Convenience function to match resume to job description.
    
    Args:
        resume_text: Full resume text
        resume_skills: Skills extracted from resume
        jd_text: Job description text
        use_embeddings: Whether to use semantic embeddings
        
    Returns:
        AdvancedJobMatchResult with detailed matching
    """
    # Parse JD using existing parser
    from .job_matcher import JobDescriptionParser
    
    parser = JobDescriptionParser()
    jd_data = parser.parse(jd_text)
    jd_data['raw_text'] = jd_text  # Include raw text for context
    
    # Perform advanced matching
    matcher = AdvancedJobMatcher(use_embeddings=use_embeddings)
    return matcher.match(resume_text, resume_skills, jd_data)
