"""Enhanced Skills Extraction and Taxonomy Module.

Phase 4 improvements for comprehensive skill analysis:
- Comprehensive skill taxonomy with categories and hierarchies
- Explicit and implicit skill extraction
- Proficiency level detection
- Skill relationship mapping
- Gap analysis between resume and job requirements
"""

import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SkillCategory(Enum):
    """Skill categories for classification."""
    PROGRAMMING_LANGUAGE = "programming_language"
    FRAMEWORK = "framework"
    DATABASE = "database"
    CLOUD_DEVOPS = "cloud_devops"
    MACHINE_LEARNING = "machine_learning"
    DATA_ENGINEERING = "data_engineering"
    FRONTEND = "frontend"
    BACKEND = "backend"
    MOBILE = "mobile"
    DEVOPS = "devops"
    TESTING = "testing"
    SOFT_SKILL = "soft_skill"
    CERTIFICATION = "certification"
    TOOL = "tool"
    METHODOLOGY = "methodology"
    DOMAIN = "domain"
    UNKNOWN = "unknown"


class ProficiencyLevel(Enum):
    """Proficiency levels for skills."""
    EXPERT = "expert"
    ADVANCED = "advanced"
    INTERMEDIATE = "intermediate"
    BEGINNER = "beginner"
    UNKNOWN = "unknown"


@dataclass
class SkillInfo:
    """Detailed information about a skill."""
    name: str
    canonical_name: str
    category: SkillCategory
    confidence: float = 0.0
    proficiency: ProficiencyLevel = ProficiencyLevel.UNKNOWN
    is_explicit: bool = True
    source_section: str = ""
    related_skills: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "canonical_name": self.canonical_name,
            "category": self.category.value,
            "confidence": self.confidence,
            "proficiency": self.proficiency.value,
            "is_explicit": self.is_explicit,
            "source_section": self.source_section,
            "related_skills": self.related_skills,
            "synonyms": self.synonyms,
        }


@dataclass
class SkillGapResult:
    """Result of skill gap analysis."""
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    partial_matches: List[Tuple[str, List[str]]] = field(default_factory=list)
    gap_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "partial_matches": [{"job_skill": j, "resume_matches": r} for j, r in self.partial_matches],
            "gap_score": self.gap_score,
            "recommendations": self.recommendations,
        }


class EnhancedSkillTaxonomy:
    """Comprehensive skill taxonomy with categories and hierarchies."""
    
    SKILLS = {
        SkillCategory.PROGRAMMING_LANGUAGE: [
            "python", "javascript", "typescript", "java", "c++", "c#", "go",
            "ruby", "php", "rust", "swift", "kotlin", "scala", "r", "matlab",
            "perl", "shell", "bash", "powershell", "groovy", "dart", "elixir",
            "haskell", "clojure", "f#", "ocaml", "lua", "julia", "ada",
        ],
        SkillCategory.FRAMEWORK: [
            "react", "react native", "vue", "angular", "next.js", "nuxt",
            "svelte", "django", "flask", "fastapi", "express", "nestjs", "spring",
            "rails", "laravel", "tensorflow", "pytorch", "keras", "scikit-learn",
            "pandas", "numpy", "jquery", "bootstrap", "tailwindcss", "sass",
            "node.js", "deno", "electron",
        ],
        SkillCategory.DATABASE: [
            "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
            "dynamodb", "cassandra", "firebase", "supabase", "neo4j",
            "oracle", "microsoft sql server", "sqlite", "mariadb",
            "cockroachdb", "clickhouse", "influxdb", "snowflake", "bigquery",
        ],
        SkillCategory.CLOUD_DEVOPS: [
            "amazon web services", "aws", "google cloud platform", "gcp",
            "microsoft azure", "docker", "kubernetes", "helm", "terraform",
            "jenkins", "gitlab ci/cd", "github actions", "circleci",
            "prometheus", "grafana", "cloudwatch", "cloudtrail",
        ],
        SkillCategory.MACHINE_LEARNING: [
            "machine learning", "deep learning", "nlp", "computer vision",
            "natural language processing", "large language models", "llm",
            "gpt", "bert", "transformers", "reinforcement learning",
            "generative ai", "diffusion models", "gan", "vae",
        ],
        SkillCategory.DATA_ENGINEERING: [
            "apache spark", "apache kafka", "apache flink", "airflow",
            "etl", "data pipeline", "data warehouse", "data lake",
            "dbt", "sql", "pandas", "dask",
        ],
        SkillCategory.FRONTEND: [
            "html", "css", "javascript", "typescript",
            "react", "vue", "angular", "webpack", "vite",
            "babel", "sass", "responsive design", "accessibility",
        ],
        SkillCategory.BACKEND: [
            "rest api", "graphql", "grpc", "microservices", "serverless",
            "authentication", "oauth", "jwt", "caching", "orm",
        ],
        SkillCategory.MOBILE: [
            "react native", "flutter", "android", "ios",
            "swift", "kotlin", "dart", "xcode", "android studio",
        ],
        SkillCategory.DEVOPS: [
            "linux", "bash", "git", "docker", "kubernetes",
            "ci/cd", "monitoring", "security", "aws", "azure",
        ],
        SkillCategory.TESTING: [
            "jest", "pytest", "junit", "selenium", "cypress",
            "unit testing", "integration testing", "tdd", "bdd",
        ],
        SkillCategory.SOFT_SKILL: [
            "communication", "teamwork", "problem solving", "leadership",
            "project management", "time management", "adaptability",
            "critical thinking", "analytical thinking", "agile", "scrum",
        ],
        SkillCategory.CERTIFICATION: [
            "AWS Certified", "Azure Certified", "PMP", "CAPM", "CSM",
            "CISSP", "CompTIA Security+", "Kubernetes Administrator",
        ],
        SkillCategory.METHODOLOGY: [
            "agile", "scrum", "kanban", "lean", "six sigma",
            "prince2", "pmp", "itil", "devops",
        ],
        SkillCategory.DOMAIN: [
            "finance", "healthcare", "e-commerce", "fintech",
            "edtech", "cybersecurity", "blockchain", "iot",
        ],
    }
    
    ALIASES = {
        "js": "javascript", "ts": "typescript", "py": "python",
        "rb": "ruby", "rs": "rust", "cpp": "c++", "cs": "c#",
        "golang": "go", "react": "react", "vuejs": "vue",
        "angularjs": "angular", "node": "node.js", "nodejs": "node.js",
        "pg": "postgresql", "postgres": "postgresql", "mongo": "mongodb",
        "aws": "amazon web services", "gcp": "google cloud platform",
        "k8s": "kubernetes", "tf": "tensorflow", "torch": "pytorch",
        "sklearn": "scikit-learn", "ml": "machine learning",
        "dl": "deep learning", "nlp": "natural language processing",
        "cv": "computer vision", "ai": "artificial intelligence",
        "devops": "devops", "ci/cd": "ci cd", "cicd": "ci cd",
        "ux": "user experience", "ui": "user interface", "pm": "project management",
    }
    
    RELATED = {
        "react": ["vue", "angular", "svelte", "next.js", "javascript"],
        "angular": ["vue", "react", "svelte", "typescript", "javascript"],
        "vue": ["react", "angular", "svelte", "javascript", "typescript"],
        "python": ["django", "flask", "pandas", "numpy", "machine learning"],
        "javascript": ["typescript", "node.js", "react", "vue"],
        "java": ["spring", "kotlin", "scala", "android"],
        "postgresql": ["mysql", "mongodb", "sqlite"],
        "aws": ["azure", "gcp", "docker", "kubernetes"],
        "docker": ["kubernetes", "containerization"],
        "kubernetes": ["docker", "helm", "terraform"],
        "tensorflow": ["pytorch", "keras", "machine learning"],
        "pytorch": ["tensorflow", "keras", "deep learning"],
    }
    
    @classmethod
    def get_category(cls, skill: str) -> SkillCategory:
        skill_lower = skill.lower().strip()
        for category, skills in cls.SKILLS.items():
            if skill_lower in skills:
                return category
        return SkillCategory.UNKNOWN
    
    @classmethod
    def normalize_skill(cls, skill: str) -> str:
        skill_lower = skill.lower().strip()
        if skill_lower in cls.ALIASES:
            return cls.ALIASES[skill_lower]
        for category, skills in cls.SKILLS.items():
            if skill_lower in skills:
                return skill_lower
        return skill_lower
    
    @classmethod
    def get_related(cls, skill: str) -> List[str]:
        skill_lower = skill.lower().strip()
        return cls.RELATED.get(skill_lower, [])
    
    @classmethod
    def get_all_categories(cls) -> List[SkillCategory]:
        """Get all available categories."""
        return list(cls.SKILLS.keys())


class EnhancedSkillExtractor:
    """Enhanced skill extractor with proficiency detection."""
    
    PROFICIENCY_INDICATORS = {
        ProficiencyLevel.EXPERT: [r'\bexpert\b', r'\bmaster\b', r'\blead\b', r'\barchitect\b'],
        ProficiencyLevel.ADVANCED: [r'\bsenior\b', r'\badvanced\b', r'\bspecialized\b'],
        ProficiencyLevel.INTERMEDIATE: [r'\bproficient\b', r'\bexperienced\b'],
        ProficiencyLevel.BEGINNER: [r'\bfamiliar\b', r'\bbasic\b', r'\blearning\b', r'\bjunior\b'],
    }
    
    def __init__(self, taxonomy: Optional[EnhancedSkillTaxonomy] = None):
        self.taxonomy = taxonomy or EnhancedSkillTaxonomy()
    
    def extract_skills(
        self,
        skills_text: Optional[str] = None,
        experience_text: Optional[str] = None,
        sections: Optional[List[Dict]] = None,
    ) -> List[SkillInfo]:
        skills = []
        
        if sections:
            for section in sections:
                section_type = section.get("section_type", "")
                raw_text = section.get("raw_text", "")
                if section_type == "skills":
                    section_skills = self._extract_from_skills_section(raw_text)
                    skills.extend(section_skills)
                elif section_type == "experience":
                    implicit_skills = self._extract_implicit_skills(raw_text)
                    existing_names = {s.canonical_name for s in skills}
                    for skill in implicit_skills:
                        if skill.canonical_name not in existing_names:
                            skill.is_explicit = False
                            skills.append(skill)
                else:
                    all_skills = self._extract_from_skills_section(raw_text)
                    for skill in all_skills:
                        if skill.canonical_name not in {s.canonical_name for s in skills}:
                            skills.append(skill)
        else:
            if skills_text:
                explicit_skills = self._extract_from_skills_section(skills_text)
                skills.extend(explicit_skills)
            
            if experience_text:
                implicit_skills = self._extract_implicit_skills(experience_text)
                existing_names = {s.canonical_name for s in skills}
                for skill in implicit_skills:
                    if skill.canonical_name not in existing_names:
                        skill.is_explicit = False
                        skills.append(skill)
        
        return skills
    
    def _extract_from_skills_section(self, text: str) -> List[SkillInfo]:
        skills = []
        separators = r'[,;|•\-·\n\r]+'
        items = re.split(separators, text, flags=re.IGNORECASE)
        
        for item in items:
            item = item.strip()
            if len(item) < 2:
                continue
            
            canonical = self.taxonomy.normalize_skill(item)
            category = self.taxonomy.get_category(canonical)
            proficiency = self._detect_proficiency(item)
            confidence = 0.95 if category != SkillCategory.UNKNOWN else 0.5
            related = self.taxonomy.get_related(canonical)
            
            skills.append(SkillInfo(
                name=item,
                canonical_name=canonical,
                category=category,
                confidence=confidence,
                proficiency=proficiency,
                is_explicit=True,
                source_section="skills",
                related_skills=related,
                synonyms=[canonical] + list(self.taxonomy.ALIASES.get(canonical.lower(), [])),
            ))
        
        return skills
    
    def _extract_implicit_skills(self, text: str) -> List[SkillInfo]:
        skills = []
        text_lower = text.lower()
        
        for category, category_skills in EnhancedSkillTaxonomy.SKILLS.items():
            for skill in category_skills:
                if skill in ["python", "java", "sql", "aws"]:
                    continue
                if re.search(rf'\b{re.escape(skill)}\b', text_lower):
                    skills.append(SkillInfo(
                        name=skill,
                        canonical_name=skill,
                        category=category,
                        confidence=0.6,
                        proficiency=ProficiencyLevel.INTERMEDIATE,
                        is_explicit=False,
                        source_section="experience",
                    ))
                    break
        
        return skills
    
    def _detect_proficiency(self, text: str) -> ProficiencyLevel:
        text_lower = text.lower()
        for level, patterns in self.PROFICIENCY_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return level
        return ProficiencyLevel.UNKNOWN


class SkillGapAnalyzer:
    """Analyze gaps between resume skills and job requirements."""
    
    def __init__(self):
        self.taxonomy = EnhancedSkillTaxonomy()
    
    def analyze(
        self,
        resume_skills: List[SkillInfo],
        job_skills: List[str],
    ) -> SkillGapResult:
        result = SkillGapResult()
        resume_skill_names = {s.canonical_name.lower() for s in resume_skills}
        
        for job_skill in job_skills:
            job_lower = job_skill.lower()
            normalized_job = self.taxonomy.normalize_skill(job_skill)
            
            if job_lower in resume_skill_names or normalized_job.lower() in resume_skill_names:
                result.matched_skills.append(job_skill)
                continue
            
            related = self.taxonomy.get_related(normalized_job)
            matched_related = [s for s in resume_skills if s.canonical_name.lower() in related]
            if matched_related:
                result.partial_matches.append((job_skill, [s.name for s in matched_related]))
            else:
                result.missing_skills.append(job_skill)
        
        if job_skills:
            match_rate = len(result.matched_skills) / len(job_skills)
            result.gap_score = (1 - match_rate) * 100
        
        if result.missing_skills:
            result.recommendations.append(f"Consider adding: {', '.join(result.missing_skills[:5])}")
        
        return result


def extract_skills_from_resume(
    skills_text: Optional[str] = None,
    experience_text: Optional[str] = None,
    sections: Optional[List[Dict]] = None,
) -> List[Dict]:
    extractor = EnhancedSkillExtractor()
    skills = extractor.extract_skills(skills_text, experience_text, sections)
    return [s.to_dict() for s in skills]


def analyze_skill_gaps(resume_skills: List[Dict], job_skills: List[str]) -> Dict:
    skill_objects = []
    for s in resume_skills:
        try:
            category = SkillCategory(s.get("category", "unknown"))
        except:
            category = SkillCategory.UNKNOWN
        skill_objects.append(SkillInfo(
            name=s.get("name", ""),
            canonical_name=s.get("canonical_name", ""),
            category=category,
            confidence=s.get("confidence", 0.0),
        ))
    
    analyzer = SkillGapAnalyzer()
    result = analyzer.analyze(skill_objects, job_skills)
    return result.to_dict()