"""Parser modules for extracting and analyzing resume content."""

from .ocr import PDFTextExtractor, extract_text_from_resume
from .layout_detector import (
    LayoutDetector,
    LayoutFeatures,
    MLLayoutDetector,
    HeuristicLayoutDetector,
    LAYOUT_DETECTION_AVAILABLE,
)
from .section_parser import SectionParser
from .language_detector import LanguageDetector
from .unified_extractor import UnifiedResumeExtractor, extract_unified
from .skill_extractor import SkillExtractor, extract_skills_from_resume

# Optional: LangExtract parser (requires pip install langextract)
try:
    from .langextract_parser import (
        LangExtractResumeParser,
        LangExtractResult,
        LANGEXTRACT_AVAILABLE,
    )
    from .langextract_constants import (
        RESUME_EXTRACTION_PROMPT,
        create_resume_examples,
        DEFAULT_CONFIG,
    )
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    LangExtractResumeParser = None
    LangExtractResult = None
    RESUME_EXTRACTION_PROMPT = None
    create_resume_examples = None
    DEFAULT_CONFIG = None

__all__ = [
    "PDFTextExtractor",
    "extract_text_from_resume",
    "LayoutDetector",
    "LayoutFeatures",
    "MLLayoutDetector",
    "HeuristicLayoutDetector",
    "LAYOUT_DETECTION_AVAILABLE",
    "SectionParser",
    "LanguageDetector",
    "UnifiedResumeExtractor",
    "extract_unified",
    "SkillExtractor",
    "extract_skills_from_resume",
    "LangExtractResumeParser",
    "LangExtractResult",
    "LANGEXTRACT_AVAILABLE",
    "RESUME_EXTRACTION_PROMPT",
    "create_resume_examples",
    "DEFAULT_CONFIG",
]
