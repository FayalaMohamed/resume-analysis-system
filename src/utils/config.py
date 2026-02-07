"""Configuration settings for the ATS Resume Parser."""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed


class Config:
    """Project configuration."""
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = DATA_DIR / "logs"
    PROCESSED_DIR = DATA_DIR / "processed"
    RESUMES_DIR = BASE_DIR / "resumes"
    
    # OCR settings
    OCR_DPI = 200
    TEMP_IMAGE_DIR = RESUMES_DIR / ".temp_ocr"
    
    # ML Layout Detection Settings
    USE_ML_LAYOUT_DETECTION = os.getenv("USE_ML_LAYOUT_DETECTION", "true").lower() == "true"
    ML_TABLE_CONFIDENCE_THRESHOLD = float(os.getenv("ML_TABLE_CONFIDENCE_THRESHOLD", "0.7"))
    ML_COLUMN_GAP_THRESHOLD = int(os.getenv("ML_COLUMN_GAP_THRESHOLD", "50"))
    ML_IMAGE_DPI = int(os.getenv("ML_IMAGE_DPI", "150"))
    
    # Scoring weights
    LAYOUT_WEIGHT = 25
    FORMAT_WEIGHT = 25
    CONTENT_WEIGHT = 25
    STRUCTURE_WEIGHT = 25
    
    # API Keys (loaded from environment)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")  # "openrouter" or "ollama"
    
    # Default language for parsing
    DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "auto")  # "auto" or specific code like "en", "fr"
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.PROCESSED_DIR.mkdir(exist_ok=True)
        cls.TEMP_IMAGE_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_resume_files(cls) -> list:
        """Get list of resume PDF files."""
        if cls.RESUMES_DIR.exists():
            return list(cls.RESUMES_DIR.glob("*.pdf"))
        return []
