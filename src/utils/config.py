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

    # Supabase storage + database
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "resume-uploads")
    SUPABASE_SCHEMA = os.getenv("SUPABASE_SCHEMA", "public")
    SUPABASE_RESUMES_TABLE = os.getenv("SUPABASE_RESUMES_TABLE", "resumes")
    SUPABASE_EXTRACTIONS_TABLE = os.getenv("SUPABASE_EXTRACTIONS_TABLE", "resume_extractions")
    SUPABASE_AUTO_UPLOAD = os.getenv("SUPABASE_AUTO_UPLOAD", "false").lower() == "true"
    RESUME_RETENTION_DAYS = int(os.getenv("RESUME_RETENTION_DAYS", "30"))
    SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)
    
    # Default language for parsing
    DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "auto")  # "auto" or specific code like "en", "fr"

    # Async job queue
    JOBS_DB = PROCESSED_DIR / "jobs.db"
    JOBS_RESULTS_DIR = PROCESSED_DIR / "jobs"
    JOB_MAX_ATTEMPTS = int(os.getenv("JOB_MAX_ATTEMPTS", "3"))
    JOB_POLL_INTERVAL = float(os.getenv("JOB_POLL_INTERVAL", "2.0"))
    JOB_RETRY_BASE_SECONDS = int(os.getenv("JOB_RETRY_BASE_SECONDS", "5"))
    JOB_RETRY_MAX_SECONDS = int(os.getenv("JOB_RETRY_MAX_SECONDS", "60"))

    # Grounding enforcement + confidence flags
    ENFORCE_GROUNDING = os.getenv("ENFORCE_GROUNDING", "true").lower() == "true"
    LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.6"))
    
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
