# ATS Resume Analyzer

A comprehensive Applicant Tracking System (ATS) evaluation tool that analyzes resumes for ATS compatibility, extracts structured information, matches against job descriptions, and provides AI-powered improvement recommendations.

## Project Overview

This system evaluates how well a resume will be parsed by automated ATS systems and provides actionable recommendations to improve ATS compatibility.

## Project Structure

```
.
├── app.py                      # Streamlit web interface
├── pipeline.py                  # Command-line pipeline script
├── requirements.txt             # Python dependencies
├── src/                        # Source code
│   ├── __init__.py
│   ├── constants.py            # Scoring constants, action verbs, section patterns
│   ├── parsers/               # Text extraction and parsing
│   │   ├── __init__.py
│   │   ├── ocr.py             # PDF text extraction (PyMuPDF + PaddleOCR)
│   │   ├── layout_detector.py  # Layout analysis (columns, tables, sections)
│   │   ├── section_parser.py  # Resume section parsing (20+ sections)
│   │   └── language_detector.py # Multi-language detection (13+ languages)
│   ├── analysis/              # Advanced analysis engines
│   │   ├── __init__.py
│   │   ├── content_analyzer.py        # Content quality analysis
│   │   ├── job_matcher.py             # Job description parsing & matching
│   │   ├── advanced_job_matcher.py    # Advanced matching with skill taxonomy
│   │   ├── recommendation_engine.py   # Prioritized recommendations
│   │   ├── ats_simulator.py           # ATS parsing simulation
│   │   └── llm_client.py               # LLM integration (OpenRouter/Ollama)
│   ├── scoring/               # ATS scoring
│   │   ├── __init__.py
│   │   └── ats_scorer.py      # ATS compatibility scoring (0-100)
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── config.py          # Configuration settings
│       └── logger.py          # Logging utilities
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_ocr.py            # OCR tests
│   └── test_language_detector.py # Language detector tests
├── resumes/                    # Resume PDFs (test data)
├── data/                       # Data storage
│   ├── logs/                  # Processing logs
│   └── processed/             # Processed outputs
├── docs/                       # Documentation
│   ├── README.md
│   ├── phase1_implementation.md
│   └── phase2_implementation.md
└── planning/                   # Project planning docs
    ├── README.md
    ├── TECHNICAL_PLAN.md
    ├── phase1_mvp.md
    ├── phase2_enhanced.md
    └── phase3_advanced.md
```

## Features

### Text Extraction & Layout Detection

- **PDF to Text**: Extract text using PyMuPDF (fast) or PaddleOCR (image-based PDFs)
- **Layout Analysis**: Detect multi-column layouts, tables, images, section headers
- **Language Detection**: Supports 13+ languages (English, French, Spanish, German, Italian, Portuguese, etc.)
- **Section Parsing**: Identify 20+ resume sections (Experience, Education, Skills, Projects, Certifications, etc.)

### ATS Scoring (0-100)

| Component | Points | Description |
|-----------|--------|-------------|
| Layout | 25 | Single-column (+25), Multi-column (-25), Tables (-20) |
| Format | 25 | Good length (+), No images (+), Too short/long (-) |
| Content | 25 | Contact info (+), Experience (+), Education (+), Skills (+) |
| Structure | 25 | Clear headers (+), Good text density (+), Standard naming (+) |

### Risk Levels

- **Low (80-100)**: ATS-friendly, minor improvements possible
- **Medium (60-79)**: Some ATS issues detected
- **High (0-59)**: Significant ATS compatibility problems

### Content Quality Analysis

- **Action Verb Detection**: Identifies strong vs weak verbs with scoring (0-25 points)
- **Quantification Detection**: Finds metrics, percentages, numbers in achievements
- **Bullet Point Analysis**: Evaluates structure, length, consistency
- **Conciseness Analysis**: Detects filler words and verbose language

### Job Description Matching

- **Keyword Extraction**: TF-IDF-like frequency analysis
- **Skill Matching**: 100+ predefined skills with synonym support
- **Semantic Similarity**: Optional embeddings using sentence-transformers
- **Skill Taxonomy**: 100+ skill synonyms and related skills for fuzzy matching

### AI-Powered Recommendations

- **Dual LLM Support**: OpenRouter API (free tier) + Ollama (local)
- **Prioritized Suggestions**: HIGH/MEDIUM/LOW priority levels
- **Impact Estimation**: Estimated score improvement for each recommendation
- **ATS Simulation**: Shows what ATS systems actually parse from resumes

### Streamlit UI

- File upload interface with OCR toggle
- 5-tab interface: Overview, Content Quality, Job Matching, ATS Simulation, Recommendations
- Real-time score visualization
- LLM status indicators
- Raw text viewer (optional)

## Quick Start

### 1. Install Dependencies

```bash
# Create conda environment
conda create -n ats-resume python=3.10
conda activate ats-resume

# Install packages
pip install -r requirements.txt
```

### 2. Configure LLM (Optional)

For AI-powered suggestions, configure either OpenRouter or Ollama:

**OpenRouter (API):**
```bash
# Get free API key at: https://openrouter.ai/keys
export OPENROUTER_API_KEY="your_api_key"
```

**Ollama (Local):**
```bash
# Install from https://ollama.com
ollama pull ministral:3b-instruct-2512-q4_K_M
```

### 3. Run Streamlit App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 4. Run Pipeline Script (Comprehensive Analysis)

The pipeline script runs ALL system components with all available variants and outputs detailed comparisons:

```bash
# Process a single resume (saves both JSON and colored terminal output by default)
python pipeline.py resumes/CV.pdf

# Output files created:
# - CV_results.json        (structured data)
# - CV_terminal.txt        (beautiful colored terminal output)

# Process with job description
python pipeline.py resumes/CV.pdf --job job_description.txt

# Custom output filenames
python pipeline.py resumes/CV.pdf --output my_analysis.json --save-terminal my_output.txt

# Only save terminal output (no JSON)
python pipeline.py resumes/CV.pdf --no-output

# Only save JSON results (no terminal file)
python pipeline.py resumes/CV.pdf --no-terminal

# Disable colors in terminal
python pipeline.py resumes/CV.pdf --no-color

# View saved terminal output with colors
less -R CV_terminal.txt
```

**What the pipeline analyzes:**
1. **Text Extraction Variants** - PyMuPDF vs OCR comparison
2. **Language Detection** - Auto-detection + all supported languages
3. **Layout Detection Variants** - ML-based vs Heuristic vs Auto-select
4. **Section Parsing** - 20+ resume sections
5. **Content Analysis** - Action verbs, quantification, bullet structure
6. **ATS Scoring** - 0-100 score with breakdown
7. **ATS Simulation** - What ATS systems actually see
8. **Recommendations** - Prioritized improvement suggestions
9. **Job Matching** - Basic and advanced matching (if job description provided)

**Pipeline Output Structure:**
- Each component runs multiple variants (e.g., ML vs heuristic layout detection)
- Shows agreement/disagreement between methods
- Timing information for each step
- Side-by-side comparisons
- Full JSON export with all raw data

## API Usage

### Basic Analysis

```python
from src.parsers import PDFTextExtractor, LayoutDetector, SectionParser
from src.scoring import ATSScorer

# Extract text
extractor = PDFTextExtractor()
result = extractor.extract_text_from_pdf("resume.pdf")
text = result["full_text"]

# Analyze layout
layout = LayoutDetector()
layout_features = layout.analyze_layout(text)

# Parse sections
parser = SectionParser()
parsed = parser.parse(text)

# Calculate ATS score
scorer = ATSScorer()
ats_score = scorer.calculate_score(text, layout_features, parsed)

print(f"ATS Score: {ats_score.overall_score}/100")
print(f"Risk Level: {ats_score.risk_level}")
```

### Content Analysis

```python
from src.analysis import ContentAnalyzer

analyzer = ContentAnalyzer()
content_score = analyzer.analyze(resume_text)

print(f"Content Score: {content_score.overall_score}/100")
print(f"Strong Verbs: {content_score.action_verbs_found}")
print(f"Quantified Achievements: {len(content_score.quantified_achievements)}")
```

### Job Matching

```python
from src.analysis import AdvancedJobMatcher

matcher = AdvancedJobMatcher()
match_result = matcher.match(
    resume_text=resume_text,
    resume_skills=resume_skills,
    jd_text=job_description
)

print(f"Match Score: {match_result.overall_match:.0%}")
print(f"Missing Skills: {match_result.missing_skills}")
```

### AI Recommendations

```python
from src.analysis import RecommendationEngine
from src.analysis import LLMClient

recommendations = RecommendationEngine()
llm = LLMClient()

# Generate recommendations
recs = recommendations.generate_recommendations(
    ats_score=ats_score,
    content_score=content_score,
    layout=layout_features,
    job_match=match_result
)

# Get AI improvement suggestion
response = llm.improve_bullet_point("Responsible for managing team")
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test
pytest tests/test_ocr.py -v
```

## Language Support

| Language | Code | Support Level |
|----------|------|---------------|
| English | en | Full |
| French | fr | Full |
| Spanish | es | Full |
| German | de | Full |
| Italian | it | Full |
| Portuguese | pt | Full |
| + 7 more | - | Basic |

## Supported Sections (20+)

Experience, Education, Skills, Projects, Certifications, Languages, Awards, Publications, Interests, References, Volunteer Work, Professional Affiliations, Speaking Engagements, Patents, Workshops, Activities, Online Presence, Research, Exhibitions, Productions, Teaching, Clinical Experience, Technical Skills

## License

Internal project for educational purposes.
