# Phase 1: MVP - Core Functionality

This directory contains the MVP implementation of the ATS Resume Analyzer.

## Project Structure

```
.
├── app.py                      # Streamlit web interface
├── pipeline.py                 # Command-line pipeline script
├── requirements.txt            # Python dependencies
├── src/                        # Source code
│   ├── __init__.py
│   ├── parsers/               # Text extraction and parsing
│   │   ├── __init__.py
│   │   ├── ocr.py            # PDF text extraction (PyMuPDF + PaddleOCR)
│   │   ├── layout_detector.py # Layout analysis (columns, tables)
│   │   └── section_parser.py  # Resume section parsing
│   ├── scoring/               # ATS scoring
│   │   ├── __init__.py
│   │   └── ats_scorer.py     # ATS compatibility scoring
│   └── utils/                 # Utilities
│       ├── __init__.py
│       ├── config.py         # Configuration settings
│       └── logger.py         # Logging utilities
├── tests/                     # Test suite
│   ├── __init__.py
│   └── test_ocr.py           # OCR tests
├── resumes/                   # Resume PDFs (test data)
├── data/                      # Data storage
│   ├── logs/                 # Processing logs
│   └── processed/            # Processed outputs
└── planning/                  # Project planning docs
```

## Quick Start

### 1. Install Dependencies

```bash
# Create conda environment
conda create -n ats-resume python=3.10
conda activate ats-resume

# Install packages
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

For AI-powered suggestions, you need an OpenRouter API key:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenRouter API key
# Get your free API key at: https://openrouter.ai/keys
```

Or set environment variable:

```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY="your_api_key_here"

# Windows CMD
set OPENROUTER_API_KEY=your_api_key_here

# Linux/Mac
export OPENROUTER_API_KEY="your_api_key_here"
```

### 3. Run Streamlit App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 4. Run Pipeline Script

```bash
# Process a single resume
python pipeline.py resumes/CV\ \(1\).pdf

# Process all resumes in directory
python pipeline.py resumes/ --summary

# Save results to JSON
python pipeline.py resumes/CV\ \(1\).pdf -o output.json
```

## Features

### Text Extraction & Layout Detection ✓

- **PDF to Text**: Extract text using PyMuPDF (fast) or PaddleOCR (image-based PDFs)
- **Layout Analysis**: Detect multi-column layouts, tables, images
- **Section Parsing**: Identify resume sections (Experience, Education, Skills, etc.)

### ATS Scoring ✓

- **Layout Score** (25 pts): Single-column layout, no tables
- **Format Score** (25 pts): Appropriate length, no images
- **Content Score** (25 pts): Contact info, required sections
- **Structure Score** (25 pts): Clear headers, good text density

### Streamlit UI ✓

- File upload interface
- Score display with visual breakdown
- Issues and recommendations
- Raw text viewer (optional)

## ATS Score Breakdown

The ATS compatibility score (0-100) is calculated as:

| Component | Points | Criteria |
|-----------|--------|----------|
| Layout | 25 | Single-column (+25), Multi-column (-25), Tables (-20) |
| Format | 25 | Good length (+), No images (+), Too short/long (-) |
| Content | 25 | Contact info (+), Experience section (+), Education section (+), Skills (+) |
| Structure | 25 | Clear headers (+), Good text density (+), Standard naming (+) |

### Risk Levels

- **Low (80-100)**: ATS-friendly, minor improvements possible
- **Medium (60-79)**: Some ATS issues detected
- **High (0-59)**: Significant ATS compatibility problems

## API Usage

```python
from src.parsers import extract_text_from_resume, LayoutDetector, SectionParser
from src.scoring import ATSScorer

# Extract text
result = extract_text_from_resume("resume.pdf")
text = result["full_text"]

# Analyze layout
layout = LayoutDetector()
layout_summary = layout.get_layout_summary(text)

# Parse sections
parser = SectionParser()
parsed = parser.parse(text)

# Calculate score
scorer = ATSScorer()
score = scorer.calculate_score(
    text, 
    layout_summary, 
    {
        "contact_info": parsed.contact_info,
        "sections": parsed.sections,
        "skills": parsed.skills
    }
)

print(f"ATS Score: {score.overall_score}/100")
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

## Known Limitations (Phase 1)

1. **Layout Detection**: Basic heuristic-based, may miss complex layouts
2. **Section Parsing**: Rule-based, may not handle creative formatting
3. **Language Support**: Optimized for English and French
4. **Image Analysis**: Cannot detect text in images/graphics
5. **Two-Column Layouts**: Detection is basic, may have false positives

## Next Steps (Phase 2)

See `planning/phase2_enhanced.md` for:
- LLM-powered section extraction
- Keyword matching
- Improved layout detection with computer vision
- Better multi-language support

## License

Internal project for educational purposes.
