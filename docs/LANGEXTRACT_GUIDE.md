# LangExtract Integration Guide

This document provides comprehensive information about Google's LangExtract library integration into the ATS Resume Analyzer.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Features](#features)
5. [Performance Comparison](#performance-comparison)
6. [Configuration](#configuration)
7. [Testing & Comparison](#testing--comparison)
8. [Troubleshooting](#troubleshooting)
9. [Data Structures](#data-structures)
10. [Future Enhancements](#future-enhancements)

---

## Overview

LangExtract has been integrated as a third extraction method alongside the existing:

- **Standard**: Rule-based parsing (fastest) - ~50ms
- **Unified**: PyMuPDF + intelligent structure detection (balanced) - ~100-200ms
- **LangExtract**: LLM-powered detailed extraction (highest quality, slower) - ~20-60s

### What is LangExtract?

LangExtract is Google's Python library for extracting structured information from unstructured text using LLMs with:
- Precise source grounding (maps extractions to exact text locations)
- Interactive visualization capabilities
- Optimized for long documents
- Support for multiple LLM providers (Gemini, OpenAI, Ollama)

### Files Created/Modified

**New Files:**
- `src/parsers/langextract_parser.py` - Main parser module
- `src/parsers/langextract_constants.py` - Prompts and examples
- `test_langextract_comparison.py` - Comparison testing script

**Modified Files:**
- `src/parsers/__init__.py` - Added exports
- `pipeline.py` - Added extraction step (Step 4C)
- `app.py` - Added extraction method selector
- `.env.example` - Added API key configuration

---

## Installation

### 1. Install Package

```bash
pip install langextract

# For OpenAI support (optional)
pip install langextract[openai]
```

### 2. Set API Key

Copy the example file and add your API key:

```bash
cp .env.example .env
```

Edit `.env` and add:
```
LANGEXTRACT_API_KEY=your-api-key-here
```

**Get your API key:** https://aistudio.google.com/app/apikey

### 3. Alternative: Environment Variable

```bash
export LANGEXTRACT_API_KEY="your-api-key-here"
```

---

## Usage

### In the Pipeline

LangExtract is automatically run as part of the full pipeline:

```bash
python pipeline.py resume.pdf
```

The pipeline will show results for all three extraction methods:
- Step 4: Section Parsing (Standard)
- Step 4B: Unified Extraction
- Step 4C: LangExtract Extraction

### In the Streamlit App

1. Upload a resume
2. Select extraction method from sidebar:
   - "Standard (Fast)" - Rule-based
   - "Unified (Enhanced Structure)" - PyMuPDF + intelligent parsing
   - "LangExtract (LLM-Powered)" - AI-powered detailed extraction

3. If LangExtract is selected, the app will:
   - Show a progress message
   - Extract detailed information using Gemini API
   - Display results with categorized skills and granular experience items

### As a Standalone Module

```python
from parsers.langextract_parser import LangExtractResumeParser

# Initialize parser
parser = LangExtractResumeParser(
    model_id="gemini-2.5-flash"  # or gemini-2.5-pro
)

# Extract from PDF
result = parser.extract_from_pdf(
    "resume.pdf",
    extraction_passes=1  # 1-3, higher = more thorough but slower
)

if result.success:
    # Access contact info
    print(f"Name: {result.contact.name}")
    print(f"Email: {result.contact.email}")
    
    # Access experience
    for exp in result.experience:
        print(f"{exp.job_title} at {exp.company}")
        print(f"  {exp.date_range}")
        for bp in exp.bullet_points:
            print(f"    • {bp['text']}")
    
    # Access skills by category
    for skill in result.skills:
        print(f"{skill.name} ({skill.category})")
```

---

## Features

### 1. Granular Experience Extraction
- Individual job positions with dates
- Bullet points with metrics detection (e.g., "40%", "1M+ users")
- Technologies used per role
- Employment type detection (Full-time, Part-time, Contract, Internship)

### 2. Categorized Skills
- Programming languages
- Frameworks/libraries
- Cloud platforms
- Databases
- DevOps tools
- Soft skills
- Languages with proficiency

### 3. Comprehensive Data Extraction
- **Contact Info**: Name, email, phone, LinkedIn, GitHub, website, location
- **Professional Summary**: Summary text and career objectives
- **Education**: Degrees, institutions, dates, GPA, coursework, honors
- **Certifications**: Name, provider, date obtained, expiration
- **Projects**: Names, descriptions, technologies, URLs
- **Additional**: Awards, publications, volunteer work

### 4. Source Grounding
Every extraction knows its exact position in the source text, useful for verification and highlighting.

---

## Performance Comparison

### Speed Comparison

| Method | Speed | Use Case |
|--------|-------|----------|
| Standard | ~50ms | Fast processing, bulk analysis |
| Unified | ~100-200ms | Balanced quality and speed |
| LangExtract | ~20-60s | Maximum detail, complex resumes |

**LangExtract is 220x slower** than the existing system due to API latency, but provides **6x more experience items** and **6.5x more skills**.

### Extraction Quality Comparison

| Metric | Existing System | LangExtract | Improvement |
|--------|----------------|-------------|-------------|
| Experience Items | 0 | **6** | +6x |
| Skills | 2 | **13** | +6.5x |
| Speed | 125ms | 27,533ms | 220x slower |

### Trade-offs

**Existing System:**
- ✅ 220x faster (local processing)
- ✅ No API costs or rate limits
- ✅ Works offline
- ✅ Good document structure detection
- ❌ Misses granular details

**LangExtract:**
- ✅ Extracts detailed experience items with metrics
- ✅ Categorizes skills by type (programming, cloud, database, etc.)
- ✅ Source grounding (knows exactly where info came from)
- ✅ Handles complex bullet points with achievements
- ❌ 20x slower due to API latency
- ❌ Rate limited (20 requests/day free)
- ❌ Costs money per API call

### When to Use Each System

**Use Existing System when:**
- You need **fast processing** (< 200ms)
- Processing **many resumes** (no rate limits)
- **Cost-sensitive** (no API fees)
- Need **document structure** (sections, layout)

**Use Unified Extraction when:**
- Need good structure detection
- Handle multi-column layouts
- Balanced approach for most standard resumes

**Use LangExtract when:**
- You need **maximum detail** (bullet points, metrics)
- **Quality > Speed** is acceptable
- Extracting **technical skills** (categorization)
- Need **source verification** (grounding)
- Processing **complex/non-standard** resumes

### Hybrid Approach (Recommended)

```python
# Tier 1: Fast filtering with existing system
if existing_system.extract(resume).is_valid():
    # Tier 2: Detailed extraction with LangExtract for complex cases
    if is_complex_resume(resume):
        detailed_data = langextract.extract(resume)
```

---

## Configuration

### Extraction Passes
- **1 pass** (default): Fast, good for most resumes
- **2 passes**: Better recall, catches more entities
- **3 passes**: Maximum coverage, slowest

```python
parser.extract_from_pdf("resume.pdf", extraction_passes=2)
```

### Buffer Size
- **2000-4000 chars**: Balance between accuracy and speed
- Smaller = better accuracy, more API calls
- Larger = faster, may miss details

### Workers
- **2-5 workers**: Parallel processing
- Limited by API rate limits
- Higher = faster for long documents

### Default Configuration
```python
DEFAULT_CONFIG = {
    'model_id': 'gemini-2.5-flash',
    'extraction_passes': 1,
    'max_workers': 2,
    'max_char_buffer': 4000,
}
```

### API Limits (Free Tier)
- **20 requests per day** for Gemini API
- Rate limit exceeded error: `429 RESOURCE_EXHAUSTED`

### Workarounds for Rate Limits
1. **Upgrade API tier**: Get higher limits from Google AI Studio
2. **Use local models** via Ollama (no rate limits):
   ```bash
   ollama pull gemma2:2b
   python test_langextract_comparison.py --model gemma2:2b
   ```
3. **Wait 24 hours**: Free tier resets daily
4. **Use Vertex AI**: For production with batch processing

### Cost Considerations
- **Free tier**: 20 requests/day (testing only)
- **Paid tier**: ~$0.50-2.00 per 1000 requests (Gemini Flash)
- **For 1000 resumes**: ~$0.50-2.00 with LangExtract vs $0 with existing system

---

## Testing & Comparison

### Running the Comparison Script

```bash
# Test on a specific resume
python test_langextract_comparison.py --resume resumes/CV_1.pdf

# Test on multiple resumes (default: 5)
python test_langextract_comparison.py --resume-dir resumes --limit 10

# Use a different model
python test_langextract_comparison.py --model gemini-2.5-pro

# Test only existing system
python test_langextract_comparison.py --skip-langextract

# Test only LangExtract
python test_langextract_comparison.py --skip-existing
```

### What Gets Compared

**Performance Metrics:**
- **Success Rate**: Percentage of successful extractions
- **Speed**: Time taken for extraction (ms)
- **API Calls**: Number of LLM calls (LangExtract only)

**Extraction Quality:**
- **Contact Info**: Name, email, phone detection
- **Sections**: Number and types of sections found
- **Experience Items**: Jobs/positions extracted
- **Education Items**: Degrees/schools extracted
- **Skills**: Technical skills and tools identified

**Overlap Analysis:**
- **Section Overlap**: Jaccard similarity of section names
- **Skills Overlap**: Jaccard similarity of extracted skills
- Shows where systems agree/disagree

### Example Output

```
================================================================================
COMPARISON SUMMARY
================================================================================

Total Resumes Tested: 5

Existing System:
   Success Rate: 100.0%
   Avg Duration: 245ms

LangExtract:
   Success Rate: 100.0%
   Avg Duration: 3200ms

Overlap Analysis:
   Avg Section Overlap: 85.3%
   Avg Skills Overlap: 62.1%

Recommendations:
   - LangExtract is significantly slower (API latency)
   - Low skills overlap suggests different extraction approaches
   - Consider LangExtract for complex documents only
```

### Output Files

After running the comparison, you'll find in `langextract_comparison_results/`:

```
langextract_comparison_results/
├── comparison_report.json       # Complete comparison data
├── langextract_viz_1.html       # Visualization for resume 1
├── langextract_viz_2.html       # Visualization for resume 2
└── ...
```

---

## Troubleshooting

### "LangExtract not available"
- Install: `pip install langextract`
- Check import: `from parsers.langextract_parser import LangExtractResumeParser`

### "API key not configured"
- Set `LANGEXTRACT_API_KEY` in `.env` file
- Or set environment variable: `export LANGEXTRACT_API_KEY=...`

### "429 RESOURCE_EXHAUSTED"
- You've hit the free tier limit (20 requests/day)
- Wait 24 hours or upgrade API tier
- Use local Ollama models instead

### "No resume PDFs found"
Place PDFs in the `resumes/` directory or specify a path:
```bash
python test_langextract_comparison.py --resume /path/to/resume.pdf
```

### Slow extraction
- Reduce `extraction_passes` to 1
- Increase `max_char_buffer` for fewer chunks
- Use `gemini-2.5-flash` instead of `gemini-2.5-pro`

### High API costs
- Use `gemini-2.5-flash` (faster, cheaper) instead of `gemini-2.5-pro`
- Test on fewer resumes first
- Use local models via Ollama

---

## Data Structures

### LangExtractResult

```python
{
    'success': bool,
    'contact': {
        'name': str,
        'email': str,
        'phone': str,
        'linkedin': str,
        'github': str,
        'website': str,
        'location': str
    },
    'summary': str,
    'objective': str,
    'experience': [{
        'job_title': str,
        'company': str,
        'date_range': str,
        'location': str,
        'employment_type': str,
        'bullet_points': [{
            'text': str,
            'has_metric': bool,
            'metric': str
        }],
        'technologies': [str]
    }],
    'education': [{
        'degree': str,
        'field': str,
        'institution': str,
        'date_range': str,
        'gpa': str,
        'coursework': [str]
    }],
    'skills': [{
        'name': str,
        'category': str,
        'parent_category': str
    }],
    'certifications': [{
        'name': str,
        'provider': str,
        'date_obtained': str,
        'expiration_date': str,
        'level': str
    }],
    'projects': [{
        'name': str,
        'description': str,
        'technologies': [str],
        'url': str
    }],
    'languages': [{
        'language': str,
        'proficiency': str
    }],
    'awards': [...],
    'publications': [...],
    'volunteer': [...]
}
```

---

## Integration Architecture

```
Resume PDF
    ↓
┌─────────────────────────────────────────┐
│  Text Extraction (PyMuPDF/OCR)          │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Layout Analysis                        │
└─────────────────────────────────────────┘
    ↓
    ├─→ Standard Parsing (SectionParser)   ─┐
    ├─→ Unified Extraction (PyMuPDF)       ─┼→ Merged Results
    └─→ LangExtract (LLM-Powered)          ─┘         ↓
                                         ATS Scoring
```

---

## Future Enhancements

When building the API:
1. **Async processing** for LangExtract to avoid blocking
2. **Caching** of extraction results
3. **Hybrid scoring** combining all three methods
4. **Confidence scores** per extraction
5. **Batch processing** with Vertex AI
6. **Fallback chain**: Standard → Unified → LangExtract

---

## Further Reading

**LangExtract Resources:**
- GitHub: https://github.com/google/langextract
- Docs: https://pypi.org/project/langextract/
- API Key: https://aistudio.google.com/app/apikey

**Internal Resources:**
- Parser: `src/parsers/langextract_parser.py`
- Constants: `src/parsers/langextract_constants.py`
- Examples: `test_langextract_comparison.py`
- Results: `langextract_comparison_results/`

---

## Example Output

```json
{
  "success": true,
  "contact": {
    "name": "John Smith",
    "email": "john@example.com",
    "phone": "(555) 123-4567",
    "linkedin": "linkedin.com/in/johnsmith"
  },
  "experience": [
    {
      "job_title": "Software Engineer",
      "company": "Google",
      "date_range": "January 2020 - Present",
      "bullet_points": [
        {
          "text": "Led development serving 1M+ users",
          "has_metric": true,
          "metric": "1M+ users"
        }
      ]
    }
  ],
  "skills": [
    {"name": "Python", "category": "programming_language"},
    {"name": "AWS", "category": "cloud_platform"}
  ]
}
```

---

*Last Updated: 2026-02-11*
