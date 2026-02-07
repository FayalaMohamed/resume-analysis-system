# Project Planning Index

Welcome to the ATS Resume Analyzer planning folder! This is your roadmap for the entire project.

## 📋 Planning Documents

| Phase | Document | Goal |
|-------|----------|------|
| **Phase 1** | [phase1_mvp.md](./phase1_mvp.md) | Core functionality - OCR, layout detection, basic ATS scoring |
| **Phase 2** | [phase2_enhanced.md](./phase2_enhanced.md) | Content analysis, job matching, recommendations |
| **Phase 3** | [phase3.md](./phase3.md) | Advanced ML & analysis - OCR excellence, scoring refinement, skills taxonomy |
| **Phase 4** | [phase4.md](./phase4.md) | Production-ready architecture for micro SaaS deployment |
| **Optional** | [optional.md](./optional.md) | Advanced learning topics - fine-tuning, APIs, visualizations |

---

## 🚀 Getting Started Checklist

### One-Time Setup Commands (Using Conda)

```bash
# Create conda environment with Python 3.10
conda create -n ats-resume python=3.10

# Activate environment
conda activate ats-resume

# Install Phase 1 dependencies
pip install paddleocr paddlepaddle-gpu pymupdf opencv-python streamlit pandas numpy

# Test PaddleOCR installation
python -c "from paddleocr import PaddleOCR; print('✓ PaddleOCR installed successfully')"

# (Optional) To deactivate when done:
# conda deactivate
```

**Note**: Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download) installed first.

---

## 📊 Project Overview

```
Phase 1: MVP
├── Environment & First Pipeline
├── Layout Detection
├── Basic ATS Scoring
└── Simple UI & Integration

Phase 2: Enhanced Analysis
├── Content Quality Analysis
├── Job Description Matching
├── Recommendation Engine
└── ATS Simulation & Testing

Phase 3: Advanced ML & Analysis Engine
├── OCR Reliability & Text Extraction
├── ATS Scoring Algorithm Refinement
├── Resume Content Understanding
├── Skills Extraction & Taxonomy
├── Job Matching System
├── Suggestion Engine Improvements
├── ML Model Diversification
└── Testing & Validation Framework

Phase 4: Production-Ready Architecture
├── Data Architecture & Persistence
├── Multi-Tenancy & User Management
├── Error Handling & Resilience
├── Performance & Scalability
├── Security Hardening
├── Observability & Analytics
├── Testing & Quality Assurance
└── Deployment Architecture

Optional: Advanced Learning
├── Fine-tuning Experiment
├── API & Export Features
├── Advanced Visualizations
└── Multi-Resume Comparison
```

### 🎯 Your Test Data

You already have **30 real resumes** in the `resumes/` folder from your former engineering school classmates - perfect for testing! These will give you realistic examples of:
- Different layouts and formatting styles
- Various ATS compatibility levels
- Actual content to analyze and improve

**Note**: Since these are real people's resumes, be mindful of privacy if you share your project publicly.

## 🎓 What You'll Learn

By completing this project, you'll understand:

**Technical Skills:**
- How OCR engines work (text detection & recognition)
- Document layout analysis concepts
- Building rule-based scoring systems
- Working with pre-trained ML models
- Local LLM integration (Ollama)
- API-based LLM integration (OpenRouter)
- Building resilient systems with fallbacks
- Building simple UIs with Streamlit
- Basic NLP and text analysis

**Conceptual Understanding:**
- How ATS systems parse resumes
- What makes resumes ATS-friendly vs. unfriendly
- Resume structure and best practices
- Semantic similarity and embeddings
- The full ML pipeline from data to deployment

---

## 💰 Costs

**Total Project Cost: $0**

All tools and models are free and open-source:
- PaddleOCR: Free
- PyMuPDF: Free
- Streamlit: Free
- Ollama: Free (local LLM)
- Ministral-3B: Free (local)
- OpenRouter: Free tier models available
- Python packages: Free

**Dual LLM Setup:**
- **Primary**: Ministral-3B via Ollama (runs on your GPU, always available)
- **Secondary**: OpenRouter API (free tier models, requires internet + API key)
- **Smart fallback**: Automatically switches if one fails


---

## 🤖 LLM Configuration

We support dual LLM providers for maximum flexibility:

### Setup Instructions

**1. Ollama (Local - Primary)**
```bash
# Install from https://ollama.com
# Then download model:
ollama pull ministral:3b-instruct-2512-q4_K_M
```

**2. OpenRouter (API - Secondary)**
```bash
# 1. Create free account at https://openrouter.ai
# 2. Get API key from dashboard
# 3. Set environment variable:
export OPENROUTER_API_KEY=your_key_here

# 4. Install Python client:
pip install openrouter
```

**3. Available Free Models**

Choose from these free models on OpenRouter:

| Model | Size | Best For |
|-------|------|----------|
| `mistralai/mistral-small-3.1-24b-instruct:free` | 24B | **Recommended** - Good balance of quality and speed |
| `meta-llama/llama-3.3-70b-instruct:free` | 70B | Higher quality, but slower |
| `openai/gpt-oss-20b:free` | 20B | OpenAI architecture, fast |
| `openai/gpt-oss-120b:free` | 120B | Largest model, most capable |
| `deepseek/deepseek-r1-0528:free` | - | Reasoning model, good for analysis |
| `z-ai/glm-4.5-air:free` | - | GLM architecture |

**Default**: `mistralai/mistral-small-3.1-24b-instruct:free` (set in the code)

### Example Code
See [`src/analysis/llm_client.py`](src/analysis/llm_client.py) for a complete implementation showing:
- Dual provider setup (Ollama + OpenRouter)
- Automatic fallback logic
- Error handling
- Provider health checks

---

## 📚 Additional Resources

### Documentation
- [PaddleOCR Docs](https://github.com/PaddlePaddle/PaddleOCR)
- [Streamlit Docs](https://docs.streamlit.io)
- [Ollama Docs](https://github.com/ollama/ollama)
- [OpenRouter Docs](https://openrouter.ai/docs)
- [OpenRouter Free Models](https://openrouter.ai/models?max_price=0)

### Learning Materials
- Sample resumes to test: Search "resume template pdf" or use your own
- ATS best practices: Search "ATS-friendly resume tips"

---

## 🎉 Let's Get Started!

Ready to begin? Head to **[Phase 1: MVP](./phase1_mvp.md)** and start with Week 1!

Remember: The goal is learning, not perfection. Have fun building! 🚀

---

**Last Updated**: 2026-02-03  
**Project Status**: Planning Phase
