# Phase 2: Enhanced Analysis

**Goal**: Add content analysis, job matching, and improvement suggestions

---

## Content Quality Analysis

### Objectives
- [ ] Analyze resume content quality (not just structure)
- [ ] Detect action verbs, quantification, bullet points
- [ ] Provide content-specific feedback

### Tasks

#### Action Verb Detection
- [ ] Create list of strong action verbs (led, developed, implemented, etc.)
- [ ] Create list of weak verbs (helped, worked on, assisted)
- [ ] Implement verb detection in Experience section
- [ ] Score based on action verb usage

#### Quantification Detection
- [ ] Build regex patterns for numbers/metrics:
  - Percentages (%, percent)
  - Dollar amounts ($, k, million)
  - Time periods (months, years)
  - Counts (users, customers, projects)
- [ ] Detect quantified achievements
- [ ] Score based on quantification density

#### Bullet Point Analysis
- [ ] Detect bullet points vs paragraphs
- [ ] Analyze bullet length (ideal: 1-2 lines)
- [ ] Check for consistency
- [ ] Score readability

#### Content Quality Score
- [ ] Combine all content metrics into score
- [ ] Weight factors:
  - Action verbs: 30%
  - Quantification: 30%
  - Bullet structure: 20%
  - Conciseness: 20%

### Deliverables
- [ ] Content quality analyzer
- [ ] Action verb detection working
- [ ] Quantification detection working
- [ ] Integrated content score

---

## Job Description Matching

### Objectives
- [ ] Extract keywords from job descriptions
- [ ] Match resume content to job requirements
- [ ] Calculate match score

### Tasks

#### Keyword Extraction
- [ ] Implement TF-IDF for keyword extraction from JD
- [ ] Extract skill keywords (using spaCy or patterns)
- [ ] Identify required vs preferred qualifications
- [ ] Build keyword importance ranking

#### Embedding Similarity (Optional but Recommended)
- [ ] Install sentence-transformers
- [ ] Load `all-MiniLM-L6-v2` model (small, fast)
- [ ] Encode resume sections and JD
- [ ] Calculate semantic similarity
- [ ] Understand how embeddings work

#### Skill Gap Analysis
- [ ] Extract skills from resume
- [ ] Compare to JD requirements
- [ ] Identify missing skills
- [ ] Calculate skill match percentage

#### Match Score Integration
- [ ] Combine keyword match + semantic similarity + skill match
- [ ] Create overall match score (0-100)
- [ ] Generate gap analysis report

### Deliverables
- [ ] Job description parser
- [ ] Keyword extraction working
- [ ] Semantic similarity calculation
- [ ] Skill gap identification
- [ ] Overall match score

---

## Recommendation Engine

### Objectives
- [ ] Generate specific improvement suggestions
- [ ] Create before/after comparisons
- [ ] Prioritize recommendations

### Tasks

#### Rule-Based Recommendations
- [ ] Map detected issues to specific suggestions:
  - Multi-column layout → "Convert to single column"
  - Tables detected → "Remove tables, use simple lists"
  - Missing action verbs → "Start bullets with strong verbs"
  - No quantification → "Add metrics to achievements"
  - Dense paragraphs → "Convert to bullet points"
- [ ] Create recommendation templates
- [ ] Implement recommendation generator

#### LLM Integration (Dual Setup)

**Part A: Local LLM with Ollama**
- [ ] Set up Ollama with Ministral-3B-Q4_K_M
- [ ] Test basic prompt completion locally
- [ ] Verify GPU utilization with small test

**Part B: OpenRouter API (Free Tier)**
- [ ] Create OpenRouter account (free)
- [ ] Get API key from OpenRouter dashboard
- [ ] Install OpenRouter Python client: `pip install openrouter`
- [ ] Configure API client with your key
- [ ] Test with free model (choose one):
  - `mistralai/mistral-small-3.1-24b-instruct:free` (recommended - 24B, fast)
  - `meta-llama/llama-3.3-70b-instruct:free` (70B, higher quality)
  - `openai/gpt-oss-20b:free` (20B, OpenAI model)
  - `openai/gpt-oss-120b:free` (120B, largest)
  - `deepseek/deepseek-r1-0528:free` (reasoning model)
  - `z-ai/glm-4.5-air:free` (GLM model)
- [ ] Create fallback logic: Try OpenRouter first, fall back to Ollama

**Part C: Unified LLM Interface**
- [ ] Create `LLMClient` class that abstracts both options:
  ```python
  class LLMClient:
      def __init__(self, primary='openrouter', fallback='ollama'):
          self.primary = primary
          self.fallback = fallback
      
      def generate(self, prompt):
          # Try primary first, fallback on failure
          pass
  ```
- [ ] Create prompt templates for content improvement:
  - Rewrite bullet points
  - Improve weak sections
  - Suggest missing keywords
- [ ] Test both providers with same prompts
- [ ] Compare quality and speed

#### Prioritization & UI
- [ ] Score recommendations by impact (high/medium/low)
- [ ] Show estimated score improvement
- [ ] Add recommendations section to UI
- [ ] Display before/after comparison

### Deliverables
- [ ] Rule-based recommendation engine
- [ ] Local LLM integration (Ministral-3B via Ollama)
- [ ] OpenRouter API integration (free tier)
- [ ] Unified LLM client with fallback logic
- [ ] Prioritized improvement suggestions
- [ ] Updated UI with recommendations

---

## ATS Simulation & Testing

### Objectives
- [ ] Simulate how ATS systems see the resume
- [ ] Identify what information might be lost
- [ ] Generate ATS-friendly version

### Tasks

#### Plain-Text Extraction Simulation
- [ ] Build simple ATS parser simulation:
  - Strip all formatting
  - Extract text in reading order
  - Remove graphics/images
- [ ] Compare structured extraction vs plain text
- [ ] Identify "lost" information

#### Lost Information Detection
- [ ] Compare original to ATS-parsed version
- [ ] Highlight missing sections
- [ ] Flag content in tables/columns that may be skipped
- [ ] Generate "what ATS sees" view

#### Testing & Validation
- [ ] Test complete pipeline on 20 resumes
- [ ] Measure accuracy against manual review
- [ ] Document failure modes
- [ ] Refine scoring weights based on results

### Deliverables
- [ ] ATS simulation view
- [ ] Lost information detection
- [ ] Tested on 20 resumes
- [ ] Refined scoring algorithm

---

## Phase 2 Success Criteria

- [ ] ✅ Content quality analysis (action verbs, quantification)
- [ ] ✅ Job description matching with semantic similarity
- [ ] ✅ Specific improvement recommendations
- [ ] ✅ Local LLM generating content suggestions
- [ ] ✅ ATS simulation showing what parsers see
- [ ] ✅ Tested and validated on 20+ resumes

---

## New Dependencies

```bash
# Content Quality Analysis
pip install spacy
python -m spacy download en_core_web_sm

# Job Description Matching
pip install sentence-transformers scikit-learn

# LLM Integration - Local
# Install Ollama from https://ollama.com
# Then run: ollama pull ministral:3b-instruct-2512-q4_K_M

# LLM Integration - OpenRouter API
pip install openrouter  # or requests if openrouter package not available
# Set OPENROUTER_API_KEY environment variable
# Available free models: mistralai/mistral-small-3.1-24b-instruct:free, meta-llama/llama-3.3-70b-instruct:free, openai/gpt-oss-20b:free, openai/gpt-oss-120b:free, deepseek/deepseek-r1-0528:free, z-ai/glm-4.5-air:free
```

---

## Learning Goals for Phase 2

By end of Phase 2, understand:
- [ ] How to analyze text quality (NLP basics)
- [ ] What makes resume content effective
- [ ] How semantic similarity/embedding models work
- [ ] How to integrate local LLMs (Ollama)
- [ ] How to work with API-based LLMs (OpenRouter)
- [ ] How to build fallback systems (resilient architecture)
- [ ] How ATS systems actually parse resumes
- [ ] How to build recommendation systems

---

## Integration Points

Phase 2 builds on Phase 1:
- Uses same OCR and layout detection
- Extends ATS scoring with content analysis
- Adds JD matching as optional feature
- Enhances UI with recommendations

---

## Notes & Decisions Log

**Content Quality Analysis:**
- 

**Job Description Matching:**
- 

**Recommendation Engine:**
- 

**ATS Simulation & Testing:**
- 

---

**Status**: Not Started  
**Prerequisites**: Phase 1 Complete  
**Started Date**:  
**Completed Date**:  
