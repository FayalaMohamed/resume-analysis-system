# ATS-Aware Resume Analysis System: Learning Project Plan

> **🎓 Learning Project Notice**: This is a **personal educational project**, not a production system. The goal is to learn how document processing, OCR, and ATS analysis work by building a working prototype using only free, open-source tools.

## Project Philosophy

### Why This Approach?
- **Zero Cost**: No subscriptions, no API bills, no cloud services
- **Learning Focus**: Understand every component deeply rather than optimizing for scale
- **Freedom to Experiment**: Break things, try different approaches, learn from mistakes
- **Build at Your Pace**: No deadlines, no stakeholder pressure
- **Portfolio Piece**: Demonstrate understanding of NLP, CV, and document processing

### What We're Avoiding (to Keep Costs $0)

| Expensive Approach | Our Free Alternative | Why It Works |
|-------------------|---------------------|--------------|
| OpenAI/Anthropic APIs | Ollama + Ministral-3B locally | Learn LLM integration basics |
| Cloud GPUs | Your laptop's CPU | Slower but educational |
| AWS/GCP/Azure | Local filesystem | No infra complexity |
| PostgreSQL + Redis | SQLite + Python dict | Simpler for single-user |
| Docker/K8s | Direct Python install | Easier to debug and learn |
| Commercial ATS APIs | Rule-based simulation | Understand how ATS works |

## Executive Summary

**Project Goal**: Build a working ATS-Aware Resume Analysis System to learn:
- How OCR engines extract text from documents
- How to analyze document layouts
- How ATS systems parse and score resumes
- How to build simple ML/NLP pipelines

**Key Design Decision**: Use pre-trained open-source models with minimal infrastructure - SQLite for storage, local CPU inference, and zero paid services.

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │   PDF Upload   │  │ Image Upload │  │   Job Desc   │                  │
│  │   (PyMuPDF)    │  │  (PNG/JPEG)  │  │    (Text)    │                  │
│  └──────┬─────────┘  └──────┬─────────┘  └──────┬─────────┘                  │
└─────────┼──────────────────┼──────────────────┼─────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     DOCUMENT PROCESSING PIPELINE                         │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Stage 1: Layout Analysis & OCR (Parallel Processing)          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │  PaddleOCR   │  │  LayoutLMv3  │  │  PDF Plumber │          │   │
│  │  │  (ppstructure)│  │ (Heuristic)  │  │  (Fallback)  │          │   │
│  │  └──────┬─────────┘  └──────┬─────────┘  └──────┬─────────┘          │   │
│  │         │                  │                  │                    │   │
│  │         └──────────────────┼──────────────────┘                    │   │
│  │                            ▼                                      │   │
│  │              ┌─────────────────────────────┐                      │   │
│  │              │   Layout Detection Output   │                      │   │
│  │              │ • Bounding boxes            │                      │   │
│  │              │ • Section classifications     │                      │   │
│  │              │ • Text regions              │                      │   │
│  │              │ • Reading order             │                      │   │
│  │              └──────────────┬──────────────┘                      │   │
│  └─────────────────────────────┼───────────────────────────────────┘   │
│                                │                                       │
│  ┌─────────────────────────────┼───────────────────────────────────┐   │
│  │ Stage 2: Text Extraction & Structure                            │   │
│  │                            ▼                                    │   │
│  │              ┌─────────────────────────────┐                    │   │
│  │              │    Structured Resume JSON   │                    │   │
│  │              │ • Contact Info                │                    │   │
│  │              │ • Experience (role, company)  │                    │   │
│  │              │ • Education                 │                    │   │
│  │              │ • Skills                    │                    │   │
│  │              │ • Projects                  │                    │   │
│  │              └──────────────┬──────────────┘                    │   │
│  └─────────────────────────────┼───────────────────────────────────┘   │
└────────────────────────────────┼────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     ATS ANALYSIS ENGINE                                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Sub-Engine 1: Structural Compatibility Scoring                 │   │
│  │  • Multi-column detection (heuristic + vision)                  │   │
│  │  • Table detection & risk assessment                            │   │
│  │  • Icon/graphic detection (computer vision)                   │   │
│  │  • Header validation against standard ATS keywords              │   │
│  │  • Reading order validation                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Sub-Engine 2: Content Quality Scoring                            │   │
│  │  • Bullet point analysis (length, action verbs)                 │   │
│  │  • Quantification detection (regex + LLM)                       │   │
│  │  • Keyword density analysis                                     │   │
│  │  • Date consistency checking                                    │   │
│  │  • Contact info validation                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Sub-Engine 3: Job Match Analysis (Optional JD input)           │   │
│  │  • TF-IDF keyword extraction from JD                            │   │
│  │  • Embedding similarity (sentence-transformers)                 │   │
│  │  • Skill gap analysis                                           │   │
│  │  • Keyword stuffing detection                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Sub-Engine 4: ATS Simulation                                     │   │
│  │  • Plain-text extraction simulation                             │   │
│  │  • Lost information identification                              │   │
│  │  • Parser-friendly version generation                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  IMPROVEMENT RECOMMENDATION ENGINE                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ • Structural fix suggestions (vision-driven)                   │   │
│  │ • Content rewrite suggestions (LLM-powered)                    │   │
│  │ • Before/after ATS score estimation                            │   │
│  │ • Prioritized action items                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      VISUALIZATION & UI LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  Streamlit   │  │  Layout      │  │   Score      │                  │
│  │   Web UI     │  │  Overlays    │  │ Dashboard    │                  │
│  │              │  │  (SVG/Canvas)│  │              │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Strategy

### 2.1 Data Sources

#### A. Public Datasets (No Training Required)
| Dataset | Purpose | Size | Usage |
|---------|---------|------|-------|
| **PubLayNet** | Layout detection training | 358k documents | Pre-trained models available |
| **DocBank** | Document structure | 500k pages | Fine-tuning optional |
| **IIT-CDIP** | Document images | 11M images | LayoutLM pre-training |
| **FunSD** | Form understanding | 199 samples | Few-shot examples |
| **CORD** | Receipt parsing | 1k receipts | Layout patterns |

#### B. Resume-Specific Data
| Source | Type | Strategy |
|--------|------|----------|
| **Synthetic Generation** | Controlled layouts | Generate 10k+ synthetic resumes with variations |
| **Kaggle Resume Dataset** | Real resumes | Use for validation/testing only |
| **Job Description Dataset** | CareerBuilder data | Keyword extraction validation |
| **LinkedIn Resume Templates** | Template variations | Augmentation patterns |

#### C. ATS Behavior Data
| Source | Purpose |
|--------|---------|
| **OpenSource ATS** (e.g., OpenCATS) | Ground truth parsing behavior |
| **Resume Parsing APIs** | Validation against commercial parsers |
| **Manual Annotation** | 500 resumes for quality assurance |

### 2.2 Synthetic Data Generation Strategy

```python
# Synthetic Resume Generator Architecture
class SyntheticResumeGenerator:
    """Generates controlled resume variations for testing"""
    
    def __init__(self):
        self.layout_templates = [
            'single_column_standard',
            'two_column_modern', 
            'table_based_skills',
            'graphic_heavy_creative',
            'minimalist_text_only'
        ]
        
    def generate_resume(self, layout_type, ats_risk_level):
        """
        Generate resume with controlled ATS compatibility issues
        
        Risk Levels:
        - LOW: Single column, standard fonts, no graphics
        - MEDIUM: Two-column, some icons, table-based sections
        - HIGH: Multi-column, heavy graphics, custom fonts, tables
        """
        pass
```

**Synthetic Data Parameters:**
- **Layout Types**: 20+ templates covering common resume designs
- **ATS Risk Profiles**: Controlled introduction of ATS-unfriendly elements
- **Content Variations**: Different industries, experience levels, skill densities
- **Image Quality**: Simulated scan quality, PDF generation artifacts

### 2.3 Data Augmentation for OCR Robustness

| Augmentation | Purpose | Implementation |
|--------------|---------|---------------|
| **Rotation** | Scan alignment issues | ±5° rotation |
| **Skew** | Document perspective | 0-3° skew |
| **Brightness/Contrast** | Lighting variations | Random adjustments |
| **Compression Artifacts** | PDF quality | JPEG compression levels |
| **Noise** | Scan quality | Gaussian noise |
| **Resolution Scaling** | Image quality | 150-300 DPI simulation |

---

## 3. Model Selection & Architecture

### 3.1 OCR Engine: PaddleOCR (Recommended)

**Why PaddleOCR:**
- ✅ **Superior accuracy**: Industry-leading text recognition
- ✅ **Document structure**: Built-in `ppstructure` for layout analysis
- ✅ **Multilingual**: 100+ languages support
- ✅ **Production-ready**: Optimized inference, CPU/GPU support
- ✅ **Active development**: Regular updates, community support
- ✅ **Resume-specific**: Already used in document parsing pipelines

**Architecture Components:**
```
PaddleOCR Pipeline:
├── Text Detection: DB (Differentiable Binarization)
│   └── Detects text regions with bounding boxes
├── Text Recognition: SVTR (Scene Text Recognition)
│   └── Converts text regions to strings
├── Layout Analysis: PP-Structure
│   ├── Table Recognition
│   ├── Key-Value Extraction  
│   └── Document Classification
└── Output: Structured JSON with bounding boxes
```

**Alternative: Tesseract 5 + LayoutLM**
- Use if: Need 100% open-source, smaller footprint
- Trade-off: Lower accuracy on complex layouts

### 3.2 Layout Analysis: Hybrid Approach

**Primary: PaddleOCR PP-Structure**
- Pre-trained on diverse document types
- Detects: tables, headers, lists, figures
- Resume-specific fine-tuning: Optional (few-shot)

**Secondary: LayoutLMv3 (Microsoft)**
- Use for: Document understanding, key-value extraction
- Fine-tuning: Not required for MVP
- Role: Semantic understanding of resume sections

**Fallback: Rule-based Heuristics**
- Y-coordinate analysis for column detection
- Font size-based header detection
- Whitespace analysis for section separation

### 3.3 ATS Scoring Models

#### A. Structural Compatibility (Rule-Based + Vision)

```python
class StructuralATSAnalyzer:
    """No ML training required - uses heuristics + pre-trained vision"""
    
    def analyze(self, layout_data):
        scores = {
            'column_score': self._detect_columns(),
            'table_risk': self._detect_tables(),
            'graphic_risk': self._detect_graphics(),
            'header_standard': self._validate_headers(),
            'reading_order': self._validate_reading_order()
        }
        return self._aggregate_score(scores)
```

**Scoring Weights:**
| Factor | Weight | Detection Method |
|--------|--------|----------------|
| Multi-column layout | 25% | Vision (PP-Structure) + Heuristics |
| Table usage | 20% | Vision + Manual rules |
| Graphics/Icons | 15% | Object detection (YOLOv8) |
| Header standardization | 20% | NLP keyword matching |
| Reading order | 20% | Geometry analysis |

#### B. Content Quality (LLM-Enhanced Rule-Based)

**Components:**
1. **Bullet Analysis**: spaCy + regex patterns
2. **Action Verb Detection**: Predefined verb lists + embeddings
3. **Quantification**: Regex patterns + LLM validation
4. **Keyword Density**: TF-IDF + readability metrics

**LLM Usage (Local Only):**
- **Model**: Ministral-3B via Ollama (Q4_K_M quantized)
- **Purpose**: Content quality assessment, rewrite suggestions
- **Cost**: $0 - runs on your GPU efficiently
- **Why this model**: Smaller, faster, fits your GPU while still capable

#### C. Job Match Analysis (Embedding-Based)

```python
from sentence_transformers import SentenceTransformer

class JobMatchAnalyzer:
    def __init__(self):
        # Pre-trained model, no training required
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def calculate_match(self, resume_text, job_description):
        resume_emb = self.model.encode(resume_text)
        job_emb = self.model.encode(job_description)
        similarity = cosine_similarity(resume_emb, job_emb)
        return similarity
```

### 3.4 Model Training Requirements Summary

| Component | Training Required | Approach | Effort |
|-----------|------------------|----------|--------|
| OCR | ❌ No | Pre-trained PaddleOCR | Zero |
| Layout Detection | ❌ No | PP-Structure | Zero |
| Table Detection | ⚠️ Optional | Fine-tune LayoutLM on 500 samples | Low |
| ATS Scoring | ❌ No | Rule-based + heuristics | Zero |
| Content Quality | ❌ No | Rule-based + LLM | Zero |
| Job Matching | ❌ No | Pre-trained embeddings | Zero |

**Total Training Burden**: Minimal to None

---

## 4. Pipeline Orchestration

### 4.1 Processing Pipeline Architecture

```python
# Simple Python Class - No orchestration framework needed

class ResumeAnalysisPipeline:
    """Simple end-to-end pipeline for learning"""
    
    def execute(self, resume_file, job_description=None):
        # Stage 1: Document Ingestion
        doc_data = self.ingest(resume_file)
        
        # Stage 2: Sequential Processing (simpler for learning)
        ocr_result = self.run_ocr(doc_data)
        layout_result = self.analyze_layout(doc_data)
        
        # Stage 3: Structure Extraction
        structured_resume = self.extract_structure(ocr_result, layout_result)
        
        # Stage 4: ATS Analysis
        ats_scores = self.run_ats_analysis(structured_resume, layout_result)
        
        # Stage 5: Job Matching (if JD provided)
        if job_description:
            match_scores = self.analyze_job_match(structured_resume, job_description)
            
        # Stage 6: Recommendation Generation
        recommendations = self.generate_recommendations(ats_scores, structured_resume)
        
        # Stage 7: Report Generation
        report = self.generate_report(
            ats_scores, recommendations, 
            layout_result, structured_resume
        )
        
        return report
```

**No orchestration frameworks needed** (no Airflow, Prefect, Celery) - just simple Python code that's easier to understand and debug.

### 4.2 Simple Caching (Optional for Learning)

Since this is a learning project with low volume, caching is optional:

| Cache Type | Purpose | Implementation |
|------------|---------|----------------|
| **Simple File Cache** | Avoid re-processing same resume | Save JSON results to disk |
| **In-Memory** | Speed up repeated analyses | Python dict during session |

**No Redis, no databases** - keep it simple!

### 4.3 No Queue Needed

For a learning project processing one resume at a time, no queue architecture needed. Just simple synchronous processing:

```
User Uploads ──▶ Process Resume ──▶ Display Results
     (Streamlit)        (Python)         (Streamlit)
```

---

## 5. Implementation Phases

### Phase 1: MVP (Weeks 1-4)

**Goal**: Core functionality - Upload, OCR, Layout Detection, Basic ATS Score

**Components:**
1. **Document Upload & Preprocessing**
   - PDF to image conversion (pdf2image)
   - Image preprocessing (deskew, enhance)
   
2. **OCR Pipeline**
   - PaddleOCR integration
   - Text extraction with bounding boxes
   
3. **Layout Detection**
   - PP-Structure integration
   - Section identification (Experience, Education, Skills)
   
4. **Structural ATS Analysis**
   - Multi-column detection
   - Table detection
   - Basic scoring algorithm
   
5. **Basic UI**
   - Streamlit upload interface
   - Score display
   - Text extraction view

**Deliverables:**
- Working prototype analyzing single-column resumes
- ATS compatibility score (0-100)
- Layout issue detection
- Plain text extraction view

### Phase 2: Enhanced Analysis (Weeks 5-8)

**Goal**: Content analysis, job matching, improvement suggestions

**Components:**
1. **Content Quality Analysis**
   - Bullet point evaluation
   - Action verb detection
   - Quantification checking
   
2. **Job Description Matching**
   - Keyword extraction (TF-IDF)
   - Embedding similarity
   - Skill gap analysis
   
3. **ATS Simulation**
   - Plain-text rendering
   - Lost information highlighting
   
4. **Recommendation Engine**
   - Rule-based suggestions
   - Score improvement estimation
   
5. **Enhanced UI**
   - Layout overlay visualization
   - Score breakdown dashboard
   - Before/after comparison

**Deliverables:**
- Full ATS analysis with content scoring
- Job description matching
- Concrete improvement suggestions
- Visual layout risk overlay

### Phase 3: Advanced Learning (Weeks 9-12) - Optional

**Goal**: Explore advanced topics based on interest

**Possible Extensions:**
1. **Experiment with Fine-tuning**
   - Try fine-tuning LayoutLM on a small resume dataset (if curious)
   - Understand transfer learning concepts
   
2. **Build Simple API**
   - Create a basic FastAPI endpoint (for learning web frameworks)
   - Understand how web services work
   
3. **Add Export Features**
   - Generate ATS-friendly PDF version
   - Learn about PDF generation

**Deliverables:**
- Deeper understanding of ML/NLP concepts
- Optional: Working API or export functionality
- Solid foundation for future projects

**Note**: This phase is flexible - focus on what interests you most!

---

## 6. Technology Stack (Zero-Cost, Local-Only)

### Core Dependencies

| Category | Technology | Purpose | Cost |
|----------|-----------|---------|------|
| **OCR** | PaddleOCR | Text extraction & layout | Free |
| **Document Processing** | PyMuPDF | PDF handling | Free |
| **Vision** | OpenCV | Image preprocessing | Free |
| **ML/DL** | PyTorch (GPU) | Model inference | Free (GPU) |
| **NLP** | spaCy | Text analysis | Free |
| **Embeddings** | sentence-transformers | Semantic similarity | Free |
| **LLM** | Ollama + Ministral-3B | Local content analysis | Free |
| **Web Framework** | Streamlit | Simple UI | Free |
| **Storage** | SQLite | Local data persistence | Free |

### Infrastructure

| Component | Choice | Notes |
|-----------|--------|-------|
| **Runtime** | Local Python environment | No Docker needed for learning |
| **Processing** | GPU OR CPU | Slower but free and sufficient |
| **Storage** | Local filesystem + SQLite | No cloud storage |
| **Models** | Download once, cache locally | ~1GB total |

### What We're NOT Using (to save money)

- ❌ No cloud services (AWS, GCP, Azure)
- ❌ No paid APIs (OpenAI, Anthropic, etc.)
- ❌ No GPU rentals or cloud GPUs
- ❌ No PostgreSQL/Redis servers
- ❌ No Docker/Kubernetes (keep it simple)
- ❌ No paid datasets or annotations

---

## 7. Training & Fine-Tuning Strategy

### 7.1 No-Training Baseline (Phase 1-2)

**Approach**: Use pre-trained models exclusively

**Pre-trained Models:**
- PaddleOCR (detection + recognition + structure)
- LayoutLMv3 (document understanding)
- Sentence-transformers (embeddings)

**Expected Performance:**
- OCR Accuracy: 95-98% on clean PDFs
- Layout Detection: 90-95% on standard resumes
- ATS Scoring: 85-90% correlation with real ATS

### 7.2 Optional Fine-Tuning (Phase 3)

**If Accuracy < 90%:**

**A. LayoutLM Fine-Tuning on Resumes**
```python
# Few-shot fine-tuning (500-1000 samples)
from transformers import LayoutLMv3ForTokenClassification

model = LayoutLMv3ForTokenClassification.from_pretrained(
    'microsoft/layoutlmv3-base',
    num_labels=len(resume_section_labels)
)

# Train on synthetic + annotated real resumes
trainer.train(resume_dataset)
```

**B. YOLOv8 for Resume Object Detection**
```python
# Train on custom resume layout dataset
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='resume_layout.yaml',
    epochs=50,
    imgsz=640
)
```

**For Learning Project**: Skip fine-tuning entirely - use pre-trained models only. If curious later, try fine-tuning on 50-100 samples just to learn the process.

### 7.3 No Continuous Learning Pipeline

This is a learning project, not a production service. Skip complex MLOps:
- No user feedback loops
- No retraining pipelines  
- No validation infrastructure
- No monitoring or alerting

**Focus on**: Understanding how the system works, not maintaining it at scale.

---

## 8. Evaluation & Metrics

### 8.1 Component-Level Metrics

| Component | Metric | Target | Validation Method |
|-----------|--------|--------|-------------------|
| **OCR** | CER (Character Error Rate) | < 2% | Ground truth comparison |
| **OCR** | WER (Word Error Rate) | < 5% | Manual review |
| **Layout Detection** | mAP@0.5 | > 0.90 | COCO-style evaluation |
| **Section Classification** | F1-Score | > 0.92 | Manual annotation |
| **Table Detection** | Accuracy | > 95% | Synthetic ground truth |

### 8.2 End-to-End ATS Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Parsing Success Rate** | % of resumes successfully analyzed | > 98% |
| **False Positive Rate** | Incorrect ATS issue detection | < 10% |
| **Score Correlation** | Correlation with real ATS systems | > 0.85 |
| **Processing Time** | End-to-end analysis | < 5 seconds |
| **User Satisfaction** | NPS score from beta users | > 50 |

### 8.3 Evaluation Dataset

**Test Set Composition:**
- 200 real resumes (diverse industries)
- 100 synthetic resumes (controlled difficulty)
- 50 edge cases (scanned, complex layouts, graphics)

**Ground Truth Generation:**
- Manual annotation of sections
- Commercial ATS parsing for comparison
- Expert review of ATS compatibility

---

## 9. Risk Mitigation & Fallbacks

### 9.1 OCR Failures

| Failure Mode | Detection | Fallback Strategy |
|--------------|-----------|-------------------|
| **Low Confidence** | Confidence < 0.8 | Switch to Tesseract |
| **Handwritten** | Font detection | Warning + manual review flag |
| **Corrupted PDF** | PyMuPDF error | Image extraction + OCR |
| **Language Mismatch** | Language detection | User notification |

### 9.2 Layout Detection Failures

| Failure Mode | Fallback |
|--------------|----------|
| **Complex Layout** | Heuristic-based column detection |
| **Poor Image Quality** | Denoising preprocessing |
| **Unknown Template** | Generic section detection |

### 9.3 Scoring Accuracy

| Risk | Mitigation |
|------|------------|
| **False Positives** | Confidence thresholds + user override |
| **Evolving ATS Systems** | Continuous monitoring + rule updates |
| **Industry Variations** | Configurable scoring weights |

---

## 10. Cost Estimation (Learning Project Budget)

### 10.1 Total Project Cost

| Item | Cost | Notes |
|------|------|-------|
| **Your Time** | Free | This is a learning project |
| **Computing** | $0 | Use your existing laptop/PC |
| **Software** | $0 | All open-source tools |
| **Models** | $0 | Pre-trained, download once |
| **Storage** | $0 | Local SQLite + filesystem |
| **APIs** | $0 | No paid APIs used |
| **Total** | **$0** | Completely free! |

### 10.2 What You Need

- A computer with 8GB+ RAM (16GB preferred)
- ~2GB free disk space for models
- Python 3.8+ installed
- Internet connection (for initial downloads only)
- Time to learn and experiment

### 10.3 Cost Savings Summary

By using only open-source tools:
- **$0** instead of $45k-80k development cost
- **$0/month** instead of $105-410/month operational cost
- Learn at your own pace without financial pressure
- No credit cards, no subscriptions, no surprise bills

---

## 11. Learning Objectives & Success Criteria

### 11.1 What You'll Learn

By the end of this project, you'll understand:

- ✅ How OCR engines work (text detection & recognition)
- ✅ Document layout analysis concepts
- ✅ How ATS systems parse resumes
- ✅ Rule-based scoring systems
- ✅ Working with pre-trained ML models
- ✅ Building simple UIs with Streamlit
- ✅ Local LLM integration
- ✅ Resume structure and best practices

### 11.2 Project Success Criteria (Learning Focus)

- ✅ Successfully process simple single-column resumes
- ✅ Detect basic layout issues (columns, tables)
- ✅ Provide reasonable ATS compatibility feedback
- ✅ Generate helpful improvement suggestions
- ✅ Working demo you can show to others
- ✅ Deep understanding of each component

**Not goals for this project**:
- ❌ Production-grade accuracy (80% is fine for learning)
- ❌ Support for complex layouts (focus on basics)
- ❌ Fast processing (30 seconds is OK on CPU)
- ❌ Enterprise scalability (single user is fine)

---

## 12. Next Steps

### Immediate Actions (Week 1) - Getting Started

1. **Set up free development environment**
   - Install Python 3.8+ on your computer
   - Create a virtual environment
   - Install PaddleOCR (free, open source)
   - Download pre-trained models (one-time download)

2. **Build your first pipeline**
   - Start with a simple Python script (not a complex app)
   - Load a PDF resume
   - Extract text using PaddleOCR
   - Print the results

3. **Experiment and learn**
   - Try 5-10 different resume PDFs
   - See what works and what doesn't
   - Take notes on what you learn

4. **Iterate gradually**
   - Add layout detection
   - Add simple scoring rules
   - Build Streamlit UI when ready

### Simplified Decisions (No Complex Trade-offs)

1. **OCR Engine**: Use PaddleOCR (free, works well, good docs)
2. **Processing**: CPU-only (your laptop is enough)
3. **LLM**: Use Ollama with Ministral-3B-Q4_K_M (completely free, GPU-efficient)
4. **Storage**: SQLite (built into Python, zero setup)
5. **Data**: Start with 10-20 real resumes you find online

---

## Appendix A: Pre-trained Model Checklist

**Download on Setup:**
```bash
# PaddleOCR Models (Auto-downloaded on first use)
# LayoutLMv3
huggingface-cli download microsoft/layoutlmv3-base
# Sentence Transformers
pip install sentence-transformers
```

**Model Sizes:**
- PaddleOCR Detection: ~5MB
- PaddleOCR Recognition: ~10MB
- PP-Structure Layout: ~20MB
- LayoutLMv3: ~500MB
- Sentence Transformer: ~100MB

**Total Model Size**: ~650MB

---

## Appendix B: Resume Template Coverage

**Target Templates for Validation:**

| Template Type | ATS Risk | Priority |
|---------------|----------|----------|
| Single-column simple | Low | MVP |
| Two-column modern | Medium | Phase 2 |
| Three-column creative | High | Phase 2 |
| Table-based skills | High | Phase 2 |
| Graphic-heavy | Very High | Phase 3 |
| Infographic | Very High | Phase 3 |
| Minimalist | Low | MVP |
| Academic CV | Low | Phase 2 |

---

**Document Version**: 1.1 (Learning Project Edition)  
**Last Updated**: 2026-02-03  
**Status**: Zero-Cost Learning Project Plan  
**Budget**: $0 (Open Source Only)
