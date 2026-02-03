# ATS-Aware Resume Analysis System: Technical Implementation Plan

## Executive Summary

This document outlines the end-to-end technical implementation plan for an ATS-Aware Resume Analysis & Improvement System. The architecture follows a **hybrid approach** combining:
- **Pre-trained Vision Models** for layout analysis (zero-shot/few-shot)
- **Rule-based ATS scoring** with ML enhancement
- **LLM-based content analysis** for semantic understanding
- **Synthetic data generation** for resume-specific training

**Key Design Decision**: Minimal custom training required by leveraging state-of-the-art pre-trained models with domain adaptation techniques.

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

**LLM Usage (Optional, API-based):**
- **Model**: GPT-4-mini or local LLaMA-3-8B
- **Purpose**: Content quality assessment, rewrite suggestions
- **Cost mitigation**: Cache common patterns, batch processing

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
# Airflow / Prefect / Custom DAG

class ResumeAnalysisPipeline:
    """End-to-end orchestration"""
    
    def execute(self, resume_file, job_description=None):
        # Stage 1: Document Ingestion
        doc_data = self.ingest(resume_file)
        
        # Stage 2: Parallel Processing
        with ThreadPoolExecutor() as executor:
            ocr_future = executor.submit(self.run_ocr, doc_data)
            layout_future = executor.submit(self.analyze_layout, doc_data)
            
        ocr_result = ocr_future.result()
        layout_result = layout_future.result()
        
        # Stage 3: Structure Extraction
        structured_resume = self.extract_structure(ocr_result, layout_result)
        
        # Stage 4: ATS Analysis (Parallel sub-analyses)
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

### 4.2 Caching Strategy

| Cache Layer | Content | TTL | Storage |
|-------------|---------|-----|---------|
| **OCR Results** | Raw text + bounding boxes | 24h | Redis |
| **Layout Analysis** | Detected regions | 24h | Redis |
| **ATS Scores** | Score components | 1h | In-memory |
| **Embeddings** | Text embeddings | 7d | Vector DB (optional) |

### 4.3 Queue Architecture (for High Volume)

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Upload    │────▶│  Redis Queue │────▶│   Worker     │
│   Endpoint  │     │  (Celery)    │     │   Pool       │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                       ┌────────────────────────┘
                       ▼
              ┌──────────────────────┐
              │   Result Storage     │
              │   (PostgreSQL/S3)    │
              └──────────────────────┘
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

### Phase 3: Production Polish (Weeks 9-12)

**Goal**: Performance, accuracy, user experience

**Components:**
1. **Performance Optimization**
   - Async processing
   - Caching layer
   - Batch processing support
   
2. **Accuracy Improvements**
   - Synthetic data validation
   - Error analysis pipeline
   - Model fallback strategies
   
3. **Advanced Features**
   - Multi-resume comparison
   - Industry-specific scoring
   - Export to ATS-friendly format
   
4. **Production Infrastructure**
   - API endpoints (FastAPI)
   - Authentication
   - Usage analytics

**Deliverables:**
- Production-ready API
- <5s processing time for typical resumes
- 95%+ layout detection accuracy
- User accounts & history

---

## 6. Technology Stack

### Core Dependencies

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **OCR** | PaddleOCR | 2.7+ | Text extraction & layout |
| **Document Processing** | PyMuPDF | 1.23+ | PDF handling |
| **Vision** | OpenCV | 4.8+ | Image preprocessing |
| **ML/DL** | PyTorch | 2.0+ | Model inference |
| **NLP** | spaCy | 3.7+ | Text analysis |
| **Embeddings** | sentence-transformers | 2.2+ | Semantic similarity |
| **LLM** | OpenAI API / Ollama | - | Content generation |
| **Web Framework** | Streamlit / FastAPI | Latest | UI & API |
| **Storage** | PostgreSQL / SQLite | - | Data persistence |
| **Cache** | Redis | 7+ | Result caching |
| **Queue** | Celery + Redis | - | Background jobs |

### Infrastructure (Production)

| Component | Recommended | Alternative |
|-----------|--------------|-------------|
| **Container** | Docker + Docker Compose | Kubernetes (scale) |
| **GPU** | NVIDIA T4 (inference) | CPU-only (slower) |
| **Storage** | AWS S3 / Local | MinIO |
| **API Gateway** | Nginx | Traefik |
| **Monitoring** | Prometheus + Grafana | Custom |

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

**Training Data Required:**
- 500-1000 annotated resumes
- Synthetic data for augmentation
- Cost: 2-3 days of training on single GPU

### 7.3 Continuous Learning Pipeline

```
User Uploads ──▶ Analysis ──▶ User Feedback
     │                              │
     ▼                              ▼
┌──────────┐               ┌──────────────┐
│  Store   │               │ Correction   │
│  Review  │               │ Interface    │
└────┬─────┘               └──────┬───────┘
     │                              │
     └──────────┬───────────────────┘
                ▼
         ┌──────────────┐
         │  Validation  │
         │  Pipeline    │
         └──────┬───────┘
                ▼
         ┌──────────────┐
         │ Retraining   │
         │ (Monthly)    │
         └──────────────┘
```

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

## 10. Cost Estimation

### 10.1 Development Costs

| Phase | Duration | Team Size | Est. Cost |
|-------|----------|-----------|-----------|
| **Phase 1 (MVP)** | 4 weeks | 1-2 devs | $10k-20k |
| **Phase 2 (Enhancement)** | 4 weeks | 2 devs | $15k-25k |
| **Phase 3 (Production)** | 4 weeks | 2-3 devs | $20k-35k |
| **Total** | 12 weeks | - | $45k-80k |

### 10.2 Operational Costs (Monthly)

| Component | Scale | Cost |
|-----------|-------|------|
| **Compute (CPU)** | 1000 resumes/day | $50-100 |
| **Compute (GPU)** | Optional acceleration | $100-200 |
| **Storage** | 10GB | $5-10 |
| **LLM API** | 1000 calls/day | $50-100 |
| **Total** | - | $105-410 |

### 10.3 Alternative: Reduced Cost Option

- **CPU-only inference**: Remove GPU costs
- **Local LLM**: Use Ollama/Llama3 instead of OpenAI
- **SQLite instead of PostgreSQL**: For low volume

**Reduced Monthly Cost**: $50-150

---

## 11. Success Criteria

### 11.1 MVP Success Metrics

- ✅ Process 95% of uploaded resumes without errors
- ✅ Detect multi-column layouts with 90% accuracy
- ✅ Provide ATS score within 10 points of commercial ATS
- ✅ Generate improvement suggestions for 80% of issues
- ✅ <10 second processing time

### 11.2 Full Product Metrics

- ✅ 98% parsing success rate
- ✅ 95% layout detection accuracy
- ✅ 90% user-reported usefulness
- ✅ <5 second average processing time
- ✅ Support for 50+ resume templates

---

## 12. Next Steps

### Immediate Actions (Week 1)

1. **Set up development environment**
   - Install PaddleOCR
   - Download pre-trained models
   - Set up Python environment

2. **Create data pipeline skeleton**
   - Document ingestion
   - OCR integration
   - Basic layout detection

3. **Build simple UI prototype**
   - Streamlit upload interface
   - Display OCR results
   - Basic scoring display

4. **Validate approach**
   - Test on 20-30 sample resumes
   - Measure baseline accuracy
   - Identify failure modes

### Key Decisions Required

1. **OCR Engine**: PaddleOCR vs Tesseract (recommend PaddleOCR)
2. **GPU Requirement**: Is GPU inference needed for target volume?
3. **LLM Strategy**: Local vs API-based (recommend hybrid)
4. **Job Matching**: MVP feature or Phase 2?
5. **Synthetic Data**: Generate 1000 samples for validation?

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

**Document Version**: 1.0  
**Last Updated**: 2026-02-03  
**Status**: Technical Planning Phase
