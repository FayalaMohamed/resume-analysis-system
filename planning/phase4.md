# Phase 4: Production-Ready Extraction System

**Goal**: Build a production-grade resume extraction system that prioritizes accuracy, completeness, and truthfulness, even if per-resume processing takes 30-60+ seconds. Scope is a micro SaaS for early traction, not large-scale throughput.

---

## Overview

Phase 4 focuses on turning the current experimental, multi-strategy codebase into a reliable, measurable, production-ready extraction pipeline. The emphasis is on extraction quality as the foundation for everything else. We accept slower processing to maximize extraction fidelity and only need to support early-stage usage (not millions of resumes).

**Key Principles:**
1. Quality > Speed (within 30-60+ seconds per resume)
2. Multi-Strategy Extraction (combine best parts of current system)
3. Confidence-Driven Decisions (choose or merge outputs based on evidence)
4. Traceability (every extracted fact should be grounded to source text)
5. Measurable Accuracy (repeatable evaluation and regression testing)

---

## 1. Current Assets to Leverage (Do Not Rebuild)

Use the existing capabilities as building blocks:

**Extraction & OCR (Already Implemented)**
- `src/parsers/enhanced_ocr.py`: Multi-engine OCR with fallback and confidence
- `src/parsers/ocr.py`: PyMuPDF + PaddleOCR hybrid
- `src/parsers/langextract_parser.py`: High-quality LLM extraction with grounding
- `src/parsers/unified_extractor.py`: Structured extraction with heuristics

**Layout & Structure (Already Implemented)**
- `src/parsers/ml_layout_detector.py`: ML layout detection
- `src/parsers/heuristic_layout_detector.py`: Rule-based layout fallback
- `src/parsers/section_parser.py`: Section classification and parsing

**Content Understanding (Already Implemented)**
- `src/analysis/content_understanding.py`: Red flags, trajectory, quantification
- `src/analysis/enhanced_skills.py`: Skill taxonomy and normalization

**Pipeline & Comparison (Already Implemented)**
- `pipeline.py`: Full analysis pipeline and variant comparison
- `docs/LANGEXTRACT_GUIDE.md`: Quality vs speed benchmarks and hybrid strategy
- `app.py`: Streamlit UI already wires OCR + unified + LangExtract

---

## 2. Target Production Architecture (Extraction-Centric)

### 2.1 Asynchronous Job Pipeline (Quality First)

```
Upload -> Preflight -> Queue -> Extraction Workers -> Aggregation -> QA -> Store -> Notify
```

**Design Intent:**
- Accept slower extraction jobs, but always complete with highest quality path
- Parallelize steps where possible (OCR, layout, and LLM extraction)
- Centralize decision logic in an Aggregation layer
- Keep infrastructure minimal (single worker pool + queue) until traffic demands more

### 2.2 Multi-Stage Extraction Stack (Matches Current Code)

Use all extraction strategies and reconcile them:

1. **Fast Text Layer (PyMuPDF)**
   - Use when text layer is valid and complete
   - Provides baseline structure

2. **Multi-Engine OCR Layer**
   - `PaddleOCR -> Tesseract -> EasyOCR -> PDF Native`
   - Generate per-word confidence (`enhanced_ocr.py` already does this)

3. **Layout & Section Layer**
   - ML layout detection with heuristic fallback
   - Reading order and column reconstruction
   - Current coverage: `ml_layout_detector.py` + `heuristic_layout_detector.py`

4. **LLM Extraction Layer (LangExtract)**
   - Use for deep structure, skills, and bullet-level extraction
   - Use source grounding for auditability
   - Current coverage: `langextract_parser.py`

### 2.3 Aggregation and Reconciliation (Implemented)

Create a single authoritative resume representation by merging results:

**Aggregation Rules:**
- Prefer grounded fields with highest confidence
- Cross-validate fields between extractors (contact, dates, titles)
- Resolve conflicts by confidence + source agreement
- Keep original extracted variants for audit/debug

**Output Schema (Canonical):**
- `contact` (+ `contact_meta` with sources/confidence)
- `summary` (+ `summary_meta` with sources/confidence)
- `experience[]` (job_title, company, dates, bullets, technologies)
- `education[]`
- `skills[]` (normalized + category)
- `projects[]`
- `certifications[]`
- `languages[]`
- `metadata` (timings, text selection rationale, OCR confidence, language)

---

## 3. Quality Controls and Truthfulness

### 3.1 Confidence and Grounding

Every field must include:
- Confidence score (0-1)
- Extraction source (OCR, LangExtract, PyMuPDF, etc.)
- Source grounding (text spans or bounding boxes)

### 3.2 Truthfulness Filters

Prevent hallucination and invented content:
- Only accept LLM output if grounded to source text
- Reject ungrounded entities
- Maintain a strict "no invention" policy for structured fields

### 3.3 Human Review Hooks (Optional)

For low-confidence results:
- Flag sections for user review
- Provide "verify or edit" flow in UI

---

## 4. Performance Strategy (30-60+ Seconds OK)

### 4.1 Target Latency

- **p50**: 30-45 seconds
- **p95**: 60-90 seconds
- **p99**: 120 seconds (acceptable for complex scans)

### 4.2 Optimization Priorities

- Model warm-up and reuse (avoid reload per job)
- OCR parallelization per page
- LLM extraction chunking tuned for accuracy (not speed)
- Cache intermediate results by file hash

### 4.3 When to Escalate to LangExtract

**Escalation Triggers:**
- Low OCR confidence (<0.85)
- Multi-column or table-heavy layout
- Missing required sections (experience, education)
- High disagreement between extractors

---

## 5. Production Infrastructure (Extraction-Focused, Micro SaaS)

### 5.1 Storage & Persistence

- Store original file (encrypted)
- Store all intermediate outputs for debug
- Store canonical output separately
- Maintain audit log for extraction decisions
- Keep data model minimal (users, resumes, extractions, jobs)

### 5.2 Job Queue & Workers

- Background workers for OCR + LangExtract
- Separate pools for heavy LLM jobs
- Retry policy for external providers
- Small worker pool (2-4) is enough initially

### 5.3 Observability

Track extraction quality as first-class metrics:
- Extraction success rate
- Section completeness
- OCR confidence distribution
- LangExtract grounding rate
- Conflict rate in aggregation
- Manual edit rate (user corrections)

---
## 6. Roadmap (Production-Ready Extraction)

### Phase 4A: Pipeline Hardening
- [x] Formalize extraction stages and interfaces
- [x] Add aggregation/reconciliation layer (`src/parsers/aggregated_extractor.py`)
- [x] Persist multi-source outputs (aggregated JSON output)
- [x] Add confidence + grounding metadata (sources + confidence per field)

### Phase 4B: Production Operations
- [x] Async job queue + worker pools (SQLite-backed queue + worker)
- [x] Storage encryption + retention policy (Supabase storage + retention job)
- [ ] Observability dashboards
- [x] Error handling + retry strategy (job retry/backoff + failure recording)

---

## 7. Success Criteria (Extraction First)

**Quality:**
- >= 95% field completeness on benchmark set
- >= 90% precision on contact/experience/education/skills
- <= 2% ungrounded LLM extractions

**Performance:**
- p95 extraction time <= 90 seconds
- System stable for early-stage traffic (tens to hundreds of resumes/day)

**Trust:**
- Every extracted field traceable to source
- Clear user review path for low-confidence content

---

## 8. Notes & Decisions Log

**Extraction Strategy:**
- Implemented aggregated pipeline that merges OCR, unified extraction, SectionParser, and LangExtract when available.

**Aggregation Rules:**
- Field-level confidence and source tracking; dedupe by title/company/date for experience and degree/institution/date for education.

**LangExtract Usage:**
- Optional but used when configured; fallback to unified and OCR outputs.

**Evaluation Corpus:**
- Removed from Phase 4 scope (tracked separately).

**Production Infrastructure:**
- Micro SaaS scope; local storage acceptable initially.

---

**Status**: In Progress  
**Prerequisites**: Phase 3 Complete  
**Started Date**:  
**Completed Date**:
