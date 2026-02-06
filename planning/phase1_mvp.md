# Phase 1: MVP - Core Functionality

**Timeline**: Weeks 1-4  
**Goal**: Build a working prototype that can analyze single-column resumes

---

## Week 1: Environment & First Pipeline

### Objectives
- [ ] Set up Python environment with all dependencies
- [ ] Get PaddleOCR working with a simple PDF
- [ ] Extract text from 3-5 sample resumes
- [ ] Understand OCR output format

### Tasks

#### Day 1-2: Environment Setup
- [ ] Create project directory structure
- [x] Create conda environment:
  ```bash
  conda create -n ats-resume python=3.10
  conda activate ats-resume
  ```
- [x] Install dependencies:
  ```bash
  pip install paddleocr paddlepaddle pymupdf opencv-python
  pip install streamlit pandas numpy
  ```
- [ ] Download PaddleOCR models (first run will auto-download)
- [ ] Test PaddleOCR on a simple image

#### Day 3-4: PDF Processing
- [ ] Write function to convert PDF to images
- [ ] Test with 3 different resume PDFs from `resumes/` folder
- [ ] Pick a mix: one simple, one complex layout
- [ ] Handle multi-page PDFs
- [ ] Save processed images for debugging

#### Day 5-7: Text Extraction Pipeline
- [ ] Build `extract_text_from_resume(pdf_path)` function
- [ ] Process 5 sample resumes from `resumes/` folder
- [ ] Start with simpler single-column layouts
- [ ] Log extraction results to understand accuracy
- [ ] Document any issues encountered (French formatting, special characters, etc.)

### Deliverables
- [ ] Working text extraction from PDF resumes
- [ ] Log of 5 test cases with accuracy notes
- [ ] Basic understanding of PaddleOCR output format

---

## Week 2: Layout Detection

### Objectives
- [ ] Detect resume sections (Experience, Education, Skills, etc.)
- [ ] Identify basic layout patterns
- [ ] Extract structured data from resumes

### Tasks

#### Day 1-2: Understanding PP-Structure
- [ ] Research PaddleOCR's PP-Structure module
- [ ] Test layout detection on sample resumes
- [ ] Understand bounding box and region classification output

#### Day 3-5: Section Classification
- [ ] Build section detector using heuristics + PP-Structure
- [ ] Identify common resume sections:
  - Contact Information
  - Experience / Work History
  - Education
  - Skills
  - Projects
  - Summary/Objective
- [ ] Handle variations in section naming

#### Day 6-7: Structured Data Extraction
- [ ] Create `ResumeParser` class
- [ ] Extract structured JSON from raw OCR text
- [ ] Handle parsing errors gracefully
- [ ] Test on 10 different resumes

### Deliverables
- [ ] Working section detection
- [ ] Structured JSON output from resumes
- [ ] Test results on 10 resumes with accuracy notes

---

## Week 3: Basic ATS Scoring

### Objectives
- [ ] Implement rule-based ATS compatibility scoring
- [ ] Detect multi-column layouts
- [ ] Identify common ATS-unfriendly elements

### Tasks

#### Day 1-2: Scoring Algorithm Design
- [ ] Define scoring criteria:
  - Single vs multi-column (-25 points if multi-column)
  - Table detection (-20 points for table-based layouts)
  - Image/graphic detection (-15 points if graphics heavy)
  - Standard headers (+10 points for standard sections)
  - Text extraction quality (+10 points if clean)
- [ ] Create `ATSScorer` class
- [ ] Implement base scoring logic

#### Day 3-4: Layout Risk Detection
- [ ] Detect multi-column layouts (use bounding box analysis)
- [ ] Detect tables in resume
- [ ] Detect images/graphics vs text
- [ ] Test detection accuracy on known layouts

#### Day 5-7: Scoring Implementation
- [ ] Implement all scoring rules
- [ ] Normalize scores to 0-100 range
- [ ] Generate score breakdown
- [ ] Test on 15 resumes (mix of good and bad)

### Deliverables
- [ ] Working ATS scoring algorithm
- [ ] Score breakdown display
- [ ] Tested on 15 resumes with validation notes

---

## Week 4: Simple UI & Integration

### Objectives
- [ ] Build Streamlit interface for uploading and viewing results
- [ ] Display OCR text, structured data, and ATS score
- [ ] Create basic visualizations

### Tasks

#### Day 1-3: Streamlit UI
- [ ] Create main app file `app.py`
- [ ] Build file upload component
- [ ] Display extracted text
- [ ] Show structured data in organized sections
- [ ] Display ATS score prominently

#### Day 4-5: Score Visualization
- [ ] Show score breakdown (bar chart or simple display)
- [ ] Highlight issues detected
- [ ] Color-code risk levels (green/yellow/red)

#### Day 6-7: Testing & Refinement
- [ ] Test UI with 5 real resumes
- [ ] Fix any UI/UX issues
- [ ] Add loading states and error handling
- [ ] Document known limitations

### Deliverables
- [ ] Working Streamlit app
- [ ] End-to-end pipeline: Upload → Process → Display results
- [ ] Documentation of MVP limitations

---

## Phase 1 Success Criteria

- [ ] ✅ Can upload and process PDF resumes
- [ ] ✅ Successfully extracts text from simple single-column resumes (80%+ accuracy)
- [ ] ✅ Detects basic layout issues (columns, tables)
- [ ] ✅ Provides ATS compatibility score (even if approximate)
- [ ] ✅ Working Streamlit UI demonstrating the pipeline

---

## Resources Needed

### Software
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)
- Python 3.10 (via conda)
- PaddleOCR (with PP-Structure)
- PyMuPDF
- Streamlit
- OpenCV

### Hardware
- Your existing laptop/PC
- ~2GB disk space for models
- 8GB+ RAM recommended

### Test Data
✅ **Already Available**: 30 real resumes in `resumes/` folder from your former classmates

**Suggested Testing Strategy:**
1. Start with 5-10 resumes to test your pipeline
2. Classify them by layout type:
   - Simple single-column (easiest for MVP)
   - Two-column modern
   - Table-based sections
   - Graphic-heavy or creative layouts
3. Note any French-specific formatting that might differ from US resumes
4. Test with a mix of good and ATS-problematic layouts

---

## Learning Goals for Phase 1

By end of Phase 1, understand:
- [ ] How PaddleOCR detects and recognizes text
- [ ] How to convert PDFs to processable images
- [ ] What makes a resume ATS-friendly vs unfriendly
- [ ] How to build rule-based scoring systems
- [ ] How to create simple ML pipelines
- [ ] How to build basic UIs with Streamlit

---

## Notes & Decisions Log

**Week 1:**
- 

**Week 2:**
- 

**Week 3:**
- 

**Week 4:**
- 

---

**Status**: Not Started  
**Started Date**:  
**Completed Date**:  
