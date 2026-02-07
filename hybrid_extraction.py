#!/usr/bin/env python3
"""
Hybrid Resume Extraction

Combines:
1. ML Layout Detection - to find regions (titles, tables, figures, text blocks)
2. Unified Extraction - to parse text with font hierarchy analysis

Usage:
    python hybrid_extraction.py resume.pdf [--output hybrid.json]
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import fitz

sys.path.insert(0, str(Path(__file__).parent / "src"))

from parsers.ml_layout_detector import MLLayoutDetector
from parsers.ocr import PDFTextExtractor
from parsers.unified_extractor import UnifiedResumeExtractor


@dataclass
class TextSpan:
    """A span of text with font information."""
    text: str
    font_name: str
    font_size: float
    is_bold: bool
    color: int
    bbox: Tuple[float, float, float, float]


@dataclass
class TextLine:
    """A line of text containing multiple spans."""
    text: str
    spans: List[TextSpan]
    bbox: Tuple[float, float, float, float]
    
    def __post_init__(self):
        if isinstance(self.spans, list) and len(self.spans) > 0:
            first = self.spans[0]
            self.is_bold = any(s.is_bold for s in self.spans)
            self.y0 = self.bbox[1]
        else:
            self.is_bold = False
            self.y0 = 0


@dataclass
class ResumeItem:
    """An item within a section."""
    title: str = ""
    subtitle: str = ""
    date_range: str = ""
    location: str = ""
    company: str = ""
    description_lines: List[str] = field(default_factory=list)
    bullet_points: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict:
        desc = self.description if self.description else '\n'.join(self.description_lines)
        return {
            'title': self.title,
            'subtitle': self.subtitle,
            'date_range': self.date_range,
            'location': self.location,
            'company': self.company,
            'description': desc,
            'bullet_points': self.bullet_points,
        }


@dataclass
class ResumeSection:
    """A section in the resume."""
    title: str = ""
    section_type: str = "unknown"
    items: List[ResumeItem] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'section_type': self.section_type,
            'items': [item.to_dict() for item in self.items],
            'raw_text': self.raw_text,
        }


@dataclass
class StructuredResume:
    """Complete structured resume."""
    name: str = ""
    contact_info: Dict[str, str] = field(default_factory=dict)
    summary: str = ""
    sections: List[ResumeSection] = field(default_factory=list)
    all_text: str = ""
    ml_regions: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'contact_info': self.contact_info,
            'summary': self.summary,
            'sections': [s.to_dict() for s in self.sections],
            'all_text': self.all_text,
            'ml_regions': self.ml_regions,
        }


class MLLayoutProcessor:
    """Process ML-detected regions and extract text from each."""
    
    def __init__(self, lang: str = 'en'):
        self.lang = lang
        self.pdf_doc = None
        self.page_width = 612
        self.page_height = 792
        self.ocr = PDFTextExtractor(use_paddle=True)
    
    def open_pdf(self, pdf_path: Path):
        """Open PDF and prepare for processing."""
        self.pdf_doc = fitz.open(str(pdf_path))
        if len(self.pdf_doc) > 0:
            rect = self.pdf_doc[0].rect
            self.page_width = rect.width
            self.page_height = rect.height
    
    def extract_text_from_region(self, page, region_bbox: Tuple[float, float, float, float]) -> List[TextLine]:
        """Extract text with font info from a specific region."""
        lines = []
        
        try:
            blocks = page.get_text("dict", clip=region_bbox)["blocks"]
        except Exception:
            return lines
        
        for block in blocks:
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    spans = []
                    line_bbox = line.get("bbox", [0, 0, 0, 0])
                    
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if not text.strip():
                            continue
                        
                        font = span.get("font", "unknown")
                        size = span.get("size", 10)
                        flags = span.get("flags", 0)
                        is_bold = bool(flags & 2**4) or "bold" in font.lower()
                        color = span.get("color", 0)
                        
                        try:
                            span_bbox = (span["x0"], span["y0"], span["x1"], span["y1"])
                        except (KeyError, TypeError):
                            span_bbox = tuple(line_bbox)
                        
                        spans.append(TextSpan(
                            text=text,
                            font_name=font,
                            font_size=size,
                            is_bold=is_bold,
                            color=color,
                            bbox=span_bbox
                        ))
                    
                    if spans:
                        line_text = "".join(s.text for s in spans)
                        line_obj = TextLine(text=line_text, spans=spans, bbox=tuple(line_bbox))
                        lines.append(line_obj)
        
        return lines
    
    def extract_text_with_ocr(self, page, region_bbox: Tuple[float, float, float, float]) -> str:
        """Extract text from region using OCR."""
        x0, y0, x1, y1 = region_bbox
        page_rect = page.rect
        
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(page_rect.width, x1)
        y1 = min(page_rect.height, y1)
        
        if x1 <= x0 or y1 <= y0:
            return ""
        
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        
        scale = 2
        region_pix = fitz.Pixmap(pix, int(x0 * scale), int(y0 * scale), int((x1 - x0) * scale), int((y1 - y0) * scale))
        
        import tempfile
        import os
        temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(temp_fd)
        region_pix.save(temp_path)
        
        try:
            result = self.ocr.extract_text_from_image(temp_path)
            if result and "full_text" in result:
                return result["full_text"]
            return ""
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def detect_regions(self, pdf_path: Path) -> List[Dict]:
        """Detect layout regions using ML."""
        detector = MLLayoutDetector(lang=self.lang)
        
        doc = fitz.open(str(pdf_path))
        temp_dir = pdf_path.parent / ".temp_ml"
        temp_dir.mkdir(exist_ok=True)
        
        image_paths = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(150/72, 150/72)
            pix = page.get_pixmap(matrix=mat)
            image_path = temp_dir / f"{pdf_path.stem}_page_{page_num + 1}.png"
            pix.save(str(image_path))
            image_paths.append(str(image_path))
        doc.close()
        
        all_regions = []
        
        try:
            for img_path in image_paths:
                regions = detector.detect_layout(img_path)
                for r in regions:
                    all_regions.append({
                        'type': r.get('type', 'text'),
                        'bbox': r.get('bbox', [0, 0, 100, 100]),
                        'confidence': r.get('score', 0.5),
                        'image_path': img_path
                    })
            
            for img_path in image_paths:
                try:
                    Path(img_path).unlink()
                except:
                    pass
            temp_dir.rmdir()
            
        except Exception as e:
            print(f"ML detection error: {e}")
        
        return all_regions


class UnifiedItemParser:
    """Parse items from text lines."""
    
    DATE_PATTERNS = [
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}(?:\s*[-–—to]+\s*(?:present|current|\d{4}))?\b',
        r'\b(?:0?[1-9]|1[0-2])\s*/\s*\d{4}(?:\s*[-–—to]+\s*(?:present|current|\d{4}))?\b',
        r'\b\d{4}\s*[-–—to]+\s*(?:present|current|\d{4})\b',
        r'\b(?:Since|From)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b',
        r'\b(?:Since|From)\s+\d{4}\b',
    ]
    
    DATE_RANGE_PATTERN = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–—to]+\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b'
    
    BULLET_MARKERS = ['•', '-', '*', '○', '◦', '▪', '▫', '→', '⇒', '➢', '✓', '✔', '●', '·']
    
    SECTION_KEYWORDS = {
        'experience': ['experience', 'employment', 'work history', 'professional experience'],
        'education': ['education', 'academic', 'degree', 'university', 'college'],
        'skills': ['skills', 'technical skills', 'technologies', 'competencies'],
        'projects': ['projects', 'personal projects'],
        'summary': ['summary', 'profile', 'objective', 'about'],
        'volunteer': ['volunteer', 'community'],
        'awards': ['awards', 'honors', 'achievements'],
        'certifications': ['certifications', 'certificates'],
        'languages': ['languages'],
        'interests': ['interests', 'hobbies'],
    }
    
    def is_section_header(self, text: str, is_bold: bool) -> Tuple[bool, str]:
        """Check if text is a section header."""
        text = text.strip()
        if len(text) > 60 or len(text) < 3:
            return False, ""
        
        is_upper = text.isupper()
        is_title = text.istitle()
        
        if not (is_upper or is_title):
            if not is_bold or len(text) > 30:
                return False, ""
        
        text_lower = text.lower()
        for section_type, keywords in self.SECTION_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return True, section_type
        
        return False, ""
    
    def extract_date(self, text: str) -> Tuple[str, str]:
        """Extract date from text."""
        text = text.strip()
        
        date_range_match = re.search(self.DATE_RANGE_PATTERN, text, re.I)
        if date_range_match:
            date = date_range_match.group(0)
            remaining = text[:date_range_match.start()].strip() + text[date_range_match.end():].strip()
            return date, remaining
        
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text, re.I)
            if match:
                date = match.group(0)
                remaining = text[:match.start()].strip() + text[match.end():].strip()
                return date, remaining
        
        return "", text
    
    def is_bullet_point(self, text: str) -> bool:
        """Check if text is a bullet point."""
        stripped = text.strip()
        return any(stripped.startswith(m) for m in self.BULLET_MARKERS)
    
    def parse_item(self, lines: List[str], is_bold_flags: List[bool], section_type: str) -> ResumeItem:
        """Parse lines into a ResumeItem."""
        if not lines:
            return ResumeItem()
        
        item = ResumeItem()
        
        for i, text in enumerate(lines):
            text = text.strip()
            if not text:
                continue
            
            is_bold = is_bold_flags[i] if i < len(is_bold_flags) else False
            
            if self.is_bullet_point(text):
                bullet_text = text
                for marker in self.BULLET_MARKERS:
                    if bullet_text.startswith(marker):
                        bullet_text = bullet_text[len(marker):].strip()
                        break
                item.bullet_points.append(bullet_text)
                continue
            
            date, remaining = self.extract_date(text)
            has_date = bool(date)
            
            if not item.title:
                if is_bold and len(text) < 80:
                    item.title = remaining or text
                    if date:
                        item.date_range = date
                    continue
            
            if has_date and not item.date_range:
                item.date_range = date
                if remaining and not item.title:
                    item.title = remaining
                continue
            
            if text not in item.description_lines:
                item.description_lines.append(text)
        
        return item


class HybridExtractor:
    """Main hybrid extractor combining ML regions with unified parsing."""
    
    def __init__(self, lang: str = 'en'):
        self.lang = lang
        self.ml_processor = MLLayoutProcessor(lang)
        self.item_parser = UnifiedItemParser()
        self.pdf_doc = None
    
    def extract(self, pdf_path: Path) -> StructuredResume:
        """Extract structured resume from PDF."""
        result = StructuredResume()
        
        self.pdf_doc = fitz.open(str(pdf_path))
        self.ml_processor.open_pdf(pdf_path)
        
        all_text = []
        ml_regions = self.ml_processor.detect_regions(pdf_path)
        
        # Process each page
        for page_num, page in enumerate(self.pdf_doc):
            print(f"Processing page {page_num + 1}/{len(self.pdf_doc)}...")
            
            # Get regions for this page (by page number)
            page_regions = []
            for r in ml_regions:
                img_path = str(r.get('image_path', ''))
                if f"_page_{page_num + 1}" in img_path:
                    page_regions.append(r)
            
            # If no ML regions, use full page text extraction
            if not page_regions:
                # Fall back to full page extraction without ML regions
                lines = self.ml_processor.extract_text_from_region(page, page.rect)
                text_lines = [line.text for line in lines if line.text.strip()]
                text = "\n".join(text_lines)
                if text.strip():
                    all_text.append(text)
                
                ml_regions.append({
                    'type': 'text',
                    'bbox': list(page.rect),
                    'confidence': 1.0,
                    'text': text[:300]
                })
            else:
                # Process ML regions
                for region in page_regions:
                    region_type = region['type']
                    bbox = region['bbox']
                    
                    if isinstance(bbox, list):
                        bbox = tuple(bbox)
                    
                    if region_type in ['table', 'figure']:
                        text = self.ml_processor.extract_text_with_ocr(page, bbox)
                    else:
                        lines = self.ml_processor.extract_text_from_region(page, bbox)
                        text_lines = [line.text for line in lines if line.text.strip()]
                        text = "\n".join(text_lines)
                    
                    if text.strip():
                        ml_regions.append({
                            'type': region_type,
                            'bbox': list(bbox) if isinstance(bbox, tuple) else bbox,
                            'confidence': region.get('confidence', 0.5),
                            'text': text[:300]
                        })
                        all_text.append(text)
        
        result.all_text = "\n\n".join(all_text)
        result.ml_regions = ml_regions
        
        # Build sections from text
        self._build_sections(result, all_text)
        
        # Extract header
        self._extract_header(result, all_text[0] if all_text else "")
        
        return result
    
    def _build_sections(self, result: StructuredResume, all_text: List[str]):
        """Build sections from extracted text."""
        full_text = "\n\n".join(all_text)
        lines = full_text.split('\n')
        
        current_section = None
        current_lines = []
        current_types = []
        current_bold_flags = []
        
        for line_text in lines:
            line_text = line_text.strip()
            if not line_text:
                continue
            
            # Check if bold (heuristic)
            is_bold = line_text.isupper() or (len(line_text) < 50 and line_text[0].isupper())
            
            is_header, section_type = self.item_parser.is_section_header(line_text, is_bold)
            
            if is_header:
                if current_section and current_lines:
                    items = self._parse_section_items(current_lines, current_bold_flags, current_section_type)
                    current_section.items = items
                    current_section.raw_text = "\n".join(current_lines)
                    result.sections.append(current_section)
                
                current_section = ResumeSection(title=line_text, section_type=section_type)
                current_lines = []
                current_types = []
                current_bold_flags = []
                current_section_type = section_type
            else:
                if current_section is None:
                    # Header content goes to first section
                    if not result.sections:
                        current_section = ResumeSection(title="Header", section_type="header")
                        current_section_type = "header"
                    else:
                        current_section = result.sections[-1]
                        current_section_type = current_section.section_type
                
                current_lines.append(line_text)
                current_bold_flags.append(is_bold)
        
        # Don't forget last section
        if current_section and current_lines:
            items = self._parse_section_items(current_lines, current_bold_flags, current_section_type)
            current_section.items = items
            current_section.raw_text = "\n".join(current_lines)
            if current_section not in result.sections:
                result.sections.append(current_section)
    
    def _parse_section_items(self, lines: List[str], bold_flags: List[bool], section_type: str) -> List[ResumeItem]:
        """Parse lines into items."""
        items = []
        current_item_lines = []
        current_item_bolds = []
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            is_new_item = False
            if section_type in ['experience', 'education', 'projects']:
                is_bold = bold_flags[i] if i < len(bold_flags) else False
                if is_bold and len(line) < 80:
                    is_new_item = True
            
            if is_new_item and current_item_lines:
                item = self.item_parser.parse_item(current_item_lines, current_item_bolds, section_type)
                if item.title or item.bullet_points or item.description_lines:
                    items.append(item)
                current_item_lines = [line]
                current_item_bolds = [bold_flags[i]] if i < len(bold_flags) else [False]
            else:
                current_item_lines.append(line)
                if i < len(bold_flags):
                    current_item_bolds.append(bold_flags[i])
                else:
                    current_item_bolds.append(False)
        
        if current_item_lines:
            item = self.item_parser.parse_item(current_item_lines, current_item_bolds, section_type)
            if item.title or item.bullet_points or item.description_lines:
                items.append(item)
        
        return items
    
    def _extract_header(self, result: StructuredResume, first_page_text: str):
        """Extract name and contact from header."""
        lines = first_page_text.split('\n')[:10]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', line)
            if email_match and not result.contact_info.get('email'):
                result.contact_info['email'] = email_match.group(0)
                continue
            
            phone_match = re.search(r'[\+]?[\d\s\-\(\)]{7,20}', line)
            if phone_match:
                phone = phone_match.group(0)
                if len(re.sub(r'\D', '', phone)) >= 7:
                    result.contact_info['phone'] = phone
                    continue
            
            if 'linkedin.com/in/' in line.lower():
                result.contact_info['linkedin'] = line
                continue
            
            if 'github.com/' in line.lower():
                result.contact_info['github'] = line
                continue
            
            if not result.name and len(line.split()) <= 4:
                if '@' not in line and 'http' not in line and not re.search(r'\d', line):
                    result.name = line


def safe_print(text):
    """Print text, handling encoding errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        clean = text.encode('ascii', 'replace').decode('ascii')
        print(clean)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid ML + Unified Resume Extraction")
    parser.add_argument("resume", help="Path to resume PDF")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--unified-only", action="store_true", help="Use unified extraction only")
    parser.add_argument("--compare", action="store_true", help="Compare hybrid vs unified extraction")
    
    args = parser.parse_args()
    
    pdf_path = Path(args.resume)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    safe_print(f"\n{'='*70}")
    safe_print(f"HYBRID RESUME EXTRACTION")
    safe_print(f"{'='*70}")
    safe_print(f"File: {pdf_path}")
    
    # Try unified extraction (always works, no ML dependencies)
    print("\n[1/2] Running Unified Extraction...")
    unified_extractor = UnifiedResumeExtractor()
    unified_result = unified_extractor.extract(pdf_path)
    unified_dict = unified_result.to_dict()
    unified_items = sum(len(s.get('items', [])) for s in unified_dict.get('sections', []))
    print(f"      Done: {len(unified_dict.get('sections', []))} sections, {unified_items} items")
    
    if args.unified_only:
        result = unified_result
        result_dict = unified_dict
    elif args.compare:
        # Also run hybrid for comparison
        print("\n[2/2] Running Hybrid Extraction (ML + Unified)...")
        try:
            extractor = HybridExtractor()
            hybrid_result = extractor.extract(pdf_path)
            hybrid_items = sum(len(s.items) for s in hybrid_result.sections)
            print(f"      Done: {len(hybrid_result.sections)} sections, {hybrid_items} items")
            
            # Compare results
            safe_print(f"\n{'='*70}")
            safe_print(f"COMPARISON")
            safe_print(f"{'='*70}")
            
            safe_print(f"\nUnified Extraction:")
            safe_print(f"  Sections: {len(unified_dict.get('sections', []))}")
            safe_print(f"  Items: {unified_items}")
            safe_print(f"  Name: {unified_dict.get('name', 'N/A')}")
            
            safe_print(f"\nHybrid Extraction:")
            safe_print(f"  Sections: {len(hybrid_result.sections)}")
            safe_print(f"  Items: {hybrid_items}")
            safe_print(f"  Name: {hybrid_result.name or 'N/A'}")
            
            if unified_items > hybrid_items:
                safe_print("\n[RESULT] Unified extracted more items")
            else:
                safe_print("\n[RESULT] Hybrid extracted more items")
            
            # Use the better result
            if hybrid_items > unified_items:
                result = hybrid_result
            else:
                result = unified_result
            result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
            
        except Exception as e:
            print(f"      Hybrid failed: {e}")
            print("      Using unified extraction result")
            result = unified_result
            result_dict = unified_dict
    else:
        # Try hybrid first, fall back to unified
        print("\n[2/2] Running Hybrid Extraction (ML + Unified)...")
        try:
            extractor = HybridExtractor()
            hybrid_result = extractor.extract(pdf_path)
            hybrid_items = sum(len(s.items) for s in hybrid_result.sections)
            print(f"      Done: {len(hybrid_result.sections)} sections, {hybrid_items} items")
            
            # Use hybrid if it has more items
            if hybrid_items >= unified_items:
                result = hybrid_result
            else:
                print("      Unified extraction has more items, using unified result")
                result = unified_result
        except Exception as e:
            print(f"      Hybrid failed: {e}")
            print("      Using unified extraction result")
            result = unified_result
        
        result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
    
    safe_print(f"\n{'='*70}")
    safe_print(f"FINAL RESULT")
    safe_print(f"{'='*70}")
    safe_print(f"Name: {result_dict.get('name', 'Not detected')}")
    safe_print(f"Contact: {result_dict.get('contact_info', {})}")
    sections = result_dict.get('sections', [])
    safe_print(f"Sections: {len(sections)}")
    total_items = sum(len(s.get('items', [])) for s in sections)
    safe_print(f"Total Items: {total_items}")
    
    # Save output
    output_path = Path(args.output) if args.output else pdf_path.parent / f"{pdf_path.stem}_hybrid.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    safe_print(f"\n[OK] Saved to: {output_path}")


if __name__ == "__main__":
    main()
