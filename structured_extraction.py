#!/usr/bin/env python3
"""
Structured Resume Extraction using ML Regions

This script extracts text from ML-detected regions and structures the resume
into sections, items, and bullet points for a clean hierarchical representation.

Usage:
    python structured_extraction.py resume.pdf [--output structured.json]
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "src"))

from parsers.ml_layout_detector import MLLayoutDetector
from parsers import extract_text_from_resume
import fitz
from PIL import Image
import numpy as np


@dataclass
class BulletPoint:
    """A bullet point within an item."""
    text: str
    level: int = 0  # Indentation level
    has_metric: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ResumeItem:
    """An item within a section (e.g., a degree, job position)."""
    title: str
    subtitle: str = ""
    date_range: str = ""
    location: str = ""
    description: str = ""
    bullet_points: List[BulletPoint] = field(default_factory=list)
    raw_text: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'subtitle': self.subtitle,
            'date_range': self.date_range,
            'location': self.location,
            'description': self.description,
            'bullet_points': [bp.to_dict() for bp in self.bullet_points],
            'raw_text': self.raw_text
        }


@dataclass
class ResumeSection:
    """A section in the resume (e.g., Education, Experience)."""
    title: str
    section_type: str  # education, experience, skills, etc.
    items: List[ResumeItem] = field(default_factory=list)
    raw_text: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'section_type': self.section_type,
            'items': [item.to_dict() for item in self.items],
            'raw_text': self.raw_text
        }


@dataclass
class StructuredResume:
    """Complete structured resume."""
    contact_info: Dict[str, str] = field(default_factory=dict)
    summary: str = ""
    sections: List[ResumeSection] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    raw_regions: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'contact_info': self.contact_info,
            'summary': self.summary,
            'sections': [section.to_dict() for section in self.sections],
            'skills': self.skills,
        }


class RegionTextExtractor:
    """Extract text from specific regions of an image."""
    
    def __init__(self):
        """Initialize PaddleOCR for text extraction."""
        self.ocr = None
        self.available = False
        
        try:
            from paddleocr import PaddleOCR
            # Try different initialization options
            try:
                self.ocr = PaddleOCR(
                    use_textline_orientation=True,
                    lang='en'
                )
            except:
                # Fallback to basic initialization
                self.ocr = PaddleOCR(lang='en')
            self.available = True
        except Exception as e:
            print(f"Warning: PaddleOCR not available: {e}")
    
    def extract_from_region(self, image: Image.Image, bbox: List[float]) -> str:
        """Extract text from a specific region of an image.
        
        Args:
            image: Full page image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Extracted text
        """
        if not self.available or self.ocr is None:
            return ""
        
        # Crop the region
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.width, x2)
        y2 = min(image.height, y2)
        
        if x2 <= x1 or y2 <= y1:
            return ""
        
        region_img = image.crop((x1, y1, x2, y2))
        
        # Convert to numpy array
        img_array = np.array(region_img)
        
        # Run OCR
        try:
            result = self.ocr.ocr(img_array)
            
            # Extract text from result
            texts = []
            if result and result[0]:
                for line in result[0]:
                    if line:
                        text = line[1][0]  # Get the text content
                        texts.append(text)
            
            return ' '.join(texts)
        except Exception as e:
            print(f"OCR error: {e}")
            return ""


class StructuredResumeExtractor:
    """Extract structured information from resume using ML regions."""
    
    # Section title patterns
    SECTION_PATTERNS = {
        'education': r'\b(education|academic|qualifications|degrees?|university|college)\b',
        'experience': r'\b(experience|work|employment|professional|career|jobs?)\b',
        'skills': r'\b(skills|technical|competencies|expertise|technologies)\b',
        'projects': r'\b(projects?|portfolio)\b',
        'certifications': r'\b(certifications?|licenses?|credentials?)\b',
        'summary': r'\b(summary|profile|objective|about)\b',
        'contact': r'\b(contact|personal)\b',
        'languages': r'\b(languages?|linguistic)\b',
        'interests': r'\b(interests?|hobbies|activities)\b',
        'awards': r'\b(awards?|honors?|achievements?)\b',
    }
    
    # Date patterns
    DATE_PATTERN = r'\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}|\d{1,2}/\d{4}|\d{4}\s*-\s*(?:present|\d{4}|current)|since\s+\d{4})\b'
    
    def __init__(self):
        """Initialize extractor."""
        self.ml_detector = MLLayoutDetector()
        self.text_extractor = RegionTextExtractor()
    
    def extract(self, pdf_path: Path) -> StructuredResume:
        """Extract structured resume from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            StructuredResume object
        """
        if not self.ml_detector.is_available():
            raise RuntimeError("ML LayoutDetection not available")
        
        print(f"\nExtracting structured data from: {pdf_path.name}")
        print("="*70)
        
        # First, extract all text from PDF using PyMuPDF
        print("\nExtracting text from PDF...")
        from parsers import extract_text_from_resume
        extraction_result = extract_text_from_resume(pdf_path, use_ocr=False)
        full_text = extraction_result["full_text"]
        
        # Convert PDF to images and detect regions
        doc = fitz.open(str(pdf_path))
        
        structured = StructuredResume()
        all_regions = []
        
        for page_num in range(len(doc)):
            print(f"\nProcessing page {page_num + 1}...")
            
            page = doc[page_num]
            mat = fitz.Matrix(150/72, 150/72)
            pix = page.get_pixmap(matrix=mat)
            
            # Save temp image
            temp_path = Path(f"temp_page_{page_num}.png")
            pix.save(str(temp_path))
            
            # Load image for text extraction
            page_image = Image.open(temp_path)
            
            # Detect regions
            regions = self.ml_detector.detect_layout(str(temp_path))
            all_regions.extend(regions)
            
            print(f"  Found {len(regions)} regions")
            
            # Extract text from each region
            for region in regions:
                bbox = region.get('bbox', [])
                if bbox and len(bbox) >= 4:
                    # Extract text from PDF page using bbox coordinates (faster than OCR)
                    text = self._extract_text_from_pdf_region(page, bbox)
                    # Only try OCR if no text found and OCR is available
                    if not text and self.text_extractor.available:
                        text = self.text_extractor.extract_from_region(page_image, bbox)
                    region['extracted_text'] = text
            
            # Group regions into sections
            page_sections = self._group_into_sections(regions)
            structured.sections.extend(page_sections)
            
            # Cleanup - close image first, then delete file
            page_image.close()
            temp_path.unlink()
        
        doc.close()
        
        structured.raw_regions = all_regions
        
        # Post-process sections
        self._post_process_sections(structured)
        
        return structured
    
    def _extract_text_from_pdf_region(self, page: fitz.Page, bbox: List[float]) -> str:
        """Extract text from a specific region of a PDF page using PyMuPDF.
        
        Args:
            page: fitz Page object
            bbox: Bounding box [x1, y1, x2, y2] in pixels
            
        Returns:
            Extracted text
        """
        try:
            # Convert pixels to points (PDF units)
            # We used 150 DPI for conversion, so 1 pixel = 72/150 points
            scale = 72 / 150
            rect = fitz.Rect(
                bbox[0] * scale,
                bbox[1] * scale,
                bbox[2] * scale,
                bbox[3] * scale
            )
            
            # Extract text from the rectangle
            text = page.get_text("text", clip=rect)
            if isinstance(text, str):
                return text.strip()
            elif isinstance(text, (list, tuple)):
                return ' '.join(str(t) for t in text).strip()
            else:
                return str(text).strip()
        except Exception as e:
            return ""
    
    def _group_into_sections(self, regions: List[Dict]) -> List[ResumeSection]:
        """Group regions into sections based on titles and spatial layout."""
        # Sort regions by y position
        sorted_regions = sorted(regions, key=lambda r: r.get('bbox', [0, 99999, 0, 0])[1])
        
        sections = []
        current_section = None
        current_items = []
        
        for region in sorted_regions:
            region_type = region.get('type', 'unknown')
            text = region.get('extracted_text', '')
            
            if not text:
                continue
            
            # Check if this is a section title
            if region_type == 'title' or self._is_section_header(text):
                # Save previous section
                if current_section:
                    current_section.items = current_items
                    sections.append(current_section)
                
                # Start new section
                section_type = self._classify_section_type(text)
                current_section = ResumeSection(
                    title=text,
                    section_type=section_type,
                    raw_text=text
                )
                current_items = []
            
            elif current_section:
                # This is content within the current section
                if self._is_bullet_point(region, text):
                    # Add to last item as bullet
                    if current_items:
                        bullet = BulletPoint(
                            text=text,
                            has_metric=self._has_metric(text)
                        )
                        current_items[-1].bullet_points.append(bullet)
                else:
                    # This might be a new item
                    item = self._parse_item(text, region)
                    current_items.append(item)
        
        # Don't forget the last section
        if current_section:
            current_section.items = current_items
            sections.append(current_section)
        
        return sections
    
    def _is_section_header(self, text: str) -> bool:
        """Check if text looks like a section header."""
        # Headers are typically short, single line, and uppercase or title case
        if len(text) > 50:
            return False
        
        # Check for section keywords
        text_lower = text.lower()
        for pattern in self.SECTION_PATTERNS.values():
            if re.search(pattern, text_lower):
                return True
        
        # Check if mostly uppercase (common for headers)
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.6:
            return True
        
        return False
    
    def _classify_section_type(self, text: str) -> str:
        """Classify the type of section from its title."""
        text_lower = text.lower()
        
        for section_type, pattern in self.SECTION_PATTERNS.items():
            if re.search(pattern, text_lower):
                return section_type
        
        return "unknown"
    
    def _is_bullet_point(self, region: Dict, text: str) -> bool:
        """Check if region is a bullet point."""
        # Check for bullet markers
        bullet_markers = ['•', '-', '*', '○', '◦', '▪', '▫', '→', '⇒', '➢', '✓', '✔']
        if any(text.strip().startswith(marker) for marker in bullet_markers):
            return True
        
        # Check region type
        if region.get('type') == 'list':
            return True
        
        # Check indentation (bullet points are typically indented)
        bbox = region.get('bbox', [])
        if len(bbox) >= 4:
            x1 = bbox[0]
            # If significantly indented (>50px), might be bullet
            if x1 > 100:
                return True
        
        return False
    
    def _has_metric(self, text: str) -> bool:
        """Check if text contains a metric or number."""
        # Look for percentages
        if re.search(r'\d+%', text):
            return True
        
        # Look for numbers with units
        if re.search(r'\d+\s*(?:k|m|b|million|billion|users|customers|team members?|people)', text, re.I):
            return True
        
        # Look for time periods
        if re.search(r'\d+\s*(?:years?|months?|weeks?|days?)', text, re.I):
            return True
        
        # Look for dollar amounts
        if re.search(r'\$[\d,]+(?:k|m)?', text, re.I):
            return True
        
        return False
    
    def _parse_item(self, text: str, region: Dict) -> ResumeItem:
        """Parse an item (degree, job, etc.) from text."""
        item = ResumeItem(title=text, raw_text=text)
        
        # Try to extract date range
        date_match = re.search(self.DATE_PATTERN, text, re.I)
        if date_match:
            item.date_range = date_match.group(1)
            # Remove date from title
            item.title = text[:date_match.start()].strip()
        
        # Try to extract location (common pattern: City, Country or City, State)
        location_match = re.search(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?,\s*(?:France|USA|UK|Germany|Paris|Lyon|London|New York))\b', text)
        if location_match:
            item.location = location_match.group(1)
        
        # Try to split title and subtitle
        lines = text.split('\n')
        if len(lines) >= 2:
            item.title = lines[0].strip()
            item.subtitle = lines[1].strip()
        
        return item
    
    def _post_process_sections(self, structured: StructuredResume):
        """Post-process extracted sections."""
        # Try to find contact info from first section or top regions
        for section in structured.sections:
            if section.section_type == 'contact' or section.section_type == 'unknown':
                # Look for email, phone, etc.
                for item in section.items:
                    text = item.raw_text
                    
                    # Email
                    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
                    if email_match and not structured.contact_info.get('email'):
                        structured.contact_info['email'] = email_match.group(0)
                    
                    # Phone
                    phone_match = re.search(r'[\+]?[\d\s\-\(\)]{7,20}', text)
                    if phone_match and not structured.contact_info.get('phone'):
                        structured.contact_info['phone'] = phone_match.group(0)
                    
                    # LinkedIn
                    linkedin_match = re.search(r'linkedin\.com/in/[\w\-]+', text, re.I)
                    if linkedin_match and not structured.contact_info.get('linkedin'):
                        structured.contact_info['linkedin'] = linkedin_match.group(0)
                
                break
        
        # Extract skills if there's a skills section
        for section in structured.sections:
            if section.section_type == 'skills':
                for item in section.items:
                    # Split by common delimiters
                    skills_text = item.raw_text
                    skills = re.split(r'[,•\|\n]', skills_text)
                    structured.skills.extend([s.strip() for s in skills if len(s.strip()) > 1])


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract structured resume data using ML regions"
    )
    parser.add_argument("resume", help="Path to resume PDF")
    parser.add_argument("--output", "-o", help="Output JSON file")
    
    args = parser.parse_args()
    
    pdf_path = Path(args.resume)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    # Extract structured data
    extractor = StructuredResumeExtractor()
    
    try:
        structured = extractor.extract(pdf_path)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"EXTRACTION SUMMARY")
        print(f"{'='*70}\n")
        
        print(f"Contact Info:")
        for key, value in structured.contact_info.items():
            print(f"  {key}: {value}")
        
        print(f"\nSections: {len(structured.sections)}")
        for i, section in enumerate(structured.sections, 1):
            print(f"\n{i}. {section.title} ({section.section_type})")
            print(f"   Items: {len(section.items)}")
            for j, item in enumerate(section.items[:3], 1):  # Show first 3
                print(f"     {j}. {item.title}")
                if item.date_range:
                    print(f"        Date: {item.date_range}")
                if item.bullet_points:
                    print(f"        Bullets: {len(item.bullet_points)}")
            if len(section.items) > 3:
                print(f"     ... and {len(section.items) - 3} more items")
        
        if structured.skills:
            print(f"\nSkills: {len(structured.skills)} found")
            print(f"  {', '.join(structured.skills[:10])}")
            if len(structured.skills) > 10:
                print(f"  ... and {len(structured.skills) - 10} more")
        
        # Create output directory based on resume name
        output_dir = pdf_path.parent / pdf_path.stem
        output_dir.mkdir(exist_ok=True)
        
        # Save to JSON
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = output_dir / "structured.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"✓ Structured data saved to: {output_path}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
