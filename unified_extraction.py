#!/usr/bin/env python3
"""
Unified Intelligent Resume Extraction

This module provides robust resume extraction by:
1. Using PyMuPDF's native text extraction with full font/spatial metadata
2. Analyzing font size hierarchy to detect document structure
3. Handling multi-column layouts through spatial clustering
4. Preserving ALL text content without loss
5. Intelligently grouping content into sections and items

Usage:
    python unified_extraction.py resume.pdf [--output unified.json]
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import fitz

sys.path.insert(0, str(Path(__file__).parent / "src"))


@dataclass
class TextSpan:
    """A span of text with font information."""
    text: str
    font_size: float
    font_name: str
    is_bold: bool
    color: int
    bbox: Tuple[float, float, float, float]


@dataclass
class TextLine:
    """A line of text composed of spans."""
    text: str
    spans: List[TextSpan]
    bbox: Tuple[float, float, float, float]
    y0: float
    y1: float
    x0: float
    x1: float

    @property
    def avg_font_size(self) -> float:
        if not self.spans:
            return 0
        return sum(s.font_size for s in self.spans) / len(self.spans)

    @property
    def is_bold(self) -> bool:
        return any(s.is_bold for s in self.spans)

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class TextBlock:
    """A block of text lines."""
    lines: List[TextLine]
    bbox: Tuple[float, float, float, float]
    page_num: int

    @property
    def text(self) -> str:
        return '\n'.join(line.text for line in self.lines if line.text.strip())

    @property
    def avg_font_size(self) -> float:
        if not self.lines:
            return 0
        total = sum(line.avg_font_size * len(line.spans) for line in self.lines if line.spans)
        total_spans = sum(len(line.spans) for line in self.lines if line.spans)
        return total / total_spans if total_spans > 0 else 0

    @property
    def is_bold(self) -> bool:
        return any(line.is_bold for line in self.lines)

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class ResumeItem:
    """An item within a section (job, degree, project, etc.)."""
    title: str = ""
    subtitle: str = ""
    date_range: str = ""
    location: str = ""
    company: str = ""
    description_lines: List[str] = field(default_factory=list)
    bullet_points: List[str] = field(default_factory=list)
    raw_lines: List[TextLine] = field(default_factory=list)

    def to_dict(self) -> Dict:
        description = '\n'.join(self.description_lines)
        return {
            'title': self.title,
            'subtitle': self.subtitle,
            'date_range': self.date_range,
            'location': self.location,
            'company': self.company,
            'description': description,
            'bullet_points': self.bullet_points,
        }


@dataclass
class ResumeSection:
    """A section in the resume."""
    title: str = ""
    section_type: str = "unknown"
    items: List[ResumeItem] = field(default_factory=list)
    header_block: Optional[TextBlock] = None
    content_blocks: List[TextBlock] = field(default_factory=list)
    raw_text: str = ""
    y_start: float = 0
    y_end: float = 0

    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'section_type': self.section_type,
            'items': [item.to_dict() for item in self.items],
            'raw_text': self.raw_text.strip() if self.raw_text else "",
        }


@dataclass
class StructuredResume:
    """Complete structured resume."""
    name: str = ""
    contact_info: Dict[str, str] = field(default_factory=dict)
    summary: str = ""
    sections: List[ResumeSection] = field(default_factory=list)
    all_text: str = ""

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'contact_info': self.contact_info,
            'summary': self.summary,
            'sections': [s.to_dict() for s in self.sections],
            'all_text': self.all_text,
        }


class FontHierarchyAnalyzer:
    """Analyze font hierarchy in a document."""

    def __init__(self, doc: fitz.Document):
        self.doc = doc
        self.font_sizes: List[float] = []
        self.size_frequencies: Dict[float, int] = defaultdict(int)
        self._analyze_fonts()

    def _analyze_fonts(self):
        """Collect all font sizes and their frequencies."""
        for page in self.doc:
            dict_output = page.get_text("dict")
            for block in dict_output.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = round(span.get("size", 0), 1)
                        if size > 0:
                            self.font_sizes.append(size)
                            self.size_frequencies[size] += 1

        self.font_sizes = sorted(set(self.font_sizes), reverse=True)

    def get_body_font_size(self) -> float:
        """Get the most common font size (body text)."""
        if not self.size_frequencies:
            return 10.0
        return max(self.size_frequencies.items(), key=lambda x: x[1])[0]

    def get_title_font_size(self) -> float:
        """Get the largest font size (typically name/header)."""
        if not self.font_sizes:
            return 18.0
        return self.font_sizes[0]

    def get_threshold(self, category: str) -> float:
        """Get font size threshold for a category."""
        body = self.get_body_font_size()
        title = self.get_title_font_size()

        thresholds = {
            'name': title * 0.9,
            'major_header': title * 0.75,
            'section_header': body * 1.15,
            'item_title': body * 1.05,
            'body': body,
        }
        return thresholds.get(category, body)

    def classify_font_size(self, size: float) -> str:
        """Classify a font size into a category."""
        body = self.get_body_font_size()
        title = self.get_title_font_size()

        if size >= self.get_threshold('name'):
            return 'name'
        elif size >= self.get_threshold('major_header'):
            return 'major_header'
        elif size >= self.get_threshold('section_header'):
            return 'section_header'
        elif size >= self.get_threshold('item_title'):
            return 'item_title'
        else:
            return 'body'


class ColumnDetector:
    """Detect and handle multi-column layouts."""

    def __init__(self, page_width: float):
        self.page_width = page_width

    def detect_columns(self, blocks: List[TextBlock]) -> List[Tuple[float, float]]:
        """Detect column boundaries based on block x-positions."""
        if not blocks:
            return [(0, self.page_width)]

        x_centers = [(b.bbox[0] + b.bbox[2]) / 2 for b in blocks]
        x0s = [b.bbox[0] for b in blocks]

        if not x_centers:
            return [(0, self.page_width)]

        min_x = min(x0s)
        max_x = max(b.bbox[2] for b in blocks)
        content_width = max_x - min_x

        if content_width < self.page_width * 0.4:
            return [(0, self.page_width)]

        mid_point = self.page_width / 2
        left_count = sum(1 for x in x_centers if x < mid_point - 30)
        right_count = sum(1 for x in x_centers if x > mid_point + 30)

        min_blocks = max(3, len(blocks) * 0.1)

        if left_count > min_blocks and right_count > min_blocks:
            gap = 30
            left_max = max(b.bbox[2] for b in blocks if b.bbox[2] < mid_point)
            right_min = min(b.bbox[0] for b in blocks if b.bbox[0] > mid_point)
            return [(0, left_max + gap), (right_min - gap, self.page_width)]

        return [(0, self.page_width)]

    def get_column(self, block: TextBlock, columns: List[Tuple[float, float]]) -> int:
        """Get the column index for a block."""
        center = (block.bbox[0] + block.bbox[2]) / 2
        for idx, (x_min, x_max) in enumerate(columns):
            if x_min <= center <= x_max:
                return idx
        return 0


class SectionClassifier:
    """Classify section types and detect headers."""

    SECTION_KEYWORDS = {
        'education': ['education', 'academic', 'qualifications', 'degrees', 'university', 'college', 'school', 'diploma'],
        'experience': ['experience', 'work', 'employment', 'professional', 'career', 'jobs', 'internship', 'position', 'role'],
        'skills': ['skills', 'technical', 'competencies', 'expertise', 'technologies', 'tools', 'techniques'],
        'projects': ['projects', 'portfolio', 'personal projects', 'academic projects'],
        'certifications': ['certifications', 'licenses', 'credentials', 'certificates'],
        'summary': ['summary', 'profile', 'objective', 'about', 'professional summary', 'introduction'],
        'contact': ['contact', 'personal information', 'personal details'],
        'languages': ['languages', 'linguistic', 'language skills', 'spoken languages'],
        'interests': ['interests', 'hobbies', 'activities', 'personal interests'],
        'awards': ['awards', 'honors', 'achievements', 'recognition', 'prizes'],
        'publications': ['publications', 'papers', 'research', 'articles'],
        'volunteer': ['volunteer', 'community', 'charity', 'social'],
    }

    DATE_PATTERNS = [
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b',
        r'\b\d{1,2}/\d{4}\b',
        r'\b\d{4}\s*[-–—to]+\s*(?:present|current|now|\d{4})\b',
        r'\bsince\s+\d{4}\b',
        r'\b\d{4}\b',
    ]

    SECTION_TITLE_PATTERNS = [
        r'^[A-Z][A-Z\s]{2,50}$',
        r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5}$',
    ]

    def is_section_header(self, block: TextBlock, font_analyzer: FontHierarchyAnalyzer) -> bool:
        """Check if a block is a section header."""
        text = block.text.strip()
        if len(text) > 80 or len(text) < 2:
            return False

        text_lower = text.lower()

        for section_type, keywords in self.SECTION_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                if len(text) < 60:
                    return True

        size_category = font_analyzer.classify_font_size(block.avg_font_size)
        if size_category in ['section_header', 'major_header', 'name']:
            if len(text) < 50:
                return True

        for pattern in self.SECTION_TITLE_PATTERNS:
            if re.match(pattern, text):
                if block.avg_font_size >= font_analyzer.get_threshold('section_header'):
                    return True

        return False

    def classify_section_type(self, text: str) -> str:
        """Classify the type of section from its title."""
        text_lower = text.lower()
        for section_type, keywords in self.SECTION_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return section_type
        return "unknown"

    def has_date(self, text: str) -> bool:
        """Check if text contains a date pattern."""
        for pattern in self.DATE_PATTERNS:
            if re.search(pattern, text, re.I):
                return True
        return False

    def extract_date(self, text: str) -> Tuple[str, str]:
        """Extract date from text and return (date, remaining_text)."""
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text, re.I)
            if match:
                date = match.group(0)
                remaining = text[:match.start()].strip() + text[match.end():].strip()
                return date, remaining
        return "", text


class ResumeItemParser:
    """Parse resume items (jobs, degrees, projects)."""

    DATE_PATTERNS = [
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b(?:\s*[-–—to]+\s*(?:present|current|\d{4}))?',
        r'\b\d{1,2}/\d{4}\b(?:\s*[-–—to]+\s*(?:present|current|\d{4}))?',
        r'\b\d{4}\s*[-–—to]+\s*(?:present|current|\d{4})\b',
        r'\b(?:Since|From)\s+\d{4}\b',
        r'\b(?:Since|From)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b',
        r'\b\d{4}\b',
    ]

    BULLET_MARKERS = ['•', '-', '*', '○', '◦', '▪', '▫', '→', '⇒', '➢', '✓', '✔', '●', '·', '›', '▪']

    LOCATION_PATTERN = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*(?:[A-Z]{2}|[A-Z][a-z]+))\b'

    def is_item_header(self, line: TextLine, section_type: str) -> bool:
        """Check if a line is an item header (job title, degree, project name)."""
        text = line.text.strip()

        if len(text) > 100:
            return False

        if line.is_bold and len(text) < 80:
            return True

        if self.has_date(text) and len(text) < 100:
            return True

        if section_type == 'education':
            if any(kw in text.lower() for kw in ['bachelor', 'master', 'phd', 'degree', 'diploma', 'engineer', 'institute', 'university']):
                return True

        if section_type == 'experience':
            if any(kw in text.lower() for kw in ['manager', 'developer', 'engineer', 'analyst', 'consultant', 'designer', 'lead', 'senior', 'junior']):
                return True

        if section_type == 'projects':
            if not any(kw in text.lower() for kw in ['project', 'portfolio']):
                return True

        return False

    def has_date(self, text: str) -> bool:
        """Check if text contains a date pattern."""
        for pattern in self.DATE_PATTERNS:
            if re.search(pattern, text, re.I):
                return True
        return False

    def is_bullet_point(self, line: TextLine) -> bool:
        """Check if a line is a bullet point."""
        text = line.text.strip()
        if any(text.startswith(marker) for marker in self.BULLET_MARKERS):
            return True
        if text.startswith('• '):
            return True
        return False

    def parse_item(self, lines: List[TextLine], section_type: str) -> ResumeItem:
        """Parse a list of lines into a ResumeItem."""
        if not lines:
            return ResumeItem()

        item = ResumeItem()
        item.raw_lines = lines

        all_text = '\n'.join(line.text for line in lines)

        for line in lines:
            text = line.text.strip()
            if not text:
                continue

            date, remaining = self.extract_date(text)
            if date:
                if not item.date_range:
                    item.date_range = date
                if remaining and not item.title:
                    self._parse_title_subtitle(remaining, item, section_type)
                continue

            if not item.title:
                if self.is_item_header(line, section_type):
                    self._parse_title_subtitle(text, item, section_type)
                    continue

            if self.is_bullet_point(line):
                bullet_text = text
                for marker in self.BULLET_MARKERS:
                    if bullet_text.startswith(marker):
                        bullet_text = bullet_text[len(marker):].strip()
                        break
                item.bullet_points.append(bullet_text)
            else:
                if text not in item.description_lines:
                    item.description_lines.append(text)

        if not item.title:
            first_meaningful = next((l.text.strip() for l in lines if l.text.strip()), "")
            if first_meaningful:
                self._parse_title_subtitle(first_meaningful, item, section_type)

        return item

    def _parse_title_subtitle(self, text: str, item: ResumeItem, section_type: str):
        """Parse text into title and subtitle."""
        lines = text.split('\n')
        if len(lines) >= 2:
            item.title = lines[0].strip()
            item.subtitle = lines[1].strip()
        else:
            item.title = text

        item.title = self._extract_location(item.title)
        item.subtitle = self._extract_location(item.subtitle) if item.subtitle else ""

    def _extract_location(self, text: str, item: Optional[ResumeItem] = None) -> str:
        """Extract location from text."""
        if not text:
            return text

        location_match = re.search(self.LOCATION_PATTERN, text)
        if location_match:
            if item:
                item.location = location_match.group(1)
            text = text[:location_match.start()].strip() + text[location_match.end():].strip()

        return text.strip()

    def extract_date(self, text: str) -> Tuple[str, str]:
        """Extract date from text."""
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text, re.I)
            if match:
                date = match.group(0)
                remaining = text[:match.start()].strip() + text[match.end():].strip()
                return date, remaining
        return "", text


class UnifiedResumeExtractor:
    """Main extractor class combining all intelligence."""

    def __init__(self):
        self.section_classifier = SectionClassifier()
        self.item_parser = ResumeItemParser()

    def extract(self, pdf_path: Path) -> StructuredResume:
        """Extract structured resume from PDF."""
        print(f"\n{'='*70}")
        print(f"UNIFIED RESUME EXTRACTION")
        print(f"{'='*70}")
        print(f"File: {pdf_path.name}\n")

        doc = fitz.open(str(pdf_path))
        structured = StructuredResume()

        font_analyzer = FontHierarchyAnalyzer(doc)

        all_blocks = []
        all_text_content = []

        for page_num in range(len(doc)):
            print(f"Processing page {page_num + 1}/{len(doc)}...")

            page = doc[page_num]
            page_blocks = self._extract_blocks(page, page_num)
            all_blocks.extend(page_blocks)

            page_text = '\n'.join(block.text for block in page_blocks if block.text.strip())
            all_text_content.append(page_text)

            print(f"  Extracted {len(page_blocks)} text blocks")

        page_width = doc[0].rect.width if len(doc) > 0 else 612
        doc.close()

        structured.all_text = '\n\n---PAGE---\n\n'.join(all_text_content)

        column_detector = ColumnDetector(page_width)
        columns = column_detector.detect_columns(all_blocks)
        print(f"Detected {len(columns)} column(s)")

        for block in all_blocks:
            block.column = column_detector.get_column(block, columns)

        all_blocks.sort(key=lambda b: (b.page_num, b.column, b.y0))

        structured.sections = self._extract_sections(all_blocks, font_analyzer)

        self._extract_header_info(structured, all_blocks)

        for section in structured.sections:
            self._extract_section_items(section)

        return structured

    def _extract_blocks(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """Extract text blocks with full font/spatial information."""
        blocks = []
        dict_output = page.get_text("dict")

        for block_data in dict_output.get("blocks", []):
            if "lines" not in block_data:
                continue

            lines = []
            block_bbox = block_data.get("bbox", [0, 0, 0, 0])

            for line_data in block_data["lines"]:
                spans = []
                line_text = ""
                line_bbox = line_data.get("bbox", [0, 0, 0, 0])

                for span_data in line_data.get("spans", []):
                    text = span_data.get("text", "")
                    if text.strip():
                        font_flags = span_data.get("flags", 0)
                        span = TextSpan(
                            text=text,
                            font_size=span_data.get("size", 10),
                            font_name=span_data.get("font", ""),
                            is_bold=bool(font_flags & 2**4),
                            color=span_data.get("color", 0),
                            bbox=tuple(span_data.get("bbox", [0, 0, 0, 0]))
                        )
                        spans.append(span)
                        line_text += text

                if spans:
                    line = TextLine(
                        text=line_text,
                        spans=spans,
                        bbox=tuple(line_bbox),
                        y0=line_bbox[1],
                        y1=line_bbox[3],
                        x0=line_bbox[0],
                        x1=line_bbox[2]
                    )
                    lines.append(line)

            if lines:
                block = TextBlock(
                    lines=lines,
                    bbox=tuple(block_bbox),
                    page_num=page_num
                )
                blocks.append(block)

        return blocks

    def _extract_sections(self, blocks: List[TextBlock], font_analyzer: FontHierarchyAnalyzer) -> List[ResumeSection]:
        """Extract sections from blocks."""
        sections = []
        current_section = None
        current_content = []

        for block in blocks:
            if not block.text.strip():
                continue

            if self.section_classifier.is_section_header(block, font_analyzer):
                if current_section:
                    current_section.content_blocks = current_content
                    current_section.raw_text = '\n'.join(b.text for b in current_content if b.text.strip())
                    sections.append(current_section)

                section_type = self.section_classifier.classify_section_type(block.text)
                current_section = ResumeSection(
                    title=block.text.strip(),
                    section_type=section_type,
                    header_block=block,
                    y_start=block.y0,
                )
                current_content = []
            elif current_section:
                current_content.append(block)
            else:
                current_section = ResumeSection(
                    title="Header",
                    section_type="header",
                    header_block=block,
                    y_start=block.y0,
                )
                current_content.append(block)

        if current_section:
            current_section.content_blocks = current_content
            current_section.raw_text = '\n'.join(b.text for b in current_content if b.text.strip())
            sections.append(current_section)

        return sections

    def _extract_header_info(self, structured: StructuredResume, blocks: List[TextBlock]):
        """Extract name and contact information."""
        header_blocks = [b for b in blocks if b.page_num == 0]
        header_blocks.sort(key=lambda b: b.y0)

        font_analyzer = FontHierarchyAnalyzer(fitz.open())

        name_candidates = []
        contact_blocks = []

        for block in header_blocks:
            text = block.text.strip()
            if not text:
                continue

            text_lower = text.lower()

            if block.avg_font_size >= font_analyzer.get_threshold('major_header') and len(text) < 60:
                name_candidates.append((block.avg_font_size, text))

            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
            if email_match:
                structured.contact_info['email'] = email_match.group(0)
                contact_blocks.append(text)

            phone_match = re.search(r'[\+]?[\d\s\-\(\)]{7,20}', text)
            if phone_match and len(phone_match.group(0)) >= 10:
                if 'phone' not in structured.contact_info:
                    structured.contact_info['phone'] = phone_match.group(0).strip()

            linkedin_match = re.search(r'linkedin\.com/in/[\w\-]+', text, re.I)
            if linkedin_match:
                structured.contact_info['linkedin'] = linkedin_match.group(0)

            github_match = re.search(r'github\.com/[\w\-]+', text, re.I)
            if github_match:
                structured.contact_info['github'] = github_match.group(0)

        if name_candidates:
            name_candidates.sort(key=lambda x: x[0], reverse=True)
            structured.name = name_candidates[0][1]

        if not structured.name and structured.sections:
            first_section = structured.sections[0]
            if first_section.section_type == 'unknown' and not first_section.items:
                potential_name = first_section.title
                if potential_name and len(potential_name) < 50 and not any(c in potential_name for c in ['@', '.com', 'http']):
                    structured.name = potential_name

        summary_blocks = [b for b in header_blocks if b.page_num == 0 and b not in contact_blocks]
        for block in summary_blocks:
            text = block.text.strip()
            if len(text) > 50 and len(text) < 600:
                if any(kw in text.lower() for kw in ['student', 'engineer', 'developer', 'experienced', 'professional', 'looking']):
                    structured.summary = text
                    break

    def _extract_section_items(self, section: ResumeSection):
        """Extract items from section content."""
        if not section.content_blocks:
            return

        all_lines = []
        for block in section.content_blocks:
            all_lines.extend(block.lines)

        if not all_lines:
            return

        current_item_lines = []
        items = []

        for line in all_lines:
            text = line.text.strip()
            if not text:
                continue

            is_header = self.item_parser.is_item_header(line, section.section_type)

            if is_header:
                if current_item_lines:
                    item = self.item_parser.parse_item(current_item_lines, section.section_type)
                    if item.title or item.bullet_points or item.description_lines:
                        items.append(item)
                current_item_lines = [line]
            else:
                current_item_lines.append(line)

        if current_item_lines:
            item = self.item_parser.parse_item(current_item_lines, section.section_type)
            if item.title or item.bullet_points or item.description_lines:
                items.append(item)

        section.items = items


def safe_print(text: str):
    """Print with Unicode error handling."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified intelligent resume extraction"
    )
    parser.add_argument("resume", help="Path to resume PDF")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    pdf_path = Path(args.resume)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    extractor = UnifiedResumeExtractor()

    try:
        structured = extractor.extract(pdf_path)

        safe_print(f"\n{'='*70}")
        safe_print(f"EXTRACTION SUMMARY")
        safe_print(f"{'='*70}\n")

        safe_print(f"Name: {structured.name or 'Not detected'}")
        safe_print(f"Contact: {structured.contact_info}")

        if structured.summary:
            safe_print(f"\nSummary: {structured.summary[:200]}{'...' if len(structured.summary) > 200 else ''}")

        total_items = sum(len(s.items) for s in structured.sections)
        safe_print(f"\nSections: {len(structured.sections)}")
        safe_print(f"Total Items: {total_items}")
        safe_print(f"Total Text Length: {len(structured.all_text)} chars\n")

        for i, section in enumerate(structured.sections, 1):
            safe_print(f"{i}. {section.title} ({section.section_type})")
            safe_print(f"   Items: {len(section.items)}")
            if args.verbose:
                for j, item in enumerate(section.items[:5], 1):
                    safe_print(f"     {j}. {item.title[:50]}{'...' if len(item.title) > 50 else ''}")
                    if item.date_range:
                        safe_print(f"        [{item.date_range}]")
                    if item.bullet_points:
                        safe_print(f"        Bullets: {len(item.bullet_points)}")
            if len(section.items) > 5:
                safe_print(f"     ... and {len(section.items) - 5} more")
            safe_print("")

        if args.verbose:
            safe_print(f"\n{'='*70}")
            safe_print(f"FULL EXTRACTED TEXT ({len(structured.all_text)} chars)")
            safe_print(f"{'='*70}")
            safe_print(structured.all_text[:2000])
            if len(structured.all_text) > 2000:
                safe_print(f"\n... and {len(structured.all_text) - 2000} more chars")

        output_path = Path(args.output) if args.output else pdf_path.parent / f"{pdf_path.stem}_unified.json"

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
