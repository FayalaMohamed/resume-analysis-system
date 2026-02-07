#!/usr/bin/env python3
"""
Comprehensive Resume Analysis Pipeline

This script runs ALL variants of every system component on a single resume
and outputs detailed comparisons for analysis and debugging.

Usage:
    python pipeline.py <resume.pdf> [--output results.json] [--job job_description.txt]
"""

import json
import sys
import time
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict, is_dataclass
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).parent / "src"))

from parsers import (
    extract_text_from_resume,
    LayoutDetector,
    MLLayoutDetector,
    HeuristicLayoutDetector,
    SectionParser,
    LanguageDetector,
    LAYOUT_DETECTION_AVAILABLE,
)
from analysis import (
    ContentAnalyzer,
    analyze_content,
    JobDescriptionParser,
    ResumeJobMatcher,
    parse_job_description,
    match_resume_to_job,
    AdvancedJobMatcher,
    match_resume_to_job_advanced,
    ATSSimulator,
    simulate_ats_parsing,
    RecommendationEngine,
)
from scoring import ATSScorer
from utils import get_logger

logger = get_logger(__name__)


class Colors:
    """Terminal colors for beautiful output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class TerminalCapture:
    """Capture terminal output with ANSI colors."""
    
    def __init__(self):
        self.buffer = io.StringIO()
        self.original_stdout = sys.stdout
        self.capturing = False
    
    def start(self):
        """Start capturing stdout."""
        self.capturing = True
        sys.stdout = self
    
    def stop(self):
        """Stop capturing and restore stdout."""
        self.capturing = False
        sys.stdout = self.original_stdout
    
    def write(self, text):
        """Write to both buffer and original stdout."""
        self.buffer.write(text)
        self.original_stdout.write(text)
    
    def flush(self):
        """Flush both outputs."""
        self.buffer.flush()
        self.original_stdout.flush()
    
    def getvalue(self):
        """Get captured content."""
        return self.buffer.getvalue()
    
    def save_to_file(self, filepath: Path):
        """Save captured content to file."""
        content = self.getvalue()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)


def print_section(title: str, color: str = Colors.HEADER):
    """Print a formatted section header."""
    width = 70
    print(f"\n{color}{Colors.BOLD}{'='*width}{Colors.END}")
    print(f"{color}{Colors.BOLD}{title.center(width)}{Colors.END}")
    print(f"{color}{Colors.BOLD}{'='*width}{Colors.END}")


def print_subsection(title: str, color: str = Colors.CYAN):
    """Print a subsection header."""
    print(f"\n{color}{Colors.BOLD}▶ {title}{Colors.END}")
    print(f"{color}{'─' * (len(title) + 3)}{Colors.END}")


def print_key_value(key: str, value: Any, indent: int = 0):
    """Print a key-value pair with formatting."""
    prefix = "  " * indent
    if isinstance(value, bool):
        color = Colors.GREEN if value else Colors.RED
        val_str = f"{color}✓ YES{Colors.END}" if value else f"{color}✗ NO{Colors.END}"
    elif isinstance(value, (int, float)):
        if isinstance(value, float) and 0 <= value <= 1:
            # Probability/percentage
            pct = value * 100
            if pct >= 80:
                color = Colors.GREEN
            elif pct >= 50:
                color = Colors.YELLOW
            else:
                color = Colors.RED
            val_str = f"{color}{value:.2%}{Colors.END}"
        else:
            val_str = f"{Colors.CYAN}{value}{Colors.END}"
    elif isinstance(value, str):
        val_str = f"{Colors.YELLOW}{value}{Colors.END}"
    elif isinstance(value, list):
        if len(value) == 0:
            val_str = f"{Colors.RED}(empty){Colors.END}"
        else:
            val_str = f"{Colors.CYAN}{len(value)} items{Colors.END}"
    elif value is None:
        val_str = f"{Colors.RED}N/A{Colors.END}"
    else:
        val_str = f"{Colors.CYAN}{value}{Colors.END}"
    
    print(f"{prefix}{Colors.BOLD}{key}:{Colors.END} {val_str}")


def print_comparison(name1: str, val1: Any, name2: str, val2: Any, label: str = ""):
    """Print a comparison between two values."""
    match = val1 == val2
    color = Colors.GREEN if match else Colors.YELLOW
    symbol = "✓" if match else "⚠"
    
    print(f"\n  {Colors.BOLD}{label or 'Comparison'}:{Colors.END} {color}{symbol}{Colors.END}")
    print(f"    {Colors.BLUE}{name1}:{Colors.END} {val1}")
    print(f"    {Colors.BLUE}{name2}:{Colors.END} {val2}")


def serialize_for_json(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if is_dataclass(obj):
        return {k: serialize_for_json(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, (set, tuple)):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, 'isoformat'):  # datetime
        return obj.isoformat()
    else:
        return obj


def extract_text_variants(pdf_path: Path) -> Dict[str, Any]:
    """Run all text extraction variants."""
    print_section("1. TEXT EXTRACTION VARIANTS", Colors.BLUE)
    
    results = {}
    
    # Variant 1: PyMuPDF (default)
    print_subsection("Method 1: PyMuPDF (Default)")
    start = time.time()
    try:
        result = extract_text_from_resume(pdf_path, use_ocr=False)
        results["pymupdf"] = {
            "success": True,
            "duration_ms": (time.time() - start) * 1000,
            "text_length": len(result["full_text"]),
            "num_pages": result["num_pages"],
            "full_text": result["full_text"],
        }
        print_key_value("Text Length", len(result["full_text"]))
        print_key_value("Pages", result["num_pages"])
        print_key_value("Duration", f"{(time.time() - start)*1000:.1f}ms")
    except Exception as e:
        results["pymupdf"] = {"success": False, "error": str(e)}
        print(f"{Colors.RED}✗ Failed: {e}{Colors.END}")
    
    # Variant 2: OCR
    print_subsection("Method 2: OCR (PaddleOCR)")
    start = time.time()
    try:
        result = extract_text_from_resume(pdf_path, use_ocr=True)
        results["ocr"] = {
            "success": True,
            "duration_ms": (time.time() - start) * 1000,
            "text_length": len(result["full_text"]),
            "num_pages": result["num_pages"],
            "full_text": result["full_text"],
        }
        print_key_value("Text Length", len(result["full_text"]))
        print_key_value("Pages", result["num_pages"])
        print_key_value("Duration", f"{(time.time() - start)*1000:.1f}ms")
    except Exception as e:
        results["ocr"] = {"success": False, "error": str(e)}
        print(f"{Colors.RED}✗ Failed: {e}{Colors.END}")
    
    # Comparison
    if results["pymupdf"].get("success") and results["ocr"].get("success"):
        print_comparison(
            "PyMuPDF", results["pymupdf"]["text_length"],
            "OCR", results["ocr"]["text_length"],
            "Text Length Comparison"
        )
    
    return results


def detect_language_variants(text: str) -> Dict[str, Any]:
    """Run all language detection variants."""
    print_section("2. LANGUAGE DETECTION VARIANTS", Colors.BLUE)
    
    results = {}
    
    # Variant 1: Auto-detection
    print_subsection("Method 1: Auto-Detection")
    start = time.time()
    try:
        lang = LanguageDetector.detect(text, default='en')
        lang_name = LanguageDetector.get_language_name(lang)
        results["auto"] = {
            "success": True,
            "detected_code": lang,
            "language_name": lang_name,
            "duration_ms": (time.time() - start) * 1000,
        }
        print_key_value("Detected", f"{lang_name} ({lang})")
        print_key_value("Duration", f"{(time.time() - start)*1000:.1f}ms")
    except Exception as e:
        results["auto"] = {"success": False, "error": str(e)}
        print(f"{Colors.RED}✗ Failed: {e}{Colors.END}")
    
    # Test specific languages
    print_subsection("Method 2: Forced Language Codes")
    languages = ['en', 'fr', 'es', 'de', 'it', 'pt']
    forced_results = {}
    for lang in languages:
        try:
            lang_name = LanguageDetector.get_language_name(lang)
            forced_results[lang] = lang_name
            print(f"  {Colors.BLUE}{lang}:{Colors.END} {lang_name}")
        except Exception as e:
            forced_results[lang] = f"Error: {e}"
    results["forced_languages"] = forced_results
    
    return results


def analyze_layout_variants(text: str, pdf_path: Path, lang_code: str) -> Dict[str, Any]:
    """Run all layout detection variants."""
    print_section("3. LAYOUT DETECTION VARIANTS", Colors.BLUE)
    
    results = {}
    
    # Variant 1: Full ML Detection
    print_subsection("Method 1: ML-Based (PaddleOCR LayoutDetection)")
    start = time.time()
    ml_detector = MLLayoutDetector(lang=lang_code)
    
    if not ml_detector.is_available():
        print(f"{Colors.YELLOW}⚠ ML LayoutDetection not available{Colors.END}")
        results["ml"] = {"success": False, "error": "ML LayoutDetection not available"}
    else:
        try:
            # Convert PDF to images
            import fitz
            temp_dir = Path(pdf_path).parent / ".temp_layout"
            temp_dir.mkdir(exist_ok=True)
            
            doc = fitz.open(str(pdf_path))
            image_paths = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                mat = fitz.Matrix(150/72, 150/72)
                pix = page.get_pixmap(matrix=mat)
                image_path = temp_dir / f"{pdf_path.stem}_layout_page_{page_num + 1}.png"
                pix.save(str(image_path))
                image_paths.append(str(image_path))
            doc.close()
            
            try:
                all_regions = []
                page_width = 612
                
                for image_path in image_paths:
                    regions = ml_detector.detect_layout(image_path)
                    all_regions.extend(regions)
                    try:
                        from PIL import Image
                        with Image.open(image_path) as img:
                            page_width = img.width
                    except:
                        pass
                
                column_result = ml_detector.analyze_columns(all_regions, page_width)
                table_result = ml_detector.analyze_tables(all_regions)
                image_result = ml_detector.analyze_images(all_regions)
                
                # Count region types
                region_breakdown = {}
                for region in all_regions:
                    rtype = region.get("type", "unknown")
                    region_breakdown[rtype] = region_breakdown.get(rtype, 0) + 1
                
                results["ml"] = {
                    "success": True,
                    "duration_ms": (time.time() - start) * 1000,
                    "is_single_column": column_result["is_single_column"],
                    "num_columns": column_result["num_columns"],
                    "column_confidence": column_result.get("confidence"),
                    "has_tables": table_result["has_tables"],
                    "table_count": table_result["table_count"],
                    "table_confidence": table_result.get("confidence"),
                    "has_images": image_result["has_images"],
                    "image_count": image_result["image_count"],
                    "total_regions": len(all_regions),
                    "region_breakdown": region_breakdown,
                    "raw_regions": all_regions[:10],  # First 10 for detail
                }
                
                print_key_value("Single Column", column_result["is_single_column"])
                print_key_value("Columns Detected", column_result["num_columns"])
                print_key_value("Column Confidence", column_result.get("confidence"))
                print_key_value("Has Tables", table_result["has_tables"])
                print_key_value("Table Count", table_result["table_count"])
                print_key_value("Has Images", image_result["has_images"])
                print_key_value("Total Regions", len(all_regions))
                print_key_value("Region Breakdown", region_breakdown)
                print_key_value("Duration", f"{(time.time() - start)*1000:.1f}ms")
                
            finally:
                # Cleanup
                for img_path in image_paths:
                    try:
                        Path(img_path).unlink()
                    except:
                        pass
                try:
                    temp_dir.rmdir()
                except:
                    pass
                    
        except Exception as e:
            results["ml"] = {"success": False, "error": str(e)}
            print(f"{Colors.RED}✗ Failed: {e}{Colors.END}")
    
    # Variant 2: Heuristic Detection
    print_subsection("Method 2: Heuristic-Based (Pattern Analysis)")
    start = time.time()
    try:
        heuristic = HeuristicLayoutDetector(language=lang_code)
        
        column_info = heuristic.detect_columns(text)
        has_tables = heuristic.detect_tables(text)
        section_headers = heuristic.detect_section_headers(text, lang_code)
        text_density = heuristic.calculate_text_density(text)
        
        lines = [line for line in text.split('\n') if line.strip()]
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        
        results["heuristic"] = {
            "success": True,
            "duration_ms": (time.time() - start) * 1000,
            "is_single_column": column_info["is_single_column"],
            "num_columns": column_info["num_columns"],
            "indent_variance": column_info.get("indent_variance"),
            "has_tables": has_tables,
            "section_headers_found": len(section_headers),
            "section_headers": section_headers[:15],
            "text_density": text_density,
            "avg_line_length": avg_line_length,
            "total_lines": len(lines),
        }
        
        print_key_value("Single Column", column_info["is_single_column"])
        print_key_value("Columns Detected", column_info["num_columns"])
        print_key_value("Indent Variance", column_info.get("indent_variance"))
        print_key_value("Has Tables", has_tables)
        print_key_value("Section Headers Found", len(section_headers))
        print_key_value("Text Density", f"{text_density:.1f}")
        print_key_value("Duration", f"{(time.time() - start)*1000:.1f}ms")
        
    except Exception as e:
        results["heuristic"] = {"success": False, "error": str(e)}
        print(f"{Colors.RED}✗ Failed: {e}{Colors.END}")
    
    # Variant 3: Combined/Auto Selection
    print_subsection("Method 3: Auto-Select (ML preferred, Heuristic fallback)")
    start = time.time()
    try:
        detector = LayoutDetector(language=lang_code, use_ml=True)
        layout_features = detector.analyze_layout(text, pdf_path=str(pdf_path), lang_code=lang_code)
        layout_summary = detector.get_layout_summary(text, pdf_path=str(pdf_path))
        
        results["auto"] = {
            "success": True,
            "duration_ms": (time.time() - start) * 1000,
            "method_used": layout_features.detection_method,
            "is_single_column": layout_features.is_single_column,
            "num_columns": layout_features.num_columns,
            "has_tables": layout_features.has_tables,
            "has_images": layout_features.has_images,
            "section_headers": layout_features.section_headers[:15],
            "risk_score": layout_features.layout_risk_score,
            "confidence": layout_features.confidence,
            "summary": layout_summary,
        }
        
        print_key_value("Method Used", layout_features.detection_method)
        print_key_value("Single Column", layout_features.is_single_column)
        print_key_value("Has Tables", layout_features.has_tables)
        print_key_value("Has Images", layout_features.has_images)
        print_key_value("Risk Score", layout_features.layout_risk_score)
        print_key_value("Duration", f"{(time.time() - start)*1000:.1f}ms")
        
    except Exception as e:
        results["auto"] = {"success": False, "error": str(e)}
        print(f"{Colors.RED}✗ Failed: {e}{Colors.END}")
    
    # Comparison
    if results.get("ml", {}).get("success") and results.get("heuristic", {}).get("success"):
        print_subsection("Comparison: ML vs Heuristic", Colors.YELLOW)
        
        ml_res = results["ml"]
        heu_res = results["heuristic"]
        
        agreements = []
        disagreements = []
        
        if ml_res["is_single_column"] == heu_res["is_single_column"]:
            agreements.append("column_layout")
        else:
            disagreements.append({
                "aspect": "columns",
                "ml": f"{ml_res['num_columns']}-column",
                "heuristic": f"{heu_res['num_columns']}-column"
            })
        
        if ml_res["has_tables"] == heu_res["has_tables"]:
            agreements.append("tables")
        else:
            disagreements.append({
                "aspect": "tables",
                "ml": f"detected={ml_res['has_tables']}",
                "heuristic": f"detected={heu_res['has_tables']}"
            })
        
        print_key_value("Agreements", agreements)
        if disagreements:
            print(f"{Colors.YELLOW}Disagreements:{Colors.END}")
            for d in disagreements:
                print(f"  • {d['aspect']}: ML={d['ml']}, Heuristic={d['heuristic']}")
        
        results["comparison"] = {
            "agreements": agreements,
            "disagreements": disagreements,
            "agreement_rate": len(agreements) / (len(agreements) + len(disagreements)),
        }
    
    return results


def parse_sections_variants(text: str, lang_code: str) -> Dict[str, Any]:
    """Run section parsing with different configurations."""
    print_section("4. SECTION PARSING VARIANTS", Colors.BLUE)
    
    results = {}
    
    print_subsection(f"Parsing with Language: {lang_code}")
    start = time.time()
    try:
        parser = SectionParser(language=lang_code)
        parsed = parser.parse(text)
        
        results["standard"] = {
            "success": True,
            "duration_ms": (time.time() - start) * 1000,
            "sections_found": len(parsed.sections),
            "section_names": [s["name"] for s in parsed.sections],
            "section_types": [s["section_type"] for s in parsed.sections],
            "contact_info": parsed.contact_info,
            "skills_count": len(parsed.skills),
            "languages_count": len(parsed.languages),
            "certifications_count": len(parsed.certifications),
            "projects_count": len(parsed.projects),
            "awards_count": len(parsed.awards),
            "publications_count": len(parsed.publications),
        }
        
        print_key_value("Sections Found", len(parsed.sections))
        print_key_value("Section Names", [s["name"] for s in parsed.sections])
        print_key_value("Contact Info", parsed.contact_info)
        print_key_value("Skills Count", len(parsed.skills))
        print_key_value("Duration", f"{(time.time() - start)*1000:.1f}ms")
        
    except Exception as e:
        results["standard"] = {"success": False, "error": str(e)}
        print(f"{Colors.RED}✗ Failed: {e}{Colors.END}")
    
    return results


def analyze_content_variants(text: str) -> Dict[str, Any]:
    """Run content analysis."""
    print_section("5. CONTENT ANALYSIS", Colors.BLUE)
    
    results = {}
    
    print_subsection("Content Quality Analysis")
    start = time.time()
    try:
        analyzer = ContentAnalyzer()
        score = analyzer.analyze(text)
        
        results["quality"] = {
            "success": True,
            "duration_ms": (time.time() - start) * 1000,
            "overall_score": score.overall_score,
            "action_verb_score": score.action_verb_score,
            "quantification_score": score.quantification_score,
            "bullet_structure_score": score.bullet_structure_score,
            "conciseness_score": score.conciseness_score,
            "strong_verbs_count": len(score.action_verbs_found),
            "weak_verbs_count": len(score.weak_verbs_found),
            "quantified_achievements": len(score.quantified_achievements),
            "bullet_points_analyzed": len(score.bullet_points),
            "recommendations_count": len(score.recommendations),
            "recommendations": score.recommendations[:10],
        }
        
        print_key_value("Overall Score", score.overall_score)
        print_key_value("Action Verb Score", score.action_verb_score)
        print_key_value("Quantification Score", score.quantification_score)
        print_key_value("Bullet Structure Score", score.bullet_structure_score)
        print_key_value("Strong Verbs Found", len(score.action_verbs_found))
        print_key_value("Weak Verbs Found", len(score.weak_verbs_found))
        print_key_value("Quantified Achievements", len(score.quantified_achievements))
        print_key_value("Duration", f"{(time.time() - start)*1000:.1f}ms")
        
    except Exception as e:
        results["quality"] = {"success": False, "error": str(e)}
        print(f"{Colors.RED}✗ Failed: {e}{Colors.END}")
    
    return results


def calculate_ats_score(text: str, layout_summary: Dict, parsed_data: Dict) -> Dict[str, Any]:
    """Run ATS scoring."""
    print_section("6. ATS SCORING", Colors.BLUE)
    
    results = {}
    
    print_subsection("Standard ATS Scorer")
    start = time.time()
    try:
        scorer = ATSScorer()
        # Create layout_features dict that scorer expects
        layout_features = {
            'is_single_column': layout_summary.get('is_single_column', True),
            'has_tables': layout_summary.get('has_tables', False),
            'has_images': layout_summary.get('has_images', False),
            'section_headers': layout_summary.get('section_headers', []),
            'text_density': layout_summary.get('text_density', 0),
        }
        score = scorer.calculate_score(text, layout_features, parsed_data)
        summary = scorer.get_score_summary(score)
        
        results["standard"] = {
            "success": True,
            "duration_ms": (time.time() - start) * 1000,
            "overall_score": summary["overall_score"],
            "grade": summary["grade"],
            "risk_level": summary["risk_level"],
            "breakdown": summary["breakdown"],
            "issues": summary["issues"],
            "recommendations": summary["recommendations"],
        }
        
        print_key_value("Overall Score", summary["overall_score"])
        print_key_value("Grade", summary["grade"])
        print_key_value("Risk Level", summary["risk_level"])
        print_key_value("Issues Found", len(summary["issues"]))
        print_key_value("Duration", f"{(time.time() - start)*1000:.1f}ms")
        
    except Exception as e:
        results["standard"] = {"success": False, "error": str(e)}
        print(f"{Colors.RED}✗ Failed: {e}{Colors.END}")
    
    return results


def simulate_ats_parsing_variants(text: str, layout_summary: Dict) -> Dict[str, Any]:
    """Run ATS simulation."""
    print_section("7. ATS SIMULATION", Colors.BLUE)
    
    results = {}
    
    print_subsection("ATS Parser Simulation")
    start = time.time()
    try:
        # Create layout_info dict
        layout_info = {
            'is_single_column': layout_summary.get('is_single_column', True),
            'has_tables': layout_summary.get('has_tables', False),
            'has_images': layout_summary.get('has_images', False),
        }
        sim_result = simulate_ats_parsing(text, layout_info)
        
        results["simulation"] = {
            "success": True,
            "duration_ms": (time.time() - start) * 1000,
            "extracted_sections": list(sim_result.extracted_sections.keys()),
            "sections_count": len(sim_result.extracted_sections),
            "detected_skills_count": len(sim_result.detected_skills),
            "detected_skills": sim_result.detected_skills[:10],
            "detected_contact": sim_result.detected_contact,
            "lost_content": sim_result.lost_content,
            "warnings": sim_result.warnings,
            "readability_score": sim_result.readability_score,
            "parsing_confidence": sim_result.parsing_confidence,
        }
        
        print_key_value("Sections Extracted", len(sim_result.extracted_sections))
        print_key_value("Detected Skills", len(sim_result.detected_skills))
        print_key_value("Contact Info", sim_result.detected_contact)
        print_key_value("Lost Content", sim_result.lost_content)
        print_key_value("Warnings", len(sim_result.warnings))
        print_key_value("Readability", sim_result.readability_score)
        print_key_value("Parsing Confidence", sim_result.parsing_confidence)
        print_key_value("Duration", f"{(time.time() - start)*1000:.1f}ms")
        
    except Exception as e:
        results["simulation"] = {"success": False, "error": str(e)}
        print(f"{Colors.RED}✗ Failed: {e}{Colors.END}")
    
    return results


def generate_recommendations(ats_score: Dict, content_score: Dict, layout_summary: Dict) -> Dict[str, Any]:
    """Generate recommendations."""
    print_section("8. RECOMMENDATIONS", Colors.BLUE)
    
    results = {}
    
    print_subsection("Recommendation Engine")
    start = time.time()
    try:
        engine = RecommendationEngine()
        recs = engine.generate_recommendations(ats_score, content_score, layout_summary)
        
        results["engine"] = {
            "success": True,
            "duration_ms": (time.time() - start) * 1000,
            "recommendations_count": len(recs),
            "high_priority": len([r for r in recs if r.priority.value == "high"]),
            "medium_priority": len([r for r in recs if r.priority.value == "medium"]),
            "low_priority": len([r for r in recs if r.priority.value == "low"]),
            "formatted": engine.format_recommendations(recs, max_items=10),
        }
        
        print_key_value("Total Recommendations", len(recs))
        print_key_value("High Priority", results["engine"]["high_priority"])
        print_key_value("Medium Priority", results["engine"]["medium_priority"])
        print_key_value("Duration", f"{(time.time() - start)*1000:.1f}ms")
        
    except Exception as e:
        results["engine"] = {"success": False, "error": str(e)}
        print(f"{Colors.RED}✗ Failed: {e}{Colors.END}")
    
    return results


def match_job_variants(text: str, job_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run job matching variants."""
    print_section("9. JOB MATCHING VARIANTS", Colors.BLUE)
    
    results = {}
    
    if not job_path or not job_path.exists():
        print(f"{Colors.YELLOW}⚠ No job description provided, skipping job matching{Colors.END}")
        return results
    
    # Parse job description
    print_subsection("Parsing Job Description")
    try:
        with open(job_path, 'r', encoding='utf-8') as f:
            job_text = f.read()
        
        job_parser = JobDescriptionParser()
        job_desc = job_parser.parse(job_text)
        
        print_key_value("Job Title", job_desc.title or "N/A")
        print_key_value("Required Skills", len(job_desc.required_skills))
        
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to parse job: {e}{Colors.END}")
        return results
    
    # Variant 1: Basic Matcher
    print_subsection("Method 1: Basic Job Matcher")
    start = time.time()
    try:
        match_result = match_resume_to_job(text, job_text)
        
        results["basic"] = {
            "success": True,
            "duration_ms": (time.time() - start) * 1000,
            "overall_match": match_result.overall_match,
            "skill_match": match_result.skill_match,
            "experience_match": match_result.experience_match,
            "education_match": match_result.education_match,
            "matched_skills": match_result.matched_skills[:10],
            "missing_skills": match_result.missing_skills[:10],
        }
        
        print_key_value("Overall Match", f"{match_result.overall_match:.1%}")
        print_key_value("Skill Match", f"{match_result.skill_match:.1%}")
        print_key_value("Duration", f"{(time.time() - start)*1000:.1f}ms")
        
    except Exception as e:
        results["basic"] = {"success": False, "error": str(e)}
        print(f"{Colors.RED}✗ Failed: {e}{Colors.END}")
    
    # Variant 2: Advanced Matcher
    print_subsection("Method 2: Advanced Job Matcher")
    start = time.time()
    try:
        adv_result = match_resume_to_job_advanced(text, job_text)
        
        results["advanced"] = {
            "success": True,
            "duration_ms": (time.time() - start) * 1000,
            "overall_match": adv_result.overall_match,
            "semantic_similarity": adv_result.semantic_similarity,
            "skill_similarity": adv_result.skill_similarity,
            "experience_similarity": adv_result.experience_similarity,
            "seniority_match": adv_result.seniority_match,
            "skill_gaps": adv_result.skill_gaps[:10],
            "suggested_improvements": adv_result.suggested_improvements[:5],
        }
        
        print_key_value("Overall Match", f"{adv_result.overall_match:.1%}")
        print_key_value("Semantic Similarity", f"{adv_result.semantic_similarity:.1%}")
        print_key_value("Seniority Match", adv_result.seniority_match)
        print_key_value("Duration", f"{(time.time() - start)*1000:.1f}ms")
        
    except Exception as e:
        results["advanced"] = {"success": False, "error": str(e)}
        print(f"{Colors.RED}✗ Failed: {e}{Colors.END}")
    
    return results


def run_full_pipeline(pdf_path: Path, job_path: Optional[Path] = None, output_path: Optional[Path] = None):
    """Run the complete analysis pipeline with all variants."""
    
    start_time = time.time()
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔" + "═"*68 + "╗")
    print("║" + " RESUME ANALYSIS PIPELINE - FULL SYSTEM REPORT ".center(68) + "║")
    print("║" + f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    print(f"{Colors.END}")
    
    print(f"\n{Colors.BOLD}Resume:{Colors.END} {pdf_path}")
    if job_path:
        print(f"{Colors.BOLD}Job Description:{Colors.END} {job_path}")
    if output_path:
        print(f"{Colors.BOLD}Output:{Colors.END} {output_path}")
    
    # Store all results
    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "resume_path": str(pdf_path),
            "job_path": str(job_path) if job_path else None,
        }
    }
    
    # Step 1: Text Extraction
    extraction_results = extract_text_variants(pdf_path)
    all_results["text_extraction"] = extraction_results
    
    # Use PyMuPDF result as primary text (or OCR if PyMuPDF failed)
    primary_text = None
    if extraction_results.get("pymupdf", {}).get("success"):
        # We need to re-extract to get the actual text
        extraction_result = extract_text_from_resume(pdf_path, use_ocr=False)
        primary_text = extraction_result["full_text"]
    elif extraction_results.get("ocr", {}).get("success"):
        extraction_result = extract_text_from_resume(pdf_path, use_ocr=True)
        primary_text = extraction_result["full_text"]
    
    if not primary_text:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ FATAL: Could not extract text from resume{Colors.END}")
        return
    
    # Step 2: Language Detection
    language_results = detect_language_variants(primary_text)
    all_results["language_detection"] = language_results
    
    # Determine language to use
    lang_code = language_results.get("auto", {}).get("detected_code", "en")
    print(f"\n{Colors.GREEN}Using language: {lang_code}{Colors.END}")
    
    # Step 3: Layout Analysis
    layout_results = analyze_layout_variants(primary_text, pdf_path, lang_code)
    all_results["layout_analysis"] = layout_results
    
    # Get layout summary for scoring
    layout_summary = None
    if layout_results.get("auto", {}).get("success"):
        layout_summary = layout_results["auto"]["summary"]
    
    # Step 4: Section Parsing
    section_results = parse_sections_variants(primary_text, lang_code)
    all_results["section_parsing"] = section_results
    
    # Prepare parsed data for scoring
    parsed_data = {}
    if section_results.get("standard", {}).get("success"):
        # Convert section names to expected format (list of dicts with section_type)
        section_names = section_results["standard"].get("section_names", [])
        sections_list = [{"section_type": name.lower()} for name in section_names]
        
        parsed_data = {
            "contact_info": section_results["standard"].get("contact_info", {}),
            "sections": sections_list,
            "skills": [],  # Would need actual skills data
        }
    
    # Step 5: Content Analysis
    content_results = analyze_content_variants(primary_text)
    all_results["content_analysis"] = content_results
    
    # Step 6: ATS Scoring
    if layout_summary:
        ats_results = calculate_ats_score(primary_text, layout_summary, parsed_data)
        all_results["ats_scoring"] = ats_results
    
    # Step 7: ATS Simulation
    ats_sim_results = simulate_ats_parsing_variants(primary_text, layout_summary if layout_summary else {})
    all_results["ats_simulation"] = ats_sim_results
    
    # Step 8: Recommendations
    if layout_summary and all_results.get("ats_scoring") and all_results.get("content_analysis"):
        # Prepare content_score dict from content analysis
        content_quality = all_results["content_analysis"].get("quality", {})
        content_score = {
            "action_verb_score": content_quality.get("action_verb_score", 25),
            "quantification_score": content_quality.get("quantification_score", 25),
            "bullet_structure_score": content_quality.get("bullet_structure_score", 25),
            "conciseness_score": content_quality.get("conciseness_score", 25),
            "weak_verbs_found": [],
            "bullets": [],
        }
        ats_score_dict = all_results["ats_scoring"].get("standard", {})
        rec_results = generate_recommendations(
            ats_score_dict, 
            content_score, 
            layout_summary
        )
        all_results["recommendations"] = rec_results
    
    # Step 9: Job Matching (if provided)
    if job_path:
        job_match_results = match_job_variants(primary_text, job_path)
        all_results["job_matching"] = job_match_results
    
    # Final Summary
    total_duration = (time.time() - start_time) * 1000
    
    print_section("PIPELINE COMPLETE", Colors.GREEN)
    print_key_value("Total Duration", f"{total_duration:.1f}ms")
    print_key_value("Components Run", len([v for v in all_results.values() if isinstance(v, dict)]))
    
    # Save to JSON if requested
    if output_path:
        all_results["metadata"]["total_duration_ms"] = total_duration
        
        # Serialize for JSON
        json_data = serialize_for_json(all_results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{Colors.GREEN}✓ Results saved to: {output_path}{Colors.END}")
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}Analysis complete!{Colors.END}\n")
    
    return all_results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive resume analysis with all system variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
By default, both JSON results and terminal output (with colors) are saved.

Examples:
  python pipeline.py resume.pdf
    → Saves: resume_results.json and resume_terminal.txt
  
  python pipeline.py resume.pdf --no-output
    → Only saves terminal output
  
  python pipeline.py resume.pdf --no-terminal
    → Only saves JSON results
  
  python pipeline.py resume.pdf --output custom.json --save-terminal custom.txt
    → Saves to custom filenames
  
  python pipeline.py resume.pdf --job job_description.txt
    → Saves both files and includes job matching analysis

Viewing saved terminal output:
  cat resume_terminal.txt              # View with colors (if terminal supports it)
  less -R resume_terminal.txt          # View with colors in less
  cat resume_terminal.txt | less       # View without colors
  
Note: The terminal output file preserves all ANSI color codes and formatting.
        """
    )
    
    parser.add_argument("resume", help="Path to resume PDF file")
    parser.add_argument("--job", "-j", help="Path to job description text file")
    parser.add_argument("--output", "-o", help="Output JSON file path (default: auto-generated)")
    parser.add_argument("--save-terminal", "-t", help="Save terminal output (with colors) to .txt file (default: auto-generated)")
    parser.add_argument("--no-output", action="store_true", help="Don't save JSON output")
    parser.add_argument("--no-terminal", action="store_true", help="Don't save terminal output")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        global Colors
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')
    
    resume_path = Path(args.resume)
    if not resume_path.exists():
        print(f"Error: Resume file not found: {resume_path}")
        sys.exit(1)
    
    job_path = Path(args.job) if args.job else None
    if job_path and not job_path.exists():
        print(f"Error: Job description file not found: {job_path}")
        sys.exit(1)
    
    # Generate default output paths based on resume name
    resume_stem = resume_path.stem
    
    # Create output directory based on resume name
    output_dir = resume_path.parent / resume_stem
    output_dir.mkdir(exist_ok=True)
    
    # Determine output paths
    if args.no_output:
        output_path = None
    elif args.output:
        output_path = Path(args.output)
    else:
        # Default: save in resume-named directory
        output_path = output_dir / "results.json"
    
    if args.no_terminal:
        terminal_path = None
    elif args.save_terminal:
        terminal_path = Path(args.save_terminal)
    else:
        # Default: save in resume-named directory
        terminal_path = output_dir / "terminal.txt"
    
    # Capture terminal output by default
    capture = TerminalCapture()
    capture.start()
    
    try:
        # Run pipeline
        run_full_pipeline(resume_path, job_path, output_path)
        
        # Save terminal output
        capture.stop()
        if terminal_path:
            capture.save_to_file(terminal_path)
            print(f"\n{Colors.GREEN}✓ Terminal output saved to: {terminal_path}{Colors.END}")
            print(f"{Colors.YELLOW}Tip: View with 'cat {terminal_path}' or 'less -R {terminal_path}'{Colors.END}")
    except Exception as e:
        capture.stop()
        raise


if __name__ == "__main__":
    main()
