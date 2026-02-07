"""Pipeline script for processing resumes end-to-end.

This script demonstrates the complete ATS resume analysis pipeline:
1. Extract text from PDF
2. Analyze layout
3. Parse sections
4. Calculate ATS score
5. Generate report
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from parsers import extract_text_from_resume, LayoutDetector, SectionParser, LanguageDetector
from scoring import ATSScorer
from utils import Config, get_logger

logger = get_logger(__name__)


def process_resume(pdf_path: Path, use_ocr: bool = False, language: str = 'auto') -> dict:
    """Process a single resume through the complete pipeline.

    Args:
        pdf_path: Path to PDF file
        use_ocr: Whether to use OCR
        language: Language code ('auto' for automatic detection, or specific like 'en', 'fr')

    Returns:
        Dictionary with complete analysis results
    """
    logger.info(f"Processing: {pdf_path.name}")

    logger.info("Step 1: Extracting text...")
    extraction_result = extract_text_from_resume(pdf_path, use_ocr=use_ocr)
    text = extraction_result["full_text"]

    logger.info("Step 2: Detecting language...")
    if language == 'auto':
        detected_lang = LanguageDetector.detect(text, default='en')
        logger.info(f"  Detected language: {LanguageDetector.get_language_name(detected_lang)}")
    else:
        detected_lang = language
        logger.info(f"  Using language: {LanguageDetector.get_language_name(detected_lang)}")

    logger.info("Step 3: Analyzing layout...")
    layout_detector = LayoutDetector(language=detected_lang)
    layout_features = layout_detector.analyze_layout(text, lang_code=detected_lang)
    layout_summary = layout_detector.get_layout_summary(text)

    logger.info("Step 4: Parsing sections...")
    parser = SectionParser(language=detected_lang)
    parsed = parser.parse(text)
    
    logger.info("Step 4: Calculating ATS score...")
    scorer = ATSScorer()
    parsed_dict = {
        "contact_info": parsed.contact_info,
        "sections": parsed.sections,
        "skills": parsed.skills,
    }
    ats_score = scorer.calculate_score(text, layout_summary, parsed_dict)
    score_summary = scorer.get_score_summary(ats_score)
    
    results = {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "num_pages": extraction_result["num_pages"],
        "detected_language": detected_lang,
        "extraction": {
            "method": "OCR" if use_ocr else "PyMuPDF",
            "text_length": len(text),
            "lines": text.count("\n"),
        },
        "layout_analysis": {
            **layout_summary,
            "has_images": layout_features.has_images,
            "text_density": layout_features.text_density,
            "avg_line_length": layout_features.avg_line_length,
        },
        "parsed_data": {
            "contact_info": parsed.contact_info,
            "sections_found": [s["name"] for s in parsed.sections],
            "section_types": [s["section_type"] for s in parsed.sections],
            "num_sections": len(parsed.sections),
            "skills_count": len(parsed.skills),
            "languages_count": len(parsed.languages),
            "certifications_count": len(parsed.certifications),
            "projects_count": len(parsed.projects),
            "awards_count": len(parsed.awards),
            "publications_count": len(parsed.publications),
            "volunteer_work_count": len(parsed.volunteer_work),
            "professional_affiliations_count": len(parsed.professional_affiliations),
        },
        "ats_score": score_summary,
    }
    
    logger.info(f"Processing complete. Score: {score_summary['overall_score']}/100")
    logger.info(results)
    
    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Process resume PDFs for ATS analysis")
    parser.add_argument("pdf_path", help="Path to PDF file or directory")
    parser.add_argument("--ocr", action="store_true", help="Use OCR extraction")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--summary", "-s", action="store_true", help="Print summary only")
    parser.add_argument("--language", "-l", default="auto",
                        help="Language code (auto, en, fr, es, de, it, pt)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)

    if pdf_path.is_dir():
        # Process all PDFs in directory
        pdf_files = list(pdf_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
    else:
        pdf_files = [pdf_path]

    all_results = []

    for pdf_file in pdf_files:
        try:
            result = process_resume(pdf_file, use_ocr=args.ocr, language=args.language)
            all_results.append(result)

            if args.summary:
                print(f"\n{result['file_name']}: {result['ats_score']['overall_score']}/100 ({result['ats_score']['grade']})")
                print(f"  Language: {result['detected_language']}")
                for issue in result['ats_score']['issues'][:3]:
                    print(f"  - {issue}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}")
            continue
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_path}")
    
    # Print summary
    if len(all_results) > 1:
        scores = [r['ats_score']['overall_score'] for r in all_results]
        avg_score = sum(scores) / len(scores)
        print(f"\n{'='*50}")
        print(f"Processed {len(all_results)} resumes")
        print(f"Average ATS Score: {avg_score:.1f}/100")
        print(f"Highest Score: {max(scores)}/100")
        print(f"Lowest Score: {min(scores)}/100")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
