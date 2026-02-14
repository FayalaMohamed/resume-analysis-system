#!/usr/bin/env python3
"""
Aggregated Resume Extraction

Combines existing extractors (PyMuPDF text, multi-engine OCR, unified parser,
SectionParser, and optional LangExtract) into a single canonical output with
confidence and source metadata.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import Config
from observability import MetricsStore

try:
    from .ocr import extract_text_from_resume
    from .enhanced_ocr import extract_text_enhanced
    from .unified_extractor import UnifiedResumeExtractor
    from .section_parser import SectionParser
    from .language_detector import LanguageDetector
    from .langextract_parser import LangExtractResumeParser, LANGEXTRACT_AVAILABLE
except Exception:  # pragma: no cover - fallback for direct execution
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.parsers.ocr import extract_text_from_resume
    from src.parsers.enhanced_ocr import extract_text_enhanced
    from src.parsers.unified_extractor import UnifiedResumeExtractor
    from src.parsers.section_parser import SectionParser
    from src.parsers.language_detector import LanguageDetector

    try:
        from src.parsers.langextract_parser import LangExtractResumeParser, LANGEXTRACT_AVAILABLE
    except Exception:
        LANGEXTRACT_AVAILABLE = False
        LangExtractResumeParser = None


SOURCE_CONFIDENCE = {
    "langextract": 0.85,
    "unified": 0.70,
    "section_parser": 0.60,
    "enhanced_skills": 0.75,
    "pymupdf_text": 0.60,
    "ocr_text": 0.55,
}


def _now_ms() -> float:
    return time.time() * 1000


def _clean_text(value: Optional[str]) -> str:
    return value.strip() if value else ""


def _add_source(
    sources: List[Dict[str, Any]],
    source: str,
    confidence: float,
    evidence: Optional[str] = None,
    grounded: bool = False,
) -> None:
    sources.append(
        {
            "source": source,
            "confidence": confidence,
            "evidence": evidence,
            "grounded": grounded,
        }
    )


def _pick_best_value(candidates: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    valid = [c for c in candidates if c.get("value")]
    if not valid:
        return "", {"confidence": 0.0, "sources": []}

    grounded = [c for c in valid if c.get("grounded")]
    pool = grounded if grounded else valid

    best = max(pool, key=lambda c: c.get("confidence", 0.0))
    sources = [
        {
            "source": c["source"],
            "confidence": c.get("confidence", 0.0),
            "evidence": c.get("evidence"),
            "grounded": c.get("grounded", False),
        }
        for c in valid
    ]
    meta = {
        "confidence": best.get("confidence", 0.0),
        "sources": sources,
        "alternates": [c["value"] for c in valid if c["value"] != best["value"]],
    }
    return best["value"], meta


def _select_text_source(pymupdf: Dict[str, Any], ocr: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    pym_text = _clean_text(pymupdf.get("full_text"))
    ocr_text = _clean_text(ocr.get("full_text"))
    ocr_conf = ocr.get("overall_confidence", 0.0)

    if not pym_text and ocr_text:
        return ocr_text, {
            "selected": "ocr",
            "reason": "pymupdf_empty",
        }
    if not ocr_text and pym_text:
        return pym_text, {
            "selected": "pymupdf",
            "reason": "ocr_empty",
        }

    if ocr_conf >= 0.75 and len(ocr_text) >= int(len(pym_text) * 0.85):
        return ocr_text, {
            "selected": "ocr",
            "reason": "ocr_confidence_and_length",
        }

    return pym_text or ocr_text, {
        "selected": "pymupdf" if pym_text else "ocr",
        "reason": "default_pymupdf",
    }


def _has_grounded_source(sources: Optional[List[Dict[str, Any]]]) -> bool:
    if not sources:
        return False
    return any(source.get("grounded") for source in sources)


def _enforce_grounding(output: Dict[str, Any]) -> Dict[str, Any]:
    rejected_counts = {
        "contact": 0,
        "summary": 0,
        "experience": 0,
        "education": 0,
        "skills": 0,
        "projects": 0,
        "certifications": 0,
        "languages": 0,
    }

    contact = output.get("contact", {})
    contact_meta = output.get("contact_meta", {})
    for field, meta in contact_meta.items():
        sources = meta.get("sources", [])
        if contact.get(field) and not _has_grounded_source(sources):
            contact[field] = ""
            meta["rejected"] = True
            meta["rejection_reason"] = "ungrounded"
            rejected_counts["contact"] += 1

    summary_meta = output.get("summary_meta", {})
    summary_sources = summary_meta.get("sources", [])
    if output.get("summary") and not _has_grounded_source(summary_sources):
        output["summary"] = ""
        summary_meta["rejected"] = True
        summary_meta["rejection_reason"] = "ungrounded"
        rejected_counts["summary"] += 1

    def _filter_items(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for item in items:
            sources = item.get("sources", [])
            if _has_grounded_source(sources):
                filtered.append(item)
            else:
                rejected_counts[key] += 1
        return filtered

    output["experience"] = _filter_items(output.get("experience", []), "experience")
    output["education"] = _filter_items(output.get("education", []), "education")
    output["skills"] = _filter_items(output.get("skills", []), "skills")
    output["projects"] = _filter_items(output.get("projects", []), "projects")
    output["certifications"] = _filter_items(output.get("certifications", []), "certifications")
    output["languages"] = _filter_items(output.get("languages", []), "languages")

    output.setdefault("metadata", {})
    output["metadata"]["grounding"] = {
        "enforced": True,
        "rejected_counts": rejected_counts,
        "low_confidence_threshold": Config.LOW_CONFIDENCE_THRESHOLD,
    }

    return output


def _dedupe_items(items: List[Dict[str, Any]], key_fields: List[str]) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        key = "|".join(_clean_text(str(item.get(f, ""))).lower() for f in key_fields)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _unified_sections_to_skills(unified_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    sections = unified_output.get("sections", [])
    skills_sections = [s for s in sections if s.get("section_type") == "skills"]
    skills_text = "\n".join(s.get("raw_text", "") for s in skills_sections)
    return [
        {
            "name": s.strip(),
            "category": "unknown",
        }
        for s in skills_text.replace("•", ",").split(",")
        if s.strip()
    ]


def aggregate_resume(
    pdf_path: Path,
    use_ocr: bool = True,
    langextract_passes: int = 2,
) -> Dict[str, Any]:
    start_ms = _now_ms()
    timings: Dict[str, float] = {}

    # Text extraction: PyMuPDF and enhanced OCR
    t0 = _now_ms()
    pymupdf_result = extract_text_from_resume(pdf_path, use_ocr=False)
    timings["pymupdf_ms"] = _now_ms() - t0

    t1 = _now_ms()
    ocr_result = extract_text_enhanced(pdf_path, use_ocr=use_ocr, min_confidence=0.7)
    timings["enhanced_ocr_ms"] = _now_ms() - t1

    primary_text, text_selection = _select_text_source(pymupdf_result, ocr_result)

    # Language detection + section parser (cheap, structured anchors)
    lang_code = LanguageDetector.detect(primary_text or "", default="en")
    section_parser = SectionParser(language=lang_code)
    parsed_sections = section_parser.parse(primary_text)

    # Unified extractor (layout-aware)
    t2 = _now_ms()
    unified = UnifiedResumeExtractor().extract(pdf_path)
    unified_output = unified.to_dict()
    timings["unified_ms"] = _now_ms() - t2

    # LangExtract (optional, quality-first)
    langextract_output = None
    langextract_used = False
    t3 = _now_ms()
    if LANGEXTRACT_AVAILABLE and LangExtractResumeParser is not None:
        try:
            parser = LangExtractResumeParser()  # type: ignore[misc]
            if parser.is_available() and primary_text.strip():
                langextract_used = True
                result = parser.extract_from_text(primary_text, extraction_passes=langextract_passes)
                langextract_output = result.to_dict()
        except Exception:
            langextract_output = None
    timings["langextract_ms"] = _now_ms() - t3

    # Contact aggregation
    contact_fields = ["name", "email", "phone", "linkedin", "github", "website", "location"]
    contact: Dict[str, str] = {}
    contact_meta: Dict[str, Any] = {}

    for field in contact_fields:
        candidates = []
        if langextract_output:
            val = _clean_text(langextract_output.get("contact", {}).get(field, ""))
            if val:
                candidates.append(
                    {
                        "value": val,
                        "source": "langextract",
                        "confidence": SOURCE_CONFIDENCE["langextract"],
                        "evidence": val,
                        "grounded": False,
                    }
                )
        val = _clean_text(unified_output.get("contact_info", {}).get(field, ""))
        if val:
            candidates.append(
                {
                    "value": val,
                    "source": "unified",
                    "confidence": SOURCE_CONFIDENCE["unified"],
                    "evidence": val,
                    "grounded": True,
                }
            )
        val = _clean_text(parsed_sections.contact_info.get(field, ""))
        if val:
            candidates.append(
                {
                    "value": val,
                    "source": "section_parser",
                    "confidence": SOURCE_CONFIDENCE["section_parser"],
                    "evidence": val,
                    "grounded": True,
                }
            )

        best_value, meta = _pick_best_value(candidates)
        if best_value:
            contact[field] = best_value
        contact_meta[field] = meta

    # Summary aggregation
    summary_candidates = []
    if langextract_output:
        summary_val = _clean_text(langextract_output.get("summary", ""))
        if summary_val:
            summary_candidates.append(
                {
                    "value": summary_val,
                    "source": "langextract",
                    "confidence": SOURCE_CONFIDENCE["langextract"],
                    "evidence": summary_val[:200],
                    "grounded": False,
                }
            )
    summary_val = _clean_text(unified_output.get("summary", ""))
    if summary_val:
        summary_candidates.append(
            {
                "value": summary_val,
                "source": "unified",
                "confidence": SOURCE_CONFIDENCE["unified"],
                "evidence": summary_val[:200],
                "grounded": True,
            }
        )
    summary_text, summary_meta = _pick_best_value(summary_candidates)

    # Experience aggregation
    experiences: List[Dict[str, Any]] = []
    if langextract_output:
        for exp in langextract_output.get("experience", []):
            item = {
                "job_title": exp.get("job_title", ""),
                "company": exp.get("company", ""),
                "date_range": exp.get("date_range", ""),
                "location": exp.get("location", ""),
                "employment_type": exp.get("employment_type", ""),
                "description": exp.get("description", ""),
                "bullets": [b.get("text", "") for b in exp.get("bullet_points", []) if b.get("text")],
                "technologies": exp.get("technologies", []),
                "confidence": SOURCE_CONFIDENCE["langextract"],
                "sources": [
                    {
                        "source": "langextract",
                        "confidence": SOURCE_CONFIDENCE["langextract"],
                        "grounded": False,
                    }
                ],
            }
            experiences.append(item)

    for section in unified_output.get("sections", []):
        if section.get("section_type") != "experience":
            continue
        for item in section.get("items", []):
            bullets = item.get("bullet_points", [])
            description = item.get("description", "")
            exp_item = {
                "job_title": item.get("title", ""),
                "company": item.get("company") or item.get("subtitle", ""),
                "date_range": item.get("date_range", ""),
                "location": item.get("location", ""),
                "employment_type": "",
                "description": description,
                "bullets": bullets,
                "technologies": [],
                "confidence": SOURCE_CONFIDENCE["unified"],
                "sources": [
                    {
                        "source": "unified",
                        "confidence": SOURCE_CONFIDENCE["unified"],
                        "grounded": True,
                        "evidence": (description or " ").strip()[:200] or item.get("title", ""),
                    }
                ],
            }
            experiences.append(exp_item)

    experiences = _dedupe_items(experiences, ["job_title", "company", "date_range"])

    # Education aggregation
    education: List[Dict[str, Any]] = []
    if langextract_output:
        for edu in langextract_output.get("education", []):
            education.append(
                {
                    "degree": edu.get("degree", ""),
                    "field_of_study": edu.get("field_of_study", "") or edu.get("field", ""),
                    "institution": edu.get("institution", ""),
                    "date_range": edu.get("date_range", ""),
                    "gpa": edu.get("gpa", ""),
                    "coursework": edu.get("coursework", []),
                    "honors": edu.get("honors", []),
                    "confidence": SOURCE_CONFIDENCE["langextract"],
                    "sources": [
                        {
                            "source": "langextract",
                            "confidence": SOURCE_CONFIDENCE["langextract"],
                            "grounded": False,
                        }
                    ],
                }
            )

    for section in unified_output.get("sections", []):
        if section.get("section_type") != "education":
            continue
        for item in section.get("items", []):
            education.append(
                {
                    "degree": item.get("title", ""),
                    "field_of_study": "",
                    "institution": item.get("subtitle", ""),
                    "date_range": item.get("date_range", ""),
                    "gpa": "",
                    "coursework": [],
                    "honors": [],
                    "confidence": SOURCE_CONFIDENCE["unified"],
                    "sources": [
                        {
                            "source": "unified",
                            "confidence": SOURCE_CONFIDENCE["unified"],
                            "grounded": True,
                            "evidence": item.get("title", ""),
                        }
                    ],
                }
            )

    education = _dedupe_items(education, ["degree", "institution", "date_range"])

    # Skills aggregation (LangExtract + EnhancedSkillExtractor)
    skills: Dict[str, Dict[str, Any]] = {}
    if langextract_output:
        for skill in langextract_output.get("skills", []):
            name = _clean_text(skill.get("name", ""))
            if not name:
                continue
            key = name.lower()
            skills[key] = {
                "name": name,
                "category": skill.get("category", ""),
                "parent_category": skill.get("parent_category", ""),
                "confidence": SOURCE_CONFIDENCE["langextract"],
                "sources": [
                    {
                        "source": "langextract",
                        "confidence": SOURCE_CONFIDENCE["langextract"],
                        "grounded": False,
                        "evidence": name,
                    }
                ],
            }

    try:
        from analysis.enhanced_skills import EnhancedSkillExtractor

        extractor = EnhancedSkillExtractor()
        unified_sections = [
            {"section_type": s.get("section_type"), "raw_text": s.get("raw_text", "")}
            for s in unified_output.get("sections", [])
        ]
        enhanced_skills = extractor.extract_skills(sections=unified_sections)
        for skill in enhanced_skills:
            key = skill.canonical_name.lower()
            if key not in skills:
                skills[key] = {
                    "name": skill.canonical_name,
                    "category": skill.category.value,
                    "parent_category": "",
                    "confidence": skill.confidence,
                    "sources": [],
                }
            _add_source(
                skills[key]["sources"],
                "enhanced_skills",
                max(skill.confidence, SOURCE_CONFIDENCE["enhanced_skills"]),
                evidence=skill.name,
                grounded=True,
            )
            skills[key]["confidence"] = max(skills[key]["confidence"], skill.confidence)
    except Exception:
        pass

    # Projects aggregation
    projects: List[Dict[str, Any]] = []
    if langextract_output:
        for proj in langextract_output.get("projects", []):
            projects.append(
                {
                    "name": proj.get("name", ""),
                    "description": proj.get("description", ""),
                    "technologies": proj.get("technologies", []),
                    "url": proj.get("url", ""),
                    "bullets": proj.get("bullet_points", []),
                    "confidence": SOURCE_CONFIDENCE["langextract"],
                    "sources": [
                        {
                            "source": "langextract",
                            "confidence": SOURCE_CONFIDENCE["langextract"],
                            "grounded": False,
                        }
                    ],
                }
            )

    for section in unified_output.get("sections", []):
        if section.get("section_type") != "projects":
            continue
        for item in section.get("items", []):
            projects.append(
                {
                    "name": item.get("title", ""),
                    "description": item.get("description", ""),
                    "technologies": [],
                    "url": "",
                    "bullets": item.get("bullet_points", []),
                    "confidence": SOURCE_CONFIDENCE["unified"],
                    "sources": [
                        {
                            "source": "unified",
                            "confidence": SOURCE_CONFIDENCE["unified"],
                            "grounded": True,
                            "evidence": item.get("title", ""),
                        }
                    ],
                }
            )

    projects = _dedupe_items(projects, ["name", "description"])

    # Certifications aggregation
    certifications: List[Dict[str, Any]] = []
    if langextract_output:
        for cert in langextract_output.get("certifications", []):
            certifications.append(
                {
                    "name": cert.get("name", ""),
                    "provider": cert.get("provider", ""),
                    "date_obtained": cert.get("date_obtained", ""),
                    "expiration_date": cert.get("expiration_date", ""),
                    "level": cert.get("level", ""),
                    "confidence": SOURCE_CONFIDENCE["langextract"],
                    "sources": [
                        {
                            "source": "langextract",
                            "confidence": SOURCE_CONFIDENCE["langextract"],
                            "grounded": False,
                        }
                    ],
                }
            )

    certifications = _dedupe_items(certifications, ["name", "provider"])

    # Languages aggregation
    languages: List[Dict[str, Any]] = []
    if langextract_output:
        for lang in langextract_output.get("languages", []):
            languages.append(
                {
                    "language": lang.get("language", ""),
                    "proficiency": lang.get("proficiency", ""),
                    "confidence": SOURCE_CONFIDENCE["langextract"],
                    "sources": [
                        {
                            "source": "langextract",
                            "confidence": SOURCE_CONFIDENCE["langextract"],
                            "grounded": False,
                        }
                    ],
                }
            )

    # Output
    output = {
        "contact": contact,
        "contact_meta": contact_meta,
        "summary": summary_text,
        "summary_meta": summary_meta,
        "experience": experiences,
        "education": education,
        "skills": list(skills.values()),
        "projects": projects,
        "certifications": certifications,
        "languages": languages,
        "metadata": {
            "extraction_time_ms": _now_ms() - start_ms,
            "text_selection": text_selection,
            "pymupdf_text_length": len(_clean_text(pymupdf_result.get("full_text"))),
            "ocr_text_length": len(_clean_text(ocr_result.get("full_text"))),
            "ocr_overall_confidence": ocr_result.get("overall_confidence", 0.0),
            "langextract_used": langextract_used,
            "language": lang_code,
            "timings_ms": timings,
        },
    }
    if Config.ENFORCE_GROUNDING:
        output = _enforce_grounding(output)

    try:
        grounding_meta = output.get("metadata", {}).get("grounding", {})
        MetricsStore().record_extraction(
            {
                "resume_path": str(pdf_path),
                "extraction_time_ms": output.get("metadata", {}).get("extraction_time_ms"),
                "ocr_overall_confidence": output.get("metadata", {}).get("ocr_overall_confidence"),
                "langextract_used": output.get("metadata", {}).get("langextract_used"),
                "grounding_enforced": grounding_meta.get("enforced", False),
                "grounding_rejected_counts": grounding_meta.get("rejected_counts", {}),
                "text_selection": output.get("metadata", {}).get("text_selection", {}),
            }
        )
    except Exception:
        pass

    return output


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate resume extraction outputs")
    parser.add_argument("resume", help="Path to resume PDF")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR")
    parser.add_argument("--passes", type=int, default=2, help="LangExtract passes (1-3)")
    args = parser.parse_args()

    pdf_path = Path(args.resume)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        return 1

    result = aggregate_resume(pdf_path, use_ocr=not args.no_ocr, langextract_passes=args.passes)

    output_path = Path(args.output) if args.output else pdf_path.with_suffix("")
    if not args.output:
        output_path = Path(f"{output_path}_aggregated.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[OK] Aggregated extraction saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
