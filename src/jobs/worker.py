"""Background worker to process resume extraction jobs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jobs.job_queue import JobQueue
from utils import Config, get_logger


logger = get_logger(__name__)


def _write_result(job_id: str, result: Dict[str, Any]) -> str:
    Config.JOBS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    job_dir = Config.JOBS_RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / "output.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return str(output_path)


def _process_job_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    extraction_method = payload.get("extraction_method", "aggregated")
    resume_path = Path(payload["resume_path"]).resolve()
    use_ocr = bool(payload.get("use_ocr", True))
    passes = int(payload.get("langextract_passes", 2))

    if extraction_method == "aggregated":
        from parsers.aggregated_extractor import aggregate_resume

        return aggregate_resume(resume_path, use_ocr=use_ocr, langextract_passes=passes)

    if extraction_method == "langextract":
        from parsers.langextract_parser import LangExtractResumeParser, LANGEXTRACT_AVAILABLE

        if not LANGEXTRACT_AVAILABLE:
            raise RuntimeError("LangExtract is not installed")
        parser = LangExtractResumeParser()
        if not parser.is_available():
            raise RuntimeError("LangExtract API key not configured")
        result = parser.extract_from_pdf(resume_path, extraction_passes=passes)
        if not result.success:
            raise RuntimeError(result.error_message or "LangExtract extraction failed")
        return result.to_dict()

    if extraction_method == "unified":
        from parsers.unified_extractor import UnifiedResumeExtractor

        extractor = UnifiedResumeExtractor()
        return extractor.extract(resume_path).to_dict()

    raise ValueError(f"Unsupported extraction method: {extraction_method}")


def _maybe_store_supabase(payload: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    if not payload.get("store_to_supabase"):
        return {}

    if not Config.SUPABASE_ENABLED:
        raise RuntimeError("Supabase is not configured")

    try:
        from storage import SupabaseStore, SUPABASE_AVAILABLE
    except Exception as exc:
        raise RuntimeError("Supabase storage module not available") from exc

    if not SUPABASE_AVAILABLE:
        raise RuntimeError("Supabase client not installed")

    resume_path = Path(payload["resume_path"]).resolve()
    extractor = payload.get("extraction_method", "unknown")
    metadata = payload.get("metadata") or {}
    store = SupabaseStore()
    stored = store.store_resume_and_extraction(
        resume_path,
        result,
        extractor=extractor,
        user_id=payload.get("user_id"),
        metadata=metadata,
    )
    return {
        "resume_id": stored.get("resume_id"),
        "extraction_id": stored.get("extraction_record", {}).get("extraction_id"),
    }


def run_worker(poll_interval: float, once: bool) -> int:
    queue = JobQueue()
    worker_id = f"worker-{int(time.time())}"

    while True:
        job = queue.reserve_next(worker_id)
        if not job:
            if once:
                return 1
            time.sleep(poll_interval)
            continue

        logger.info("Processing job %s (attempt %s)", job.id, job.attempts)
        try:
            result = _process_job_payload(job.payload)
            supabase_info = _maybe_store_supabase(job.payload, result)
            if supabase_info:
                result = dict(result)
                result["_supabase"] = supabase_info
            output_path = _write_result(job.id, result)
            queue.mark_complete(job.id, output_path)
            logger.info("Job %s complete", job.id)
        except Exception as exc:
            queue.mark_failed(job.id, str(exc), job.attempts)
            logger.exception("Job %s failed", job.id)

        if once:
            return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume extraction job worker")
    parser.add_argument("--once", action="store_true", help="Process a single job and exit")
    parser.add_argument("--poll", type=float, default=Config.JOB_POLL_INTERVAL, help="Polling interval in seconds")
    args = parser.parse_args()

    return run_worker(poll_interval=args.poll, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
