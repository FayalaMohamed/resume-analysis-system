"""CLI helper to enqueue resume extraction jobs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jobs.job_queue import JobQueue


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit a resume extraction job")
    parser.add_argument("resume", help="Path to resume PDF")
    parser.add_argument(
        "--method",
        default="aggregated",
        choices=["aggregated", "langextract", "unified"],
        help="Extraction method",
    )
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR")
    parser.add_argument("--passes", type=int, default=2, help="LangExtract passes")
    args = parser.parse_args()

    resume_path = Path(args.resume).resolve()
    if not resume_path.exists():
        print(f"Resume not found: {resume_path}")
        return 1

    payload = {
        "job_type": "resume_extraction",
        "resume_path": str(resume_path),
        "extraction_method": args.method,
        "use_ocr": not args.no_ocr,
        "langextract_passes": args.passes,
    }

    queue = JobQueue()
    job_id = queue.enqueue(payload)
    print(f"Queued job {job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
