# Phase 4 Async Job Queue

This adds a lightweight SQLite-backed job queue with retries for background OCR and extraction.

## Environment Variables

- `JOB_MAX_ATTEMPTS` (default: `3`)
- `JOB_POLL_INTERVAL` (default: `2.0` seconds)
- `JOB_RETRY_BASE_SECONDS` (default: `5`)
- `JOB_RETRY_MAX_SECONDS` (default: `60`)

## Enqueue a Job

```bash
python src/jobs/submit_job.py resumes/test_resume.pdf --method aggregated
```

## Run a Worker

```bash
python src/jobs/worker.py --poll 2
```

## Output

Results are stored at:

```
data/processed/jobs/<job_id>/output.json
```

If `SUPABASE_AUTO_UPLOAD=true`, the worker will also persist the resume + extraction
to Supabase and append a `_supabase` block to the output JSON.
