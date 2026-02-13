# Phase 4 Supabase Setup

This adds storage + database persistence for resumes and extracted JSON.

## Environment Variables

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_BUCKET` (default: `resume-uploads`)
- `SUPABASE_SCHEMA` (default: `public`)
- `SUPABASE_RESUMES_TABLE` (default: `resumes`)
- `SUPABASE_EXTRACTIONS_TABLE` (default: `resume_extractions`)
- `SUPABASE_AUTO_UPLOAD` (default: `false`)
- `RESUME_RETENTION_DAYS` (default: `30`)

## SQL Schema

```sql
create table if not exists public.resumes (
  id uuid primary key,
  user_id uuid null,
  original_filename text not null,
  storage_bucket text not null,
  storage_path text not null,
  content_type text,
  size_bytes bigint,
  sha256 text,
  uploaded_at timestamptz not null default now(),
  retention_days int not null default 30,
  expires_at timestamptz not null,
  deleted_at timestamptz,
  metadata jsonb
);

create table if not exists public.resume_extractions (
  id uuid primary key,
  resume_id uuid not null references public.resumes(id) on delete cascade,
  extractor text not null,
  status text not null default 'complete',
  extraction_json jsonb not null,
  created_at timestamptz not null default now(),
  metadata jsonb
);

create index if not exists resumes_expires_at_idx
  on public.resumes (expires_at)
  where deleted_at is null;

create index if not exists resume_extractions_resume_id_idx
  on public.resume_extractions (resume_id);
```

## Storage Bucket

Create a storage bucket named `resume-uploads` (or set `SUPABASE_BUCKET`).

## Retention Job

Run the cleanup job on a schedule:

```bash
python src/storage/retention_job.py
```
