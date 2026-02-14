# Phase 5: Product Surface and Delivery

**Goal**: Decide and implement the product surface for the micro SaaS launch, balancing speed-to-market with maintainability.

---

## Overview

Phase 5 focuses on product surface decisions, launch hardening, and delivery UX. The codebase now has async jobs, Supabase persistence, grounding enforcement, and basic observability, so the focus shifts to user-facing delivery and go-to-market readiness.

---

## Product Surface Decision

We will move away from Streamlit and build a production UI + API:
- Backend: FastAPI (Python)
- Frontend: Next.js

**Pros**
- Real API, better auth + billing
- Flexible UI and branding
- Easier to expand later

**Cons**
- Higher initial build time

**Suggested lean stack:**
- Backend: FastAPI + RQ or Celery for background jobs
- DB: Postgres (Supabase or Railway)
- Storage: S3-compatible (R2 or Supabase Storage)
- Frontend: Next.js + Tailwind
- Auth: Supabase Auth or Clerk

**Migration path from Streamlit:**
1. Wrap extraction pipeline into a FastAPI service
2. Build a Next.js upload + results UI
3. Add async job polling and result viewing
4. Retire Streamlit from the launch surface

---

## Recommended Decision

- **Short-term**: Build FastAPI + Next.js for launch
- **Mid-term**: Scale infrastructure and add billing/teams

---

## Phase 5 Roadmap

### Phase 5A: API + Web App Launch Path (First Iteration)
- [ ] Define API contracts (upload, job status, result payload)
- [ ] Create FastAPI project layout (routers, schemas, services)
- [ ] Port extraction pipeline into API service layer
- [ ] Add file upload endpoint (PDF) with validation
- [ ] Store uploads in Supabase Storage (or S3-compatible)
- [ ] Persist job + extraction records in Postgres
- [ ] Replace SQLite queue with Redis + RQ (or Celery)
- [ ] Add worker process for OCR/LangExtract
- [ ] Add job status endpoint (queued/running/failed/complete)
- [ ] Add job result endpoint (canonical JSON + metadata)
- [ ] Add grounding enforcement toggle + thresholds in API config
- [ ] Add metrics logging (extraction time, OCR confidence, grounding rejects)
- [ ] Add healthcheck + basic error handling
- [ ] Add API auth (Supabase Auth or Clerk JWT)
- [ ] Add usage limits / quotas (soft limits)

### Phase 5B: Next.js Frontend (First Iteration)
- [ ] Scaffold Next.js app with Tailwind
- [ ] Configure API client + env vars
- [ ] Build upload page (drag-and-drop + file validation)
- [ ] Build job status page with polling
- [ ] Build results page (canonical JSON + flags)
- [ ] Add download/export (JSON)
- [ ] Add error states and retry actions
- [ ] Add basic navigation and empty states
- [ ] Add auth flow (login/logout)

### Phase 5C: Delivery UX
- [ ] Add shareable result link (signed URL or token)
- [ ] Add email delivery (result link or JSON)
- [ ] Add PDF download of original upload (if permitted)
- [ ] Add retention notice + delete action

### Phase 5D: Deployment + Ops
- [ ] Configure production env vars
- [ ] Add Dockerfiles for API + worker
- [ ] Add migration tooling (alembic)
- [ ] Add basic logging + request tracing
- [ ] Add simple uptime check

---

## Notes & Decisions Log

**Product Surface Choice:**
- FastAPI backend + Next.js frontend

**Streamlit Hardening:**
- Not planned (Streamlit is deprecated for launch)

**API Migration:**
- Primary delivery path (Phase 5)

---

**Status**: Not Started  
**Prerequisites**: Phase 4 Complete  
**Started Date**:  
**Completed Date**:
