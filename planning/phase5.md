# Phase 5: Product Surface and Delivery

**Goal**: Decide and implement the product surface for the micro SaaS launch, balancing speed-to-market with maintainability.

---

## Overview

You already have a feature-rich Streamlit app in `app.py` with OCR, unified extraction, and LangExtract. Phase 5 focuses on choosing the right product surface for launch and implementing the minimum viable delivery stack.

---

## Option A: Keep Streamlit (Fastest to Launch)

**Pros**
- Minimal engineering
- Features already wired
- Lowest time to market

**Cons**
- Limited auth and multi-user workflows
- Harder to brand and customize UI deeply
- Not ideal for billing or production workflows

**Best Use**
- Private beta
- First paid users to validate demand

**How to harden Streamlit for micro SaaS:**
- Add basic auth (Streamlit Cloud auth or reverse proxy)
- Add async job status UI (polling or periodic refresh)
- Persist results in a lightweight DB (SQLite -> Postgres later)
- Wrap LangExtract in background jobs to avoid blocking UI

---

## Option B: FastAPI + Lightweight Frontend (Recommended if monetizing)

**Pros**
- Real API, better auth + billing
- Flexible UI and branding
- Easier to expand later

**Cons**
- Higher initial build time

**Best Use**
- If you plan Stripe billing and self-serve onboarding

**Suggested lean stack:**
- Backend: FastAPI + RQ or Celery for background jobs
- DB: Postgres (Supabase or Railway)
- Storage: Local disk or S3-compatible (R2 or Supabase Storage)
- Frontend: Next.js or simple React + Tailwind
- Auth: Supabase Auth or Clerk

**Migration path from Streamlit:**
1. Wrap extraction pipeline into a FastAPI service
2. Keep Streamlit for internal testing only
3. Build minimal web UI for upload + results
4. Add async job polling and result viewing

---

## Recommended Decision

- **Short-term**: Launch with Streamlit to validate traction
- **Mid-term**: Migrate to FastAPI + simple frontend once usage proves demand

---

## Phase 5 Roadmap

### Phase 5A: Streamlit Launch Path
- [ ] Add basic auth
- [ ] Add async job status view
- [ ] Persist extracted results
- [ ] Add usage limits or soft quotas

### Phase 5B: API Migration Path
- [ ] Build FastAPI extraction endpoint
- [ ] Add job queue + workers
- [ ] Build minimal upload/results UI
- [ ] Add auth and billing stub

---

## Notes & Decisions Log

**Product Surface Choice:**
- 

**Streamlit Hardening:**
- 

**API Migration:**
- 

---

**Status**: Not Started  
**Prerequisites**: Phase 4 Complete  
**Started Date**:  
**Completed Date**:
