"""SQLite-backed job queue with retries."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

from utils import Config, get_logger


logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


@dataclass
class JobRecord:
    id: str
    status: str
    payload: Dict[str, Any]
    created_at: str
    updated_at: str
    attempts: int
    max_attempts: int
    result_path: Optional[str] = None
    last_error: Optional[str] = None
    worker_id: Optional[str] = None
    next_run_at: Optional[str] = None


class JobQueue:
    """Simple SQLite job queue."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or Config.JOBS_DB
        Config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists jobs (
                  id text primary key,
                  status text not null,
                  payload_json text not null,
                  result_path text,
                  created_at text not null,
                  updated_at text not null,
                  attempts integer not null default 0,
                  max_attempts integer not null,
                  last_error text,
                  worker_id text,
                  next_run_at text
                )
                """
            )
            conn.execute("create index if not exists jobs_status_idx on jobs(status)")
            conn.execute("create index if not exists jobs_next_run_idx on jobs(next_run_at)")
            conn.execute("create index if not exists jobs_created_at_idx on jobs(created_at)")

    def enqueue(self, payload: Dict[str, Any], max_attempts: Optional[int] = None) -> str:
        job_id = str(uuid4())
        now = _iso(_utc_now())
        attempts = 0
        max_attempts_val = max_attempts or Config.JOB_MAX_ATTEMPTS
        payload_json = json.dumps(payload)

        with self._connect() as conn:
            conn.execute(
                """
                insert into jobs (id, status, payload_json, created_at, updated_at, attempts, max_attempts)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, "queued", payload_json, now, now, attempts, max_attempts_val),
            )

        logger.info("Queued job %s", job_id)
        return job_id

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._connect() as conn:
            row = conn.execute("select * from jobs where id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_recent(self, limit: int = 20) -> Iterable[JobRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from jobs order by created_at desc limit ?",
                (limit,),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def reserve_next(self, worker_id: str) -> Optional[JobRecord]:
        now = _iso(_utc_now())

        with self._connect() as conn:
            conn.isolation_level = None
            conn.execute("begin immediate")

            row = conn.execute(
                """
                select * from jobs
                where (
                  status = 'queued'
                  or (
                    status = 'failed'
                    and attempts < max_attempts
                    and (next_run_at is null or next_run_at <= ?)
                  )
                )
                order by created_at asc
                limit 1
                """,
                (now,),
            ).fetchone()

            if not row:
                conn.execute("commit")
                return None

            attempts = int(row["attempts"]) + 1
            conn.execute(
                """
                update jobs
                set status = ?, attempts = ?, updated_at = ?, worker_id = ?, last_error = null
                where id = ?
                """,
                ("running", attempts, now, worker_id, row["id"]),
            )
            conn.execute("commit")

        return self.get(row["id"])

    def mark_complete(self, job_id: str, result_path: Optional[str] = None) -> None:
        now = _iso(_utc_now())
        with self._connect() as conn:
            conn.execute(
                """
                update jobs
                set status = ?, result_path = ?, updated_at = ?
                where id = ?
                """,
                ("complete", result_path, now, job_id),
            )

    def mark_failed(self, job_id: str, error: str, attempts: int) -> None:
        now = _utc_now()
        backoff = min(
            Config.JOB_RETRY_BASE_SECONDS * (2 ** max(attempts - 1, 0)),
            Config.JOB_RETRY_MAX_SECONDS,
        )
        next_run = now + timedelta(seconds=backoff)

        with self._connect() as conn:
            conn.execute(
                """
                update jobs
                set status = ?, last_error = ?, updated_at = ?, next_run_at = ?
                where id = ?
                """,
                ("failed", error, _iso(now), _iso(next_run), job_id),
            )

    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=row["id"],
            status=row["status"],
            payload=json.loads(row["payload_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            result_path=row["result_path"],
            last_error=row["last_error"],
            worker_id=row["worker_id"],
            next_run_at=row["next_run_at"],
        )
