"""SQLite-backed metrics store for extraction observability."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from utils import Config, get_logger


logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


@dataclass
class MetricEvent:
    id: str
    event_type: str
    created_at: str
    payload: Dict[str, Any]


class MetricsStore:
    """Persist and query metrics events."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or Config.METRICS_DB
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
                create table if not exists metrics_events (
                  id text primary key,
                  event_type text not null,
                  created_at text not null,
                  payload_json text not null
                )
                """
            )
            conn.execute("create index if not exists metrics_events_type_idx on metrics_events(event_type)")
            conn.execute("create index if not exists metrics_events_created_idx on metrics_events(created_at)")

    def record_event(self, event_type: str, payload: Dict[str, Any]) -> str:
        from uuid import uuid4

        event_id = str(uuid4())
        now = _iso(_utc_now())
        payload_json = json.dumps(payload)

        with self._connect() as conn:
            conn.execute(
                """
                insert into metrics_events (id, event_type, created_at, payload_json)
                values (?, ?, ?, ?)
                """,
                (event_id, event_type, now, payload_json),
            )

        return event_id

    def record_extraction(self, payload: Dict[str, Any]) -> str:
        return self.record_event("extraction", payload)

    def list_events(self, limit: int = 200, event_type: Optional[str] = None) -> Iterable[MetricEvent]:
        with self._connect() as conn:
            if event_type:
                rows = conn.execute(
                    """
                    select * from metrics_events
                    where event_type = ?
                    order by created_at desc
                    limit ?
                    """,
                    (event_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "select * from metrics_events order by created_at desc limit ?",
                    (limit,),
                ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def _row_to_event(self, row: sqlite3.Row) -> MetricEvent:
        return MetricEvent(
            id=row["id"],
            event_type=row["event_type"],
            created_at=row["created_at"],
            payload=json.loads(row["payload_json"]),
        )
