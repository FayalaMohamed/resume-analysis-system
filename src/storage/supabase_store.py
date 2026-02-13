"""Supabase storage + database integration."""

from __future__ import annotations

import hashlib
import mimetypes
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from utils import Config, get_logger

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Client = Any
    SUPABASE_AVAILABLE = False


logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _guess_content_type(path: Path) -> str:
    content_type, _ = mimetypes.guess_type(path.name)
    return content_type or "application/octet-stream"


class SupabaseStore:
    """Supabase storage + DB operations for resumes and extractions."""

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        bucket: Optional[str] = None,
        schema: Optional[str] = None,
        resumes_table: Optional[str] = None,
        extractions_table: Optional[str] = None,
        client: Optional[Client] = None,
    ) -> None:
        if not SUPABASE_AVAILABLE:
            raise RuntimeError("Supabase client not installed")

        self.url = url or Config.SUPABASE_URL
        self.key = key or Config.SUPABASE_SERVICE_ROLE_KEY
        self.bucket = bucket or Config.SUPABASE_BUCKET
        self.schema = schema or Config.SUPABASE_SCHEMA
        self.resumes_table = resumes_table or Config.SUPABASE_RESUMES_TABLE
        self.extractions_table = extractions_table or Config.SUPABASE_EXTRACTIONS_TABLE

        if not self.url or not self.key:
            raise ValueError("Supabase URL and service role key are required")

        self.client = client or create_client(self.url, self.key)

    def upload_resume_pdf(
        self,
        pdf_path: Path,
        storage_path: str,
        content_type: Optional[str] = None,
        upsert: bool = True,
    ) -> Dict[str, Any]:
        if not pdf_path.exists():
            raise FileNotFoundError(f"Resume not found: {pdf_path}")

        resolved_type = content_type or _guess_content_type(pdf_path)
        file_options = {"content-type": resolved_type}
        if upsert:
            file_options["x-upsert"] = "true"

        data = pdf_path.read_bytes()
        response = self.client.storage.from_(self.bucket).upload(
            storage_path,
            data,
            file_options=file_options,
        )

        return {
            "bucket": self.bucket,
            "path": storage_path,
            "content_type": resolved_type,
            "response": response,
        }

    def create_resume_record(
        self,
        resume_id: str,
        original_filename: str,
        storage_path: str,
        content_type: str,
        size_bytes: int,
        sha256: str,
        retention_days: int,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = _utc_now()
        expires_at = now + timedelta(days=retention_days)

        payload = {
            "id": resume_id,
            "user_id": user_id,
            "original_filename": original_filename,
            "storage_bucket": self.bucket,
            "storage_path": storage_path,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "uploaded_at": _iso(now),
            "retention_days": retention_days,
            "expires_at": _iso(expires_at),
        }
        if metadata:
            payload["metadata"] = metadata

        response = self.client.table(self.resumes_table).insert(payload).execute()
        return {
            "resume_id": resume_id,
            "data": response.data,
        }

    def create_extraction_record(
        self,
        resume_id: str,
        extraction_json: Dict[str, Any],
        extractor: str,
        status: str = "complete",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record_id = str(uuid4())
        payload = {
            "id": record_id,
            "resume_id": resume_id,
            "extractor": extractor,
            "status": status,
            "extraction_json": extraction_json,
            "created_at": _iso(_utc_now()),
        }
        if metadata:
            payload["metadata"] = metadata

        response = self.client.table(self.extractions_table).insert(payload).execute()
        return {
            "extraction_id": record_id,
            "data": response.data,
        }

    def store_resume_and_extraction(
        self,
        pdf_path: Path,
        extraction_json: Dict[str, Any],
        extractor: str,
        user_id: Optional[str] = None,
        retention_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        storage_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        resume_id = str(uuid4())
        resolved_storage_path = storage_path or f"resumes/{resume_id}/{pdf_path.name}"
        resolved_retention = retention_days or Config.RESUME_RETENTION_DAYS

        sha256 = _sha256_file(pdf_path)
        size_bytes = pdf_path.stat().st_size
        content_type = _guess_content_type(pdf_path)

        logger.info("Uploading resume to Supabase storage")
        upload_result = self.upload_resume_pdf(
            pdf_path,
            resolved_storage_path,
            content_type=content_type,
        )

        logger.info("Creating resume record in Supabase")
        resume_record = self.create_resume_record(
            resume_id=resume_id,
            original_filename=pdf_path.name,
            storage_path=resolved_storage_path,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=sha256,
            retention_days=resolved_retention,
            user_id=user_id,
            metadata=metadata,
        )

        logger.info("Creating extraction record in Supabase")
        extraction_record = self.create_extraction_record(
            resume_id=resume_id,
            extraction_json=extraction_json,
            extractor=extractor,
            metadata=metadata,
        )

        return {
            "resume_id": resume_id,
            "storage": upload_result,
            "resume_record": resume_record,
            "extraction_record": extraction_record,
        }

    def apply_retention_policy(self, now: Optional[datetime] = None, limit: int = 200) -> Dict[str, Any]:
        current = now or _utc_now()
        query = (
            self.client.table(self.resumes_table)
            .select("id,storage_path,storage_bucket,expires_at")
            .lt("expires_at", _iso(current))
            .is_("deleted_at", "null")
            .limit(limit)
        )
        response = query.execute()
        rows = response.data or []

        deleted_ids = []
        for row in rows:
            resume_id = row.get("id")
            storage_bucket = row.get("storage_bucket") or self.bucket
            storage_path = row.get("storage_path")

            if storage_path:
                self.client.storage.from_(storage_bucket).remove([storage_path])

            self.client.table(self.resumes_table).update(
                {"deleted_at": _iso(current)}
            ).eq("id", resume_id).execute()
            deleted_ids.append(resume_id)

        return {
            "deleted_count": len(deleted_ids),
            "resume_ids": deleted_ids,
        }
