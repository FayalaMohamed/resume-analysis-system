"""Retention policy runner for Supabase storage."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import Config, get_logger
from storage import SupabaseStore, SUPABASE_AVAILABLE


logger = get_logger(__name__)


def main() -> int:
    if not Config.SUPABASE_ENABLED:
        print("Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")
        return 1
    if not SUPABASE_AVAILABLE:
        print("Supabase client not installed. Add the 'supabase' package.")
        return 1

    store = SupabaseStore()
    result = store.apply_retention_policy()
    print(f"Deleted {result['deleted_count']} expired resumes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
