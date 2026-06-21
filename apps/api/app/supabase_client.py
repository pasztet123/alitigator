from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from supabase import Client, create_client


def is_supabase_configured() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SECRET_KEY"))


@lru_cache(maxsize=1)
def get_supabase_service_client() -> Optional[Client]:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_secret_key = os.getenv("SUPABASE_SECRET_KEY")

    if not supabase_url or not supabase_secret_key:
        return None

    return create_client(supabase_url, supabase_secret_key)
