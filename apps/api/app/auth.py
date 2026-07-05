from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import httpx
from fastapi import Header, HTTPException

from app.supabase_client import is_supabase_configured


@dataclass(frozen=True)
class AuthenticatedUser:
    id: str
    email: Optional[str]
    full_name: Optional[str]


def is_admin_user(user: AuthenticatedUser) -> bool:
    configured = os.getenv("ALITIGATOR_ADMIN_EMAILS", "")
    allowed_emails = {
        email.strip().lower()
        for email in configured.split(",")
        if email.strip()
    }
    return bool(user.email and user.email.strip().lower() in allowed_emails)


def _extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Brak naglowka Authorization.")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Oczekiwano tokenu Bearer.")

    return token.strip()


async def get_current_user(authorization: Optional[str] = Header(default=None)) -> AuthenticatedUser:
    if not is_supabase_configured():
        raise HTTPException(status_code=503, detail="Supabase auth nie jest skonfigurowany.")

    token = _extract_bearer_token(authorization)
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_secret_key = os.getenv("SUPABASE_SECRET_KEY")

    if not supabase_url or not supabase_secret_key:
        raise HTTPException(status_code=503, detail="Supabase auth nie jest skonfigurowany.")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{supabase_url.rstrip('/')}/auth/v1/user",
                headers={
                    "apikey": supabase_secret_key,
                    "Authorization": f"Bearer {token}",
                },
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Nie udalo sie zweryfikowac sesji uzytkownika.") from exc

    if response.status_code in {401, 403}:
        raise HTTPException(status_code=401, detail="Sesja wygasla albo jest nieprawidlowa.")
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail="Supabase auth zwrocil blad podczas weryfikacji sesji.")

    payload = response.json()
    user_id = str(payload.get("id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Nie znaleziono identyfikatora uzytkownika w sesji.")

    metadata = payload.get("user_metadata") or {}
    full_name = metadata.get("full_name")
    return AuthenticatedUser(
        id=user_id,
        email=payload.get("email"),
        full_name=str(full_name).strip() if full_name else None,
    )
