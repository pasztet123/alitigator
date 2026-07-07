from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional
from datetime import datetime, timezone
from uuid import uuid4

import httpx
from fastapi import HTTPException
from postgrest.exceptions import APIError

from app.auth import AuthenticatedUser
from app.supabase_client import get_supabase_service_client


@dataclass(frozen=True)
class CreditPack:
    id: str
    name: str
    credit_amount: int
    price_gross: int
    currency: str
    description: str


DEFAULT_CREDIT_PACKS = [
    {
        "id": "solo-1",
        "name": "1 kredyt",
        "credit_amount": 1,
        "price_gross": 200,
        "currency": "pln",
        "description": "1 kredyt do wykorzystania w aplikacji.",
    },
    {
        "id": "mini-5",
        "name": "5 kredytow",
        "credit_amount": 5,
        "price_gross": 1000,
        "currency": "pln",
        "description": "5 kredytow do wykorzystania w aplikacji.",
    },
    {
        "id": "pro-20",
        "name": "20 kredytow",
        "credit_amount": 20,
        "price_gross": 4000,
        "currency": "pln",
        "description": "20 kredytow do wykorzystania w aplikacji.",
    },
]
DEFAULT_CREDIT_COST_PER_QUERY = 1
DEFAULT_CREDIT_UNIT_PRICE_GROSS = 200
DEFAULT_CREDIT_CURRENCY = "pln"


def _parse_json_env(name: str, fallback: Any) -> Any:
    raw = os.getenv(name)
    if not raw:
        return fallback

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {name}") from exc


@lru_cache(maxsize=1)
def get_credit_packs() -> list[CreditPack]:
    parsed = _parse_json_env(
        "ALITIGATOR_CREDIT_PACKS_JSON",
        _parse_json_env("ALITIGATOR_TOKEN_PACKS_JSON", DEFAULT_CREDIT_PACKS),
    )
    packs: list[CreditPack] = []
    for item in parsed:
        credit_amount = item.get("credit_amount", item.get("token_amount"))
        packs.append(
            CreditPack(
                id=str(item["id"]).strip(),
                name=str(item["name"]).strip(),
                credit_amount=max(1, int(credit_amount)),
                price_gross=max(1, int(item["price_gross"])),
                currency=str(item.get("currency", "pln")).strip().lower() or "pln",
                description=str(item.get("description", "")).strip(),
            )
        )

    return packs


def get_credit_cost_per_query() -> int:
    return max(1, int(os.getenv("ALITIGATOR_CREDIT_COST_PER_QUERY", str(DEFAULT_CREDIT_COST_PER_QUERY))))


def get_credit_unit_price_gross() -> int:
    return max(1, int(os.getenv("ALITIGATOR_CREDIT_UNIT_PRICE_GROSS", str(DEFAULT_CREDIT_UNIT_PRICE_GROSS))))


def get_credit_currency() -> str:
    currency = str(os.getenv("ALITIGATOR_CREDIT_CURRENCY", DEFAULT_CREDIT_CURRENCY)).strip().lower()
    return currency or DEFAULT_CREDIT_CURRENCY


def find_credit_pack(pack_id: str) -> CreditPack:
    for pack in get_credit_packs():
        if pack.id == pack_id:
            return pack
    raise HTTPException(status_code=404, detail="Nie znaleziono wybranego pakietu kredytow.")


def build_credit_pack_for_amount(credit_amount: int) -> CreditPack:
    normalized_credit_amount = max(1, int(credit_amount))
    price_gross = normalized_credit_amount * get_credit_unit_price_gross()
    suffix = "kredyt" if normalized_credit_amount == 1 else "kredytow"
    return CreditPack(
        id=f"custom-{normalized_credit_amount}",
        name=f"{normalized_credit_amount} {suffix}",
        credit_amount=normalized_credit_amount,
        price_gross=price_gross,
        currency=get_credit_currency(),
        description=f"{normalized_credit_amount} {suffix} do wykorzystania w aplikacji.",
    )


def is_stripe_configured() -> bool:
    return bool(
        os.getenv("STRIPE_SECRET_KEY")
        and os.getenv("STRIPE_WEBHOOK_SECRET")
        and os.getenv("ALITIGATOR_STRIPE_SUCCESS_URL")
        and os.getenv("ALITIGATOR_STRIPE_CANCEL_URL")
    )


def get_stripe_product_id() -> Optional[str]:
    product_id = os.getenv("STRIPE_TOKEN_PRODUCT_ID")
    if not product_id:
        return None

    normalized = product_id.strip()
    return normalized or None


def require_supabase_client():
    client = get_supabase_service_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Supabase nie jest skonfigurowany.")
    return client


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_supabase_auth_admin_config() -> tuple[str, str]:
    supabase_url = str(os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    supabase_secret_key = str(os.getenv("SUPABASE_SECRET_KEY") or "").strip()
    if not supabase_url or not supabase_secret_key:
        raise HTTPException(status_code=503, detail="Supabase auth nie jest skonfigurowany.")
    return supabase_url, supabase_secret_key


def _list_auth_users() -> list[dict[str, Any]]:
    supabase_url, supabase_secret_key = _get_supabase_auth_admin_config()
    users: list[dict[str, Any]] = []
    page = 1
    per_page = 200

    try:
        with httpx.Client(timeout=15.0) as client:
            while True:
                response = client.get(
                    f"{supabase_url}/auth/v1/admin/users",
                    params={"page": page, "per_page": per_page},
                    headers={
                        "apikey": supabase_secret_key,
                        "Authorization": f"Bearer {supabase_secret_key}",
                    },
                )
                if response.status_code >= 400:
                    raise HTTPException(
                        status_code=502,
                        detail="Supabase auth zwrocil blad podczas pobierania listy uzytkownikow.",
                    )

                payload = response.json()
                page_users = payload.get("users") if isinstance(payload, dict) else None
                if not isinstance(page_users, list):
                    break

                users.extend(user for user in page_users if isinstance(user, dict))
                if len(page_users) < per_page:
                    break
                page += 1
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Nie udalo sie pobrac listy uzytkownikow z Supabase auth.",
        ) from exc

    return users


def _upsert_profile_from_auth_payload(auth_user: dict[str, Any]) -> Optional[dict[str, Any]]:
    user_id = str(auth_user.get("id") or "").strip()
    email = str(auth_user.get("email") or "").strip() or None
    if not user_id:
        return None

    metadata = auth_user.get("user_metadata") or {}
    full_name = metadata.get("full_name")
    client = require_supabase_client()
    try:
        response = client.table("profiles").upsert(
            {
                "id": user_id,
                "email": email,
                "full_name": str(full_name).strip() if full_name else None,
            }
        ).execute()
    except APIError as exc:
        _raise_supabase_http_error(exc)

    rows = response.data or []
    return rows[0] if rows else None


def _raise_supabase_http_error(exc: APIError) -> None:
    message = str(getattr(exc, "message", "") or "")
    code = str(getattr(exc, "code", "") or "")
    details = str(getattr(exc, "details", "") or "")
    hint = str(getattr(exc, "hint", "") or "")
    combined = " ".join(part for part in [message, details, hint] if part).lower()

    if code == "PGRST205" or "could not find the table" in combined:
        raise HTTPException(
            status_code=503,
            detail=(
                "Brakuje tabel billing/auth w Supabase. "
                "Uruchom schema z apps/api/sql/auth_billing_schema.sql i odswiez cache API."
            ),
        ) from exc

    raise HTTPException(
        status_code=502,
        detail="Supabase zwrocil blad podczas obslugi konta lub billingow.",
    ) from exc


def ensure_profile(user: AuthenticatedUser) -> dict[str, Any]:
    client = require_supabase_client()
    payload = {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
    }
    try:
        response = client.table("profiles").upsert(payload).execute()
    except APIError as exc:
        _raise_supabase_http_error(exc)
    rows = response.data or []
    if rows:
        return rows[0]

    try:
        fallback = client.table("profiles").select("*").eq("id", user.id).limit(1).execute()
    except APIError as exc:
        _raise_supabase_http_error(exc)
    fallback_rows = fallback.data or []
    if not fallback_rows:
        raise HTTPException(status_code=500, detail="Nie udalo sie przygotowac profilu uzytkownika.")
    return fallback_rows[0]


def get_profile(user_id: str) -> dict[str, Any]:
    client = require_supabase_client()
    try:
        response = client.table("profiles").select("*").eq("id", user_id).limit(1).execute()
    except APIError as exc:
        _raise_supabase_http_error(exc)
    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Profil uzytkownika nie istnieje.")
    return rows[0]


def get_profile_by_email(email: str) -> dict[str, Any]:
    normalized_email = email.strip().lower()
    if not normalized_email:
        raise HTTPException(status_code=400, detail="E-mail uzytkownika jest wymagany.")

    client = require_supabase_client()
    try:
        response = (
            client.table("profiles")
            .select("*")
            .ilike("email", normalized_email)
            .limit(1)
            .execute()
        )
    except APIError as exc:
        _raise_supabase_http_error(exc)

    rows = response.data or []
    if rows:
        return rows[0]

    for auth_user in _list_auth_users():
        auth_email = str(auth_user.get("email") or "").strip().lower()
        if auth_email != normalized_email:
            continue

        profile = _upsert_profile_from_auth_payload(auth_user)
        if profile:
            return profile
        break

    if not rows:
        raise HTTPException(status_code=404, detail="Nie znaleziono uzytkownika o podanym e-mailu.")
    return rows[0]


def update_profile(user_id: str, *, full_name: Optional[str], law_firm: Optional[str]) -> dict[str, Any]:
    client = require_supabase_client()
    payload = {
        "updated_at": utc_now_iso(),
        "full_name": (full_name or "").strip() or None,
        "law_firm": (law_firm or "").strip() or None,
    }
    try:
        response = client.table("profiles").update(payload).eq("id", user_id).execute()
    except APIError as exc:
        _raise_supabase_http_error(exc)
    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=500, detail="Nie udalo sie zapisac profilu.")
    return rows[0]


def get_credit_balance(user_id: str) -> int:
    client = require_supabase_client()
    try:
        response = client.table("credit_ledger").select("amount").eq("user_id", user_id).execute()
    except APIError as exc:
        _raise_supabase_http_error(exc)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail="Supabase billing jest chwilowo niedostepny.",
        ) from exc
    rows = response.data or []
    return sum(int(row.get("amount") or 0) for row in rows)


def create_ledger_entry(
    *,
    user_id: str,
    amount: int,
    entry_type: str,
    source_type: str,
    source_id: str,
    description: str,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    client = require_supabase_client()
    payload = {
        "user_id": user_id,
        "amount": amount,
        "entry_type": entry_type,
        "source_type": source_type,
        "source_id": source_id,
        "description": description,
        "metadata": metadata or {},
    }

    existing = (
        client.table("credit_ledger")
        .select("*")
        .eq("source_type", source_type)
        .eq("source_id", source_id)
        .limit(1)
        .execute()
    )
    existing_rows = existing.data or []
    if existing_rows:
        return existing_rows[0]

    response = client.table("credit_ledger").insert(payload).execute()
    rows = response.data or []
    return rows[0] if rows else None


def grant_credits_to_user(
    *,
    admin_user: AuthenticatedUser,
    target_email: str,
    credit_amount: int,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    if credit_amount <= 0:
        raise HTTPException(status_code=400, detail="Liczba kredytow musi byc dodatnia.")

    target_profile = get_profile_by_email(target_email)
    target_user_id = str(target_profile["id"])
    entry = create_ledger_entry(
        user_id=target_user_id,
        amount=credit_amount,
        entry_type="adjustment",
        source_type="admin_grant",
        source_id=str(uuid4()),
        description=reason.strip() if reason and reason.strip() else f"Grant admina od {admin_user.email or admin_user.id}",
        metadata={
            "granted_by_user_id": admin_user.id,
            "granted_by_email": admin_user.email,
            "target_email": target_profile.get("email"),
            "reason": reason.strip() if reason else None,
        },
    )
    return {
        "profile": target_profile,
        "credit_balance": get_credit_balance(target_user_id),
        "ledger_entry": entry,
    }


def list_profiles_with_credit_balances() -> list[dict[str, Any]]:
    client = require_supabase_client()

    try:
        profiles_response = client.table("profiles").select("*").order("created_at", desc=False).execute()
    except APIError as exc:
        _raise_supabase_http_error(exc)

    try:
        ledger_response = client.table("credit_ledger").select("user_id,amount").execute()
    except APIError as exc:
        message = str(getattr(exc, "message", "") or "")
        code = str(getattr(exc, "code", "") or "")
        details = str(getattr(exc, "details", "") or "")
        hint = str(getattr(exc, "hint", "") or "")
        combined = " ".join(part for part in [message, details, hint] if part).lower()
        if code == "PGRST205" or "could not find the table" in combined:
            ledger_rows = []
        else:
            _raise_supabase_http_error(exc)
            ledger_rows = []

    profiles = profiles_response.data or []
    if "ledger_rows" not in locals():
        ledger_rows = ledger_response.data or []
    auth_users = _list_auth_users()
    balances: dict[str, int] = {}
    for row in ledger_rows:
        user_id = str(row.get("user_id") or "").strip()
        if not user_id:
            continue
        balances[user_id] = balances.get(user_id, 0) + int(row.get("amount") or 0)

    profiles_by_id = {
        str(profile.get("id") or "").strip(): profile
        for profile in profiles
        if str(profile.get("id") or "").strip()
    }
    result: list[dict[str, Any]] = []
    seen_user_ids: set[str] = set()

    for auth_user in auth_users:
        user_id = str(auth_user.get("id") or "").strip()
        if not user_id:
            continue

        profile = profiles_by_id.get(user_id, {})
        metadata = auth_user.get("user_metadata") or {}
        seen_user_ids.add(user_id)
        result.append(
            {
                **profile,
                "id": user_id,
                "email": profile.get("email") or auth_user.get("email"),
                "full_name": profile.get("full_name") or metadata.get("full_name"),
                "is_admin": bool(profile.get("is_admin")),
                "created_at": profile.get("created_at") or auth_user.get("created_at"),
                "credit_balance": balances.get(user_id, 0),
            }
        )

    for profile in profiles:
        user_id = str(profile.get("id") or "").strip()
        if not user_id or user_id in seen_user_ids:
            continue
        result.append(
            {
                **profile,
                "is_admin": bool(profile.get("is_admin")),
                "credit_balance": balances.get(user_id, 0),
            }
        )

    result.sort(
        key=lambda profile: (
            str(profile.get("created_at") or ""),
            str(profile.get("email") or ""),
            str(profile.get("id") or ""),
        )
    )
    return result


def ensure_sufficient_credits(user_id: str, required_credits: int) -> None:
    if required_credits <= 0:
        return

    balance = get_credit_balance(user_id)
    if balance < required_credits:
        raise HTTPException(
            status_code=402,
            detail=(
                f"Za malo kredytow. Potrzeba {required_credits}, a na koncie jest {balance}. "
                "Doladuj kredyty i sprobuj ponownie."
            ),
        )


def consume_credit_for_chat(*, user_id: str, model: str, chat_id: str, request_id: str) -> None:
    credit_cost = get_credit_cost_per_query()
    ensure_sufficient_credits(user_id, credit_cost)
    create_ledger_entry(
        user_id=user_id,
        amount=-credit_cost,
        entry_type="usage",
        source_type="chat_completion",
        source_id=request_id,
        description=f"Koszt zapytania w watku {chat_id}",
        metadata={"model": model, "chat_id": chat_id, "credit_cost": credit_cost},
    )


def _load_stripe_module():
    try:
        import stripe  # type: ignore
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Biblioteka Stripe nie jest jeszcze zainstalowana na backendzie.",
        ) from exc
    return stripe


def create_checkout_session(
    *,
    user: AuthenticatedUser,
    pack: CreditPack,
    success_url: Optional[str] = None,
    cancel_url: Optional[str] = None,
) -> dict[str, str]:
    if not is_stripe_configured():
        raise HTTPException(status_code=503, detail="Stripe nie jest jeszcze skonfigurowany.")

    stripe = _load_stripe_module()
    stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
    client = require_supabase_client()
    profile = get_profile(user.id)
    order_id = str(uuid4())

    stripe_customer_id = profile.get("stripe_customer_id")
    if not stripe_customer_id:
        customer = stripe.Customer.create(
            email=user.email,
            name=profile.get("full_name") or user.full_name or user.email,
            metadata={"user_id": user.id},
        )
        stripe_customer_id = customer["id"]
        client.table("profiles").update({"stripe_customer_id": stripe_customer_id}).eq("id", user.id).execute()

    stripe_product_id = get_stripe_product_id()
    price_data: dict[str, Any] = {
        "currency": pack.currency,
        "unit_amount": pack.price_gross,
    }
    if stripe_product_id:
        price_data["product"] = stripe_product_id
    else:
        price_data["product_data"] = {
            "name": pack.name,
            "description": pack.description or f"Doladowanie {pack.credit_amount} kredytow do aLitigatora",
        }

    session = stripe.checkout.Session.create(
        mode="payment",
        customer=stripe_customer_id,
        success_url=success_url or os.getenv("ALITIGATOR_STRIPE_SUCCESS_URL"),
        cancel_url=cancel_url or os.getenv("ALITIGATOR_STRIPE_CANCEL_URL"),
        client_reference_id=user.id,
        metadata={
            "order_id": order_id,
            "user_id": user.id,
            "pack_id": pack.id,
            "credit_amount": str(pack.credit_amount),
        },
        line_items=[
            {
                "quantity": 1,
                "price_data": price_data,
            }
        ],
    )

    client.table("credit_orders").insert(
        {
            "id": order_id,
            "user_id": user.id,
            "pack_id": pack.id,
            "pack_name": pack.name,
            "token_amount": pack.credit_amount,
            "currency": pack.currency,
            "unit_amount": pack.price_gross,
            "stripe_customer_id": stripe_customer_id,
            "stripe_checkout_session_id": session["id"],
            "checkout_url": session["url"],
            "status": "pending",
            "metadata": {"description": pack.description, "credit_amount": pack.credit_amount},
        }
    ).execute()

    return {
        "order_id": order_id,
        "checkout_url": session["url"],
        "checkout_session_id": session["id"],
    }


def mark_order_status(*, order_id: str, status: str, payment_intent_id: Optional[str] = None) -> None:
    client = require_supabase_client()
    payload: dict[str, Any] = {"status": status, "updated_at": utc_now_iso()}
    if payment_intent_id:
        payload["stripe_payment_intent_id"] = payment_intent_id
    client.table("credit_orders").update(payload).eq("id", order_id).execute()


def apply_topup_from_checkout_session(session: dict[str, Any]) -> None:
    client = require_supabase_client()
    metadata = session.get("metadata") or {}
    order_id = str(metadata.get("order_id") or "").strip()
    user_id = str(metadata.get("user_id") or session.get("client_reference_id") or "").strip()

    if not order_id or not user_id:
        raise HTTPException(status_code=400, detail="Webhook Stripe nie zawiera identyfikatorow zamowienia.")

    response = client.table("credit_orders").select("*").eq("id", order_id).limit(1).execute()
    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Nie znaleziono zamowienia Stripe do rozliczenia.")

    order = rows[0]
    credit_amount = int(
        order.get("token_amount")
        or metadata.get("credit_amount")
        or metadata.get("token_amount")
        or 0
    )
    if credit_amount <= 0:
        raise HTTPException(status_code=400, detail="Zamowienie Stripe nie ma poprawnej liczby kredytow.")

    create_ledger_entry(
        user_id=user_id,
        amount=credit_amount,
        entry_type="topup",
        source_type="stripe_checkout",
        source_id=str(session.get("id")),
        description=f"Doladowanie kredytow pakietem {order.get('pack_name') or order.get('pack_id')}",
        metadata={
            "order_id": order_id,
            "stripe_payment_intent_id": session.get("payment_intent"),
            "stripe_customer_id": session.get("customer"),
            "credit_amount": credit_amount,
        },
    )

    client.table("credit_orders").update(
        {
            "status": "credited",
            "updated_at": utc_now_iso(),
            "credited_at": utc_now_iso(),
            "stripe_payment_intent_id": session.get("payment_intent"),
        }
    ).eq("id", order_id).execute()


def get_checkout_session(session_id: str) -> dict[str, Any]:
    normalized_session_id = session_id.strip()
    if not normalized_session_id:
        raise HTTPException(status_code=400, detail="Brakuje identyfikatora sesji Stripe.")

    if not is_stripe_configured():
        raise HTTPException(status_code=503, detail="Stripe nie jest jeszcze skonfigurowany.")

    stripe = _load_stripe_module()
    stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

    try:
        session = stripe.checkout.Session.retrieve(normalized_session_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Nie udalo sie pobrac sesji Stripe.") from exc

    return dict(session)
