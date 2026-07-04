from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException

from app.auth import AuthenticatedUser
from app.supabase_client import get_supabase_service_client


@dataclass(frozen=True)
class TokenPack:
    id: str
    name: str
    token_amount: int
    price_gross: int
    currency: str
    description: str


DEFAULT_TOKEN_PACKS = [
    {
        "id": "starter-25k",
        "name": "Starter 25k",
        "token_amount": 25000,
        "price_gross": 4900,
        "currency": "pln",
        "description": "Pakiet startowy do pojedynczych researchy i konsultacji.",
    },
    {
        "id": "pro-120k",
        "name": "Pro 120k",
        "token_amount": 120000,
        "price_gross": 19900,
        "currency": "pln",
        "description": "Najlepszy do regularnej pracy zespolu kancelarii.",
    },
    {
        "id": "team-350k",
        "name": "Team 350k",
        "token_amount": 350000,
        "price_gross": 49900,
        "currency": "pln",
        "description": "Wiekszy zapas do intensywnej pracy nad opiniami i pismami.",
    },
]

DEFAULT_MODEL_TOKEN_COSTS = {
    "claude-haiku-4-5-20251001": 2500,
    "claude-sonnet-4-6": 9000,
    "claude-opus-4-8": 18000,
}


def _parse_json_env(name: str, fallback: Any) -> Any:
    raw = os.getenv(name)
    if not raw:
        return fallback

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {name}") from exc


@lru_cache(maxsize=1)
def get_token_packs() -> list[TokenPack]:
    parsed = _parse_json_env("ALITIGATOR_TOKEN_PACKS_JSON", DEFAULT_TOKEN_PACKS)
    packs: list[TokenPack] = []
    for item in parsed:
        packs.append(
            TokenPack(
                id=str(item["id"]).strip(),
                name=str(item["name"]).strip(),
                token_amount=max(1, int(item["token_amount"])),
                price_gross=max(1, int(item["price_gross"])),
                currency=str(item.get("currency", "pln")).strip().lower() or "pln",
                description=str(item.get("description", "")).strip(),
            )
        )

    return packs


@lru_cache(maxsize=1)
def get_model_token_costs() -> dict[str, int]:
    parsed = _parse_json_env("ALITIGATOR_MODEL_TOKEN_COSTS_JSON", DEFAULT_MODEL_TOKEN_COSTS)
    return {str(model): max(0, int(cost)) for model, cost in parsed.items()}


def get_model_token_cost(model: str) -> int:
    return get_model_token_costs().get(model, get_model_token_costs().get("claude-sonnet-4-6", 9000))


def find_token_pack(pack_id: str) -> TokenPack:
    for pack in get_token_packs():
        if pack.id == pack_id:
            return pack
    raise HTTPException(status_code=404, detail="Nie znaleziono wybranego pakietu tokenow.")


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


def ensure_profile(user: AuthenticatedUser) -> dict[str, Any]:
    client = require_supabase_client()
    payload = {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
    }
    response = client.table("profiles").upsert(payload).execute()
    rows = response.data or []
    if rows:
        return rows[0]

    fallback = client.table("profiles").select("*").eq("id", user.id).limit(1).execute()
    fallback_rows = fallback.data or []
    if not fallback_rows:
        raise HTTPException(status_code=500, detail="Nie udalo sie przygotowac profilu uzytkownika.")
    return fallback_rows[0]


def get_profile(user_id: str) -> dict[str, Any]:
    client = require_supabase_client()
    response = client.table("profiles").select("*").eq("id", user_id).limit(1).execute()
    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Profil uzytkownika nie istnieje.")
    return rows[0]


def update_profile(user_id: str, *, full_name: Optional[str], law_firm: Optional[str]) -> dict[str, Any]:
    client = require_supabase_client()
    payload = {
        "updated_at": utc_now_iso(),
        "full_name": (full_name or "").strip() or None,
        "law_firm": (law_firm or "").strip() or None,
    }
    response = client.table("profiles").update(payload).eq("id", user_id).execute()
    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=500, detail="Nie udalo sie zapisac profilu.")
    return rows[0]


def get_token_balance(user_id: str) -> int:
    client = require_supabase_client()
    response = client.table("credit_ledger").select("amount").eq("user_id", user_id).execute()
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


def ensure_sufficient_tokens(user_id: str, required_tokens: int) -> None:
    if required_tokens <= 0:
        return

    balance = get_token_balance(user_id)
    if balance < required_tokens:
        raise HTTPException(
            status_code=402,
            detail=(
                f"Za malo tokenow. Potrzeba {required_tokens}, a na koncie jest {balance}. "
                "Doladuj pakiet i sprobuj ponownie."
            ),
        )


def consume_tokens_for_chat(*, user_id: str, model: str, chat_id: str, request_id: str) -> None:
    token_cost = get_model_token_cost(model)
    ensure_sufficient_tokens(user_id, token_cost)
    create_ledger_entry(
        user_id=user_id,
        amount=-token_cost,
        entry_type="usage",
        source_type="chat_completion",
        source_id=request_id,
        description=f"Koszt odpowiedzi modelu {model} w watku {chat_id}",
        metadata={"model": model, "chat_id": chat_id},
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
    pack: TokenPack,
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
            "description": pack.description or f"Doladowanie {pack.token_amount} tokenow do aLitigatora",
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
            "token_amount": str(pack.token_amount),
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
            "token_amount": pack.token_amount,
            "currency": pack.currency,
            "unit_amount": pack.price_gross,
            "stripe_customer_id": stripe_customer_id,
            "stripe_checkout_session_id": session["id"],
            "checkout_url": session["url"],
            "status": "pending",
            "metadata": {"description": pack.description},
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
    token_amount = int(order.get("token_amount") or metadata.get("token_amount") or 0)
    if token_amount <= 0:
        raise HTTPException(status_code=400, detail="Zamowienie Stripe nie ma poprawnej liczby tokenow.")

    create_ledger_entry(
        user_id=user_id,
        amount=token_amount,
        entry_type="topup",
        source_type="stripe_checkout",
        source_id=str(session.get("id")),
        description=f"Doladowanie pakietem {order.get('pack_name') or order.get('pack_id')}",
        metadata={
            "order_id": order_id,
            "stripe_payment_intent_id": session.get("payment_intent"),
            "stripe_customer_id": session.get("customer"),
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
