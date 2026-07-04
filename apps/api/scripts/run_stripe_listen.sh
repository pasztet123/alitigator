#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STRIPE_BIN="${ROOT_DIR}/.tools/stripe-cli/stripe"
FORWARD_URL="${STRIPE_FORWARD_URL:-http://127.0.0.1:8000/api/billing/webhooks/stripe}"

if [[ ! -x "${STRIPE_BIN}" ]]; then
  echo "Nie znaleziono Stripe CLI w ${STRIPE_BIN}"
  echo "Pobierz binarke albo popraw sciezke."
  exit 1
fi

echo "Forwarduje webhooki Stripe do ${FORWARD_URL}"
echo "Po starcie skopiuj wypisany whsec_... do apps/api/.env jako STRIPE_WEBHOOK_SECRET."

exec "${STRIPE_BIN}" listen --forward-to "${FORWARD_URL}"
