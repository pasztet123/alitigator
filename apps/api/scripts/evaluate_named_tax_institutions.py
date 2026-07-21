"""Emit a JSON report for the deterministic named-institution regression set."""

from __future__ import annotations

import json

from app.legal_institutions.evaluate import evaluate_dictionary_cases
from tests.fixtures.named_institution_cases import negative_cases, positive_cases


if __name__ == "__main__":
    print(
        json.dumps(
            evaluate_dictionary_cases(positive_cases(), negative_cases()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
