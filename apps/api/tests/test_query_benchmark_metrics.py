from __future__ import annotations

from scripts.run_legal_rag_v2_ab import aggregate_metrics, precision_at


def test_precision_at_uses_only_the_requested_top_k() -> None:
    assert precision_at(["A", "B"], ["A", "X", "B"], 2) == 0.5
    assert precision_at([], ["A"], 5) is None


def test_aggregate_metrics_reports_activation_measurements_per_variant() -> None:
    summary = aggregate_metrics([
        {
            "variants": [
                {
                    "variant": "A",
                    "retrieval": {"authority_recall_at_5": 0.5, "precision_at_5": 0.25},
                    "quality": {"wrong_neighbor_rate_at_5": None, "lock_preservation_rate": None, "evidence_rate": None, "no_result": 0.0},
                    "operational": {"latency_ms": 10},
                },
                {
                    "variant": "B",
                    "retrieval": {"authority_recall_at_5": 1.0, "precision_at_5": 0.8},
                    "quality": {"wrong_neighbor_rate_at_5": 0.2, "lock_preservation_rate": 1.0, "evidence_rate": 0.75, "no_result": 0.0},
                    "operational": {"latency_ms": 20},
                },
            ]
        }
    ])

    assert summary["A"]["recall_at_5"] == 0.5
    assert summary["A"]["latency_ms_p95"] == 10
    assert summary["B"]["precision_at_5"] == 0.8
    assert summary["B"]["lock_preservation_rate"] == 1.0
    assert summary["B"]["evidence_rate"] == 0.75
