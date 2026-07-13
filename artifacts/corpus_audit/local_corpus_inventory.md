# Local corpus inventory

| Source | Valid records | Unique docs | Types |
|---|---:|---:|---|
| /Users/stas/alitigator/apps/api/data/processed/eureka_interpretations.jsonl | 11425 | 10864 | interpretation=11425 |
| /Users/stas/alitigator/apps/api/data/laws/processed/excise_act_DU_2026_412.jsonl | 456 | 456 | statute=456 |
| /Users/stas/alitigator/apps/api/data/laws/processed/vat_act_DU_2025_775_codified_2026-05-05.jsonl | 418 | 418 | statute=418 |
| /Users/stas/alitigator/apps/api/data/laws/processed/vat_act_DU_2025_775.jsonl | 550 | 550 | statute=550 |
| /Users/stas/alitigator/apps/api/data/laws/processed/cit_act_DU_2026_554.jsonl | 400 | 400 | statute=400 |
| /Users/stas/alitigator/apps/api/data/laws/processed/pit_act_DU_2025_163.jsonl | 452 | 452 | statute=452 |
| /Users/stas/alitigator/apps/api/data/laws/processed/pit_act_DU_2026_592.jsonl | 454 | 454 | statute=454 |
| /Users/stas/alitigator/apps/api/data/laws/processed/pcc_act_DU_2026_191.jsonl | 22 | 22 | statute=22 |
| /Users/stas/alitigator/apps/api/data/laws/processed/inheritance_gift_tax_act_DU_2026_478.jsonl | 34 | 34 | statute=34 |
| /Users/stas/alitigator/apps/api/data/laws/processed/tax_ordinance_DU_2026_622.jsonl | 786 | 786 | statute=786 |
| /Users/stas/alitigator/apps/api/data/laws/processed/local_taxes_act_DU_2025_707.jsonl | 49 | 49 | statute=49 |
| /Users/stas/alitigator/apps/api/data/laws/processed/tax_treaties_core.jsonl | 351 | 351 | statute=351 |
| /Users/stas/alitigator/apps/api/data/laws/processed/ksef_2_0_current_bundle.jsonl | 4 | 4 | statute=4 |
| /Users/stas/alitigator/apps/api/data/laws/processed/family_foundation_primary_bundle.jsonl | 4 | 4 | statute=4 |
| /Users/stas/alitigator/apps/api/data/processed/cbosa_nsa_fsk_judgments.jsonl | 2365 | 2365 | judgment=2365 |

## Active backend

```json
{
  "runtime": {
    "read_backend": "mysql",
    "write_backend": "mysql",
    "fallback_backend": null
  },
  "backend": {
    "backend": "mysql",
    "available": true,
    "documents": 5896,
    "chunks": 98776,
    "by_type_subtype": {
      "interpretation:individual": 2275,
      "statute:codified_text": 418,
      "statute:consolidated_text": 3203
    },
    "samples": [
      {
        "document_id": "673351",
        "source_type": "interpretation",
        "source_subtype": "individual",
        "signature": "0115-KDIT3.4011.876.2025.2.RS"
      },
      {
        "document_id": "673352",
        "source_type": "interpretation",
        "source_subtype": "individual",
        "signature": "0113-KDIPT1-2.4012.1132.2025.1.KC"
      },
      {
        "document_id": "673374",
        "source_type": "interpretation",
        "source_subtype": "individual",
        "signature": "0112-KDSL1-1.4011.672.2025.2.DS"
      },
      {
        "document_id": "673375",
        "source_type": "interpretation",
        "source_subtype": "individual",
        "signature": "0115-KDIT3.4011.888.2025.1.KP"
      },
      {
        "document_id": "673376",
        "source_type": "interpretation",
        "source_subtype": "individual",
        "signature": "0112-KDIL1-3.4012.798.2025.2.KK"
      },
      {
        "document_id": "673386",
        "source_type": "interpretation",
        "source_subtype": "individual",
        "signature": "0111-KDIB3-1.4012.915.2025.1.IK"
      },
      {
        "document_id": "673409",
        "source_type": "interpretation",
        "source_subtype": "individual",
        "signature": "0113-KDIPT1-3.4012.1066.2025.1.KAK"
      },
      {
        "document_id": "673410",
        "source_type": "interpretation",
        "source_subtype": "individual",
        "signature": "0111-KDIB3-2.4012.832.2025.2.ASZ"
      },
      {
        "document_id": "673415",
        "source_type": "interpretation",
        "source_subtype": "individual",
        "signature": "0114-KDIP1-2.4012.715.2025.1.AP"
      },
      {
        "document_id": "673421",
        "source_type": "interpretation",
        "source_subtype": "individual",
        "signature": "0111-KDIB3-2.4012.868.2025.2.AR"
      },
      {
        "document_id": "673424",
        "source_type": "interpretation",
        "source_subtype": "individual",
        "signature": "0111-KDIB2-3.4014.594.2025.1.ASZ"
      },
      {
        "document_id": "673425",
        "source_type": "interpretation",
        "source_subtype": "individual",
        "signature": "0111-KDIB3-2.4012.844.2025.2.MGO"
      }
    ]
  },
  "status": "degraded",
  "missing_required_source_types": [
    "judgment"
  ]
}
```
