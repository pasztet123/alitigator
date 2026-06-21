from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.eureka_ingest import DEFAULT_CONCURRENCY, DEFAULT_PAGE_SIZE, DEFAULT_REQUEST_TIMEOUT, FetchConfig, fetch_latest_interpretations

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = BASE_DIR / "data" / "processed" / "eureka_batch_state.json"


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0

    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def format_error(error: Exception) -> str:
    message = str(error).strip()
    if message:
        return f"{type(error).__name__}: {message}"
    return type(error).__name__


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Eureka imports in smaller resumable page batches.")
    parser.add_argument("--target-count", type=int, default=10000)
    parser.add_argument("--start-page", type=int, default=0)
    parser.add_argument("--batch-pages", type=int, default=10)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--retry-count", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    parser.add_argument("--category", default="Interpretacja indywidualna")
    parser.add_argument("--sort", default="DT_WYD,desc")
    parser.add_argument("--raw-output", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-empty-batches", type=int, default=5)
    return parser.parse_args()


async def run_batches(args: argparse.Namespace) -> dict[str, Any]:
    base_config = FetchConfig(
        page_size=max(1, args.page_size),
        concurrency=max(1, args.concurrency),
        retry_count=max(1, args.retry_count),
        request_timeout=max(1.0, args.request_timeout),
        pause_seconds=max(0.0, args.pause_seconds),
        category=args.category.strip() or None,
        sort=args.sort,
        raw_output_path=args.raw_output,
        output_path=args.output_path,
        overwrite=False,
        limit=None,
    )

    processed_path = base_config.processed_output_path
    raw_path = base_config.resolved_raw_output_path
    state_path = Path(args.state_path)

    if args.overwrite:
        if processed_path.exists():
            processed_path.unlink()
        if raw_path.exists():
            raw_path.unlink()
        if state_path.exists():
            state_path.unlink()

    saved_state = load_state(state_path) if args.resume and not args.overwrite else None

    if saved_state is not None:
        current_total = int(saved_state.get("current_total") or 0)
        page_start = max(0, int(saved_state.get("next_start_page") or args.start_page))
    else:
        current_total = count_jsonl_rows(processed_path)
        page_start = max(0, args.start_page)

    empty_batches = 0
    batch_index = 0
    last_result: dict[str, Any] | None = None

    write_state(state_path, {
        "status": "running",
        "current_total": current_total,
        "next_start_page": page_start,
        "target_count": args.target_count,
    })

    while current_total < args.target_count:
        batch_end = page_start + max(1, args.batch_pages)
        batch_config = replace(
            base_config,
            start_page=page_start,
            max_pages=batch_end,
            overwrite=False,
            limit=None,
        )

        print(json.dumps({
            "event": "batch_start",
            "batch_index": batch_index,
            "page_start": page_start,
            "page_end": batch_end,
            "current_total": current_total,
            "target_count": args.target_count,
        }, ensure_ascii=False), flush=True)

        try:
            result = await fetch_latest_interpretations(batch_config)
            last_result = result
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({
                "event": "batch_error",
                "batch_index": batch_index,
                "page_start": page_start,
                "page_end": batch_end,
                "error": format_error(exc),
            }, ensure_ascii=False), flush=True)
            write_state(state_path, {
                "status": "running",
                "current_total": current_total,
                "next_start_page": batch_end,
                "target_count": args.target_count,
                "last_error": format_error(exc),
            })
            page_start = batch_end
            batch_index += 1
            continue

        added = int(result.get("count") or 0)
        next_total = current_total + added

        print(json.dumps({
            "event": "batch_done",
            "batch_index": batch_index,
            "page_start": page_start,
            "page_end": batch_end,
            "added": added,
            "current_total": next_total,
            "failed_ids": len(result.get("failed_ids", [])),
        }, ensure_ascii=False), flush=True)

        if added <= 0:
            empty_batches += 1
        else:
            empty_batches = 0

        if empty_batches >= max(1, args.max_empty_batches):
            break

        current_total = next_total
        page_start = batch_end
        write_state(state_path, {
            "status": "running",
            "current_total": current_total,
            "next_start_page": page_start,
            "target_count": args.target_count,
        })
        batch_index += 1
        if args.pause_seconds:
            await asyncio.sleep(args.pause_seconds)

    final_result = {
        "target_count": args.target_count,
        "current_total": current_total,
        "processed_output_path": str(processed_path),
        "raw_output_path": str(raw_path),
        "last_batch": last_result,
        "next_start_page": page_start,
    }
    write_state(state_path, {
        "status": "finished" if current_total >= args.target_count else "stopped",
        "current_total": current_total,
        "next_start_page": page_start,
        "target_count": args.target_count,
    })
    return final_result


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_batches(args))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
