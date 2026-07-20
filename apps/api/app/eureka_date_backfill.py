from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from app.eureka_ingest import (
    DEFAULT_CONCURRENCY,
    DEFAULT_PAGE_SIZE,
    DEFAULT_REQUEST_TIMEOUT,
    FetchConfig,
    fetch_latest_interpretations,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = BASE_DIR / "data" / "processed" / "eureka_date_backfill_state.json"


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


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_dates(start: date, end: date, *, descending: bool = False) -> list[str]:
    if not descending and start > end:
        raise ValueError("start date must be less than or equal to end date")
    if descending and start < end:
        raise ValueError("for descending mode start date must be greater than or equal to end date")

    current = start
    values: list[str] = []
    step = -1 if descending else 1
    comparator = (lambda current_date: current_date >= end) if descending else (lambda current_date: current_date <= end)
    while comparator(current):
        values.append(current.isoformat())
        current += timedelta(days=step)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable Eureka backfill over a date range.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--start-page", type=int, default=0)
    parser.add_argument("--page-batch-size", type=int, default=2)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--retry-count", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    parser.add_argument("--date-pause-seconds", type=float, default=1.0)
    parser.add_argument("--category", default="Interpretacja indywidualna")
    parser.add_argument("--sort", default="DT_WYD,desc")
    parser.add_argument("--raw-output", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-state", action="store_true")
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--allow-failed-details", action="store_true")
    parser.add_argument("--stop-on-empty-page", action="store_true", default=True)
    parser.add_argument("--skip-error-dates", action="store_true")
    return parser.parse_args()


async def run_backfill(args: argparse.Namespace) -> dict[str, Any]:
    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    all_dates = iter_dates(start_date, end_date, descending=args.descending)

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
    state_path = Path(args.state_path)
    saved_state = load_state(state_path) if args.resume and not args.overwrite_state else None

    if saved_state is not None:
        current_total = int(saved_state.get("current_total") or 0)
        next_date = str(saved_state.get("next_date") or all_dates[0])
        next_page = max(0, int(saved_state.get("next_page") or args.start_page))
    else:
        current_total = count_jsonl_rows(processed_path)
        next_date = all_dates[0]
        next_page = max(0, args.start_page)

    write_state(
        state_path,
        {
            "status": "running",
            "current_total": current_total,
            "target_count": args.target_count,
            "next_date": next_date,
            "next_page": next_page,
            "start_date": args.start_date,
            "end_date": args.end_date,
        },
    )

    date_index = all_dates.index(next_date)
    last_result: dict[str, Any] | None = None

    while date_index < len(all_dates):
        run_date = all_dates[date_index]
        page_start = next_page if run_date == next_date else 0

        print(
            json.dumps(
                {
                    "event": "date_start",
                    "run_date": run_date,
                    "page_start": page_start,
                    "current_total": current_total,
                    "target_count": args.target_count,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        while True:
            batch_end = page_start + max(1, args.page_batch_size)
            batch_config = replace(
                base_config,
                start_page=page_start,
                max_pages=batch_end,
                published_dates=[run_date],
            )

            try:
                result = await fetch_latest_interpretations(batch_config)
                last_result = result
            except Exception as exc:  # noqa: BLE001
                error = format_error(exc)
                print(
                    json.dumps(
                        {
                            "event": "date_error",
                            "run_date": run_date,
                            "page_start": page_start,
                            "page_end": batch_end,
                            "error": error,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if not args.skip_error_dates:
                    write_state(
                        state_path,
                        {
                            "status": "stopped",
                            "current_total": current_total,
                            "target_count": args.target_count,
                            "next_date": run_date,
                            "next_page": page_start,
                            "start_date": args.start_date,
                            "end_date": args.end_date,
                            "last_error": error,
                        },
                    )
                    raise
                break

            added = int(result.get("count") or 0)
            failed_ids = result.get("failed_ids", [])
            if failed_ids and not args.allow_failed_details:
                error = f"Detail fetch failed for {len(failed_ids)} document(s): {failed_ids[:10]}"
                write_state(
                    state_path,
                    {
                        "status": "stopped",
                        "current_total": current_total,
                        "target_count": args.target_count,
                        "next_date": run_date,
                        "next_page": page_start,
                        "start_date": args.start_date,
                        "end_date": args.end_date,
                        "last_error": error,
                    },
                )
                raise RuntimeError(error)

            current_total += added
            print(
                json.dumps(
                    {
                        "event": "page_batch_done",
                        "run_date": run_date,
                        "page_start": page_start,
                        "page_end": batch_end,
                        "added": added,
                        "current_total": current_total,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

            if args.target_count is not None and current_total >= args.target_count:
                write_state(
                    state_path,
                    {
                        "status": "finished",
                        "current_total": current_total,
                        "target_count": args.target_count,
                        "next_date": run_date,
                        "next_page": batch_end,
                        "start_date": args.start_date,
                        "end_date": args.end_date,
                    },
                )
                return {
                    "status": "finished",
                    "current_total": current_total,
                    "target_count": args.target_count,
                    "processed_output_path": str(processed_path),
                    "next_date": run_date,
                    "next_page": batch_end,
                    "last_batch": last_result,
                }

            if args.stop_on_empty_page and result.get("reached_empty_page"):
                break

            page_start = batch_end
            write_state(
                state_path,
                {
                    "status": "running",
                    "current_total": current_total,
                    "target_count": args.target_count,
                    "next_date": run_date,
                    "next_page": page_start,
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                },
            )

        date_index += 1
        if date_index >= len(all_dates):
            break

        next_date = all_dates[date_index]
        next_page = 0
        write_state(
            state_path,
            {
                "status": "running",
                "current_total": current_total,
                "target_count": args.target_count,
                "next_date": next_date,
                "next_page": next_page,
                "start_date": args.start_date,
                "end_date": args.end_date,
            },
        )
        if args.date_pause_seconds:
            await asyncio.sleep(max(0.0, args.date_pause_seconds))

    write_state(
        state_path,
        {
            "status": "finished",
            "current_total": current_total,
            "target_count": args.target_count,
            "next_date": all_dates[-1],
            "next_page": 0,
            "start_date": args.start_date,
            "end_date": args.end_date,
        },
    )
    return {
        "status": "finished",
        "current_total": current_total,
        "target_count": args.target_count,
        "processed_output_path": str(processed_path),
        "next_date": all_dates[-1],
        "next_page": 0,
        "last_batch": last_result,
    }


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_backfill(args))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
