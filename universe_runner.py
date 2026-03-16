from __future__ import annotations

"""
universe_runner.py

Batch stage runner for v2:
- ingest tickers from list or CSV
- generate lean reports
- persist score_rows for ranking/backtest prep
"""

import argparse
import datetime as dt
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from llm_pipeline import ensure_score_rows_dir, generate_report_v2
from report_schema import SCORE_DIMENSIONS, scorecard_to_row


def parse_tickers(raw: str) -> List[str]:
    out: List[str] = []
    for token in re.split(r"[,\n;\t]+", raw or ""):
        t = token.strip().upper()
        if t:
            out.append(t)
    return sorted(set(out))


def _resolve_ticker_column(df: pd.DataFrame, ticker_col: str = "ticker") -> str:
    col = ticker_col
    if col in df.columns:
        return col
    candidates = [c for c in df.columns if c.lower() in {"ticker", "symbol"}]
    if candidates:
        return candidates[0]
    raise ValueError(f"Ticker column '{ticker_col}' not found.")


def tickers_from_dataframe(df: pd.DataFrame, ticker_col: str = "ticker") -> List[str]:
    if df is None or df.empty:
        return []
    col = _resolve_ticker_column(df, ticker_col=ticker_col)
    values = [str(x).strip().upper() for x in df[col].tolist()]
    normalized = [x for x in values if x and x != "NAN"]
    return sorted(set(normalized))


def tickers_from_csv(path: str, ticker_col: str = "ticker") -> List[str]:
    df = pd.read_csv(path)
    return tickers_from_dataframe(df, ticker_col=ticker_col)


def _safe_report_path(path_obj: Optional[Path]) -> str:
    return str(path_obj) if path_obj is not None else ""


def _score_row_from_report(report: Dict[str, Any], report_path: Optional[Path]) -> Dict[str, Any]:
    meta = report.get("report_metadata") or {}
    return scorecard_to_row(
        as_of_date=str(meta.get("as_of_date") or ""),
        ticker=str(meta.get("ticker") or ""),
        company=str(meta.get("company") or ""),
        scorecard=report.get("scorecard") or {},
        aggregate_score=int(report.get("aggregate_score") or 0),
        recommendation=str(report.get("recommendation") or "Hold"),
        valuation_flag=report.get("valuation_flag") or {},
        risk_flags=report.get("risk_flags") or [],
        model_name=str(meta.get("model_name") or ""),
        prompt_version=str(meta.get("prompt_version") or "v2"),
        news_enabled=bool(meta.get("news_enabled")),
        report_path=_safe_report_path(report_path),
    )


def run_universe_scan(
    tickers: Iterable[str],
    *,
    max_workers: int = 4,
    save_reports: bool = True,
) -> pd.DataFrame:
    ticker_list = sorted(set([str(t).strip().upper() for t in tickers if str(t).strip()]))
    if not ticker_list:
        raise ValueError("No valid tickers provided.")

    rows: List[Dict[str, Any]] = []
    worker_count = max(1, min(max_workers, len(ticker_list)))
    with ThreadPoolExecutor(max_workers=worker_count) as ex:
        futures = {
            ex.submit(generate_report_v2, ticker, report_mode="lean", save=save_reports): ticker
            for ticker in ticker_list
        }
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                report, path = fut.result()
                rows.append(_score_row_from_report(report, path))
            except Exception as exc:
                rows.append(
                    {
                        "as_of_date": dt.date.today().isoformat(),
                        "ticker": ticker,
                        "company": ticker,
                        **{dim: 0 for dim in SCORE_DIMENSIONS},
                        "aggregate_score": 0,
                        "recommendation": "Hold",
                        "valuation_flag": "reasonable",
                        "risk_flags": [{"type": "execution_risk", "severity": "high", "summary": str(exc)}],
                        "model_name": "",
                        "prompt_version": "v2",
                        "news_enabled": False,
                        "report_path": "",
                    }
                )

    df = pd.DataFrame(rows)
    df = df.sort_values(by="aggregate_score", ascending=False).reset_index(drop=True)
    return df


def persist_score_rows(df: pd.DataFrame) -> Dict[str, Path]:
    out_dir = ensure_score_rows_dir()
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    csv_path = out_dir / f"score_rows_{stamp}.csv"
    json_path = out_dir / f"score_rows_{stamp}.json"

    csv_df = df.copy()
    if "risk_flags" in csv_df.columns:
        csv_df["risk_flags"] = csv_df["risk_flags"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else str(x)
        )
    csv_df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2 universe scan and persist ranking rows.")
    parser.add_argument("--tickers", help="Comma-separated ticker list, e.g. AAPL,MSFT,NVDA")
    parser.add_argument("--csv", help="CSV file path containing a ticker column")
    parser.add_argument("--ticker-col", default="ticker", help="Ticker column name when using --csv")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--no-save-reports", action="store_true", help="Do not save per-ticker lean reports")
    args = parser.parse_args()

    tickers: List[str] = []
    if args.tickers:
        tickers.extend(parse_tickers(args.tickers))
    if args.csv:
        tickers.extend(tickers_from_csv(args.csv, ticker_col=args.ticker_col))
    tickers = sorted(set(tickers))

    if not tickers:
        raw = input("Enter comma-separated tickers: ").strip()
        tickers = parse_tickers(raw)
    if not tickers:
        raise ValueError("No tickers provided.")

    df = run_universe_scan(tickers, max_workers=args.max_workers, save_reports=not args.no_save_reports)
    paths = persist_score_rows(df)
    print(df.to_string(index=False))
    print(f"\nSaved score rows CSV: {paths['csv']}")
    print(f"Saved score rows JSON: {paths['json']}")


if __name__ == "__main__":
    main()
