"""
llm_pipeline.py

v2 orchestration entry point:
- Stage 1 default: lean scorecard report (single main LLM call)
- Stage 2 optional: full report extension (one extra cheaper synthesis call)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd
import yfinance as yf
try:
    from google import genai
except ImportError:  # pragma: no cover - environment-dependent
    genai = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - environment-dependent
    OpenAI = None  # type: ignore[assignment]

from archetype import archetype_to_dict, classify_archetype
from catalyst_pipeline import build_catalyst_inputs
import fundamental_pipeline as fund_mod
from llm_clients import GeminiRunner, OpenAIRunner
from llm_config import load_llm_config
from prompt_templates import FULL_REPORT_PROMPT_V2, SCORECARD_PROMPT_V2, SYSTEM_PROMPT_V2
from report_schema import (
    PROMPT_VERSION,
    REPORT_VERSION,
    compute_aggregate_score,
    normalize_scorecard,
    recommendation_from_score,
)
from report_validation import validate_v2_report
import technical_pipeline as tech_mod

THIS_DIR = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]

    load_dotenv(THIS_DIR / ".env")
except ImportError:
    pass

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


def df_to_csv_str(df: pd.DataFrame, max_rows: int = 12, max_cols: int = 25) -> str:
    if df is None or df.empty:
        return "N/A (empty dataframe)"
    _df = df.copy()
    try:
        _df.index = pd.to_datetime(_df.index).strftime("%Y-%m-%d")
    except Exception:
        _df.index = _df.index.astype(str)
    if _df.shape[1] > max_cols:
        _df = _df.iloc[:, :max_cols]
    if _df.shape[0] > max_rows:
        _df = _df.tail(max_rows)
    return _df.to_csv(index=True)


def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [f"{prefix}{c}" for c in out.columns]
    return out


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def ensure_outputs_dir() -> Path:
    out_dir = THIS_DIR / "outputs"
    out_dir.mkdir(exist_ok=True)
    return out_dir


def ensure_score_rows_dir() -> Path:
    out_dir = THIS_DIR / "score_rows"
    out_dir.mkdir(exist_ok=True)
    return out_dir


def build_unique_output_path(
    ticker: str,
    analysis_date: str,
    suffix: str = "report",
    output_dir: Optional[Path] = None,
) -> Path:
    out_dir = output_dir if output_dir is not None else ensure_outputs_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    time_tag = dt.datetime.now().strftime("%H%M")
    base = f"{ticker}_{analysis_date}_{time_tag}_{suffix}"
    candidate = out_dir / f"{base}.json"
    idx = 2
    while candidate.exists():
        candidate = out_dir / f"{base}_run{idx}.json"
        idx += 1
    return candidate


RATING_ALIASES = {
    "overweight": "Overweight",
    "equal-weight": "Equal-weight",
    "equalweight": "Equal-weight",
    "hold": "Hold",
    "underweight": "Underweight",
    "reduce": "Reduce",
    "buy": "Overweight",
    "sell": "Reduce",
    "neutral": "Equal-weight",
}


def normalize_rating(report: Dict[str, Any]) -> None:
    raw = (report.get("rating") or "").strip()
    canonical = "Hold"
    if raw:
        raw_lower = raw.lower()
        for alias, value in RATING_ALIASES.items():
            if alias in raw_lower:
                canonical = value
                break
    report["rating"] = canonical


def normalize_price_target_matrix(report: Dict[str, Any]) -> None:
    matrix = report.get("price_target_matrix")
    if not isinstance(matrix, list):
        return
    by_key: Dict[str, Dict[str, Any]] = {}
    for row in matrix:
        if not isinstance(row, dict):
            continue
        key = str(row.get("scenario") or "").strip().lower()
        if key == "consensus":
            key = "base"
        if key in {"bear", "base", "bull"} and key not in by_key:
            fixed = dict(row)
            if key == "base" and str(fixed.get("scenario") or "").strip().lower() == "consensus":
                fixed["scenario"] = "Consensus"
            elif key == "base" and not fixed.get("scenario"):
                fixed["scenario"] = "Base"
            by_key[key] = fixed
    ordered = []
    if "bear" in by_key:
        ordered.append(by_key["bear"])
    if "base" in by_key:
        ordered.append(by_key["base"])
    if "bull" in by_key:
        ordered.append(by_key["bull"])
    report["price_target_matrix"] = ordered


def fetch_technical_snapshot(ticker: str, analysis_date: str, lookback_days: int = 365) -> Dict[str, Any]:
    tk = yf.Ticker(ticker)
    end = pd.to_datetime(analysis_date) + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback_days + 30)
    hist = tk.history(start=start, end=end, interval="1d")
    if hist is None or hist.empty:
        return {"ticker": ticker, "analysis_date": analysis_date, "note": "No price history returned."}
    hist = hist.rename(columns={c: c.replace(" ", "") for c in hist.columns})
    return tech_mod.build_technical_snapshot_dict(hist, ticker, analysis_date)


def fetch_top_indicator_chart_data(
    ticker: str,
    analysis_date: str,
    lookback_days: int = 220,
) -> Dict[str, list[Dict[str, Any]]]:
    """
    Build compact timeseries payload for lightweight full-report visualization.
    """
    tk = yf.Ticker(ticker)
    end = pd.to_datetime(analysis_date) + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback_days + 30)
    hist = tk.history(start=start, end=end, interval="1d")
    if hist is None or hist.empty:
        return {
            "price": [],
            "sma_20": [],
            "sma_50": [],
            "sma_200": [],
            "ema_20": [],
            "bb_high": [],
            "bb_low": [],
            "macd": [],
            "macd_signal": [],
            "macd_hist": [],
            "rsi_14": [],
            "volume": [],
        }

    hist = hist.rename(columns={c: c.replace(" ", "") for c in hist.columns})
    feat = tech_mod.build_technical_features(hist)
    feat = feat.tail(180).copy()
    if feat.empty:
        return {
            "price": [],
            "sma_20": [],
            "sma_50": [],
            "sma_200": [],
            "ema_20": [],
            "bb_high": [],
            "bb_low": [],
            "macd": [],
            "macd_signal": [],
            "macd_hist": [],
            "rsi_14": [],
            "volume": [],
        }

    def _series(col: str) -> list[Dict[str, Any]]:
        if col not in feat.columns:
            return []
        ser = pd.to_numeric(feat[col], errors="coerce").dropna()
        out: list[Dict[str, Any]] = []
        for idx, val in ser.items():
            out.append({"time": pd.to_datetime(idx).strftime("%Y-%m-%d"), "value": float(val)})
        return out

    return {
        "price": _series("Close"),
        "sma_20": _series("sma_20"),
        "sma_50": _series("sma_50"),
        "sma_200": _series("sma_200"),
        "ema_20": _series("ema_20"),
        "bb_high": _series("bb_high"),
        "bb_low": _series("bb_low"),
        "macd": _series("macd"),
        "macd_signal": _series("macd_signal"),
        "macd_hist": _series("macd_diff"),
        "rsi_14": _series("rsi_14"),
        "volume": _series("Volume"),
    }


def fetch_quarterly_5_periods(ticker: str, periods: int = 5) -> Dict[str, pd.DataFrame]:
    tk = yf.Ticker(ticker)
    income_raw = fund_mod.merge_statement_sources(tk, ["quarterly_income_stmt", "quarterly_financials"])
    balance_raw = fund_mod.merge_statement_sources(tk, ["quarterly_balance_sheet", "quarterly_balancesheet"])
    cash_raw = fund_mod.merge_statement_sources(tk, ["quarterly_cashflow", "quarterly_cash_flow"])
    income_rows = fund_mod.statement_to_rows(income_raw)
    balance_rows = fund_mod.statement_to_rows(balance_raw)
    cash_rows = fund_mod.statement_to_rows(cash_raw)

    def tail_n(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        return df.sort_index().tail(periods)

    return {
        "quarterly_income_statement": tail_n(income_rows),
        "quarterly_balance_sheet": tail_n(balance_rows),
        "quarterly_cash_flow": tail_n(cash_rows),
    }


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if pd.isna(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _latest(metrics_df: pd.DataFrame, col: str) -> Optional[float]:
    if metrics_df is None or metrics_df.empty or col not in metrics_df.columns:
        return None
    return _float_or_none(pd.to_numeric(metrics_df[col], errors="coerce").iloc[-1])


def _prev(metrics_df: pd.DataFrame, col: str) -> Optional[float]:
    if metrics_df is None or metrics_df.empty or col not in metrics_df.columns or len(metrics_df.index) < 2:
        return None
    return _float_or_none(pd.to_numeric(metrics_df[col], errors="coerce").iloc[-2])


def summarize_annual_trend(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    out = {
        "revenue_growth_yoy_last": _latest(metrics_df, "Revenue_Growth_YoY"),
        "revenue_growth_yoy_prev": _prev(metrics_df, "Revenue_Growth_YoY"),
        "net_income_growth_yoy_last": _latest(metrics_df, "NetIncome_Growth_YoY"),
        "operating_margin_last": _latest(metrics_df, "Operating_Margin"),
        "net_margin_last": _latest(metrics_df, "Net_Margin"),
        "debt_to_equity_last": _latest(metrics_df, "Debt_to_Equity"),
        "free_cash_flow_last": _latest(metrics_df, "Free_Cash_Flow"),
        "capex_intensity_last": _latest(metrics_df, "Capex_Intensity"),
        "rd_intensity_last": _latest(metrics_df, "RD_Intensity"),
    }
    out["asset_growth_yoy_last"] = _latest(metrics_df, "Asset_Growth_YoY")
    out["debt_growth_yoy_last"] = _latest(metrics_df, "Debt_Growth_YoY")
    out["effective_tax_rate_last"] = _latest(metrics_df, "Effective_Tax_Rate")
    return out


def summarize_quarterly_delta(q_out: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    qis = q_out.get("quarterly_income_statement", pd.DataFrame())
    qbs = q_out.get("quarterly_balance_sheet", pd.DataFrame())
    if qis.empty and qbs.empty:
        return {"status": "insufficient_data"}

    def _delta(df: pd.DataFrame, col_candidates: list[str]) -> Optional[float]:
        for col in col_candidates:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) >= 2 and s.iloc[-2] != 0:
                    return float((s.iloc[-1] - s.iloc[-2]) / abs(s.iloc[-2]))
        return None

    return {
        "qoq_revenue_change": _delta(qis, ["Total Revenue", "TotalRevenue", "Revenue"]),
        "qoq_net_income_change": _delta(
            qis, ["Net Income", "NetIncome", "Net Income Common Stockholders", "NetIncomeCommonStockholders"]
        ),
        "qoq_debt_change": _delta(qbs, ["Total Debt", "TotalDebt"]),
    }


def build_company_context(meta_df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    row = meta_df.iloc[0].to_dict() if isinstance(meta_df, pd.DataFrame) and not meta_df.empty else {}
    info = {}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    officers = info.get("companyOfficers")
    top_officers = []
    if isinstance(officers, list):
        for x in officers[:3]:
            if not isinstance(x, dict):
                continue
            top_officers.append(
                {
                    "name": x.get("name"),
                    "title": x.get("title"),
                    "yearBorn": x.get("yearBorn"),
                }
            )

    return {
        "shortName": row.get("ShortName") or info.get("shortName") or info.get("longName"),
        "longBusinessSummary": info.get("longBusinessSummary"),
        "sector": row.get("Sector") or info.get("sector"),
        "industry": row.get("Industry") or info.get("industry"),
        "marketCap": row.get("MarketCap") or info.get("marketCap"),
        "currency": row.get("Currency") or info.get("currency"),
        "exchange": row.get("Exchange") or info.get("exchange"),
        "country": row.get("Country") or info.get("country"),
        "officers": top_officers,
    }


def build_valuation_packet(metrics_df: pd.DataFrame, company_context: Mapping[str, Any], ticker: str) -> Dict[str, Any]:
    info = {}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    market_cap = _float_or_none(company_context.get("marketCap"))
    fcf = _latest(metrics_df, "Free_Cash_Flow")
    p_fcf = None
    if market_cap and fcf and fcf > 0:
        p_fcf = market_cap / fcf

    packet = {
        "market_cap": market_cap,
        "trailing_pe": _float_or_none(info.get("trailingPE")),
        "forward_pe": _float_or_none(info.get("forwardPE")),
        "ev_to_ebitda": _float_or_none(info.get("enterpriseToEbitda")),
        "ev_to_sales": _float_or_none(info.get("enterpriseToRevenue")),
        "price_to_sales": _float_or_none(info.get("priceToSalesTrailing12Months")),
        "p_fcf_proxy": p_fcf,
        "pre_profit": bool((_latest(metrics_df, "NetIncome") or 0) < 0),
    }
    return packet


def build_valuation_flag(valuation_packet: Mapping[str, Any]) -> Dict[str, Any]:
    pe = _float_or_none(valuation_packet.get("trailing_pe"))
    ps = _float_or_none(valuation_packet.get("price_to_sales"))
    pfcf = _float_or_none(valuation_packet.get("p_fcf_proxy"))
    pre_profit = bool(valuation_packet.get("pre_profit"))

    label = "reasonable"
    confidence = "medium"
    summary = "Valuation appears broadly balanced relative to available fundamentals."

    if pre_profit:
        if ps is not None and ps >= 12:
            label = "stretched"
            summary = "Pre-profit profile with elevated sales multiple."
        elif ps is not None and ps <= 3:
            label = "cheap"
            summary = "Pre-profit profile with modest sales multiple."
    else:
        if (pe is not None and pe >= 40) or (pfcf is not None and pfcf >= 45):
            label = "stretched"
            summary = "Earnings/cash-flow multiples imply rich valuation."
        elif (pe is not None and pe <= 12) or (pfcf is not None and pfcf <= 12):
            label = "cheap"
            summary = "Earnings/cash-flow multiples screen inexpensive."
        elif pe is not None and pe >= 28:
            label = "full"
            summary = "Multiples are above long-run market averages."

    if pe is None and ps is None and pfcf is None:
        confidence = "low"
        summary = "Limited valuation inputs available."

    return {"label": label, "confidence": confidence, "summary": summary}


def build_risk_flags(
    metrics_df: pd.DataFrame,
    tech_snapshot: Mapping[str, Any],
    quarterly_delta: Mapping[str, Any],
    archetype_type: Optional[str] = None,
) -> list[Dict[str, str]]:
    flags: list[Dict[str, str]] = []
    d2e = _latest(metrics_df, "Debt_to_Equity")
    fcf = _latest(metrics_df, "Free_Cash_Flow")
    ni = _latest(metrics_df, "NetIncome")
    r3m = _float_or_none(tech_snapshot.get("return_3m"))
    trend = str(tech_snapshot.get("trend_regime") or "")
    q_rev = _float_or_none(quarterly_delta.get("qoq_revenue_change"))

    if d2e is not None and d2e > 2.0:
        flags.append(
            {"type": "balance_sheet_risk", "severity": "high", "summary": "Leverage remains elevated versus equity base."}
        )
    if ni is not None and ni < 0 and (fcf is None or fcf < 0):
        flags.append(
            {"type": "dilution_risk", "severity": "medium", "summary": "Losses with weak cash generation may increase financing pressure."}
        )
    if trend in {"downtrend", "strong_downtrend"} or (r3m is not None and r3m < -0.25):
        flags.append(
            {"type": "technical_breakdown_risk", "severity": "medium", "summary": "Price action indicates weak trend or recent breakdown risk."}
        )
    if q_rev is not None and q_rev < -0.15:
        flags.append(
            {"type": "execution_risk", "severity": "medium", "summary": "Recent quarterly revenue decline raises near-term execution risk."}
        )
    # Liquidity: negative FCF with elevated leverage suggests refinancing/cash strain.
    if fcf is not None and fcf < 0 and d2e is not None and d2e > 1.0:
        flags.append(
            {"type": "liquidity_risk", "severity": "medium", "summary": "Negative free cash flow with meaningful debt may pressure liquidity."}
        )
    # Cyclicality: archetype signals exposure to industrial/materials cycles.
    if archetype_type == "cyclical_industrial":
        flags.append(
            {"type": "cyclicality_risk", "severity": "low", "summary": "Sector classified as cyclical; earnings may be more volatile through cycles."}
        )

    if not flags:
        flags.append(
            {"type": "execution_risk", "severity": "low", "summary": "No acute red flag from current structured inputs."}
        )
    return flags


def build_top_signal_views(scorecard: Mapping[str, Mapping[str, Any]]) -> Dict[str, list[Dict[str, Any]]]:
    technical = [
        {"label": "trend_regime", **dict(scorecard.get("technical_setup") or {})},
        {"label": "momentum_rsi", **dict(scorecard.get("technical_setup") or {})},
        {"label": "macd_confirmation", **dict(scorecard.get("technical_setup") or {})},
    ]
    fundamental_keys = [
        "leadership",
        "competitive_advantage",
        "growth_perspective",
        "balance_sheet_health",
        "business_segment_quality",
    ]
    fundamentals: list[Dict[str, Any]] = []
    for key in fundamental_keys:
        row = scorecard.get(key) or {}
        item = {"label": key, **dict(row)}
        fundamentals.append(item)

    def _abs_score(item: Mapping[str, Any]) -> int:
        try:
            return abs(int(item.get("score", 0)))
        except (TypeError, ValueError):
            return 0

    fundamentals = sorted(fundamentals, key=_abs_score, reverse=True)[:3]
    return {
        "top_technical_signals": technical,
        "top_fundamental_signals": fundamentals,
    }


def build_fundamental_indicator_chart_data(metrics_df: pd.DataFrame) -> Dict[str, list[Dict[str, Any]]]:
    """
    Build compact time-series payload for fundamental signal visualization.
    """
    if metrics_df is None or metrics_df.empty:
        return {
            "revenue_growth_yoy": [],
            "net_income_growth_yoy": [],
            "net_margin": [],
            "operating_margin": [],
            "debt_to_equity": [],
            "free_cash_flow": [],
        }

    frame = metrics_df.copy()
    frame = frame.tail(8)

    def _series(col: str) -> list[Dict[str, Any]]:
        if col not in frame.columns:
            return []
        ser = pd.to_numeric(frame[col], errors="coerce").dropna()
        out: list[Dict[str, Any]] = []
        for idx, val in ser.items():
            out.append({"time": pd.to_datetime(idx).strftime("%Y-%m-%d"), "value": float(val)})
        return out

    return {
        "revenue_growth_yoy": _series("Revenue_Growth_YoY"),
        "net_income_growth_yoy": _series("NetIncome_Growth_YoY"),
        "net_margin": _series("Net_Margin"),
        "operating_margin": _series("Operating_Margin"),
        "debt_to_equity": _series("Debt_to_Equity"),
        "free_cash_flow": _series("Free_Cash_Flow"),
    }


def enrich_top_technical_signals(
    top_signals: list[Dict[str, Any]],
    tech_snapshot: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """
    Attach data-backed descriptions so top technical signals are not purely textual.
    """
    out: list[Dict[str, Any]] = []
    trend = str(tech_snapshot.get("trend_regime") or "unknown")
    rsi = _float_or_none(tech_snapshot.get("rsi_14"))
    macd = _float_or_none(tech_snapshot.get("macd"))
    macd_sig = _float_or_none(tech_snapshot.get("macd_signal"))
    dist_high = _float_or_none(tech_snapshot.get("dist_to_high_52w"))

    for row in top_signals:
        label = str(row.get("label") or "")
        item = dict(row)
        if label == "trend_regime":
            item["thesis"] = f"Trend regime currently classified as {trend}."
            item["evidence"] = f"dist_to_high_52w={dist_high:.3f}" if dist_high is not None else "Trend classification from moving-average structure."
        elif label == "momentum_rsi":
            item["thesis"] = "Momentum signal from RSI14."
            item["evidence"] = f"rsi_14={rsi:.2f}" if rsi is not None else "RSI unavailable."
        elif label == "macd_confirmation":
            item["thesis"] = "MACD line versus signal line confirms or weakens momentum."
            if macd is not None and macd_sig is not None:
                item["evidence"] = f"macd={macd:.4f}, macd_signal={macd_sig:.4f}, spread={macd - macd_sig:.4f}"
            else:
                item["evidence"] = "MACD data unavailable."
        out.append(item)
    return out


def build_technical_snapshot_compact(tech_snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact technical snapshot for scorecard: last values + summary text. Used for technical_setup and liquidity."""
    if not tech_snapshot:
        return {"technical_summary_text": "No technical data available."}
    keys = [
        "last_close", "ma_50", "ma_200", "sma_20", "ema_20",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "stoch_k", "stoch_d", "williams_r", "atr_14", "bb_width", "bb_pos",
        "volume_last", "avg_volume_20", "support", "resistance_near", "resistance_major",
        "data_confidence",
    ]
    compact = {k: tech_snapshot.get(k) for k in keys if tech_snapshot.get(k) is not None}
    compact["technical_summary_text"] = tech_snapshot.get("technical_summary_text") or "No summary."
    return compact


def build_analysis_packet(
    *,
    ticker: str,
    analysis_date: str,
    annual_out: Dict[str, Any],
    quarterly_out: Dict[str, pd.DataFrame],
    tech_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    metrics_df = annual_out.get("metrics", pd.DataFrame())
    meta_df = annual_out.get("meta", pd.DataFrame())
    company_context = build_company_context(meta_df, ticker)
    archetype = archetype_to_dict(classify_archetype(meta_df, metrics_df))
    annual_trend = summarize_annual_trend(metrics_df)
    quarterly_delta = summarize_quarterly_delta(quarterly_out)
    valuation_packet = build_valuation_packet(metrics_df, company_context, ticker)
    catalyst_inputs = build_catalyst_inputs(
        ticker=ticker,
        analysis_date=analysis_date,
        fundamentals=annual_out,
        tech_snapshot=tech_snapshot,
    )
    technical_snapshot_compact = build_technical_snapshot_compact(tech_snapshot)

    return {
        "ticker": ticker,
        "analysis_date": analysis_date,
        "company_context": company_context,
        "archetype": archetype,
        "annual_trend_summary": annual_trend,
        "quarterly_delta_summary": quarterly_delta,
        "technical_regime": {
            "trend_regime": tech_snapshot.get("trend_regime"),
            "volatility_regime": tech_snapshot.get("volatility_regime"),
            "dist_to_high_52w": tech_snapshot.get("dist_to_high_52w"),
            "dist_to_low_52w": tech_snapshot.get("dist_to_low_52w"),
            "return_1m": tech_snapshot.get("return_1m"),
            "return_3m": tech_snapshot.get("return_3m"),
        },
        "technical_snapshot_compact": technical_snapshot_compact,
        "valuation_packet": valuation_packet,
        "risk_inputs": {
            "debt_to_equity_last": annual_trend.get("debt_to_equity_last"),
            "free_cash_flow_last": annual_trend.get("free_cash_flow_last"),
            "net_income_growth_yoy_last": annual_trend.get("net_income_growth_yoy_last"),
            "qoq_revenue_change": quarterly_delta.get("qoq_revenue_change"),
            "trend_regime": tech_snapshot.get("trend_regime"),
        },
        "catalyst_summary": {
            "news_enabled": bool(catalyst_inputs.get("news_enabled")),
            "company_news": catalyst_inputs.get("company_news", []),
            "fundamental_inferred": catalyst_inputs.get("fundamental_inferred", []),
            "technical_inferred": catalyst_inputs.get("technical_inferred", []),
            "industry_candidates": catalyst_inputs.get("industry_candidates", []),
        },
    }


def _model_name_from_cfg(cfg: Any) -> str:
    if cfg.backend == "openai":
        return cfg.openai_model
    return cfg.gemini_model


def create_runner() -> tuple[Any, str, str]:
    cfg = load_llm_config()
    if cfg.backend == "openai":
        if OpenAI is None:
            raise RuntimeError("openai package is required when LLM_BACKEND=openai")
        client = OpenAI(base_url=cfg.openai_base_url, api_key=cfg.openai_api_key)
        runner = OpenAIRunner(model=cfg.openai_model, client=client, system_prompt=SYSTEM_PROMPT_V2)
    elif cfg.backend == "gemini-vertex":
        if genai is None:
            raise RuntimeError("google-genai package is required for Gemini backends")
        if not cfg.gcp_project:
            raise ValueError("GOOGLE_CLOUD_PROJECT must be set when LLM_BACKEND=gemini-vertex")
        client = genai.Client(vertexai=True, project=cfg.gcp_project, location=cfg.gcp_location)
        runner = GeminiRunner(model=cfg.gemini_model, client=client, system_prompt=SYSTEM_PROMPT_V2)
    else:
        if genai is None:
            raise RuntimeError("google-genai package is required for Gemini backends")
        client = genai.Client()
        runner = GeminiRunner(model=cfg.gemini_model, client=client, system_prompt=SYSTEM_PROMPT_V2)
    return runner, cfg.backend, _model_name_from_cfg(cfg)


def generate_report_v2(
    ticker: str,
    *,
    analysis_date: Optional[str] = None,
    report_mode: str = "lean",
    save: bool = True,
    output_dir: Optional[Path] = None,
) -> tuple[Dict[str, Any], Optional[Path]]:
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")
    mode = str(report_mode or "lean").strip().lower()
    if mode not in {"lean", "full"}:
        mode = "lean"

    as_of_date = analysis_date or os.getenv("ANALYSIS_DATE") or dt.date.today().isoformat()
    annual_out = fund_mod.fetch_annual_5_periods_with_metrics(ticker, periods=5)
    quarterly_out = fetch_quarterly_5_periods(ticker, periods=5)
    tech_snapshot = fetch_technical_snapshot(ticker, as_of_date)
    packet = build_analysis_packet(
        ticker=ticker,
        analysis_date=as_of_date,
        annual_out=annual_out,
        quarterly_out=quarterly_out,
        tech_snapshot=tech_snapshot,
    )
    valuation_flag = build_valuation_flag(packet.get("valuation_packet", {}))
    risk_flags = build_risk_flags(
        annual_out.get("metrics", pd.DataFrame()),
        tech_snapshot,
        packet.get("quarterly_delta_summary", {}),
        archetype_type=(packet.get("archetype") or {}).get("type"),
    )

    runner, backend_name, model_name = create_runner()
    score_prompt = SCORECARD_PROMPT_V2.format(
        TICKER=ticker,
        ANALYSIS_PACKET_JSON=safe_json_dumps(packet),
    )
    llm_score = runner.run_json(score_prompt, temperature=0.1)
    scorecard = normalize_scorecard(llm_score.get("scorecard") if isinstance(llm_score, dict) else {})

    aggregate_score = compute_aggregate_score(scorecard)
    recommendation = recommendation_from_score(aggregate_score)
    company_name = str((packet.get("company_context") or {}).get("shortName") or ticker)

    report: Dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "report_metadata": {
            "ticker": ticker,
            "company": company_name,
            "as_of_date": as_of_date,
            "model_name": model_name,
            "llm_backend": backend_name,
            "prompt_version": PROMPT_VERSION,
            "report_mode": mode,
            "news_enabled": bool((packet.get("catalyst_summary") or {}).get("news_enabled", False)),
        },
        "scorecard": scorecard,
        "aggregate_score": aggregate_score,
        "recommendation": recommendation,
        "valuation_flag": valuation_flag,
        "risk_flags": risk_flags,
    }

    if mode == "full":
        full_prompt = FULL_REPORT_PROMPT_V2.format(
            TICKER=ticker,
            LEAN_REPORT_JSON=safe_json_dumps(report),
            ANALYSIS_PACKET_JSON=safe_json_dumps(packet),
        )
        full_result = runner.run_json(full_prompt, temperature=0.0)
        if isinstance(full_result, dict):
            if isinstance(full_result.get("overall_outlook"), str):
                report["overall_outlook"] = full_result["overall_outlook"]
            if isinstance(full_result.get("price_target_matrix"), dict):
                matrix = dict(full_result["price_target_matrix"])
                if "consensus" not in matrix and "base" in matrix:
                    matrix["consensus"] = matrix["base"]
                report["price_target_matrix"] = matrix
        report.update(build_top_signal_views(scorecard))
        report["top_technical_signals"] = enrich_top_technical_signals(
            report.get("top_technical_signals", []),
            tech_snapshot=tech_snapshot,
        )
        report["top_indicator_chart_data"] = fetch_top_indicator_chart_data(ticker, as_of_date)
        report["fundamental_indicator_chart_data"] = build_fundamental_indicator_chart_data(
            annual_out.get("metrics", pd.DataFrame())
        )

    report["validation"] = validate_v2_report(report, report_mode=mode)

    out_path: Optional[Path] = None
    if save:
        suffix = "full_report" if mode == "full" else "report"
        out_path = build_unique_output_path(ticker, as_of_date, suffix=suffix, output_dir=output_dir)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="TickerAnalysis v2: lean scan report + optional full report.")
    parser.add_argument("ticker", nargs="?", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--report-mode", choices=["lean", "full"], default=os.getenv("REPORT_MODE", "lean"))
    parser.add_argument("--no-save", action="store_true", help="Do not write JSON file")
    args = parser.parse_args()

    ticker = (args.ticker or input("Enter ticker (e.g., AAPL): ").strip()).upper()
    report, out_path = generate_report_v2(
        ticker,
        report_mode=args.report_mode,
        save=not args.no_save,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if out_path:
        print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
