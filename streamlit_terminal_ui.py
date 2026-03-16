from __future__ import annotations

# Copyright (c) 2026 TickerAnalysis.
# UI ownership: terminal-style workspace implementation.

import datetime as dt
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from llm_pipeline import generate_report_v2
from report_schema import DIMENSION_DISPLAY_NAMES, DIMENSION_SCORE_BOUNDS, SCORE_DIMENSIONS, scorecard_to_row
from universe_runner import parse_tickers, tickers_from_dataframe


THIS_DIR = Path(__file__).resolve().parent


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", (text or "").strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "scan"


def _ensure_outputs_dir() -> Path:
    out = THIS_DIR / "outputs"
    out.mkdir(exist_ok=True)
    return out


def _create_scan_output_dir(scan_label: str) -> tuple[str, Path]:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{_slugify(scan_label)}_{stamp}"
    out_dir = _ensure_outputs_dir() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_id, out_dir


def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "workspace_page": "Sector Screener",
        "workspace_page_selector": "Sector Screener",
        "ranking_df": pd.DataFrame(),
        "selected_ticker": None,
        "comparison_selection": [],
        "scan_label": "",
        "scan_timestamp": None,
        "scan_runtime_seconds": None,
        "scan_workers": 4,
        "scan_tickers_text": "AAPL\nMSFT\nNVDA",
        "scan_csv": None,
        "scan_id": None,
        "scan_output_dir": None,
        "detail_reports_by_scan": {},  # scan_id -> {ticker: (report, path)}
        "full_report_runtime_by_scan": {},  # scan_id -> {ticker: seconds}
        "full_report_generating_for": None,
        "detail_select_ticker": None,
        "drill_on_select": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _inject_theme() -> None:
    theme_base = str(st.get_option("theme.base") or "dark").lower()
    if theme_base == "light":
        bg = "#f8fafc"
        panel = "#ffffff"
        panel2 = "#f1f5f9"
        text = "#0f172a"
        muted = "#334155"
        border = "#cbd5e1"
        chip_bg = "#eef2ff"
    else:
        bg = "#05070d"
        panel = "#0a101c"
        panel2 = "#0d1525"
        text = "#e5e7eb"
        muted = "#9ca3af"
        border = "#1f2a3d"
        chip_bg = "#0f172a"

    st.markdown(
        f"""
        <style>
        .stApp {{ background:{bg}; color:{text}; }}
        .block-container {{
            padding-top:1.15rem;
            padding-bottom:1.0rem;
            max-width:1500px;
        }}
        h1, h2, h3 {{ margin-top:0.15rem; margin-bottom:0.35rem; }}
        .stDataFrame, .stTable {{ font-size:0.82rem; }}
        .stButton>button {{
            border-radius:4px;
            padding:0.28rem 0.72rem;
            font-size:0.82rem;
            border:1px solid {border};
            background:{panel2};
        }}
        .terminal-card {{
            border:1px solid {border};
            border-radius:8px;
            background:{panel};
            padding:0.62rem 0.72rem;
            margin-bottom:0.42rem;
        }}
        .terminal-chip {{
            display:inline-block;
            border:1px solid {border};
            border-radius:999px;
            background:{chip_bg};
            color:{text};
            padding:0.12rem 0.5rem;
            margin:0.08rem 0.25rem 0.08rem 0;
            font-size:0.76rem;
        }}
        .terminal-muted {{
            color:{muted};
            font-size:0.80rem;
        }}
        .score-pos {{ color:#16a34a; font-weight:600; }}
        .score-neg {{ color:#dc2626; font-weight:600; }}
        .score-neu {{ color:{muted}; font-weight:600; }}
        .footer-row {{
            margin-top:1.1rem;
            padding-top:0.65rem;
            border-top:1px solid {border};
            color:{muted};
            font-size:0.75rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _pretty_dimension_name(dim: str) -> str:
    return DIMENSION_DISPLAY_NAMES.get(dim, dim.replace("_", " ").title())


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _risk_flags_list(risk_flags: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if isinstance(risk_flags, list):
        for item in risk_flags:
            if isinstance(item, Mapping):
                out.append(
                    {
                        "type": str(item.get("type") or ""),
                        "severity": str(item.get("severity") or ""),
                        "summary": str(item.get("summary") or ""),
                    }
                )
    return out


def _format_risk_flags_summary(risk_flags: Any) -> str:
    rows = _risk_flags_list(risk_flags)
    if not rows:
        return "none"
    return ", ".join([f"{r['type']} ({r['severity']})" for r in rows[:3] if r["type"]]) or "none"


def _max_risk_severity(risk_flags: Any) -> str:
    rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    best = "none"
    for row in _risk_flags_list(risk_flags):
        sev = row["severity"].lower()
        if sev in rank and rank[sev] > rank[best]:
            best = sev
    return best


def _risk_chip_html(risk_flags: Any, max_items: int = 4) -> str:
    rows = _risk_flags_list(risk_flags)[:max_items]
    if not rows:
        return "<span class='terminal-chip'>none</span>"
    out: List[str] = []
    for row in rows:
        text = f"{row['type']} ({row['severity']})" if row["type"] else "risk"
        out.append(f"<span class='terminal-chip'>{text}</span>")
    return "".join(out)


def _recommendation_tag(rec: str) -> str:
    m = {
        "Overweight": "▲ Overweight",
        "Equal-weight": "■ Equal-weight",
        "Hold": "• Hold",
        "Underweight": "▼ Underweight",
        "Reduce": "▼ Reduce",
    }
    return m.get(rec, rec or "N/A")


def _valuation_tag(v: str) -> str:
    val = str(v or "").lower()
    if val == "cheap":
        return "Cheap"
    if val == "reasonable":
        return "Reasonable"
    if val == "full":
        return "Full"
    if val == "stretched":
        return "Stretched"
    return str(v or "N/A")


def _score_sign_text(score: int) -> str:
    if score > 0:
        return "upside"
    if score < 0:
        return "downside"
    return "neutral"


def _score_class(score: int) -> str:
    if score > 0:
        return "score-pos"
    if score < 0:
        return "score-neg"
    return "score-neu"


def _zero_center_score_bar_html(score: int, lower: int, upper: int) -> str:
    neg_min = min(0, lower)
    pos_max = max(0, upper)
    left_fill = 0.0
    right_fill = 0.0
    if score < 0 and neg_min < 0:
        left_fill = min(1.0, abs(score) / abs(neg_min))
    if score > 0 and pos_max > 0:
        right_fill = min(1.0, score / pos_max)
    return f"""
    <div style="display:flex;align-items:center;gap:8px;">
      <div style="position:relative;flex:1;height:12px;border:1px solid #334155;border-radius:6px;background:#0f172a;overflow:hidden;">
        <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#6b7280;"></div>
        <div style="position:absolute;right:50%;top:0;bottom:0;width:{left_fill*50:.2f}%;background:#dc2626;"></div>
        <div style="position:absolute;left:50%;top:0;bottom:0;width:{right_fill*50:.2f}%;background:#16a34a;"></div>
      </div>
    </div>
    """


def _driver_pairs(row: Mapping[str, Any], top_n: int = 2) -> List[Tuple[str, int]]:
    pairs = [(dim, _safe_int(row.get(dim), 0)) for dim in SCORE_DIMENSIONS]
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_n]


def _short_rationale_from_row(row: Mapping[str, Any]) -> str:
    rec = str(row.get("recommendation") or "N/A")
    score = _safe_int(row.get("aggregate_score"), 0)
    val = str(row.get("valuation_flag") or "N/A")
    drivers = _driver_pairs(row, top_n=2)
    if drivers:
        dtxt = ", ".join([f"{_pretty_dimension_name(d)} ({s:+d})" for d, s in drivers if s != 0]) or "mixed drivers"
    else:
        dtxt = "mixed drivers"
    risks = _format_risk_flags_summary(row.get("risk_flags"))
    base = f"{rec} ({score:+d}) | {dtxt} | valuation: {val}"
    if risks != "none":
        base += f" | risks: {risks}"
    return base


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _find_latest_report(
    *,
    ticker: str,
    suffix: str,
    preferred_dir: Optional[Path] = None,
    allow_global_fallback: bool = False,
) -> tuple[Optional[Dict[str, Any]], Optional[Path]]:
    folders: List[Path] = []
    if preferred_dir and preferred_dir.exists():
        folders.append(preferred_dir)
    if allow_global_fallback:
        folders.append(_ensure_outputs_dir())

    pattern = f"{ticker}_*_{suffix}*.json"
    for folder in folders:
        matches = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in matches:
            payload = _load_json(p)
            if payload is not None:
                return payload, p
    return None, None


def _load_lean_from_row(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    pstr = str(row.get("report_path") or "").strip()
    if not pstr:
        return None
    p = Path(pstr)
    if not p.exists():
        return None
    return _load_json(p)


def _score_row_from_report(report: Dict[str, Any], path_obj: Optional[Path]) -> Dict[str, Any]:
    meta = report.get("report_metadata") or {}
    return scorecard_to_row(
        as_of_date=str(meta.get("as_of_date") or ""),
        ticker=str(meta.get("ticker") or ""),
        company=str(meta.get("company") or ""),
        scorecard=report.get("scorecard") or {},
        aggregate_score=_safe_int(report.get("aggregate_score"), 0),
        recommendation=str(report.get("recommendation") or "Hold"),
        valuation_flag=report.get("valuation_flag") or {},
        risk_flags=report.get("risk_flags") or [],
        model_name=str(meta.get("model_name") or ""),
        prompt_version=str(meta.get("prompt_version") or "v2"),
        news_enabled=bool(meta.get("news_enabled")),
        report_path=str(path_obj) if path_obj else "",
    )


def _scan_cache_bucket(scan_id: Optional[str]) -> Dict[str, Tuple[Dict[str, Any], Optional[Path]]]:
    sid = scan_id or "__default__"
    store = st.session_state["detail_reports_by_scan"]
    if sid not in store:
        store[sid] = {}
    return store[sid]


def _runtime_bucket(scan_id: Optional[str]) -> Dict[str, float]:
    sid = scan_id or "__default__"
    store = st.session_state["full_report_runtime_by_scan"]
    if sid not in store:
        store[sid] = {}
    return store[sid]


def _run_scan_with_progress(
    *,
    tickers: Sequence[str],
    max_workers: int,
    output_dir: Path,
) -> pd.DataFrame:
    uniq = sorted(set([str(t).strip().upper() for t in tickers if str(t).strip()]))
    if not uniq:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    total = len(uniq)
    progress = st.progress(0)
    status = st.empty()
    status.caption(f"Queued {total} tickers.")
    completed = 0
    worker_count = max(1, min(max_workers, total))

    with ThreadPoolExecutor(max_workers=worker_count) as ex:
        futures = {
            ex.submit(generate_report_v2, ticker, report_mode="lean", save=True, output_dir=output_dir): ticker for ticker in uniq
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
            completed += 1
            progress.progress(int(100 * completed / total))
            status.caption(f"Scan progress: {completed}/{total}")
    status.caption(f"Scan completed: {total}/{total}")
    progress.progress(100)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("aggregate_score", ascending=False).reset_index(drop=True)
    return df


def _enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["risk_summary"] = out["risk_flags"].apply(_format_risk_flags_summary)
    out["risk_severity"] = out["risk_flags"].apply(_max_risk_severity)
    out["has_risk"] = out["risk_summary"].apply(lambda x: str(x).lower() != "none")
    out["high_risk"] = out["risk_severity"].eq("high")
    out["short_rationale"] = out.apply(lambda r: _short_rationale_from_row(r.to_dict()), axis=1)
    out["recommendation_tag"] = out["recommendation"].apply(_recommendation_tag)
    out["valuation_tag"] = out["valuation_flag"].apply(_valuation_tag)
    return out


def _render_summary_strip(df: pd.DataFrame) -> None:
    total = len(df)
    avg_score = float(pd.to_numeric(df["aggregate_score"], errors="coerce").fillna(0).mean())
    high_risk_count = int(df["high_risk"].sum())
    stretched_full = int(df["valuation_flag"].astype(str).str.lower().isin({"stretched", "full"}).sum())
    rec_dist = df["recommendation"].astype(str).value_counts().to_dict()
    rec_line = ", ".join([f"{k}:{v}" for k, v in rec_dist.items()]) if rec_dist else "N/A"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tickers", total)
    c2.metric("Avg Score", f"{avg_score:.2f}")
    c3.metric("High Risk", high_risk_count)
    c4.metric("Full/Stretched", stretched_full)
    c5.metric("Reco Buckets", len(rec_dist))
    st.caption(f"Recommendation distribution: {rec_line}")


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    c1, c2, c3, c4, c5, c6 = st.columns([1.35, 1.1, 1.05, 1.0, 1.0, 0.95])
    with c1:
        q = st.text_input("Search ticker/company", key="flt_query").strip().lower()
    with c2:
        rec_opts = sorted(work["recommendation"].astype(str).unique().tolist())
        rec_sel = st.multiselect("Recommendation", rec_opts, default=rec_opts, key="flt_rec")
    with c3:
        val_opts = sorted(work["valuation_flag"].astype(str).unique().tolist())
        val_sel = st.multiselect("Valuation", val_opts, default=val_opts, key="flt_val")
    with c4:
        risk_mode = st.selectbox("Risk", ["All", "Has Risk", "High Risk", "Medium+", "None"], key="flt_risk")
    with c5:
        mn = int(pd.to_numeric(work["aggregate_score"], errors="coerce").fillna(0).min())
        mx = int(pd.to_numeric(work["aggregate_score"], errors="coerce").fillna(0).max())
        score_rng = st.slider("Score Range", min_value=mn, max_value=mx, value=(mn, mx), key="flt_score")
    with c6:
        top_n = st.number_input("Top N", min_value=0, max_value=max(0, len(work)), value=0, step=5, key="flt_topn")

    if q:
        work = work[
            work["ticker"].astype(str).str.lower().str.contains(q, na=False)
            | work["company"].astype(str).str.lower().str.contains(q, na=False)
        ]
    if rec_sel:
        work = work[work["recommendation"].astype(str).isin(rec_sel)]
    if val_sel:
        work = work[work["valuation_flag"].astype(str).isin(val_sel)]
    if risk_mode == "Has Risk":
        work = work[work["has_risk"]]
    elif risk_mode == "High Risk":
        work = work[work["risk_severity"] == "high"]
    elif risk_mode == "Medium+":
        work = work[work["risk_severity"].isin(["medium", "high"])]
    elif risk_mode == "None":
        work = work[~work["has_risk"]]
    work = work[
        pd.to_numeric(work["aggregate_score"], errors="coerce").fillna(0).between(score_rng[0], score_rng[1], inclusive="both")
    ]
    if top_n and top_n > 0:
        work = work.head(int(top_n))
    return work


def _render_ranking_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        st.info("No rows under current filters.")
        return df
    s1, s2, s3 = st.columns([1.25, 0.95, 0.95])
    with s1:
        sort_field = st.selectbox("Sort by", ["aggregate_score", "ticker", *SCORE_DIMENSIONS], key="sort_field")
    with s2:
        desc = st.toggle("Descending", value=True, key="sort_desc")
    with s3:
        show_dims = st.toggle("Show 8-dim columns", value=False, key="show_dims")

    sorted_df = df.sort_values(sort_field, ascending=not desc).reset_index(drop=True)
    view = sorted_df[
        ["ticker", "company", "aggregate_score", "recommendation_tag", "valuation_tag", "risk_summary", "short_rationale"]
    ].rename(
        columns={
            "aggregate_score": "score",
            "recommendation_tag": "recommendation",
            "valuation_tag": "valuation",
            "risk_summary": "risk",
            "short_rationale": "rationale",
        }
    )
    if show_dims:
        for dim in SCORE_DIMENSIONS:
            view[_pretty_dimension_name(dim)] = sorted_df[dim]
    st.dataframe(view, use_container_width=True, hide_index=True, height=430)
    return sorted_df


def _render_comparison_panel(df: pd.DataFrame) -> None:
    st.markdown("#### Comparison (Snapshot Only)")
    if df.empty:
        st.caption("No comparison candidates.")
        return
    tickers = df["ticker"].astype(str).tolist()
    default_sel = [x for x in st.session_state["comparison_selection"] if x in tickers]
    picked = st.multiselect("Select tickers", tickers, default=default_sel, key="cmp_pick")
    st.session_state["comparison_selection"] = picked
    if len(picked) < 2:
        st.caption("Select at least two tickers to compare.")
        return
    comp = df[df["ticker"].isin(picked)].copy()
    st.dataframe(
        comp[["ticker", "aggregate_score", "recommendation", "valuation_flag", "risk_summary"]],
        use_container_width=True,
        hide_index=True,
    )
    heat = comp[["ticker", *SCORE_DIMENSIONS]].set_index("ticker")
    heat = heat.rename(columns={d: _pretty_dimension_name(d) for d in SCORE_DIMENSIONS})
    try:
        st.dataframe(heat.style.background_gradient(cmap="RdYlGn", axis=None, vmin=-5, vmax=5), use_container_width=True)
    except Exception:
        st.dataframe(heat, use_container_width=True)


def _screener_drilldown_controls(sorted_df: pd.DataFrame) -> None:
    if sorted_df.empty:
        return
    c1, c2, c3 = st.columns([1.4, 1.0, 1.3])
    with c1:
        selected = st.selectbox("Selected ticker", sorted_df["ticker"].tolist(), key="detail_select_ticker")
    with c2:
        auto_open = st.toggle("Auto-open detail", value=bool(st.session_state.get("drill_on_select", True)), key="drill_on_select")
    with c3:
        st.caption("Sorting/filtering/comparison do not trigger inference.")

    if auto_open and st.session_state.get("selected_ticker") != selected:
        st.session_state["selected_ticker"] = selected
        st.session_state["workspace_page"] = "Ticker Detail"
        st.session_state["workspace_page_selector"] = "Ticker Detail"
        st.rerun()

    if st.button("Open Detail Workspace", type="primary"):
        st.session_state["selected_ticker"] = selected
        st.session_state["workspace_page"] = "Ticker Detail"
        st.session_state["workspace_page_selector"] = "Ticker Detail"
        st.rerun()


def _render_screener_page() -> None:
    st.subheader("Sector Screener")
    st.caption("Rank and filter first, then drill into selected tickers.")

    with st.container(border=True):
        st.markdown("### Scan Controls")
        st.caption("Use a scan label to group this run's artifacts under one output folder.")
        left, right = st.columns([1.15, 1.85])
        with left:
            st.text_input(
                "Scan label",
                value=st.session_state.get("scan_label") or "",
                placeholder="e.g. semis_q1_screen",
                key="scan_label_input",
                help="Used for output folder grouping and later retrieval.",
            )
            st.slider("Max workers", min_value=1, max_value=8, value=int(st.session_state.get("scan_workers", 4)), key="scan_workers")
            st.caption("Higher workers are faster but may increase API/rate-limit pressure.")
            st.file_uploader("Optional CSV (ticker/symbol column)", type=["csv"], key="scan_csv")
        with right:
            st.text_area(
                "Ticker list (newline/comma/semicolon)",
                value=st.session_state.get("scan_tickers_text", ""),
                height=130,
                key="scan_tickers_text",
            )
            run_scan = st.button("Run Universe Scan", type="primary")

        if run_scan:
            raw_label = str(st.session_state.get("scan_label_input") or "").strip()
            scan_label = raw_label or "unlabeled_scan"
            scan_id, scan_dir = _create_scan_output_dir(scan_label)

            tickers = parse_tickers(st.session_state.get("scan_tickers_text", ""))
            upload = st.session_state.get("scan_csv")
            if upload is not None:
                try:
                    upload_df = pd.read_csv(upload)
                    tickers.extend(tickers_from_dataframe(upload_df, ticker_col="ticker"))
                except Exception as exc:
                    st.warning(f"CSV parse failed: {exc}")
            tickers = sorted(set(tickers))

            if not tickers:
                st.warning("Please provide at least one valid ticker.")
            else:
                t0 = time.perf_counter()
                df = _run_scan_with_progress(
                    tickers=tickers,
                    max_workers=int(st.session_state.get("scan_workers", 4)),
                    output_dir=scan_dir,
                )
                runtime = round(time.perf_counter() - t0, 2)
                st.session_state["ranking_df"] = df
                st.session_state["scan_runtime_seconds"] = runtime
                st.session_state["scan_label"] = scan_label
                st.session_state["scan_timestamp"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["scan_id"] = scan_id
                st.session_state["scan_output_dir"] = str(scan_dir)
                st.session_state["selected_ticker"] = str(df.iloc[0]["ticker"]) if not df.empty else None
                st.session_state["comparison_selection"] = []
                # isolate per-scan cache to avoid stale leakage
                _scan_cache_bucket(scan_id)
                _runtime_bucket(scan_id)
                st.success(f"Scan completed in {runtime:.2f}s for {len(tickers)} tickers.")

    base_df = st.session_state["ranking_df"]
    if base_df.empty:
        st.info("No scan results yet. Run a scan to populate ranking workspace.")
        return

    enriched = _enrich_df(base_df)
    with st.container(border=True):
        st.markdown("### Results Workspace")
        label = st.session_state.get("scan_label") or "unlabeled_scan"
        runtime = st.session_state.get("scan_runtime_seconds")
        ts = st.session_state.get("scan_timestamp")
        out_dir = st.session_state.get("scan_output_dir")
        if runtime is not None:
            st.caption(f"Run: {label} | runtime: {runtime:.2f}s | updated: {ts}")
        if out_dir:
            st.caption(f"Current scan artifacts: `{out_dir}`")
        _render_summary_strip(enriched)

        filtered = _apply_filters(enriched)
        sorted_df = _render_ranking_table(filtered)
        _screener_drilldown_controls(sorted_df)
        _render_comparison_panel(filtered if not filtered.empty else enriched)


def _extract_top_points(row: Mapping[str, Any]) -> tuple[List[str], List[str]]:
    positives: List[str] = []
    negatives: List[str] = []
    sorted_high = sorted([(d, _safe_int(row.get(d), 0)) for d in SCORE_DIMENSIONS], key=lambda x: x[1], reverse=True)
    sorted_low = sorted([(d, _safe_int(row.get(d), 0)) for d in SCORE_DIMENSIONS], key=lambda x: x[1])
    for d, s in sorted_high[:3]:
        if s > 0:
            positives.append(f"{_pretty_dimension_name(d)} ({s:+d})")
    for d, s in sorted_low[:3]:
        if s < 0:
            negatives.append(f"{_pretty_dimension_name(d)} ({s:+d})")
    for rf in _risk_flags_list(row.get("risk_flags"))[:2]:
        negatives.append(f"{rf['type']} ({rf['severity']})")
    return positives, negatives


def _deterministic_investment_view(row: Mapping[str, Any], lean_report: Optional[Mapping[str, Any]]) -> str:
    out = _short_rationale_from_row(row)
    if isinstance(lean_report, Mapping):
        scorecard = lean_report.get("scorecard")
        if isinstance(scorecard, Mapping):
            gp = scorecard.get("growth_perspective") or {}
            ts = scorecard.get("technical_setup") or {}
            snippet = " ".join([str(gp.get("evidence") or ""), str(ts.get("evidence") or "")]).strip()
            if snippet:
                out = f"{out}. {snippet}"
    return out


def _render_ticker_header(row: Mapping[str, Any], lean_report: Optional[Mapping[str, Any]]) -> None:
    ticker = str(row.get("ticker") or "")
    company = str(row.get("company") or ticker)
    score = _safe_int(row.get("aggregate_score"), 0)
    rec = str(row.get("recommendation") or "N/A")
    valuation = str(row.get("valuation_flag") or "N/A")
    valuation_summary = ""
    if isinstance(lean_report, Mapping):
        vf = lean_report.get("valuation_flag")
        if isinstance(vf, Mapping):
            valuation_summary = str(vf.get("summary") or "")

    a, b = st.columns([1.5, 1.0])
    with a:
        st.markdown(f"### {ticker} - {company}")
        st.markdown(
            f"**Recommendation:** `{_recommendation_tag(rec)}`  |  "
            f"**Aggregate Score:** `<span class='{_score_class(score)}'>{score:+d}</span>`",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Valuation:** `{_valuation_tag(valuation)}`")
        if valuation_summary:
            st.caption(valuation_summary)
    with b:
        st.markdown("**Risk Summary**")
        st.markdown(_risk_chip_html(row.get("risk_flags")), unsafe_allow_html=True)
        st.caption(_format_risk_flags_summary(row.get("risk_flags")))


def _build_scenario_cards(ptm: Mapping[str, Any]) -> None:
    st.caption("Timeline = expected time horizon for each scenario/range to play out.")
    cols = st.columns(3)
    key_order = [("bear", "Bear"), ("consensus", "Consensus"), ("bull", "Bull")]
    for idx, (key, label) in enumerate(key_order):
        with cols[idx]:
            item = ptm.get(key) if isinstance(ptm, Mapping) else None
            if not isinstance(item, Mapping):
                st.markdown(f"<div class='terminal-card'><b>{label}</b><br/>No scenario data.</div>", unsafe_allow_html=True)
                continue
            rng = item.get("price_target_range") or {}
            low = rng.get("low")
            high = rng.get("high")
            timeline = item.get("timeline") or "N/A"
            assumption = item.get("key_assumption") or "N/A"
            st.markdown(
                f"<div class='terminal-card'><b>{label}</b><br/>"
                f"Target range: {low} - {high}<br/>"
                f"Timeline: {timeline}<br/>"
                f"<span class='terminal-muted'>{assumption}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_dimension_explanations(row: Mapping[str, Any], scorecard: Optional[Mapping[str, Any]]) -> None:
    rows: List[Tuple[str, int, str, str]] = []
    for dim in SCORE_DIMENSIONS:
        if isinstance(scorecard, Mapping) and isinstance(scorecard.get(dim), Mapping):
            item = scorecard.get(dim) or {}
            score = _safe_int(item.get("score"), 0)
            thesis = str(item.get("thesis") or "")
            evidence = str(item.get("evidence") or "")
        else:
            score = _safe_int(row.get(dim), 0)
            thesis = ""
            evidence = ""
        rows.append((dim, score, thesis, evidence))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)

    for dim, score, thesis, evidence in rows:
        lower, upper = DIMENSION_SCORE_BOUNDS.get(dim, (-5, 5))
        name = _pretty_dimension_name(dim)
        sign_txt = _score_sign_text(score)
        st.markdown(
            f"<div class='terminal-card'>"
            f"<b>{name}</b> | "
            f"<span class='{_score_class(score)}'>{score:+d}</span> ({sign_txt})"
            f"{_zero_center_score_bar_html(score, lower, upper)}"
            f"<div class='terminal-muted'>Thesis: {thesis or 'N/A'}</div>"
            f"<div class='terminal-muted'>Evidence: {evidence or 'N/A'}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_lightweight_chart(chart_data: Dict[str, Any]) -> None:
    price = chart_data.get("price", [])
    if not isinstance(price, list) or len(price) < 2:
        st.caption("No technical chart data available.")
        return
    sma20 = chart_data.get("sma_20", [])
    sma50 = chart_data.get("sma_50", [])
    sma200 = chart_data.get("sma_200", [])
    ema20 = chart_data.get("ema_20", [])
    bb_high = chart_data.get("bb_high", [])
    bb_low = chart_data.get("bb_low", [])
    macd = chart_data.get("macd", [])
    macd_signal = chart_data.get("macd_signal", [])
    macd_hist = chart_data.get("macd_hist", [])
    rsi14 = chart_data.get("rsi_14", [])
    volume = chart_data.get("volume", [])

    html = f"""
<div style="display:flex;flex-direction:column;gap:10px;">
  <div id="price-chart" style="height:280px;"></div>
  <div id="macd-chart" style="height:120px;"></div>
  <div id="rsi-chart" style="height:120px;"></div>
  <div id="volume-chart" style="height:110px;"></div>
</div>
<script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
<script>
const priceData = {json.dumps(price)};
const sma20Data = {json.dumps(sma20)};
const sma50Data = {json.dumps(sma50)};
const sma200Data = {json.dumps(sma200)};
const ema20Data = {json.dumps(ema20)};
const bbHighData = {json.dumps(bb_high)};
const bbLowData = {json.dumps(bb_low)};
const macdData = {json.dumps(macd)};
const macdSignalData = {json.dumps(macd_signal)};
const macdHistData = {json.dumps(macd_hist)};
const rsiData = {json.dumps(rsi14)};
const volumeData = {json.dumps(volume)};
const common = {{
  layout: {{ background: {{ color: '#0b1220' }}, textColor: '#d1d5db' }},
  grid: {{ vertLines: {{ color: '#1f2937' }}, horzLines: {{ color: '#1f2937' }} }},
  rightPriceScale: {{ borderColor: '#374151' }},
  timeScale: {{ borderColor: '#374151' }},
}};
function addLine(chart, options) {{
  if (typeof chart.addLineSeries === 'function') return chart.addLineSeries(options);
  return chart.addSeries(LightweightCharts.LineSeries, options);
}}
function addHistogram(chart, options) {{
  if (typeof chart.addHistogramSeries === 'function') return chart.addHistogramSeries(options);
  return chart.addSeries(LightweightCharts.HistogramSeries, options);
}}
const pc = LightweightCharts.createChart(document.getElementById('price-chart'), common);
const ps = addLine(pc, {{ color:'#60a5fa', lineWidth:2, title:'Price' }}); ps.setData(priceData);
if (sma20Data.length) {{ const s = addLine(pc, {{ color:'#22c55e', lineWidth:1, title:'SMA20' }}); s.setData(sma20Data); }}
if (sma50Data.length) {{ const s = addLine(pc, {{ color:'#f59e0b', lineWidth:1, title:'SMA50' }}); s.setData(sma50Data); }}
if (sma200Data.length) {{ const s = addLine(pc, {{ color:'#ef4444', lineWidth:1, title:'SMA200' }}); s.setData(sma200Data); }}
if (ema20Data.length) {{ const s = addLine(pc, {{ color:'#a78bfa', lineWidth:1, title:'EMA20' }}); s.setData(ema20Data); }}
if (bbHighData.length) {{ const s = addLine(pc, {{ color:'#64748b', lineWidth:1, lineStyle:2, title:'BB High' }}); s.setData(bbHighData); }}
if (bbLowData.length) {{ const s = addLine(pc, {{ color:'#64748b', lineWidth:1, lineStyle:2, title:'BB Low' }}); s.setData(bbLowData); }}
pc.timeScale().fitContent();
const mc = LightweightCharts.createChart(document.getElementById('macd-chart'), common);
if (macdHistData.length) {{
  const h = addHistogram(mc, {{ title:'MACD Hist' }});
  h.setData(macdHistData.map(p => ({{ time:p.time, value:p.value, color:p.value >= 0 ? '#34d399' : '#f87171' }})));
}}
if (macdData.length) {{ const s = addLine(mc, {{ color:'#60a5fa', lineWidth:1, title:'MACD' }}); s.setData(macdData); }}
if (macdSignalData.length) {{ const s = addLine(mc, {{ color:'#f59e0b', lineWidth:1, title:'Signal' }}); s.setData(macdSignalData); }}
mc.timeScale().fitContent();
const rc = LightweightCharts.createChart(document.getElementById('rsi-chart'), common);
if (rsiData.length) {{
  const rs = addLine(rc, {{ color:'#34d399', lineWidth:1, title:'RSI14' }});
  rs.setData(rsiData);
}}
rc.timeScale().fitContent();
const vc = LightweightCharts.createChart(document.getElementById('volume-chart'), common);
if (volumeData.length) {{
  const v = addHistogram(vc, {{ color:'#60a5fa', title:'Volume' }});
  v.setData(volumeData);
}}
vc.timeScale().fitContent();
</script>
"""
    components.html(html, height=700, scrolling=False)


def _get_full_report_for_current_scan(ticker: str) -> tuple[Optional[Dict[str, Any]], Optional[Path]]:
    scan_id = st.session_state.get("scan_id")
    bucket = _scan_cache_bucket(scan_id)
    if ticker in bucket:
        return bucket[ticker]

    scan_dir_str = st.session_state.get("scan_output_dir")
    scan_dir = Path(scan_dir_str) if scan_dir_str else None
    payload, path = _find_latest_report(
        ticker=ticker,
        suffix="full_report",
        preferred_dir=scan_dir,
        allow_global_fallback=False,
    )
    if payload is not None:
        bucket[ticker] = (payload, path)
        return payload, path
    return None, None


def _load_historical_full_report(ticker: str) -> tuple[Optional[Dict[str, Any]], Optional[Path]]:
    return _find_latest_report(
        ticker=ticker,
        suffix="full_report",
        preferred_dir=None,
        allow_global_fallback=True,
    )


def _generate_full_report_explicit(ticker: str, row: Mapping[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[Path]]:
    existing, existing_path = _get_full_report_for_current_scan(ticker)
    if existing is not None:
        return existing, existing_path

    scan_dir_str = st.session_state.get("scan_output_dir")
    output_dir = Path(scan_dir_str) if scan_dir_str else _ensure_outputs_dir()
    st.session_state["full_report_generating_for"] = ticker
    try:
        with st.spinner(f"Generating full report for {ticker}..."):
            t0 = time.perf_counter()
            report, path = generate_report_v2(ticker, report_mode="full", save=True, output_dir=output_dir)
            report["aggregate_score"] = _safe_int(row.get("aggregate_score"), _safe_int(report.get("aggregate_score"), 0))
            report["recommendation"] = str(row.get("recommendation") or report.get("recommendation") or "Hold")
            if isinstance(report.get("valuation_flag"), dict):
                report["valuation_flag"]["label"] = str(
                    row.get("valuation_flag") or report["valuation_flag"].get("label") or "reasonable"
                )
            if isinstance(row.get("risk_flags"), list):
                report["risk_flags"] = row.get("risk_flags")
            elapsed = round(time.perf_counter() - t0, 2)

        sid = st.session_state.get("scan_id")
        _scan_cache_bucket(sid)[ticker] = (report, path)
        _runtime_bucket(sid)[ticker] = elapsed
        return report, path
    finally:
        st.session_state["full_report_generating_for"] = None


def _render_risk_summary_block(row: Mapping[str, Any]) -> None:
    st.markdown(_risk_chip_html(row.get("risk_flags")), unsafe_allow_html=True)
    st.caption(_format_risk_flags_summary(row.get("risk_flags")))
    with st.expander("Risk details", expanded=False):
        rows = _risk_flags_list(row.get("risk_flags"))
        if not rows:
            st.caption("No risk flags.")
        for r in rows:
            st.markdown(f"- **{r['type']}** ({r['severity']}): {r['summary'] or 'N/A'}")


def _render_detail_page() -> None:
    st.subheader("Ticker Detail")
    base_df = st.session_state["ranking_df"]
    if base_df.empty:
        st.info("No screener dataset loaded. Run a scan first.")
        return

    tickers = base_df["ticker"].astype(str).tolist()
    selected = st.session_state.get("selected_ticker")
    if selected not in tickers:
        selected = tickers[0]
    selected = st.selectbox("Ticker", tickers, index=tickers.index(selected), key="detail_ticker_selector")
    st.session_state["selected_ticker"] = selected

    row = base_df[base_df["ticker"] == selected].iloc[0].to_dict()
    lean_report = _load_lean_from_row(row)
    full_report, full_path = _get_full_report_for_current_scan(selected)

    _render_ticker_header(row, lean_report)
    positives, negatives = _extract_top_points(row)
    iview = _deterministic_investment_view(row, lean_report)

    tabs = st.tabs(
        ["Overview", "Score Drivers", "Catalysts & Risks", "Technical", "Scenario / Valuation", "Full Report", "Raw JSON / Developer"]
    )

    with tabs[0]:
        st.markdown("#### Executive View")
        st.info(iview)
        a, b = st.columns(2)
        with a:
            st.markdown("**Top Positives**")
            for p in positives or ["No strong positive drivers."]:
                st.markdown(f"- {p}")
        with b:
            st.markdown("**Top Risks**")
            for n in negatives or ["No acute downside drivers."]:
                st.markdown(f"- {n}")

    with tabs[1]:
        st.markdown("#### 8-Dimension Score Explanation")
        scorecard = None
        if isinstance(lean_report, Mapping):
            sc = lean_report.get("scorecard")
            if isinstance(sc, Mapping):
                scorecard = sc
        _render_dimension_explanations(row, scorecard)

    with tabs[2]:
        st.markdown("#### Catalysts & Risks")
        if isinstance(lean_report, Mapping) and isinstance(lean_report.get("scorecard"), Mapping):
            sc = lean_report.get("scorecard") or {}
            pos = sc.get("positive_catalysts") or {}
            neg = sc.get("negative_catalysts") or {}
            x, y = st.columns(2)
            with x:
                st.markdown("**Positive Catalysts**")
                st.caption(f"Thesis: {pos.get('thesis') or 'N/A'}")
                st.caption(f"Evidence: {pos.get('evidence') or 'N/A'}")
            with y:
                st.markdown("**Negative Catalysts**")
                st.caption(f"Thesis: {neg.get('thesis') or 'N/A'}")
                st.caption(f"Evidence: {neg.get('evidence') or 'N/A'}")
        st.markdown("**Risk Summary**")
        _render_risk_summary_block(row)

    with tabs[3]:
        st.markdown("#### Technical Context")
        if isinstance(full_report, Mapping):
            chart_data = full_report.get("top_indicator_chart_data")
            if isinstance(chart_data, dict):
                _render_lightweight_chart(chart_data)
            else:
                st.caption("No technical chart payload in full report.")
        else:
            st.caption("Generate/load full report to display detailed technical charts.")
        if isinstance(lean_report, Mapping):
            tech = (lean_report.get("scorecard") or {}).get("technical_setup") or {}
            if isinstance(tech, Mapping):
                st.markdown("**Technical setup snapshot**")
                st.caption(f"Thesis: {tech.get('thesis') or 'N/A'}")
                st.caption(f"Evidence: {tech.get('evidence') or 'N/A'}")

    with tabs[4]:
        st.markdown("#### Scenario & Valuation")
        valuation = (lean_report.get("valuation_flag") or {}) if isinstance(lean_report, Mapping) else {}
        st.markdown(
            f"<div class='terminal-card'><b>Valuation:</b> {_valuation_tag(str(valuation.get('label') or row.get('valuation_flag') or 'N/A'))}<br/>"
            f"Confidence: {valuation.get('confidence') or 'N/A'}<br/>"
            f"{valuation.get('summary') or 'No valuation summary available.'}</div>",
            unsafe_allow_html=True,
        )
        ptm = (full_report.get("price_target_matrix") or {}) if isinstance(full_report, Mapping) else {}
        if isinstance(ptm, Mapping) and ptm:
            if "consensus" not in ptm and "base" in ptm:
                ptm["consensus"] = ptm["base"]
            _build_scenario_cards(ptm)
        else:
            st.caption("No scenario matrix loaded in current scan context.")

    with tabs[5]:
        st.markdown("#### Full Report (On-Demand)")
        if isinstance(full_report, Mapping):
            st.success("Current-scan full report loaded.")
            st.info(str(full_report.get("overall_outlook") or "N/A"))
            st.markdown("**Scenario Matrix**")
            _build_scenario_cards(full_report.get("price_target_matrix") or {})

            rt = _runtime_bucket(st.session_state.get("scan_id")).get(selected)
            if rt is not None:
                st.caption(f"Generation runtime: {rt:.2f}s")
            if isinstance(full_path, Path):
                st.caption(f"Source: {full_path}")
        else:
            st.caption("No full report loaded for current scan.")

        c1, c2 = st.columns([1.0, 1.2])
        with c1:
            if st.button("Generate Full Report", type="primary", key=f"gen_full_{selected}"):
                rep, _ = _generate_full_report_explicit(selected, row)
                if rep is not None:
                    st.success("Full report generated and cached under current scan.")
                    st.rerun()
                else:
                    st.error("Full report generation failed.")
        with c2:
            if st.button("Load latest historical full report", key=f"load_hist_{selected}"):
                hist, hist_path = _load_historical_full_report(selected)
                if hist is not None:
                    _scan_cache_bucket(st.session_state.get("scan_id"))[selected] = (hist, hist_path)
                    st.success("Historical report loaded into current view.")
                    st.rerun()
                else:
                    st.info("No historical full report found.")

    with tabs[6]:
        st.markdown("#### Developer / Raw Data")
        with st.expander("Lean Snapshot JSON", expanded=False):
            st.json(lean_report or row)
        with st.expander("Full Report JSON", expanded=False):
            st.json(full_report or {})
        st.markdown("**Validation**")
        if isinstance(full_report, Mapping):
            st.json(full_report.get("validation"))
        elif isinstance(lean_report, Mapping):
            st.json(lean_report.get("validation"))
        else:
            st.caption("No validation payload.")


def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## TickerAnalysis v2")
        page = st.radio("Workspace", ["Sector Screener", "Ticker Detail"], key="workspace_page_selector")
        st.session_state["workspace_page"] = page

        st.markdown("### Current Scan")
        st.caption(f"Label: {st.session_state.get('scan_label') or 'N/A'}")
        st.caption(f"Scan ID: {st.session_state.get('scan_id') or 'N/A'}")
        if st.session_state.get("scan_runtime_seconds") is not None:
            st.caption(f"Runtime: {st.session_state['scan_runtime_seconds']:.2f}s")
        if st.session_state.get("scan_output_dir"):
            st.caption(f"Artifacts: {st.session_state['scan_output_dir']}")

        with st.expander("How to Use", expanded=False):
            st.markdown(
                "1. Enter a sector/scan label for this batch.\n"
                "2. Paste ticker list or upload CSV.\n"
                "3. Run screener to produce lean snapshots.\n"
                "4. Filter/sort/compare to shortlist names.\n"
                "5. Open ticker detail for shortlisted names.\n"
                "6. Generate full report only when deeper view is needed.\n"
                "7. Reuse existing reports when available."
            )
            st.caption("Sorting/filtering/comparison do not trigger new inference.")
            st.caption("Full report generation is explicit and on-demand.")
            st.caption("Current run artifacts stay grouped under active scan folder.")
    return page


def _render_footer() -> None:
    st.markdown(
        "<div class='footer-row'>© 2026 TickerAnalysis. All rights reserved.</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    # Copyright (c) 2026 TickerAnalysis.
    st.set_page_config(page_title="TickerAnalysis v2 Terminal", layout="wide")
    _init_state()
    _inject_theme()

    page = _render_sidebar()
    if page == "Sector Screener":
        _render_screener_page()
    else:
        _render_detail_page()
    _render_footer()


if __name__ == "__main__":
    main()

