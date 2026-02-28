from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 1) 数学/安全处理
# ============================================================

def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """安全除法：分母0/NaN 或结果inf -> NaN"""
    n = pd.to_numeric(numer, errors="coerce")
    d = pd.to_numeric(denom, errors="coerce").replace(0, np.nan)
    out = n / d
    return out.where(np.isfinite(out), np.nan)


def yoy_pct_change(series: pd.Series) -> pd.Series:
    """年报同比：不做隐式填充，缺失就缺失（避免 FutureWarning）"""
    s = pd.to_numeric(series, errors="coerce")
    return s.pct_change(fill_method=None)


def pick_first_available(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """在 df.columns 里找第一个存在的列；都没有就返回 NaN Series"""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(index=df.index, data=np.nan, dtype="float64")


# ============================================================
# 2) yfinance 取数：同一张表多来源合并互补
# ============================================================

def safe_get_df(tk: yf.Ticker, name: str) -> pd.DataFrame | None:
    """安全获取单个 DataFrame 属性（无布尔歧义）"""
    val = getattr(tk, name, None)
    if isinstance(val, pd.DataFrame) and not val.empty:
        return val
    return None


def merge_statement_sources(tk: yf.Ticker, candidates: list[str]) -> pd.DataFrame | None:
    """
    对同一张 statement 的多个候选来源：
    - 全部取回来（非空的）
    - 用 combine_first 做互补合并（左边缺值就用右边补）
    - 返回合并后的原始形状（items x dates）
    """
    dfs: list[pd.DataFrame] = []
    for name in candidates:
        df = safe_get_df(tk, name)
        if df is not None:
            dfs.append(df)

    if not dfs:
        return None

    # 统一 index（items）取并集，columns（dates）取并集
    # combine_first 要求对齐，所以先对齐到并集
    all_index = dfs[0].index
    all_cols = dfs[0].columns
    for d in dfs[1:]:
        all_index = all_index.union(d.index)
        all_cols = all_cols.union(d.columns)

    aligned = []
    for d in dfs:
        aligned.append(d.reindex(index=all_index, columns=all_cols))

    merged = aligned[0]
    for d in aligned[1:]:
        merged = merged.combine_first(d)

    # 有些来源会产生全空列，顺手去掉
    merged = merged.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return merged if not merged.empty else None


def statement_to_rows(raw: pd.DataFrame | None) -> pd.DataFrame:
    """
    把 raw（items x dates）转换为 rows（dates x items）：
    - index: 财年期末日期（Timestamp）
    - columns: 科目
    """
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.columns = pd.to_datetime(df.columns, errors="coerce")
    df = df.loc[:, df.columns.notna()]
    if df.empty:
        return pd.DataFrame()

    df = df.T.sort_index()
    df.insert(0, "FiscalYear", df.index.year.astype(int))
    return df


# ============================================================
# 3) anchor_dates：只用“实际返回的日期”，不足5才往前补
# ============================================================

def build_anchor_dates(
    income_idx: pd.DatetimeIndex,
    balance_idx: pd.DatetimeIndex,
    cash_idx: pd.DatetimeIndex,
    periods: int = 5,
) -> pd.DatetimeIndex:
    """
    anchor_dates = 三表日期并集（union）的最新 periods 个
    如果 union 本身不足 periods，则只在最早日期之前逐年补齐（同月同日），直到 periods 个
    """
    union_dates = sorted(set(income_idx).union(set(balance_idx)).union(set(cash_idx)))
    if not union_dates:
        return pd.DatetimeIndex([])

    anchor = list(union_dates[-periods:])

    if len(anchor) < periods:
        oldest = anchor[0]
        mm, dd = oldest.month, oldest.day
        y = oldest.year
        while len(anchor) < periods:
            y -= 1
            anchor.insert(0, pd.Timestamp(year=y, month=mm, day=dd))

    return pd.DatetimeIndex(anchor)


def coverage_report(raw_rows: pd.DataFrame, anchor_dates: pd.DatetimeIndex, label: str) -> pd.DataFrame:
    """报告：对 anchor_dates 哪些日期该表缺失（缺失=整行不存在）"""
    have = set(raw_rows.index) if not raw_rows.empty else set()
    missing = [d for d in anchor_dates if d not in have]
    return pd.DataFrame([{
        "Statement": label,
        "AvailablePeriods": 0 if raw_rows.empty else len(raw_rows.index),
        "AnchorPeriods": len(anchor_dates),
        "MissingAnchorDates": ", ".join([d.strftime("%Y-%m-%d") for d in missing]) if missing else "",
    }])


# ============================================================
# 4) 主函数：annual only + 输出固定5期 + 计算 metrics
# ============================================================

def fetch_annual_5_periods_with_metrics(ticker: str, periods: int = 5) -> dict[str, pd.DataFrame]:
    ticker = ticker.strip().upper()
    tk = yf.Ticker(ticker)

    # ---- (1) 对每张表：把可能来源都抓下来并互补合并 ----
    # Income: income_stmt 与 financials 可能互补
    income_raw = merge_statement_sources(tk, ["income_stmt", "financials"])
    # Balance: balance_sheet 与 balancesheet 可能互补
    balance_raw = merge_statement_sources(tk, ["balance_sheet", "balancesheet"])
    # Cash: cashflow 与 cash_flow（不同版本命名）可能互补
    cash_raw = merge_statement_sources(tk, ["cashflow", "cash_flow"])

    # ---- (2) 转成 rows=日期 ----
    income_rows_raw = statement_to_rows(income_raw)
    balance_rows_raw = statement_to_rows(balance_raw)
    cash_rows_raw = statement_to_rows(cash_raw)

    if income_rows_raw.empty and balance_rows_raw.empty and cash_rows_raw.empty:
        meta = pd.DataFrame([{"Ticker": ticker, "Note": "No annual statements returned by yfinance."}])
        return {"meta": meta, "income_statement": pd.DataFrame(), "balance_sheet": pd.DataFrame(),
                "cash_flow": pd.DataFrame(), "metrics": pd.DataFrame(), "coverage": pd.DataFrame()}

    # ---- (3) anchor_dates：完全基于“实际返回日期”，不造中间年份 ----
    anchor_dates = build_anchor_dates(
        income_rows_raw.index if not income_rows_raw.empty else pd.DatetimeIndex([]),
        balance_rows_raw.index if not balance_rows_raw.empty else pd.DatetimeIndex([]),
        cash_rows_raw.index if not cash_rows_raw.empty else pd.DatetimeIndex([]),
        periods=periods,
    )

    # ---- (4) reindex 到 anchor_dates（固定输出5行；缺的=NaN） ----
    income = income_rows_raw.reindex(anchor_dates)
    balance = balance_rows_raw.reindex(anchor_dates)
    cash = cash_rows_raw.reindex(anchor_dates)

    # FiscalYear 确保 padding 行也有
    for df in (income, balance, cash):
        if "FiscalYear" in df.columns:
            df["FiscalYear"] = df.index.year.astype(int)
        else:
            df.insert(0, "FiscalYear", df.index.year.astype(int))

    # ---- (5) 取字段 + 算 metrics（除不了就 NaN） ----
    # Balance inputs
    total_assets = pick_first_available(balance, ["Total Assets", "TotalAssets"])
    total_equity = pick_first_available(
        balance,
        ["Total Stockholder Equity", "TotalStockholderEquity", "Stockholders Equity", "StockholdersEquity",
         "Total Equity Gross Minority Interest"],
    )
    total_debt = pick_first_available(balance, ["Total Debt", "TotalDebt"])
    if total_debt.isna().all():
        ltd = pick_first_available(balance, ["Long Term Debt", "LongTermDebt"])
        std = pick_first_available(balance, ["Short Long Term Debt", "ShortLongTermDebt", "Current Debt", "CurrentDebt"])
        total_debt = pd.to_numeric(ltd, errors="coerce") + pd.to_numeric(std, errors="coerce")

    # Income inputs
    total_revenue = pick_first_available(income, ["Total Revenue", "TotalRevenue", "Revenue"])
    operating_income = pick_first_available(income, ["Operating Income", "OperatingIncome"])
    net_income = pick_first_available(income, ["Net Income", "NetIncome", "Net Income Common Stockholders", "NetIncomeCommonStockholders"])
    pretax_income = pick_first_available(income, ["Pretax Income", "PretaxIncome", "Income Before Tax", "IncomeBeforeTax"])
    tax_provision = pick_first_available(income, ["Tax Provision", "TaxProvision", "Income Tax Expense", "IncomeTaxExpense"])

    metrics = pd.DataFrame(
        {
            "FiscalYear": anchor_dates.year.astype(int),
            "TotalRevenue": pd.to_numeric(total_revenue, errors="coerce"),
            "OperatingIncome": pd.to_numeric(operating_income, errors="coerce"),
            "NetIncome": pd.to_numeric(net_income, errors="coerce"),
            "PretaxIncome": pd.to_numeric(pretax_income, errors="coerce"),
            "TaxProvision": pd.to_numeric(tax_provision, errors="coerce"),
            "TotalAssets": pd.to_numeric(total_assets, errors="coerce"),
            "TotalDebt": pd.to_numeric(total_debt, errors="coerce"),
            "TotalEquity": pd.to_numeric(total_equity, errors="coerce"),
            "Debt_to_Equity": safe_div(total_debt, total_equity),
            "Asset_Growth_YoY": yoy_pct_change(total_assets),
            "Debt_Growth_YoY": yoy_pct_change(total_debt),
            "Operating_Margin": safe_div(operating_income, total_revenue),
            "Net_Margin": safe_div(net_income, total_revenue),
            "Effective_Tax_Rate": safe_div(tax_provision, pretax_income),
        },
        index=anchor_dates,
    )

    # ---- (6) coverage：缺“整行”的情况（注意：即使不缺行，也可能某些字段是 NaN） ----
    cov = pd.concat(
        [
            coverage_report(income_rows_raw, anchor_dates, "IncomeStatement"),
            coverage_report(balance_rows_raw, anchor_dates, "BalanceSheet"),
            coverage_report(cash_rows_raw, anchor_dates, "CashFlow"),
        ],
        ignore_index=True,
    )

    # 额外给一个“关键字段缺失率”提示（解释你为啥还会看到 NaN）
    key_field_missing = pd.DataFrame([{
        "AnchorDates": ", ".join([d.strftime("%Y-%m-%d") for d in anchor_dates]),
        "TotalRevenue_isna": int(pd.to_numeric(total_revenue, errors="coerce").isna().sum()),
        "TotalAssets_isna": int(pd.to_numeric(total_assets, errors="coerce").isna().sum()),
        "NetIncome_isna": int(pd.to_numeric(net_income, errors="coerce").isna().sum()),
    }])

    info = getattr(tk, "info", {}) or {}
    meta = pd.DataFrame([{
        "Ticker": ticker,
        "ShortName": info.get("shortName"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Currency": info.get("currency"),
        "MarketCap": info.get("marketCap"),
        "Exchange": info.get("exchange"),
        "Country": info.get("country"),
        "PeriodsRequested": periods,
        "AnchorDates": ", ".join([d.strftime("%Y-%m-%d") for d in anchor_dates]),
        "IncomeRawPeriods": 0 if income_raw is None else income_raw.shape[1],
        "BalanceRawPeriods": 0 if balance_raw is None else balance_raw.shape[1],
        "CashRawPeriods": 0 if cash_raw is None else cash_raw.shape[1],
    }])

    return {
        "meta": meta,
        "coverage": cov,
        "key_field_missing": key_field_missing,
        "income_statement": income,
        "balance_sheet": balance,
        "cash_flow": cash,
        "metrics": metrics,
    }


# ============================================================
# 5) Run
# ============================================================

if __name__ == "__main__":
    ticker_in = input("Enter ticker (e.g., ORCL, AAPL, MSFT): ").strip().upper()
    if not ticker_in:
        raise ValueError("Ticker cannot be empty.")

    out = fetch_annual_5_periods_with_metrics(ticker_in, periods=5)

    print("\n=== META ===")
    print(out["meta"].to_string(index=False))

    print("\n=== COVERAGE (missing entire rows for anchor dates) ===")
    print(out["coverage"].to_string(index=False))

    print("\n=== KEY FIELD MISSING COUNTS (explains why NaN can still appear) ===")
    print(out["key_field_missing"].to_string(index=False))

    print("\n=== INCOME STATEMENT (annual; anchor-based 5 periods; missing -> NaN) ===")
    print(out["income_statement"])

    print("\n=== BALANCE SHEET (annual; anchor-based 5 periods; missing -> NaN) ===")
    print(out["balance_sheet"])

    print("\n=== CASH FLOW (annual; anchor-based 5 periods; missing -> NaN) ===")
    print(out["cash_flow"])

    print("\n=== METRICS (annual; anchor-based 5 periods; missing -> NaN) ===")
    print(out["metrics"].round(6))
