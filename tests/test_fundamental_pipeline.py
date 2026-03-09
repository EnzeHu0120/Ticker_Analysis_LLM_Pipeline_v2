from __future__ import annotations

import pandas as pd

from fundamental_pipeline import (
    safe_div,
    yoy_pct_change,
    pick_first_available,
    fetch_annual_5_periods_with_metrics,
)


def test_safe_div_handles_zero_and_nan() -> None:
    numer = pd.Series([1.0, 2.0, None])
    denom = pd.Series([1.0, 0.0, 2.0])

    out = safe_div(numer, denom)

    assert out.iloc[0] == 1.0
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[2])


def test_yoy_pct_change_does_not_forward_fill() -> None:
    s = pd.Series([100.0, None, 121.0])
    pct = yoy_pct_change(s)

    # First entry NaN by definition, middle stays NaN, last has a valid change only to previous valid point
    assert pd.isna(pct.iloc[0])
    assert pd.isna(pct.iloc[1])


def test_pick_first_available_prefers_first_existing_column() -> None:
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "C": [10, 20, 30],
        }
    )
    out = pick_first_available(df, ["B", "A", "C"])
    assert out.tolist() == [1, 2, 3]


def test_fetch_annual_5_periods_with_metrics_basic_shape() -> None:
    """
    Light integration test against yfinance.
    Confirms that we always return a metrics DataFrame with 5 rows and key columns present.
    """
    out = fetch_annual_5_periods_with_metrics("ORCL", periods=5)
    metrics = out["metrics"]

    assert metrics.shape[0] == 5
    for col in [
        "TotalRevenue",
        "OperatingIncome",
        "NetIncome",
        "TotalAssets",
        "TotalDebt",
        "TotalEquity",
        "Debt_to_Equity",
        "Operating_Margin",
        "Net_Margin",
    ]:
        assert col in metrics.columns

