from __future__ import annotations

import numpy as np
import pandas as pd

from technical_pipeline import (
    build_technical_features,
    build_technical_snapshot_dict,
)


def _make_dummy_ohlcv(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = np.linspace(100, 120, n)
    high = close + 1.0
    low = close - 1.0
    open_ = close - 0.5
    volume = np.full(n, 1_000_000, dtype=float)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )


def test_build_technical_features_adds_expected_columns() -> None:
    df = _make_dummy_ohlcv()
    out = build_technical_features(df)

    expected_cols = [
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_20",
        "macd",
        "macd_signal",
        "macd_diff",
        "rsi_14",
        "stoch_k",
        "stoch_d",
        "williams_r",
        "atr_14",
        "bb_mid",
        "bb_high",
        "bb_low",
        "bb_width",
        "bb_pos",
        "obv",
        "cmf",
        "mfi_14",
        "vwap_14",
    ]

    for col in expected_cols:
        assert col in out.columns

    # Latest row should have concrete values for the key indicators
    last = out.iloc[-1]
    for col in ["sma_20", "sma_50", "ema_20", "rsi_14", "atr_14"]:
        assert not np.isnan(last[col])


def test_build_technical_snapshot_dict_shape_and_summary() -> None:
    df = _make_dummy_ohlcv()
    snap = build_technical_snapshot_dict(df, ticker="TEST", analysis_date="2099-01-01")

    # Core keys
    for key in [
        "ticker",
        "analysis_date",
        "last_close",
        "support",
        "resistance_near",
        "resistance_major",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr_14",
        "bb_width",
        "bb_pos",
        "volume_last",
        "avg_volume_20",
        "data_confidence",
        "technical_summary_text",
    ]:
        assert key in snap

    assert snap["ticker"] == "TEST"
    assert snap["analysis_date"] == "2099-01-01"
    assert isinstance(snap["technical_summary_text"], str)
    assert len(snap["technical_summary_text"]) > 0

