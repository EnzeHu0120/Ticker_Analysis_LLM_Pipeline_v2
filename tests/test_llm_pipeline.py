from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_pipeline import (
    df_to_csv_str,
    safe_json_dumps,
    build_unique_output_path,
    normalize_rating,
    normalize_price_target_matrix,
)


def test_df_to_csv_str_basic() -> None:
    df = pd.DataFrame({"A": [1, 2]}, index=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])
    s = df_to_csv_str(df, max_rows=5, max_cols=5)
    # Should contain header and both dates
    assert "A" in s
    assert "2024-01-01" in s
    assert "2024-01-02" in s


def test_safe_json_dumps_roundtrip() -> None:
    obj = {"x": 1, "y": "测试"}  # non-ascii should be preserved
    s = safe_json_dumps(obj)
    loaded = json.loads(s)
    assert loaded == obj


def test_build_unique_output_path_never_overwrites(tmp_path: Path, monkeypatch) -> None:
    # Force outputs directory to be under tmp_path
    from llm_pipeline import THIS_DIR as _THIS_DIR  # type: ignore
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    # Monkeypatch ensure_outputs_dir to use tmp_path / "outputs"
    from llm_pipeline import ensure_outputs_dir as _ensure

    def _ensure_override() -> Path:
        out = tmp_path / "outputs"
        out.mkdir(exist_ok=True)
        return out

    import llm_pipeline as llm_mod

    llm_mod.ensure_outputs_dir = _ensure_override  # type: ignore[assignment]

    p1 = build_unique_output_path("TEST", "2099-01-01")
    p1.write_text("{}", encoding="utf-8")
    p2 = build_unique_output_path("TEST", "2099-01-01")
    assert p1 != p2


def test_normalize_rating_aliases() -> None:
    report = {"rating": "Strong buy / Overweight view"}
    normalize_rating(report)
    assert report["rating"] == "Overweight"

    report2 = {"rating": "reduce / take profits"}
    normalize_rating(report2)
    assert report2["rating"] == "Reduce"


def test_normalize_price_target_matrix_deduplicates_and_orders() -> None:
    report = {
        "price_target_matrix": [
            {"scenario": "Consensus", "timeline": "12m"},
            {"scenario": "Bear", "timeline": "12m"},
            {"scenario": "Bear", "timeline": "duplicated"},
            {"scenario": "Bull", "timeline": "12m"},
        ]
    }
    normalize_price_target_matrix(report)
    matrix = report["price_target_matrix"]
    assert [row["scenario"] for row in matrix] == ["Bear", "Consensus", "Bull"]
    assert matrix[0]["timeline"] == "12m"

