from __future__ import annotations

"""
report_schema.py

Shared schema constants and scoring helpers for v2 reports.
"""

from typing import Any, Dict, Iterable, List, Mapping


REPORT_VERSION = "2.0"
PROMPT_VERSION = "v2"

SCORE_DIMENSIONS: List[str] = [
    "leadership",
    "competitive_advantage",
    "growth_perspective",
    "balance_sheet_health",
    "business_segment_quality",
    "technical_setup",
    "positive_catalysts",
    "negative_catalysts",
]

# Simple, readable labels for UI and prompts (keys unchanged in JSON).
DIMENSION_DISPLAY_NAMES: Dict[str, str] = {
    "leadership": "Leadership",
    "competitive_advantage": "Competitive Advantage",
    "growth_perspective": "Growth Perspective",
    "balance_sheet_health": "Balance Sheet Health",
    "business_segment_quality": "Business Segment Quality",
    "technical_setup": "Technical & Liquidity Analysis",
    "positive_catalysts": "Positive Catalysts",
    "negative_catalysts": "Negative Catalysts",
}

DIMENSION_SCORE_BOUNDS: Dict[str, tuple[int, int]] = {
    "leadership": (-5, 5),
    "competitive_advantage": (-5, 5),
    "growth_perspective": (-5, 5),
    "balance_sheet_health": (-5, 5),
    "business_segment_quality": (-5, 5),
    "technical_setup": (-5, 5),
    "positive_catalysts": (0, 5),
    "negative_catalysts": (-5, 0),
}

CONFIDENCE_LEVELS = {"low", "medium", "high"}
VALUATION_LABELS = {"cheap", "reasonable", "full", "stretched"}
RISK_TYPES = {
    "execution_risk",
    "dilution_risk",
    "balance_sheet_risk",
    "concentration_risk",
    "customer_concentration_risk",
    "regulatory_risk",
    "technical_breakdown_risk",
    "cyclicality_risk",
    "governance_risk",
    "liquidity_risk",
}
RISK_SEVERITIES = {"low", "medium", "high"}
RECOMMENDATIONS = {"Overweight", "Equal-weight", "Hold", "Underweight", "Reduce"}


def recommendation_from_score(score: int) -> str:
    if score >= 18:
        return "Overweight"
    if 8 <= score <= 17:
        return "Equal-weight"
    if -7 <= score <= 7:
        return "Hold"
    if -17 <= score <= -8:
        return "Underweight"
    return "Reduce"


def build_default_dimension() -> Dict[str, Any]:
    return {
        "score": 0,
        "confidence": "medium",
        "thesis": "Insufficient evidence for a directional call.",
        "evidence": "Limited or mixed upstream signals.",
    }


def normalize_score(raw_score: Any, lower: int, upper: int) -> int:
    try:
        score = int(raw_score)
    except (TypeError, ValueError):
        score = 0
    return max(lower, min(upper, score))


def normalize_confidence(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in CONFIDENCE_LEVELS:
        return text
    return "medium"


def normalize_scorecard(scorecard: Mapping[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    source = scorecard if isinstance(scorecard, Mapping) else {}

    for dim in SCORE_DIMENSIONS:
        base = build_default_dimension()
        row = source.get(dim) if isinstance(source, Mapping) else None
        if not isinstance(row, Mapping):
            row = {}

        lower, upper = DIMENSION_SCORE_BOUNDS[dim]
        base["score"] = normalize_score(row.get("score"), lower, upper)
        base["confidence"] = normalize_confidence(row.get("confidence"))
        thesis = str(row.get("thesis") or "").strip()
        evidence = str(row.get("evidence") or "").strip()
        if thesis:
            base["thesis"] = thesis
        if evidence:
            base["evidence"] = evidence
        normalized[dim] = base

    return normalized


def compute_aggregate_score(
    scorecard: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, float] | None = None,
) -> int:
    """
    v2.0 uses equal weights by default, but this function supports configurable
    weights to keep future changes straightforward.
    """
    total = 0.0
    for dim in SCORE_DIMENSIONS:
        row = scorecard.get(dim) or {}
        try:
            score = int(row.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        weight = 1.0 if weights is None else float(weights.get(dim, 1.0))
        total += score * weight
    return int(round(total))


def scorecard_to_row(
    *,
    as_of_date: str,
    ticker: str,
    company: str,
    scorecard: Mapping[str, Mapping[str, Any]],
    aggregate_score: int,
    recommendation: str,
    valuation_flag: Mapping[str, Any],
    risk_flags: Iterable[Mapping[str, Any]],
    model_name: str,
    prompt_version: str,
    news_enabled: bool,
    report_path: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "as_of_date": as_of_date,
        "ticker": ticker,
        "company": company,
        "aggregate_score": aggregate_score,
        "recommendation": recommendation,
        "valuation_flag": valuation_flag.get("label"),
        "risk_flags": [dict(x) for x in risk_flags],
        "model_name": model_name,
        "prompt_version": prompt_version,
        "news_enabled": bool(news_enabled),
        "report_path": report_path,
    }
    for dim in SCORE_DIMENSIONS:
        row[dim] = int((scorecard.get(dim) or {}).get("score", 0))
    return row
