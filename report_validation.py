from __future__ import annotations

"""
report_validation.py

Validation for v2 lean/full report schema.
"""

from typing import Any, Dict, List, Mapping

from report_schema import (
    CONFIDENCE_LEVELS,
    DIMENSION_SCORE_BOUNDS,
    PROMPT_VERSION,
    RECOMMENDATIONS,
    REPORT_VERSION,
    RISK_SEVERITIES,
    RISK_TYPES,
    SCORE_DIMENSIONS,
    VALUATION_LABELS,
    compute_aggregate_score,
    recommendation_from_score,
)


def _is_non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_v2_report(report: Dict[str, Any], report_mode: str = "lean") -> Dict[str, Any]:
    errors: List[str] = []

    if not isinstance(report, dict):
        return {
            "schema_valid": False,
            "score_bounds_valid": False,
            "aggregate_matches_components": False,
            "recommendation_matches_score": False,
            "errors": ["report must be a JSON object"],
        }

    score_bounds_valid = True
    aggregate_matches_components = True
    recommendation_matches_score = True

    if report.get("report_version") != REPORT_VERSION:
        errors.append(f"report_version must be {REPORT_VERSION}")

    metadata = report.get("report_metadata")
    if not isinstance(metadata, dict):
        errors.append("report_metadata must be an object")
    else:
        if metadata.get("prompt_version") != PROMPT_VERSION:
            errors.append(f"report_metadata.prompt_version must be {PROMPT_VERSION}")
        if metadata.get("report_mode") not in {"lean", "full"}:
            errors.append("report_metadata.report_mode must be lean|full")

    scorecard = report.get("scorecard")
    if not isinstance(scorecard, dict):
        errors.append("scorecard must be an object")
        scorecard = {}

    for dim in SCORE_DIMENSIONS:
        row = scorecard.get(dim)
        if not isinstance(row, dict):
            errors.append(f"scorecard.{dim} must be an object")
            score_bounds_valid = False
            continue

        raw_score = row.get("score")
        if not isinstance(raw_score, int):
            errors.append(f"scorecard.{dim}.score must be an integer")
            score_bounds_valid = False
        else:
            lower, upper = DIMENSION_SCORE_BOUNDS[dim]
            if not (lower <= raw_score <= upper):
                errors.append(f"scorecard.{dim}.score out of bounds [{lower}, {upper}]")
                score_bounds_valid = False

        conf = str(row.get("confidence") or "").strip().lower()
        if conf not in CONFIDENCE_LEVELS:
            errors.append(f"scorecard.{dim}.confidence must be one of {sorted(CONFIDENCE_LEVELS)}")

        if not _is_non_empty_text(row.get("thesis")):
            errors.append(f"scorecard.{dim}.thesis must be non-empty string")
        if not _is_non_empty_text(row.get("evidence")):
            errors.append(f"scorecard.{dim}.evidence must be non-empty string")

    expected_aggregate = compute_aggregate_score(scorecard if isinstance(scorecard, Mapping) else {})
    if report.get("aggregate_score") != expected_aggregate:
        aggregate_matches_components = False
        errors.append(
            f"aggregate_score mismatch: expected {expected_aggregate}, got {report.get('aggregate_score')!r}"
        )

    reco = report.get("recommendation")
    if reco not in RECOMMENDATIONS:
        recommendation_matches_score = False
        errors.append(f"recommendation must be one of {sorted(RECOMMENDATIONS)}")
    elif isinstance(report.get("aggregate_score"), int):
        expected_reco = recommendation_from_score(int(report["aggregate_score"]))
        if reco != expected_reco:
            recommendation_matches_score = False
            errors.append(
                f"recommendation mismatch: expected {expected_reco}, got {reco}"
            )

    valuation_flag = report.get("valuation_flag")
    if not isinstance(valuation_flag, dict):
        errors.append("valuation_flag must be an object")
    else:
        label = str(valuation_flag.get("label") or "").strip().lower()
        if label not in VALUATION_LABELS:
            errors.append(f"valuation_flag.label must be one of {sorted(VALUATION_LABELS)}")
        conf = str(valuation_flag.get("confidence") or "").strip().lower()
        if conf not in CONFIDENCE_LEVELS:
            errors.append(f"valuation_flag.confidence must be one of {sorted(CONFIDENCE_LEVELS)}")
        if not _is_non_empty_text(valuation_flag.get("summary")):
            errors.append("valuation_flag.summary must be non-empty string")

    risk_flags = report.get("risk_flags")
    if not isinstance(risk_flags, list):
        errors.append("risk_flags must be a list")
    else:
        for i, item in enumerate(risk_flags):
            if not isinstance(item, dict):
                errors.append(f"risk_flags[{i}] must be object")
                continue
            rtype = str(item.get("type") or "").strip()
            sev = str(item.get("severity") or "").strip().lower()
            if rtype not in RISK_TYPES:
                errors.append(f"risk_flags[{i}].type invalid: {rtype!r}")
            if sev not in RISK_SEVERITIES:
                errors.append(f"risk_flags[{i}].severity invalid: {sev!r}")
            if not _is_non_empty_text(item.get("summary")):
                errors.append(f"risk_flags[{i}].summary must be non-empty string")

    mode_value = report_mode
    if not mode_value and isinstance(metadata, dict):
        mode_value = metadata.get("report_mode")
    mode = str(mode_value or "lean").lower()
    if mode == "full":
        if not _is_non_empty_text(report.get("overall_outlook")):
            errors.append("overall_outlook must be non-empty in full mode")
        matrix = report.get("price_target_matrix")
        if not isinstance(matrix, dict):
            errors.append("price_target_matrix must be object with bear/consensus/bull in full mode")
        else:
            # Backward compatibility: accept "base" but prefer "consensus".
            if "consensus" not in matrix and "base" in matrix:
                matrix["consensus"] = matrix["base"]
            for key in ("bear", "consensus", "bull"):
                row = matrix.get(key)
                if not isinstance(row, dict):
                    errors.append(f"price_target_matrix.{key} must be an object")
                    continue
                ptr = row.get("price_target_range")
                if not isinstance(ptr, dict) or "low" not in ptr or "high" not in ptr:
                    errors.append(f"price_target_matrix.{key}.price_target_range must have low/high")

    schema_valid = len(errors) == 0
    return {
        "schema_valid": schema_valid,
        "score_bounds_valid": score_bounds_valid,
        "aggregate_matches_components": aggregate_matches_components,
        "recommendation_matches_score": recommendation_matches_score,
        "errors": errors,
    }


def validate_report(report: Dict[str, Any]) -> List[str]:
    """
    Backward-compatible shim returning only error list.
    """
    return validate_v2_report(report).get("errors", [])

