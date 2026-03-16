from __future__ import annotations

"""
prompt_templates.py

Prompt templates for v2 lean scorecard and optional full report synthesis.
"""

SYSTEM_PROMPT_V2 = (
    "You are a disciplined equity research analyst. "
    "Return valid JSON only, no markdown, no prose outside JSON. "
    "Use only upstream packet evidence. Never claim live web retrieval. "
    "Be concise, evidence-first, and avoid promotional language. "
    "Do not double count structural factors as catalysts."
)


SCORECARD_PROMPT_V2 = """You are scoring {TICKER} using a compact analysis packet.

Scoring objective:
- Produce a strict 8-dimension scorecard for ranking.
- Structural dimensions and event dimensions must stay separate.
- Do not let catalysts repeat growth/balance sheet conclusions.

Dimensions (use these exact keys in JSON; labels for readability):
- leadership (Leadership & Execution): quality and execution credibility of leadership.
- competitive_advantage (Competitive Moat): moat and defensibility.
- growth_perspective (Growth Outlook): structural growth outlook.
- balance_sheet_health (Financial Resilience): current funding and financial resilience.
- business_segment_quality (Segment Quality): segment mix quality and concentration.
- technical_setup (Technical Setup & Liquidity): market structure, trend, risk/reward, and liquidity/trading quality (e.g. volume vs average, breadth). Use technical_regime, technical_snapshot_compact, and technical_summary_text.
- positive_catalysts (Upside Catalysts): 3-12 month event-driven upside triggers only.
- negative_catalysts (Downside Catalysts): 3-12 month event-driven downside triggers only.

Data source discipline:
- leadership, competitive_advantage, business_segment_quality -> mostly company_context (+ archetype)
- growth_perspective -> annual_trend_summary + quarterly_delta_summary
- balance_sheet_health -> structured balance-sheet/funding metrics
- technical_setup -> technical_regime + technical_snapshot_compact + technical_summary_text (include liquidity and volume context)
- positive/negative catalysts -> filtered catalyst/event summary only
- News (if present) is only optional catalyst evidence, not direct thesis driver.

Packet:
{ANALYSIS_PACKET_JSON}

Return JSON with this exact shape:
{{
  "scorecard": {{
    "leadership": {{"score": int[-5..5], "confidence": "low|medium|high", "thesis": string, "evidence": string}},
    "competitive_advantage": {{"score": int[-5..5], "confidence": "low|medium|high", "thesis": string, "evidence": string}},
    "growth_perspective": {{"score": int[-5..5], "confidence": "low|medium|high", "thesis": string, "evidence": string}},
    "balance_sheet_health": {{"score": int[-5..5], "confidence": "low|medium|high", "thesis": string, "evidence": string}},
    "business_segment_quality": {{"score": int[-5..5], "confidence": "low|medium|high", "thesis": string, "evidence": string}},
    "technical_setup": {{"score": int[-5..5], "confidence": "low|medium|high", "thesis": string, "evidence": string}},
    "positive_catalysts": {{"score": int[0..5], "confidence": "low|medium|high", "thesis": string, "evidence": string}},
    "negative_catalysts": {{"score": int[-5..0], "confidence": "low|medium|high", "thesis": string, "evidence": string}}
  }}
}}
"""


FULL_REPORT_PROMPT_V2 = """You are extending an existing lean scorecard report for {TICKER}.

Use only the report and packet below. Do not invent external facts.
Keep it concise.

Lean report:
{LEAN_REPORT_JSON}

Packet (for context):
{ANALYSIS_PACKET_JSON}

Return only:
{{
  "overall_outlook": string,
  "price_target_matrix": {{
    "bear": {{"timeline": string, "price_target_range": {{"low": number, "high": number}}, "key_assumption": string}},
    "consensus": {{"timeline": string, "price_target_range": {{"low": number, "high": number}}, "key_assumption": string}},
    "bull": {{"timeline": string, "price_target_range": {{"low": number, "high": number}}, "key_assumption": string}}
  }}
}}
"""
