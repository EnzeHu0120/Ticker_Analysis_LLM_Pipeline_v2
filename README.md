# TickerAnalysis LLM Pipeline (v2)

Two-stage research workflow optimized for performance, cost control, and readability:

1. **Universe Scan / Ranking (default)**  
   Batch lean scorecard generation for many tickers.
2. **Deep Dive / Full Report (on demand)**  
   Extra synthesis only for selected names.

Example reports are written to `outputs/`.  
Example ranking rows are written to `score_rows/`.

---

## Product workflow

### Stage 1 (default): Universe Scan
- Input: ticker list or CSV ticker column
- Output per ticker: lean structured report
- Output batch: sortable ranking table (`score_rows_*.csv/.json`)
- LLM cost model: **one main LLM call per ticker**

### Stage 2 (optional): Full Report
- Triggered only when explicitly requested for selected tickers
- Adds:
  - `overall_outlook`
  - `price_target_matrix` (`bear` / `base` / `bull`)
- LLM cost model: **one extra cheaper synthesis call per selected ticker**

---

## v2 default output schema (lean)

Top-level fields:
- `report_version`
- `report_metadata`
- `scorecard` (exactly 8 dimensions)
- `aggregate_score`
- `recommendation`
- `valuation_flag`
- `risk_flags`
- `validation`

Excluded in lean mode:
- `aggregate_score_pct`
- `sector_rank`
- `deep_dive_priority`
- `catalyst_inputs_debug`

---

## Scorecard design

Exact 8 dimensions:
- `leadership`
- `competitive_advantage`
- `growth_perspective`
- `balance_sheet_health`
- `business_segment_quality`
- `technical_setup`
- `positive_catalysts`
- `negative_catalysts`

Each dimension has:
- `score` (integer, bounded by dimension)
- `confidence` (`low|medium|high`)
- `thesis`
- `evidence`

Score ranges:
- First 6 dimensions: `-5..5`
- `positive_catalysts`: `0..5`
- `negative_catalysts`: `-5..0`

---

## Aggregate and recommendation

v2.0 aggregate:
- `aggregate_score = sum(8 dimension scores)` (equal-weight now; weighting-ready implementation)

Recommendation mapping:
- `>= 18`: `Overweight`
- `8..17`: `Equal-weight`
- `-7..7`: `Hold`
- `-17..-8`: `Underweight`
- `<= -18`: `Reduce`

---

## News handling (cost + quality)

- Default: `ENABLE_NEWS=false`
- If enabled, news is strictly filtered:
  - lookback constrained to `7..14` days
  - deduplicated
  - noisy titles removed
  - max `5` items
- News is optional catalyst evidence only; it does not directly drive structural thesis.

---

## Setup

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Configure `.env` (never commit secrets):

```bash
cp .env.example .env
```

Key env options:
- `LLM_BACKEND=gemini | gemini-vertex | openai`
- `GEMINI_MODEL=...`
- `ENABLE_NEWS=false` (default recommended)
- `REPORT_MODE=lean` (default)

---

## Run

### Single ticker (lean default)

```bash
python llm_pipeline.py AAPL
```

### Single ticker full mode

```bash
python llm_pipeline.py AAPL --report-mode full
```

### Universe scan / ranking

```bash
python universe_runner.py --tickers AAPL,MSFT,NVDA
```

or

```bash
python universe_runner.py --csv your_universe.csv --ticker-col ticker
```

---

## Lightweight UI

```bash
streamlit run streamlit_app.py
```

UI flow:
1. Universe Scan
2. Ranking Table
3. Ticker Detail Snapshot
4. Explicit `Generate Full Report` action

---

## Repo layout (v2)

| Path | Purpose |
|------|--------|
| `fundamental_pipeline.py` | Reused fundamental fetch + engineered metrics |
| `technical_pipeline.py` | Reused technical indicators + regime snapshot |
| `archetype.py` | Reused archetype classification |
| `catalyst_pipeline.py` | Catalyst/event inputs; news optional and filtered |
| `llm_config.py` | Central backend/model config |
| `report_schema.py` | v2 schema constants, score bounds, aggregation helpers |
| `prompt_templates.py` | v2 prompt templates (lean scorecard + full extension) |
| `llm_clients.py` | LLM JSON runners (Gemini/OpenAI) |
| `report_validation.py` | v2 lean/full schema validation |
| `llm_pipeline.py` | v2 single-ticker orchestration entry point |
| `universe_runner.py` | Batch scan/ranking runner + score_rows persistence |
| `streamlit_app.py` | Lightweight ranking/deep-dive UI |
| `outputs/` | Per-ticker reports |
| `score_rows/` | Persistent ranking rows |

---

## Migration notes

- Replaced multi-step 1A/1B/1C/1D+2 default path with compact v2 packet + one main scorecard call.
- Default workflow is now scan/ranking first; full report is opt-in.
- Validation now enforces v2 scorecard bounds, aggregate/recommendation consistency, valuation/risk schemas, and mode-specific checks.
- `fundamental_pipeline.py`, `technical_pipeline.py`, `archetype.py`, and `llm_config.py` are preserved and reused.
