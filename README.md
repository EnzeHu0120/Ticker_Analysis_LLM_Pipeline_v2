# Ticker Analysis LLM Pipeline

A pipeline that pulls fundamental and price data for a given ticker, engineers features, and uses an LLM (Gemini) to produce a structured **fundamental + technical synthesis report** in JSON.

**End goal:** A fully automated workflow to analyze and record a ticker’s fundamentals and technicals, with reports that can be versioned and compared over time. The design is intended to evolve with LLM capabilities (better models, prompts, and tool use) so the pipeline stays current.

---

## What it does

1. **Data** — Fetches from Yahoo Finance:
   - Annual financials: income statement, balance sheet, cash flow (last 5 fiscal years)
   - Quarterly financials (last 5 quarters)
   - Price history for technicals

2. **Features** — Computes:
   - Fundamental metrics: margins, leverage, growth rates, etc.
   - Technical snapshot: RSI, MACD, moving averages, support/resistance, volume

3. **LLM analysis** — Sends the above to Gemini in three parallel prompts, then one synthesis:
   - **1A** — Annual fundamental signals (top 5)
   - **1B** — Quarterly deviation / acceleration / reversal signals
   - **1C** — Technical signals from the computed snapshot
   - **2** — Synthesis: fundamental vs technical stance, combined signals, price target matrix (Bear / Consensus / Bull), risks, catalysts

4. **Output** — Writes a single JSON report to `outputs/{TICKER}_{date}_report.json`.

---

## Setup

- **Python:** 3.10+ (tested on 3.13).

- **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

- **Secrets (never commit):**  
  Copy `.env.example` to `.env` and set your Gemini API key:
  ```bash
  cp .env.example .env
  # Edit .env: set GEMINI_API_KEY=your_key
  ```
  `.env` is in `.gitignore`; only `.env.example` (no real keys) is in the repo.

---

## Run

```bash
python gemini_pipeline.py AAPL
```
Or run without arguments to be prompted for a ticker.

Reports are saved under `outputs/` (also gitignored so local runs don’t pollute the repo).

---

## Repo layout

| Path | Purpose |
|------|--------|
| `data_pipeline.py` | Yahoo Finance fetch, statement merge, anchor dates, annual metrics |
| `gemini_pipeline.py` | Technical snapshot, quarterly fetch, prompts, Gemini client, orchestration |
| `.env.example` | Template for local `.env` (copy to `.env`, add `GEMINI_API_KEY`) |
| `requirements.txt` | Python dependencies |
| `outputs/` | Generated reports (gitignored) |

---

## Keeping it current

- The pipeline is built to **stay updated as LLMs improve**: swap models via `GEMINI_MODEL` (or code), refine prompts, and later add tool use / agents if useful.
- Extend with more data sources, metrics, or prompt stages as needed; the goal is a single, repeatable flow from ticker → stored analysis.
