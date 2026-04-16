# 🔭 Clairvoyant — Global Sector Intelligence Agent

An agentic AI system that performs sector deep-dives for solo investors. Select a sector, a ranking metric, and a universe of regions — the agent autonomously discovers companies, fetches live financial data across 15 currencies, converts everything to USD for fair comparison, enriches each company with Wikipedia context, and delivers a GPT-written comparison table and investor briefing.

> **Portfolio project** by [Valérie Vienne](https://valerie-vienne.com) — demonstrating agentic tool-calling, multi-source data pipelines, FX normalisation, and financial prompt engineering.

---

## Live Demo

🔗 [Open on Streamlit Cloud](https://clairvoyant-stocks-sector.streamlit.app/)

---

## What makes this different from a generic finance chatbot?

Three things set Clairvoyant apart:

**Data-first, not LLM-first.** The agent discovers and ranks companies from real market data before GPT touches anything. The LLM is used only for writing the analysis — it never invents a ticker, a market cap, or a revenue figure.

**Genuine global coverage.** US, European, Asian, Indian, and Chinese stocks are all in scope, with region-aware ticker handling (`.NS` for NSE India, `.T` for Tokyo, `.HK` for Hong Kong, `.PA/.DE/.L` for Europe).

**Currency-normalised ranking.** A critical bug in most global finance tools is comparing raw local-currency figures — Toyota's ¥55 trillion market cap would rank ahead of Apple's $3 trillion if not converted. Clairvoyant fetches live FX rates and normalises all monetary metrics to USD before ranking.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  PHASE 00 — FX Rates (runs once, cached 1 hour)         │
│                                                         │
│  fetch_fx_rates()                                       │
│  → yfinance forex tickers (JPY=X, INR=X, EURUSD=X...)  │
│  → returns {currency: rate_to_USD} for 15 currencies    │
│  @st.cache_data(ttl=3600) — never fetched twice/hour    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  PHASE 01 — Discovery & Ranking                         │
│                                                         │
│  SECTOR_TICKERS[sector][region]                         │
│  → curated list of tickers per sector × region          │
│       │                                                 │
│       ▼  (for each ticker)                              │
│  fetch_ticker_metrics(ticker, metric_key, fx_rates)     │
│  → yfinance: price, market cap, revenue, P/E, margins   │
│  → to_usd(): converts monetary values using FX rates    │
│  → fiscal year + quarter from lastFiscalYearEnd         │
│       │                                                 │
│       ▼                                                 │
│  rank_companies()                                       │
│  → sorts by USD-normalised metric_value                 │
│  → returns top 5                                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  PHASE 02 — Enrichment                                  │
│                                                         │
│  enrich_company() × 5                                   │
│  → Wikipedia API: company background, founding, context │
│  → fallback to yfinance longBusinessSummary             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  PHASE 03 — Analysis                                    │
│                                                         │
│  generate_analysis(companies, sector, metric, client)   │
│  → GPT-5.4-mini receives USD-normalised data            │
│    + fiscal year labels on every metric                 │
│  → writes comparison table with Fiscal Year column      │
│  → writes 150-200 word investor briefing                │
│  → assigns HIGH/MEDIUM/LOW confidence per company       │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
clairvoyant/
├── clairvoyant_app.py     # main Streamlit app — all logic here
├── requirements.txt       # dependencies
├── .gitignore             # keeps secrets.toml off GitHub
├── .streamlit/
│   ├── secrets.toml       # API key — never commit this
│   └── config.toml        # Observatory dark theme config
└── README.md
```

---

## Key Components

### `SECTOR_TICKERS` — the company universe
A curated dictionary mapping each sector to a list of tickers per region. This is the source of truth for company discovery — the agent scans these tickers and ranks them by real data, rather than asking the LLM to guess which companies are biggest.

```python
SECTOR_TICKERS = {
    "technology": {
        "US":    ["AAPL", "MSFT", "NVDA", ...],
        "Europe":["ASML.AS", "SAP.DE", ...],
        "Asia":  ["9984.T", "005930.KS", ...],
        "India": ["TCS.NS", "INFY.NS", ...],
        "China": ["BIDU", "BABA", ...],
    },
    ...  # 8 sectors total
}
```

Covers 8 sectors: Technology, Energy, Finance, Healthcare, Consumer, Industrials, Automotive, E-commerce.

### `fetch_fx_rates()` — currency normalisation
Fetches live exchange rates for 15 currencies from Yahoo Finance using forex tickers (`JPY=X`, `INR=X`, `EURUSD=X` etc.). Cached for 1 hour via `@st.cache_data` so it doesn't re-fetch on every user interaction.

```python
@st.cache_data(ttl=3600)
def fetch_fx_rates() -> dict[str, float]:
    # Returns e.g. {"JPY": 0.00667, "INR": 0.01198, "EUR": 1.085, ...}
```

Two quoting conventions are handled separately:
- **Standard** (`JPY=X` → 149.5): means 1 USD = ¥149.5 → multiply by `1/149.5` to get USD
- **Inverted** (`EURUSD=X` → 1.085): means 1 EUR = $1.085 → multiply directly

> **Why this matters:** Without FX normalisation, Toyota's ¥55,000,000,000,000 market cap would numerically rank ahead of Apple's $3,000,000,000,000 — even though Toyota is roughly 10x smaller in USD terms. This is a real, common bug in global finance tools.

### `to_usd()` — per-value conversion
Simple multiplier applied to every monetary field (market cap, revenue) before ranking. Ratios like P/E, revenue growth, and profit margin are currency-neutral and pass through unchanged.

```python
def to_usd(value, currency, fx_rates) -> float | None:
    rate = fx_rates.get(currency.upper())
    return value * rate if rate else None
```

### `fetch_ticker_metrics()` — data fetcher
Calls yfinance for a single ticker and returns both raw local-currency values (for display) and USD-normalised values (for ranking). Also extracts fiscal year dates from Unix timestamps:

```python
fiscal_year   = datetime.fromtimestamp(info["lastFiscalYearEnd"]).strftime("FY %Y")
most_recent_q = f"Q{quarter} {year}"   # e.g. "Q3 2024"
```

Every metric in the output carries its time period label so the analysis is never ambiguous about which year it refers to.

### `generate_analysis()` — the only place GPT runs
Receives clean, USD-normalised, date-labelled data and writes two outputs:
- A markdown comparison table with a Fiscal Year column and confidence scores (HIGH/MEDIUM/LOW based on data completeness)
- A 150-200 word conversational briefing that always cites the fiscal period for each figure

`temperature=0.4` is intentionally low — financial analysis benefits from consistency over creativity.

---

## FX Coverage

| Currency | Ticker used | Region |
|---|---|---|
| JPY | `JPY=X` | Japan |
| INR | `INR=X` | India |
| HKD | `HKD=X` | Hong Kong |
| KRW | `KRW=X` | South Korea |
| TWD | `TWD=X` | Taiwan |
| CNY | `CNY=X` | China |
| EUR | `EURUSD=X` | Eurozone |
| GBP | `GBPUSD=X` | UK |
| CHF | `CHFUSD=X` | Switzerland |
| SEK | `SEK=X` | Sweden |
| NOK | `NOK=X` | Norway |
| AUD | `AUDUSD=X` | Australia |
| CAD | `CAD=X` | Canada |
| SGD | `SGD=X` | Singapore |
| BRL | `BRL=X` | Brazil |

---

## Sectors & Ranking Metrics

**Sectors:** Technology · Energy · Finance · Healthcare · Consumer · Industrials · Automotive · E-commerce

**Ranking metrics:**
- Market Cap — total company value, USD-normalised
- Revenue — annual revenue, USD-normalised from most recent fiscal year
- P/E Ratio — trailing price-to-earnings (ratio, no FX conversion needed)
- Revenue Growth — YoY revenue growth percentage
- Profit Margin — net profit as % of revenue

---

## Cost Model

All costs as of April 2026:

| Operation | Model | Price | Per analysis |
|---|---|---|---|
| FX rates fetch | yfinance (free) | $0 | $0 |
| Ticker metrics × ~30 | yfinance (free) | $0 | $0 |
| Wikipedia × 5 | Wikipedia API (free) | $0 | $0 |
| GPT analysis | gpt-5.4-mini input | $0.75/1M | ~$0.002 |
| GPT analysis | gpt-5.4-mini output | $4.50/1M | ~$0.009 |
| **Total per deep-dive** | | | **~$0.011** |

The analysis uses `max_tokens=2000` to allow for the full comparison table. FX rates are cached for 1 hour — repeated analyses within the same hour cost nothing on the data side.

### Cost calculation in code
```python
input_tokens  = response.usage.prompt_tokens
output_tokens = response.usage.completion_tokens
cost = (
    (input_tokens  / 1_000_000) * 0.75 +
    (output_tokens / 1_000_000) * 4.50
)
```

---

## Observatory Theme

The UI is built with a custom Observatory/Cosmic aesthetic injected via `st.markdown(unsafe_allow_html=True)`:

- **Deep space background** with indigo/violet nebula gradients and a 10-point star field
- **Fonts:** Syne (headings) + Space Mono (data labels, tool calls, metrics) from Google Fonts
- **Phase panels** with left accent borders showing the agent's live pipeline progress
- **Terminal-style tool call boxes** with colour-coded syntax (purple = function, amber = argument, green = success)
- **Rank cards** with gradient top borders and clickable expanders showing full metrics + Wikipedia summary
- **Streamlit theme** set via `.streamlit/config.toml` for native components (dropdowns, progress bars, top ribbon)

`.streamlit/config.toml`:
```toml
[theme]
base = "dark"
backgroundColor = "#04050f"
secondaryBackgroundColor = "#0d1230"
primaryColor = "#22d3ee"
textColor = "#e2e8f0"
```

---

## Data & Privacy

| Data | Where stored | Lifetime |
|---|---|---|
| FX rates | `@st.cache_data` in RAM | 1 hour TTL |
| Ticker metrics | Python list in RAM | Session only |
| Wikipedia summaries | Python dict in RAM | Session only |
| GPT analysis result | `st.session_state` | Session only |
| API key | Streamlit secrets / `.streamlit/secrets.toml` | Never in code |

Nothing is persisted to disk or any external database. All data resets when the app restarts or the session ends.

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/ValerieVienne/clairvoyant.git
cd clairvoyant

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
mkdir .streamlit
echo 'OPENAI_API_KEY = "sk-..."' > .streamlit/secrets.toml

# 5. Add the theme config
cat > .streamlit/config.toml << EOF
[theme]
base = "dark"
backgroundColor = "#04050f"
secondaryBackgroundColor = "#0d1230"
primaryColor = "#22d3ee"
textColor = "#e2e8f0"
EOF

# 6. Run
streamlit run clairvoyant_app.py
```

App opens at `http://localhost:8501`. Start with **Technology · Market Cap · All regions** for the richest demo.

---

## Deployment (Streamlit Cloud)

1. Push to a public GitHub repo — make sure `.streamlit/secrets.toml` is in `.gitignore` but `.streamlit/config.toml` is committed
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo, branch `master`, main file `clairvoyant_app.py`
4. Advanced settings → paste your secret: `OPENAI_API_KEY = "sk-..."`
5. Deploy → live URL in ~2 minutes

---

## Adapting for a Real Client

To turn this into a production tool for a fintech, hedge fund, or investment platform:

1. **Expand `SECTOR_TICKERS`** with more tickers per region or add new sectors (e.g. "real estate", "crypto", "commodities")
2. **Replace the curated ticker list** with a dynamic screener API (Finnhub, FMP) for truly live discovery
3. **Add persistent storage** (Supabase, PostgreSQL) to save analyses and compare them over time
4. **Increase `TOP_N`** from 5 to 10 or 20 for broader coverage
5. **Add more ranking metrics** — EV/EBITDA, dividend yield, debt-to-equity
6. **Add a PDF export** button using `reportlab` or `weasyprint` to deliver institutional-style reports

---

## Tech Stack

| Tool | Role |
|---|---|
| Streamlit | UI framework & deployment |
| OpenAI Python SDK | GPT-5.4-mini for analysis writing |
| yfinance | Stock metrics + live FX rates |
| AKShare | Chinese & Asian market data (available, reserved for extended coverage) |
| Wikipedia API | Company background enrichment |
| Pandas | Data handling |
| NumPy | Numerical operations |
| GPT-5.4-mini | Comparison table + investor briefing (March 2026) |

---

## Author

**Valérie Vienne** — AI Automation Engineer  
[valerie-vienne.com](https://valerie-vienne.com) · [GitHub](https://github.com/ValerieVienne)  
Available for freelance on [Upwork](https://upwork.com)

---

## Credits

This project was co-created with [Claude AI](https://claude.ai) (Anthropic) — used for architecture design, code generation, prompt engineering, FX normalisation logic, and documentation.
