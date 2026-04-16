import streamlit as st
import yfinance as yf
import akshare as ak
import wikipediaapi
import pandas as pd
import numpy as np
from openai import OpenAI
import json
import time

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
MODEL          = "gpt-5.4-mini"
MAX_TOKENS     = 2000
TOP_N          = 5

INPUT_PRICE_PER_M  = 0.75
OUTPUT_PRICE_PER_M = 4.50

# ─────────────────────────────────────────
#  SECTOR → TICKERS MAPPING
#  Curated global list per sector
#  yfinance tickers: plain = US, .NS = India NSE,
#  .HK = Hong Kong, .PA/.AS/.DE = Europe
# ─────────────────────────────────────────
SECTOR_TICKERS = {
    "technology": {
        "US":     ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSM", "AVGO", "ORCL", "CRM"],
        "Europe": ["ASML.AS", "SAP.DE", "STM.PA", "CAP.PA", "DASSF"],
        "Asia":   ["9984.T", "2330.TW", "005930.KS", "700.HK", "9618.HK"],
        "India":  ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
        "China":  ["BIDU", "BABA", "JD", "PDD", "NTES"],
    },
    "energy": {
        "US":     ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "PSX", "VLO", "OXY"],
        "Europe": ["TTE.PA", "BP.L", "SHEL.L", "ENI.MI", "EQNR.OL"],
        "Asia":   ["0883.HK", "1088.HK", "5020.T", "010950.KS"],
        "India":  ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "NTPC.NS"],
        "China":  ["SNP", "PTR", "CEO"],
    },
    "finance": {
        "US":     ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "USB", "PNC"],
        "Europe": ["HSBA.L", "BNP.PA", "SAN.MC", "DBK.DE", "ACA.PA"],
        "Asia":   ["8306.T", "8316.T", "105560.KS", "0939.HK", "1398.HK"],
        "India":  ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
        "China":  ["CICHY", "BACHF", "IDCBY"],
    },
    "healthcare": {
        "US":     ["JNJ", "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "PFE", "DHR", "BMY"],
        "Europe": ["NOVN.SW", "ROG.SW", "SAN.PA", "AZN.L", "GSK.L"],
        "Asia":   ["4519.T", "4568.T", "128940.KS", "1177.HK"],
        "India":  ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
        "China":  ["ZLAB", "BeiGene", "HTHT"],
    },
    "consumer": {
        "US":     ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "COST", "TJX"],
        "Europe": ["MC.PA", "OR.PA", "CDI.PA", "ITX.MC", "RMS.PA"],
        "Asia":   ["7203.T", "7267.T", "005380.KS", "9988.HK"],
        "India":  ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "TITAN.NS", "MARUTI.NS"],
        "China":  ["NIO", "LI", "XPEV", "9618.HK"],
    },
    "industrials": {
        "US":     ["GE", "HON", "CAT", "BA", "RTX", "DE", "UPS", "LMT", "MMM", "EMR"],
        "Europe": ["SIE.DE", "AIR.PA", "ABB.ST", "VLVLY", "ATLKY"],
        "Asia":   ["6301.T", "6326.T", "7011.T", "042660.KS"],
        "India":  ["LT.NS", "SIEMENS.NS", "ABB.NS", "BHEL.NS", "ADANIPORTS.NS"],
        "China":  ["CCCGY", "601766.SS"],
    },
    "automotive": {
        "US":     ["TSLA", "GM", "F", "RIVN", "LCID"],
        "Europe": ["VOW3.DE", "BMW.DE", "MBG.DE", "STLA.PA", "RNO.PA"],
        "Asia":   ["7203.T", "7267.T", "7201.T", "005380.KS", "000270.KS"],
        "India":  ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
        "China":  ["NIO", "LI", "XPEV", "BYD"],
    },
    "ecommerce": {
        "US":     ["AMZN", "SHOP", "EBAY", "ETSY", "W"],
        "Europe": ["ZALANDO.DE", "ASOS.L"],
        "Asia":   ["9618.HK", "9988.HK"],
        "India":  ["NYKAA.NS", "ZOMATO.NS", "PAYTM.NS"],
        "China":  ["BABA", "JD", "PDD", "VIPS"],
    },
}

RANKING_METRICS = {
    "Market Cap":      "marketCap",
    "Revenue":         "totalRevenue",
    "P/E Ratio":       "trailingPE",
    "Revenue Growth":  "revenueGrowth",
    "Profit Margin":   "profitMargins",
}

# ─────────────────────────────────────────
#  DATA FETCHING FUNCTIONS
# ─────────────────────────────────────────
def get_all_tickers_for_sector(sector_key: str) -> list[str]:
    """Return all tickers across all regions for a sector."""
    mapping = SECTOR_TICKERS.get(sector_key, {})
    all_tickers = []
    for region_tickers in mapping.values():
        all_tickers.extend(region_tickers)
    return all_tickers


# ── FX conversion layer ───────────────────────────────────────────────────────
# Maps ISO currency code → yfinance forex ticker (all = units per 1 USD)
CURRENCY_TO_FX_TICKER = {
    "JPY": "JPY=X",    # ¥ per $1
    "INR": "INR=X",    # ₹ per $1
    "HKD": "HKD=X",    # HK$ per $1
    "KRW": "KRW=X",    # ₩ per $1
    "TWD": "TWD=X",    # NT$ per $1
    "CNY": "CNY=X",    # ¥ per $1
    "EUR": "EURUSD=X", # $ per €1  ← inverted, handled below
    "GBP": "GBPUSD=X", # $ per £1  ← inverted
    "CHF": "CHFUSD=X", # $ per CHF ← inverted
    "SEK": "SEK=X",
    "NOK": "NOK=X",
    "AUD": "AUDUSD=X", # inverted
    "CAD": "CAD=X",
    "SGD": "SGD=X",
    "BRL": "BRL=X",
    "MXN": "MXN=X",
}
# Currencies quoted as "USD per 1 unit" (inverted — multiply instead of divide)
INVERTED_CURRENCIES = {"EUR", "GBP", "CHF", "AUD"}


@st.cache_data(ttl=3600)  # cache FX rates for 1 hour
def fetch_fx_rates() -> dict[str, float]:
    """
    Fetch live FX rates for all relevant currencies via yfinance.
    Returns dict: currency_code → rate_to_multiply_to_get_USD
    e.g. {"JPY": 0.00667, "INR": 0.01198, "EUR": 1.085, ...}
    USD itself is always 1.0.
    """
    rates = {"USD": 1.0}
    for currency, fx_ticker in CURRENCY_TO_FX_TICKER.items():
        try:
            data = yf.Ticker(fx_ticker).fast_info
            price = data.last_price
            if price and price > 0:
                if currency in INVERTED_CURRENCIES:
                    # e.g. EURUSD=X returns 1.085 meaning 1 EUR = $1.085
                    rates[currency] = price
                else:
                    # e.g. JPY=X returns 149.5 meaning 1 USD = ¥149.5
                    # so 1 JPY = 1/149.5 USD
                    rates[currency] = 1.0 / price
        except Exception:
            pass
    return rates


def to_usd(value: float | None, currency: str, fx_rates: dict) -> float | None:
    """Convert a monetary value from its local currency to USD."""
    if value is None:
        return None
    rate = fx_rates.get(currency.upper(), None)
    if rate is None:
        return None  # unknown currency — exclude from ranking
    return value * rate


def detect_region(ticker: str) -> str:
    """Detect region from ticker suffix."""
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "🇮🇳 India"
    elif ticker.endswith(".HK"):
        return "🇭🇰 Hong Kong"
    elif any(ticker.endswith(s) for s in [".PA", ".AS", ".DE", ".L", ".SW", ".MC", ".MI", ".ST", ".OL"]):
        return "🇪🇺 Europe"
    elif any(ticker.endswith(s) for s in [".T", ".KS", ".TW", ".SS"]):
        return "🇯🇵🇰🇷 Asia"
    else:
        return "🇺🇸 US / Global"


def fetch_ticker_metrics(ticker: str, metric_key: str, fx_rates: dict) -> dict:
    """Fetch key metrics for a single ticker via yfinance, converting to USD."""
    try:
        import datetime
        t = yf.Ticker(ticker)
        info = t.info

        # ── Date helpers ──────────────────────────────────────────────
        def ts_to_year(ts):
            if ts:
                try:
                    return datetime.datetime.fromtimestamp(ts).strftime("FY %Y")
                except Exception:
                    pass
            return "N/A"

        def ts_to_quarter(ts):
            if ts:
                try:
                    d = datetime.datetime.fromtimestamp(ts)
                    q = (d.month - 1) // 3 + 1
                    return f"Q{q} {d.year}"
                except Exception:
                    pass
            return "N/A"

        fiscal_year   = ts_to_year(info.get("lastFiscalYearEnd"))
        most_recent_q = ts_to_quarter(info.get("mostRecentQuarter"))
        data_as_of    = fiscal_year if fiscal_year != "N/A" else most_recent_q
        currency      = info.get("currency", "USD")

        # ── Raw local-currency values ─────────────────────────────────
        raw_market_cap = info.get("marketCap")
        raw_revenue    = info.get("totalRevenue")
        raw_metric     = info.get(metric_key)

        # ── USD-converted values for ranking ─────────────────────────
        # P/E, revenueGrowth, profitMargins are ratios — no conversion needed
        monetary_metrics = {"marketCap", "totalRevenue"}
        if metric_key in monetary_metrics:
            metric_usd = to_usd(raw_metric, currency, fx_rates)
        else:
            metric_usd = raw_metric  # ratios/percentages are currency-neutral

        fx_rate = fx_rates.get(currency.upper(), 1.0)

        return {
            "ticker":          ticker,
            "name":            info.get("longName") or info.get("shortName") or ticker,
            "region":          detect_region(ticker),
            "sector":          info.get("sector", "N/A"),
            "country":         info.get("country", "N/A"),
            "currency":        currency,
            "fx_rate_to_usd":  fx_rate,
            # Raw local values (for display with currency label)
            "marketCap":       raw_market_cap,
            "totalRevenue":    raw_revenue,
            # USD-normalised values (for display in comparison)
            "marketCap_usd":   to_usd(raw_market_cap, currency, fx_rates),
            "totalRevenue_usd": to_usd(raw_revenue, currency, fx_rates),
            # Ratios — no conversion needed
            "trailingPE":      info.get("trailingPE"),
            "revenueGrowth":   info.get("revenueGrowth"),
            "profitMargins":   info.get("profitMargins"),
            "52wHigh":         info.get("fiftyTwoWeekHigh"),
            "52wLow":          info.get("fiftyTwoWeekLow"),
            "currentPrice":    info.get("currentPrice") or info.get("regularMarketPrice"),
            "employees":       info.get("fullTimeEmployees"),
            "summary":         info.get("longBusinessSummary", "")[:300],
            "metric_value":    metric_usd,   # always USD-normalised for ranking
            "fiscal_year":     fiscal_year,
            "most_recent_q":   most_recent_q,
            "data_as_of":      data_as_of,
        }
    except Exception as e:
        return {"ticker": ticker, "name": ticker, "error": str(e), "metric_value": None}


def get_wikipedia_summary(company_name: str) -> str:
    """Fetch a short Wikipedia summary for a company."""
    try:
        wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent="Clairvoyant/1.0 (portfolio demo)"
        )
        # Try exact name first, then cleaned version
        for name in [company_name, company_name.split(" Inc")[0].split(" Corp")[0].strip()]:
            page = wiki.page(name)
            if page.exists():
                return page.summary[:400]
        return ""
    except Exception:
        return ""


def get_akshare_china_info(ticker: str) -> dict:
    """Try to get additional info for Chinese stocks via akshare."""
    try:
        # For US-listed Chinese ADRs, akshare may not help much
        # But for HK stocks we can try
        if ticker.endswith(".HK"):
            code = ticker.replace(".HK", "")
            return {"source": "akshare_hk", "code": code}
        return {}
    except Exception:
        return {}


# ─────────────────────────────────────────
#  AGENT LOGIC
# ─────────────────────────────────────────
def rank_companies(tickers: list[str], metric_key: str,
                   status_container, progress_bar,
                   fx_rates: dict) -> list[dict]:
    """
    Fetch metrics for all tickers, convert to USD, and rank by chosen metric.
    Shows live progress in the UI.
    """
    results = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        status_container.markdown(
            f'<div class="tool-call">'
            f'<span class="fn">get_metrics</span>('
            f'<span class="arg">"{ticker}"</span>, '
            f'<span class="arg">"{metric_key}"</span>'
            f') &nbsp;·&nbsp; fetching + converting to USD...</div>',
            unsafe_allow_html=True
        )
        progress_bar.progress((i + 1) / total)

        data = fetch_ticker_metrics(ticker, metric_key, fx_rates)
        if data.get("metric_value") is not None:
            results.append(data)
        time.sleep(0.1)

    results.sort(key=lambda x: x.get("metric_value") or 0, reverse=True)
    return results[:TOP_N]


def enrich_company(company: dict, status_container) -> dict:
    """Fetch Wikipedia summary for a company."""
    name = company.get("name", company["ticker"])
    status_container.markdown(
        f'<div class="tool-call">'
        f'<span class="fn">get_wikipedia</span>('
        f'<span class="arg">"{name}"</span>'
        f') &nbsp;·&nbsp; fetching background...</div>',
        unsafe_allow_html=True
    )
    wiki_summary = get_wikipedia_summary(name)
    company["wiki_summary"] = wiki_summary
    time.sleep(0.2)
    return company


def generate_analysis(companies: list[dict], sector: str,
                       metric_label: str, client: OpenAI) -> tuple[str, str, float]:
    """
    Ask GPT to write the comparison table and conversational briefing.
    Returns (table_markdown, briefing, cost)
    """
    # Prepare clean data for GPT — no raw Python objects
    companies_clean = []
    for c in companies:
        companies_clean.append({
            "name":           c.get("name", c["ticker"]),
            "ticker":         c["ticker"],
            "region":         c.get("region", "N/A"),
            "country":        c.get("country", "N/A"),
            "local_currency": c.get("currency", "USD"),
            "fiscal_year":    c.get("fiscal_year", "N/A"),
            "most_recent_q":  c.get("most_recent_q", "N/A"),
            # All monetary values in USD for fair comparison
            "marketCap_usd":      f"${c['marketCap_usd']:,.0f} USD" if c.get("marketCap_usd") else "N/A",
            "revenue_usd":        f"${c['totalRevenue_usd']:,.0f} USD ({c.get('fiscal_year','N/A')})" if c.get("totalRevenue_usd") else "N/A",
            "pe_ratio":           f"{round(c['trailingPE'], 2)}x (trailing, as of {c.get('most_recent_q','N/A')})" if c.get("trailingPE") else "N/A",
            "revenue_growth":     f"{c['revenueGrowth']*100:.1f}% ({c.get('fiscal_year','N/A')})" if c.get("revenueGrowth") else "N/A",
            "profit_margin":      f"{c['profitMargins']*100:.1f}% ({c.get('fiscal_year','N/A')})" if c.get("profitMargins") else "N/A",
            "52w_range_local":    f"{c.get('52wLow','?')} – {c.get('52wHigh','?')} {c.get('currency','')}",
            "current_price":      f"{c.get('currentPrice','N/A')} {c.get('currency','')}",
            "employees":          f"{c['employees']:,}" if c.get("employees") else "N/A",
            "wiki_summary":       c.get("wiki_summary", "")[:200],
            "yfinance_summary":   c.get("summary", "")[:200],
        })

    prompt = f"""You are a financial analyst assistant helping a solo investor research the {sector} sector.

Here is real data for the top {TOP_N} companies ranked by {metric_label}.
Each metric includes the fiscal year or quarter it refers to in parentheses — always mention these dates in your analysis.

{json.dumps(companies_clean, indent=2)}

Your task has TWO parts:

PART 1 — COMPARISON TABLE
Write a clean markdown table comparing all {TOP_N} companies with these columns:
| Company | Region | Fiscal Year | Market Cap | Revenue | P/E | Revenue Growth | Profit Margin | Confidence |

- Include the fiscal year in the Fiscal Year column (e.g. FY 2024)
- For the Confidence column, assign HIGH / MEDIUM / LOW based on:
  HIGH: all key metrics available, well-known company
  MEDIUM: some metrics missing or company less covered
  LOW: multiple metrics missing or data sparse

PART 2 — INVESTOR BRIEFING
Write a 150-200 word conversational briefing as if you're a knowledgeable friend explaining this sector over coffee.
- Always mention which fiscal year or period the data refers to (e.g. "based on FY 2024 results...")
- What stands out in the data?
- Which company looks most interesting and why?
- Any red flags or things to watch?
- Keep it warm, direct, and jargon-light.
- Start with "Here's what caught my eye in the {sector} sector..."

Format your response EXACTLY as:
## Comparison Table
[table here]

## Investor Briefing
[briefing here]

IMPORTANT: Base your analysis ONLY on the data provided. Do not invent figures. Always cite the time period for each metric you mention."""

    response = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=MAX_TOKENS,
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content

    # Calculate cost
    input_tokens  = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = (
        (input_tokens  / 1_000_000) * INPUT_PRICE_PER_M +
        (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_M
    )

    # Split table and briefing
    parts = content.split("## Investor Briefing")
    table    = parts[0].replace("## Comparison Table", "").strip()
    briefing = parts[1].strip() if len(parts) > 1 else content

    return table, briefing, cost


# ─────────────────────────────────────────
#  PAGE SETUP — OBSERVATORY THEME
# ─────────────────────────────────────────
st.set_page_config(page_title="Clairvoyant", page_icon="🔭", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;800&display=swap');

/* ── Root palette ── */
:root {
    --void:      #04050f;
    --deep:      #080c1e;
    --panel:     #0d1230;
    --surface:   #111830;
    --rim:       #1e2a50;
    --indigo:    #3d52d5;
    --violet:    #7c3aed;
    --nebula:    #a855f7;
    --cyan:      #22d3ee;
    --gold:      #f59e0b;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --dim:       #334155;
}

/* ── Global background ── */
.stApp {
    background: var(--void);
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(61,82,213,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(124,58,237,0.14) 0%, transparent 55%),
        radial-gradient(ellipse 40% 30% at 50% 50%, rgba(34,211,238,0.04) 0%, transparent 70%);
    font-family: 'Syne', sans-serif;
}

/* ── Star field ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(1px 1px at 15% 20%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 35% 55%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 55% 15%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 72% 72%, rgba(255,255,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 88% 35%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 8%  80%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 92% 88%, rgba(255,255,255,0.3) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 45% 90%, rgba(168,85,247,0.7) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 25% 40%, rgba(34,211,238,0.6) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 78% 5%,  rgba(245,158,11,0.5) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* ── Main content wrapper ── */
.block-container {
    max-width: 1200px !important;
    padding: 2rem 2rem 4rem !important;
    position: relative;
    z-index: 1;
}

/* ── Typography ── */
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}
p, li, span, div, label {
    font-family: 'Syne', sans-serif;
    color: var(--text);
}

/* ── Header hero ── */
.clairvoyant-hero {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--rim);
    margin-bottom: 2rem;
}
.hero-lens {
    width: 64px; height: 64px;
    border-radius: 50%;
    background: conic-gradient(from 180deg, #3d52d5, #7c3aed, #22d3ee, #3d52d5);
    display: flex; align-items: center; justify-content: center;
    font-size: 28px;
    box-shadow: 0 0 32px rgba(124,58,237,0.5), 0 0 64px rgba(61,82,213,0.2);
    animation: pulse-lens 4s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes pulse-lens {
    0%,100% { box-shadow: 0 0 32px rgba(124,58,237,0.5), 0 0 64px rgba(61,82,213,0.2); }
    50%      { box-shadow: 0 0 48px rgba(168,85,247,0.7), 0 0 96px rgba(61,82,213,0.3); }
}
.hero-text h1 {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #e2e8f0 0%, #a855f7 50%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 !important; padding: 0 !important;
    line-height: 1.1;
}
.hero-text p {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 6px 0 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--deep) !important;
    border-right: 1px solid var(--rim) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
}
.sidebar-brand {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--nebula);
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--rim);
}
.sidebar-section {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.2rem 0 0.5rem;
}

/* ── Selectbox & multiselect ── */
[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--rim) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted) !important;
}
/* Selected value and dropdown option text — force bright readable colour */
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-baseweb="popover"] li,
[data-baseweb="popover"] span,
[data-testid="stMultiSelect"] span {
    color: #e2e8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
}
/* The actual selected value text inside the box */
[data-baseweb="select"] [data-testid="stWidgetLabel"] + div span,
[data-baseweb="select"] div[class*="placeholder"],
[data-baseweb="select"] div[class*="singleValue"],
[data-baseweb="select"] div[class*="value"],
div[data-baseweb="select"] > div > div > div {
    color: #e2e8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
}
/* Dropdown arrow icon */
[data-testid="stSelectbox"] svg { fill: #22d3ee !important; }
/* Multiselect tags */
[data-baseweb="tag"] {
    background: rgba(61,82,213,0.35) !important;
    border: 1px solid var(--indigo) !important;
}
[data-baseweb="tag"] span { color: #e2e8f0 !important; }

/* ── Primary button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #3d52d5, #7c3aed) !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: white !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 0 20px rgba(124,58,237,0.35);
    transition: all 0.2s ease;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 0 32px rgba(124,58,237,0.6);
    transform: translateY(-1px);
}

/* ── Phase panels ── */
.phase-panel {
    background: var(--panel);
    border: 1px solid var(--rim);
    border-left: 3px solid var(--indigo);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--cyan);
}
.phase-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.phase-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text);
}

/* ── Tool call log ── */
.tool-call {
    background: var(--void);
    border: 1px solid var(--rim);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--cyan);
    margin: 0.4rem 0;
}
.tool-call .fn   { color: var(--nebula); }
.tool-call .arg  { color: var(--gold); }
.tool-call .ok   { color: #4ade80; }

/* ── Company rank cards ── */
.rank-card {
    background: var(--panel);
    border: 1px solid var(--rim);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    transition: border-color 0.2s;
    position: relative;
    overflow: hidden;
}
.rank-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--indigo), var(--violet), var(--cyan));
}
.rank-card:hover { border-color: var(--indigo); }
.rank-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.rank-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text);
    margin: 4px 0 2px;
    line-height: 1.3;
}
.rank-ticker {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--cyan);
}
.rank-region {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 4px;
}

/* ── Result section ── */
.result-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0.5rem 0;
    border-top: 1px solid var(--rim);
    border-bottom: 1px solid var(--rim);
    margin: 2rem 0 1rem;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom-color: var(--cyan) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--panel) !important;
    border: 1px solid var(--rim) !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    color: var(--cyan) !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--indigo), var(--violet), var(--cyan)) !important;
}

/* ── Divider ── */
hr { border-color: var(--rim) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--rim) !important;
    border-radius: 8px !important;
    overflow: hidden;
}

/* ── Landing feature cards ── */
.feature-card {
    background: var(--panel);
    border: 1px solid var(--rim);
    border-radius: 10px;
    padding: 1.5rem;
    height: 100%;
}
.feature-icon {
    font-size: 1.5rem;
    margin-bottom: 0.75rem;
}
.feature-title {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    color: var(--text);
    margin-bottom: 0.4rem;
}
.feature-desc {
    font-size: 0.85rem;
    color: var(--muted);
    line-height: 1.6;
}

/* ── Expander (company detail panels) ── */
[data-testid="stExpander"] {
    background: var(--panel) !important;
    border: 1px solid var(--rim) !important;
    border-radius: 8px !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
    color: #22d3ee !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stExpander"] summary:hover {
    color: #a855f7 !important;
}
[data-testid="stExpander"] > div {
    padding: 0.5rem 1rem 1rem !important;
}
.briefing-box {
    background: var(--panel);
    border: 1px solid var(--rim);
    border-left: 3px solid var(--nebula);
    border-radius: 0 10px 10px 0;
    padding: 1.5rem 1.75rem;
    font-size: 0.95rem;
    line-height: 1.8;
    color: var(--text);
}

/* ── Status badge ── */
.status-ok {
    display: inline-block;
    background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.3);
    color: #4ade80;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 10px;
    border-radius: 99px;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

# ── Hero header ──────────────────────────
st.markdown("""
<div class="clairvoyant-hero">
    <div class="hero-lens">🔭</div>
    <div class="hero-text">
        <h1>CLAIRVOYANT</h1>
        <p>Global Sector Intelligence · US · Europe · Asia · India · China · Real data + GPT-5.4-mini</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
for key, val in [("results", None), ("total_cost", 0.0), ("run_count", 0)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────
#  SIDEBAR — OBSERVATORY THEME
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🔭 &nbsp; Clairvoyant &nbsp; v1.0</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Target Sector</div>', unsafe_allow_html=True)
    sector_label = st.selectbox(
        "Sector",
        options=list(SECTOR_TICKERS.keys()),
        format_func=lambda x: x.upper(),
        label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-section">Ranking Metric</div>', unsafe_allow_html=True)
    metric_label = st.selectbox(
        "Rank companies by",
        options=list(RANKING_METRICS.keys()),
        label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-section">Universe</div>', unsafe_allow_html=True)
    regions = st.multiselect(
        "Include regions",
        options=["US", "Europe", "Asia", "India", "China"],
        default=["US", "Europe", "Asia", "India", "China"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-section">Agent Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#64748b;line-height:2;">
▸ discover_companies()<br>
▸ rank_by_metric()<br>
▸ get_wikipedia()<br>
▸ generate_analysis()
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Runs", st.session_state.run_count)
    with col_b:
        st.metric("Cost", f"${st.session_state.total_cost:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔭  INITIATE SCAN", type="primary", use_container_width=True)

# ─────────────────────────────────────────
#  MAIN — RUN AGENT
# ─────────────────────────────────────────
if run_btn:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    sector_data = SECTOR_TICKERS.get(sector_label, {})
    tickers = []
    for region in regions:
        tickers.extend(sector_data.get(region, []))
    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        st.error("No tickers found for this sector + region combination.")
        st.stop()

    metric_key = RANKING_METRICS[metric_label]

    # ── Scan header ──────────────────────
    st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.12em;
            text-transform:uppercase;color:#64748b;margin:1.5rem 0 0.5rem;">
    ◈ &nbsp; initiating deep-dive
</div>
<div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:700;
            color:#e2e8f0;margin-bottom:0.25rem;">
    {sector_label.upper()} SECTOR
</div>
<div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#64748b;">
    Scanning {len(tickers)} companies &nbsp;·&nbsp; Ranking by {metric_label} &nbsp;·&nbsp; Regions: {' · '.join(regions)}
</div>
<hr style="margin:1rem 0;">
""", unsafe_allow_html=True)

    # ── PHASE 0: Fetch live FX rates ─────
    st.markdown("""
<div class="phase-panel">
    <div class="phase-label">Phase 00</div>
    <div class="phase-title">Fetching live FX rates for USD normalisation</div>
</div>""", unsafe_allow_html=True)

    fx_status = st.empty()
    fx_status.markdown("""
<div class="tool-call">
    <span class="fn">fetch_fx_rates</span>(<span class="arg">"JPY, INR, HKD, KRW, EUR, GBP, TWD, CNY..."</span>)
    &nbsp;·&nbsp; fetching live rates via yfinance...
</div>""", unsafe_allow_html=True)

    fx_rates = fetch_fx_rates()

    rate_preview = " &nbsp;·&nbsp; ".join([
        f"<span style='color:#f59e0b'>{k}</span>→<span style='color:#22d3ee'>${v:.5f}</span>"
        for k, v in list(fx_rates.items())[:8] if k != "USD"
    ])
    fx_status.markdown(
        f'<div class="tool-call"><span class="ok">✓</span> &nbsp; Rates loaded: &nbsp; {rate_preview} &nbsp;·&nbsp; '
        f'<span style="color:#64748b">{len(fx_rates)} currencies normalised to USD</span></div>',
        unsafe_allow_html=True
    )

    # ── PHASE 1 ──────────────────────────
    st.markdown("""
<div class="phase-panel">
    <div class="phase-label">Phase 01</div>
    <div class="phase-title">Discovering & ranking companies</div>
</div>""", unsafe_allow_html=True)

    status1   = st.empty()
    progress1 = st.progress(0)

    top_companies = rank_companies(tickers, metric_key, status1, progress1, fx_rates)

    if not top_companies:
        st.error("Could not fetch data for any tickers. Check your internet connection.")
        st.stop()

    status1.markdown(f'<span class="status-ok">✓ &nbsp; TOP {len(top_companies)} IDENTIFIED</span>', unsafe_allow_html=True)
    progress1.progress(1.0)

    # Company rank cards — clickable expanders
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(len(top_companies))
    for i, c in enumerate(top_companies):
        with cols[i]:
            st.markdown(f"""
<div class="rank-card">
    <div class="rank-num">#{i+1}</div>
    <div class="rank-name">{c.get('name', c['ticker'])[:22]}</div>
    <div class="rank-ticker">{c['ticker']}</div>
    <div class="rank-region">{c.get('region','')}</div>
    <div class="rank-region" style="color:#a855f7;margin-top:4px;font-size:0.65rem;">
        {c.get('data_as_of','N/A')}
    </div>
</div>""", unsafe_allow_html=True)

    # Expandable detail panels below the cards
    st.markdown("<br>", unsafe_allow_html=True)
    for i, c in enumerate(top_companies):
        name = c.get("name", c["ticker"])
        with st.expander(f"#{i+1} · {name} ({c['ticker']}) — click for details"):

            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:0.6rem;
            letter-spacing:0.12em;text-transform:uppercase;
            color:#64748b;margin-bottom:0.75rem;">
    ◈ &nbsp; Key Metrics &nbsp;·&nbsp; {c.get('data_as_of','N/A')}
</div>""", unsafe_allow_html=True)

                def fmt_val(val, prefix="$", suffix="", pct=False, decimals=0):
                    if val is None:
                        return "N/A"
                    if pct:
                        return f"{val*100:.1f}%"
                    if decimals:
                        return f"{prefix}{val:,.{decimals}f}{suffix}"
                    return f"{prefix}{val:,.0f}{suffix}"

                currency = c.get("currency", "USD")
                cur_sym  = {"USD":"$","EUR":"€","GBP":"£","JPY":"¥",
                            "INR":"₹","HKD":"HK$","KRW":"₩","TWD":"NT$",
                            "CNY":"¥","AUD":"A$","CAD":"C$"}.get(currency, currency+" ")

                def fmt_dual(local_val, usd_val, currency, cur_sym):
                    """Show local value + USD equivalent for monetary fields."""
                    if local_val is None:
                        return "N/A"
                    local_str = f"{cur_sym}{local_val:,.0f} {currency}"
                    if currency == "USD" or usd_val is None:
                        return local_str
                    return f"{local_str} <span style='color:#64748b;font-size:0.65rem'>(≈ ${usd_val:,.0f} USD)</span>"

                metrics = [
                    ("Current Price",  f"{cur_sym}{c.get('currentPrice','N/A')} {currency}"),
                    ("Market Cap (local)", fmt_dual(c.get("marketCap"), c.get("marketCap_usd"), currency, cur_sym)),
                    ("Market Cap (USD)",   f"${c.get('marketCap_usd'):,.0f}" if c.get("marketCap_usd") else "N/A"),
                    ("Revenue (local)",    fmt_dual(c.get("totalRevenue"), c.get("totalRevenue_usd"), currency, cur_sym) + f"  <span style='color:#a855f7;font-size:0.65rem'>({c.get('fiscal_year','N/A')})</span>"),
                    ("Revenue (USD)",      f"${c.get('totalRevenue_usd'):,.0f}" if c.get("totalRevenue_usd") else "N/A"),
                    ("P/E Ratio",      fmt_val(c.get("trailingPE"), prefix="", suffix="x", decimals=2) + f"  ({c.get('most_recent_q','')})"),
                    ("Revenue Growth", fmt_val(c.get("revenueGrowth"), prefix="", pct=True) + f"  ({c.get('fiscal_year','N/A')})"),
                    ("Profit Margin",  fmt_val(c.get("profitMargins"), prefix="", pct=True) + f"  ({c.get('fiscal_year','N/A')})"),
                    ("52w Range",      f"{c.get('52wLow','?')} – {c.get('52wHigh','?')} {currency}"),
                    ("Employees",      fmt_val(c.get("employees"), prefix="")),
                    ("Country",        c.get("country", "N/A")),
                    ("Currency",       currency),
                ]
                for label, value in metrics:
                    st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:baseline;
            border-bottom:1px solid #1e2a50;padding:5px 0;">
    <span style="font-family:'Space Mono',monospace;font-size:0.65rem;
                 color:#64748b;text-transform:uppercase;">{label}</span>
    <span style="font-family:'Space Mono',monospace;font-size:0.75rem;
                 color:#e2e8f0;text-align:right;">{value}</span>
</div>""", unsafe_allow_html=True)

            with col_right:
                st.markdown(f"""
<div style="font-family:'Space Mono',monospace;font-size:0.6rem;
            letter-spacing:0.12em;text-transform:uppercase;
            color:#64748b;margin-bottom:0.75rem;">
    ◈ &nbsp; About
</div>""", unsafe_allow_html=True)

                wiki = c.get("wiki_summary") or c.get("summary") or ""
                if wiki:
                    st.markdown(f"""
<div style="font-size:0.85rem;line-height:1.7;color:#94a3b8;
            border-left:2px solid #3d52d5;padding-left:1rem;">
    {wiki[:450]}
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div style="font-size:0.85rem;color:#334155;font-style:italic;">
    No Wikipedia summary available for this company.
</div>""", unsafe_allow_html=True)

    # ── PHASE 2 ──────────────────────────
    st.markdown("""
<div class="phase-panel">
    <div class="phase-label">Phase 02</div>
    <div class="phase-title">Enriching with Wikipedia intelligence</div>
</div>""", unsafe_allow_html=True)

    status2 = st.empty()
    enriched = []
    for company in top_companies:
        enriched_company = enrich_company(company, status2)
        enriched.append(enriched_company)

    status2.markdown(f'<span class="status-ok">✓ &nbsp; BACKGROUND FETCHED FOR {len(enriched)} COMPANIES</span>', unsafe_allow_html=True)

    # ── PHASE 3 ──────────────────────────
    st.markdown("""
<div class="phase-panel">
    <div class="phase-label">Phase 03</div>
    <div class="phase-title">Generating analysis with GPT-5.4-mini</div>
</div>""", unsafe_allow_html=True)

    status3 = st.empty()
    status3.markdown("""
<div class="tool-call">
    <span class="fn">generate_analysis</span>(<span class="arg">"comparison_table"</span>,
    <span class="arg">"investor_briefing"</span>, <span class="arg">"confidence_scores"</span>)
    &nbsp;·&nbsp; running...
</div>""", unsafe_allow_html=True)

    with st.spinner(""):
        try:
            table_md, briefing, cost = generate_analysis(
                enriched, sector_label, metric_label, client
            )
            st.session_state.total_cost += cost
            st.session_state.run_count  += 1
        except Exception as e:
            st.error(f"GPT analysis failed: {e}")
            st.stop()

    status3.markdown(f'<span class="status-ok">✓ &nbsp; ANALYSIS COMPLETE &nbsp;·&nbsp; COST: ${cost:.5f}</span>', unsafe_allow_html=True)

    # ── OUTPUT ────────────────────────────
    st.markdown(f"""
<div class="result-header">
    ◈ &nbsp; {sector_label.upper()} · TOP {TOP_N} BY {metric_label.upper()} · DEEP-DIVE REPORT
</div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["◈  COMPARISON TABLE", "◈  INVESTOR BRIEFING", "◈  RAW DATA"])

    with tab1:
        st.markdown(table_md)

    with tab2:
        st.markdown(f'<div class="briefing-box">{briefing}</div>', unsafe_allow_html=True)

    with tab3:
        raw_rows = []
        for c in enriched:
            raw_rows.append({
                "Ticker":           c["ticker"],
                "Company":          c.get("name", c["ticker"]),
                "Region":           c.get("region", "N/A"),
                "Country":          c.get("country", "N/A"),
                "Currency":         c.get("currency", "USD"),
                "Fiscal Year":      c.get("fiscal_year", "N/A"),
                "As of Quarter":    c.get("most_recent_q", "N/A"),
                "Mkt Cap (USD)":    f"${c['marketCap_usd']:,.0f}" if c.get("marketCap_usd") else "N/A",
                "Revenue (USD)":    f"${c['totalRevenue_usd']:,.0f}" if c.get("totalRevenue_usd") else "N/A",
                "Mkt Cap (local)":  f"{c['marketCap']:,.0f}" if c.get("marketCap") else "N/A",
                "Revenue (local)":  f"{c['totalRevenue']:,.0f}" if c.get("totalRevenue") else "N/A",
                "P/E":              round(c["trailingPE"], 2) if c.get("trailingPE") else "N/A",
                "Rev Growth":       f"{c['revenueGrowth']*100:.1f}%" if c.get("revenueGrowth") else "N/A",
                "Profit Margin":    f"{c['profitMargins']*100:.1f}%" if c.get("profitMargins") else "N/A",
                "Price (local)":    c.get("currentPrice", "N/A"),
            })
        st.dataframe(pd.DataFrame(raw_rows), use_container_width=True)

    st.session_state.results = {
        "sector": sector_label, "metric": metric_label,
        "companies": enriched, "table": table_md, "briefing": briefing
    }

elif st.session_state.results:
    r = st.session_state.results
    st.markdown(f"""
<div class="result-header">
    ◈ &nbsp; {r['sector'].upper()} · TOP {TOP_N} BY {r['metric'].upper()} · PREVIOUS REPORT
</div>""", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["◈  COMPARISON TABLE", "◈  INVESTOR BRIEFING"])
    with tab1:
        st.markdown(r["table"])
    with tab2:
        st.markdown(f'<div class="briefing-box">{r["briefing"]}</div>', unsafe_allow_html=True)

else:
    # ── Landing state ────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    cards = [
        ("🌍", "Global Universe", "US, Europe, Asia, India & China — real tickers, real data, no guessing"),
        ("🔬", "Data-First Agent", "Companies ranked from live market metrics — GPT never invents figures"),
        ("👁️", "Full Transparency", "Every tool call shown live as the agent works through each phase"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(f"""
<div class="feature-card">
    <div class="feature-icon">{icon}</div>
    <div class="feature-title">{title}</div>
    <div class="feature-desc">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div style="text-align:center;font-family:'Space Mono',monospace;font-size:0.7rem;
            color:#334155;letter-spacing:0.1em;text-transform:uppercase;padding:2rem 0;">
    ◂ &nbsp; SELECT SECTOR · METRIC · UNIVERSE &nbsp; THEN &nbsp; INITIATE SCAN &nbsp; ▸
</div>""", unsafe_allow_html=True)