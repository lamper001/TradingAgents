"""WeStock data provider implementation.

Wraps the `npx -y westock-data-skillhub@1.0.3` CLI to provide stock data
for A-shares, Hong Kong stocks, and US stocks as a drop-in replacement for
the yfinance data layer.

Ticker format conversion:
  - US:  "AAPL"    → "usAAPL"
  - SH:  "600000"  → "sh600000"
  - SZ:  "000001"  → "sz000001"
  - HK:  "00700"   → "hk00700"

Inputs that already have a market prefix (sh/sz/hk/us/bj) are passed
through unchanged.
"""

from __future__ import annotations

import re
import subprocess
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Annotated

logger = logging.getLogger(__name__)

# The published CLI package version
_WESTOCK_PKG = "westock-data-skillhub@1.0.3"


# ---------------------------------------------------------------------------
# Ticker conversion helpers
# ---------------------------------------------------------------------------

_HK_RE = re.compile(r"^\d{4,5}$")          # e.g. "00700" / "9988"
_SH_RE = re.compile(r"^(6\d{5}|68\d{4})$") # 6xxxxx or 688xxx (STAR)
_SZ_RE = re.compile(r"^[0-3]\d{5}$")        # 0xxxxx / 1xxxxx / 2xxxxx / 3xxxxx
_BJ_RE = re.compile(r"^[4-8]\d{5}$")        # 4xxxxx – 8xxxxx (Beijing)
_ALREADY_PREFIXED = re.compile(r"^(sh|sz|bj|hk|us)", re.IGNORECASE)


def to_westock_code(ticker: str) -> str:
    """Convert a raw ticker string to the westock code format.

    Examples
    --------
    >>> to_westock_code("AAPL")
    'usAAPL'
    >>> to_westock_code("600000")
    'sh600000'
    >>> to_westock_code("000001")
    'sz000001'
    >>> to_westock_code("00700")
    'hk00700'
    >>> to_westock_code("usAAPL")   # already prefixed
    'usAAPL'
    """
    ticker = ticker.strip()
    if _ALREADY_PREFIXED.match(ticker):
        return ticker

    if _SH_RE.fullmatch(ticker):
        return f"sh{ticker}"
    if _SZ_RE.fullmatch(ticker):
        return f"sz{ticker}"
    if _BJ_RE.fullmatch(ticker):
        return f"bj{ticker}"
    if _HK_RE.fullmatch(ticker):
        return f"hk{ticker.zfill(5)}"

    # Default: treat as US ticker
    return f"us{ticker}"


# ---------------------------------------------------------------------------
# Low-level CLI runner
# ---------------------------------------------------------------------------

def _run_westock(*args: str, timeout: int = 30) -> str:
    """Execute a westock CLI command and return its stdout as a string.

    Raises RuntimeError on non-zero exit code or timeout.
    """
    cmd = ["npx", "-y", _WESTOCK_PKG, *args]
    logger.debug("westock cmd: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"westock command timed out after {timeout}s: {' '.join(args)}")

    if result.returncode != 0:
        raise RuntimeError(
            f"westock command failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Core stock data  (replaces get_YFin_data_online / load_ohlcv)
# ---------------------------------------------------------------------------

def get_stock_data_westock(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Get OHLCV data via westock kline command.

    Returns a Markdown table with columns:
        date | open | last | high | low | volume | amount | exchange
    """
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    code = to_westock_code(symbol)

    # Calculate days between start and end + buffer for non-trading days
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    delta_days = (end_dt - start_dt).days
    # Request 1.5x days to ensure coverage (weekends/holidays)
    limit = max(int(delta_days * 1.5), 10)

    try:
        output = _run_westock(
            "kline", code,
            "--period", "day",
            "--limit", str(limit),
            "--fq", "qfq",          # forward-adjusted prices
        )
    except RuntimeError as e:
        return f"Error fetching stock data for {symbol}: {e}"

    if not output:
        return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

    # Filter rows by date range
    lines = output.splitlines()
    header_lines = [l for l in lines if l.startswith("| date") or l.startswith("| ---")]
    data_lines = [l for l in lines if l.startswith("|") and not l.startswith("| date") and not l.startswith("| ---")]

    filtered = []
    for line in data_lines:
        cols = [c.strip() for c in line.strip("|").split("|")]
        if cols and len(cols) >= 1:
            date_val = cols[0]
            try:
                row_dt = datetime.strptime(date_val, "%Y-%m-%d")
                if start_dt <= row_dt <= end_dt:
                    filtered.append(line)
            except ValueError:
                pass

    if not filtered:
        return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

    result_lines = header_lines + filtered
    table = "\n".join(result_lines)

    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(filtered)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + table


# ---------------------------------------------------------------------------
# Technical indicators  (replaces get_stock_stats_indicators_window)
# ---------------------------------------------------------------------------

# Map stockstats indicator names → westock group names + field names
_INDICATOR_MAP = {
    # Moving Averages → group "ma"
    "close_50_sma":  ("ma",   "ma.MA_50",   "50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals."),
    "close_200_sma": ("ma",   "ma.MA_250",  "200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."),
    "close_10_ema":  ("ma",   "ma.EMA_12",  "10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."),
    # MACD
    "macd":  ("macd", "macd.DIF",  "MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets."),
    "macds": ("macd", "macd.DEA",  "MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives."),
    "macdh": ("macd", "macd.MACD", "MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets."),
    # RSI
    "rsi":   ("rsi",  "rsi.RSI_6", "RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."),
    # Bollinger Bands
    "boll":    ("boll", "boll.BOLL_MID",   "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."),
    "boll_ub": ("boll", "boll.BOLL_UPPER", "Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends."),
    "boll_lb": ("boll", "boll.BOLL_LOWER", "Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals."),
    # ATR — westock does not expose ATR directly; fall back gracefully
    "atr":  (None, None, "ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy."),
    # VWMA — not in westock; fall back gracefully
    "vwma": (None, None, "VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."),
    # MFI — not in westock; fall back gracefully
    "mfi":  (None, None, "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure."),
}


def _parse_markdown_table(text: str) -> list[dict]:
    """Parse a westock Markdown table into a list of row dicts."""
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        return []

    header_cells = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:   # skip header + separator
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(header_cells):
            rows.append(dict(zip(header_cells, cells)))
    return rows


def get_stock_stats_indicators_window_westock(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Get technical indicator values via westock technical command."""
    if indicator not in _INDICATOR_MAP:
        raise ValueError(
            f"Indicator '{indicator}' is not supported. Choose from: {list(_INDICATOR_MAP.keys())}"
        )

    group, field, description = _INDICATOR_MAP[indicator]

    code = to_westock_code(symbol)
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)
    start_str = start_dt.strftime("%Y-%m-%d")

    if group is None:
        # Indicator not supported by westock — return informative message
        return (
            f"## {indicator} values from {start_str} to {curr_date}:\n\n"
            f"N/A: This indicator is not available from the westock data source.\n\n"
            f"{description}"
        )

    try:
        output = _run_westock(
            "technical", code,
            "--group", group,
            "--start", start_str,
            "--end", curr_date,
        )
    except RuntimeError as e:
        return f"Error fetching indicator {indicator} for {symbol}: {e}"

    rows = _parse_markdown_table(output)
    if not rows:
        return f"No indicator data found for {symbol} ({indicator}) between {start_str} and {curr_date}"

    ind_string = ""
    for row in rows:
        date_val = row.get("date", "")
        value = row.get(field, "N/A")
        # Skip rows outside our date window
        try:
            row_dt = datetime.strptime(date_val, "%Y-%m-%d")
            if row_dt < start_dt or row_dt > curr_dt:
                continue
        except ValueError:
            continue
        ind_string += f"{date_val}: {value}\n"

    if not ind_string:
        ind_string = "N/A: No trading data found for this period.\n"

    return (
        f"## {indicator} values from {start_str} to {curr_date}:\n\n"
        + ind_string
        + "\n\n"
        + description
    )


# ---------------------------------------------------------------------------
# Fundamental data
# ---------------------------------------------------------------------------

def get_fundamentals_westock(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (unused, for API compatibility)"] = None,
) -> str:
    """Get company fundamental overview via westock profile command."""
    code = to_westock_code(ticker)
    try:
        output = _run_westock("profile", code)
    except RuntimeError as e:
        return f"Error retrieving fundamentals for {ticker}: {e}"

    if not output:
        return f"No fundamentals data found for symbol '{ticker}'"

    header = f"# Company Fundamentals for {ticker.upper()}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + output


def get_balance_sheet_westock(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get balance sheet data via westock finance command."""
    code = to_westock_code(ticker)
    # westock finance --type: A-share uses 'zcfz', US/HK uses 'balance'
    fin_type = _finance_type(code, "balance")
    num = 4 if freq.lower() == "quarterly" else 2
    try:
        output = _run_westock("finance", code, "--type", fin_type, "--num", str(num))
    except RuntimeError as e:
        return f"Error retrieving balance sheet for {ticker}: {e}"

    if not output:
        return f"No balance sheet data found for symbol '{ticker}'"

    # Optional: filter columns by curr_date to avoid look-ahead
    if curr_date:
        output = _filter_finance_by_date(output, curr_date)

    header = f"# Balance Sheet data for {ticker.upper()} ({freq})\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + output


def get_cashflow_westock(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get cash flow data via westock finance command."""
    code = to_westock_code(ticker)
    fin_type = _finance_type(code, "cashflow")
    num = 4 if freq.lower() == "quarterly" else 2
    try:
        output = _run_westock("finance", code, "--type", fin_type, "--num", str(num))
    except RuntimeError as e:
        return f"Error retrieving cash flow for {ticker}: {e}"

    if not output:
        return f"No cash flow data found for symbol '{ticker}'"

    if curr_date:
        output = _filter_finance_by_date(output, curr_date)

    header = f"# Cash Flow data for {ticker.upper()} ({freq})\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + output


def get_income_statement_westock(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get income statement data via westock finance command."""
    code = to_westock_code(ticker)
    fin_type = _finance_type(code, "income")
    num = 4 if freq.lower() == "quarterly" else 2
    try:
        output = _run_westock("finance", code, "--type", fin_type, "--num", str(num))
    except RuntimeError as e:
        return f"Error retrieving income statement for {ticker}: {e}"

    if not output:
        return f"No income statement data found for symbol '{ticker}'"

    if curr_date:
        output = _filter_finance_by_date(output, curr_date)

    header = f"# Income Statement data for {ticker.upper()} ({freq})\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + output


# ---------------------------------------------------------------------------
# News & insider transactions (stubs — westock has no news/insider data)
# ---------------------------------------------------------------------------

def get_news_westock(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """News data is not available from the westock provider."""
    return (
        f"## {ticker} News, from {start_date} to {end_date}:\n\n"
        "News data is not available from the westock data source. "
        "Configure a different news_data vendor (e.g. yfinance or alpha_vantage) "
        "for news analysis."
    )


def get_global_news_westock(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
) -> str:
    """Global news is not available from the westock provider."""
    return (
        "## Global Market News:\n\n"
        "Global news data is not available from the westock data source. "
        "Configure a different news_data vendor (e.g. yfinance or alpha_vantage) "
        "for global news analysis."
    )


def get_insider_transactions_westock(
    ticker: str,
) -> str:
    """Insider transactions are not available from the westock provider."""
    return (
        f"## Insider Transactions for {ticker}:\n\n"
        "Insider transaction data is not available from the westock data source. "
        "Configure a different news_data vendor (e.g. yfinance or alpha_vantage) "
        "for insider transaction data."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _finance_type(code: str, category: str) -> str:
    """Return the westock finance --type value for the given market and category.

    A-share finance types: lrb | zcfz | xjll
    HK/US finance types:  income | balance | cashflow
    """
    if code.startswith(("sh", "sz", "bj")):
        mapping = {"income": "lrb", "balance": "zcfz", "cashflow": "xjll"}
    else:
        mapping = {"income": "income", "balance": "balance", "cashflow": "cashflow"}
    return mapping.get(category, category)


def _filter_finance_by_date(table_text: str, curr_date: str) -> str:
    """Remove Markdown table columns whose header date is after curr_date.

    westock finance tables have dates in the ``_date`` column (rows, not
    columns), so we simply drop rows whose _date > curr_date.
    """
    cutoff = datetime.strptime(curr_date, "%Y-%m-%d")
    lines = table_text.splitlines()
    result = []
    for line in lines:
        if not line.startswith("|"):
            result.append(line)
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if not cols:
            result.append(line)
            continue
        first = cols[0]
        # Keep header/separator rows and rows whose date <= curr_date
        try:
            row_dt = datetime.strptime(first, "%Y-%m-%d")
            if row_dt <= cutoff:
                result.append(line)
        except ValueError:
            result.append(line)  # header or separator
    return "\n".join(result)
