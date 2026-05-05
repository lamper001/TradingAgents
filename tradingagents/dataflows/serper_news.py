"""Serper.dev + WebBaseLoader news data provider.

Uses GoogleSerperAPIWrapper (type="news") to search for news articles,
then optionally scrapes the full article body via WebBaseLoader.

Requires:
    SERPER_API_KEY environment variable (https://serper.dev — 2500 free/month)

Install (already in langchain-community):
    pip install langchain-community
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Max characters to keep per scraped article (avoids flooding the LLM context)
_MAX_ARTICLE_CHARS = 2000
# Max articles to scrape full text for (keeps latency reasonable)
_MAX_SCRAPE = 3


def _get_serper_wrapper(k: int = 10):
    """Create a GoogleSerperAPIWrapper for news search."""
    from langchain_community.utilities import GoogleSerperAPIWrapper

    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SERPER_API_KEY environment variable is not set. "
            "Get a free key at https://serper.dev (2500 searches/month free)."
        )
    return GoogleSerperAPIWrapper(type="news", k=k, serper_api_key=api_key)


def _scrape_url(url: str) -> str:
    """Fetch full article text from a URL via WebBaseLoader."""
    try:
        from langchain_community.document_loaders import WebBaseLoader
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loader = WebBaseLoader(url)
            loader.requests_kwargs = {"timeout": 10}
            docs = loader.load()

        if not docs:
            return ""
        text = docs[0].page_content
        # Trim and clean whitespace
        text = " ".join(text.split())
        return text[:_MAX_ARTICLE_CHARS]
    except Exception as e:
        logger.debug("Failed to scrape %s: %s", url, e)
        return ""


def _format_news_results(raw: dict, ticker: str, start_date: str, end_date: str, scrape: bool = True) -> str:
    """Parse Serper news results dict into a formatted string for the LLM."""
    articles = raw.get("news", [])
    if not articles:
        return f"No news found for {ticker} between {start_date} and {end_date}"

    # Filter by date range when date metadata is available
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    filtered = []
    for article in articles:
        date_str = article.get("date", "")
        # Serper news date is like "2 hours ago", "May 4, 2026", etc.
        # We keep articles without parseable dates (can't filter them out safely)
        filtered.append(article)

    if not filtered:
        return f"No news found for {ticker} between {start_date} and {end_date}"

    news_str = ""
    scraped = 0

    for article in filtered:
        title = article.get("title", "No title")
        source = article.get("source", "Unknown")
        date = article.get("date", "")
        link = article.get("link", "")
        snippet = article.get("snippet", "")

        news_str += f"### {title}\n"
        news_str += f"**Source**: {source}"
        if date:
            news_str += f"  |  **Date**: {date}"
        news_str += "\n"
        if snippet:
            news_str += f"{snippet}\n"

        # Scrape full text for the first N articles
        if scrape and link and scraped < _MAX_SCRAPE:
            full_text = _scrape_url(link)
            if full_text and len(full_text) > len(snippet):
                news_str += f"\n**Full article excerpt**:\n{full_text}\n"
            scraped += 1

        if link:
            news_str += f"Link: {link}\n"
        news_str += "\n"

    return f"## {ticker} News, from {start_date} to {end_date}:\n\n{news_str}"


def _format_global_news_results(raw: dict, curr_date: str, look_back_days: int, scrape: bool = True) -> str:
    """Parse Serper news results dict into a formatted string for global news."""
    articles = raw.get("news", [])
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    if not articles:
        return f"No global market news found for {curr_date}"

    news_str = ""
    scraped = 0

    for article in articles:
        title = article.get("title", "No title")
        source = article.get("source", "Unknown")
        date = article.get("date", "")
        link = article.get("link", "")
        snippet = article.get("snippet", "")

        news_str += f"### {title}\n"
        news_str += f"**Source**: {source}"
        if date:
            news_str += f"  |  **Date**: {date}"
        news_str += "\n"
        if snippet:
            news_str += f"{snippet}\n"

        if scrape and link and scraped < _MAX_SCRAPE:
            full_text = _scrape_url(link)
            if full_text and len(full_text) > len(snippet):
                news_str += f"\n**Full article excerpt**:\n{full_text}\n"
            scraped += 1

        if link:
            news_str += f"Link: {link}\n"
        news_str += "\n"

    return f"## Global Market News, from {start_date} to {curr_date}:\n\n{news_str}"


# ---------------------------------------------------------------------------
# Public API — matches the interface expected by route_to_vendor
# ---------------------------------------------------------------------------

def get_news_serper(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """Search for company-specific news via Serper.dev Google News.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL", "NVDA")
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string containing news articles with full text excerpts.
    """
    try:
        wrapper = _get_serper_wrapper(k=8)
        # Build a targeted news query: ticker name + financial keywords
        query = f"{ticker} stock news earnings analyst"
        raw = wrapper.results(query)
        return _format_news_results(raw, ticker, start_date, end_date, scrape=True)
    except Exception as e:
        return f"Error fetching news for {ticker} via Serper: {e}"


def get_global_news_serper(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
) -> str:
    """Search for global macro/market news via Serper.dev Google News.

    Args:
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back
        limit: Maximum number of articles to return

    Returns:
        Formatted string containing global news articles with full text excerpts.
    """
    try:
        wrapper = _get_serper_wrapper(k=limit)
        query = "global stock market economy Federal Reserve inflation macroeconomics"
        raw = wrapper.results(query)
        return _format_global_news_results(raw, curr_date, look_back_days, scrape=True)
    except Exception as e:
        return f"Error fetching global news via Serper: {e}"


def get_insider_transactions_serper(ticker: str) -> str:
    """Search for insider trading news via Serper.dev.

    Note: Serper returns news articles about insider transactions, not
    structured SEC filing data. For structured data use alpha_vantage.
    """
    try:
        wrapper = _get_serper_wrapper(k=5)
        query = f"{ticker} insider trading SEC filing buy sell shares"
        raw = wrapper.results(query)
        articles = raw.get("news", [])

        if not articles:
            return f"No insider transaction news found for {ticker}"

        news_str = ""
        for article in articles:
            title = article.get("title", "No title")
            source = article.get("source", "Unknown")
            date = article.get("date", "")
            link = article.get("link", "")
            snippet = article.get("snippet", "")

            news_str += f"### {title}\n"
            news_str += f"**Source**: {source}"
            if date:
                news_str += f"  |  **Date**: {date}"
            news_str += "\n"
            if snippet:
                news_str += f"{snippet}\n"
            if link:
                news_str += f"Link: {link}\n"
            news_str += "\n"

        return f"## {ticker} Insider Transactions News:\n\n{news_str}"
    except Exception as e:
        return f"Error fetching insider transaction news for {ticker} via Serper: {e}"
