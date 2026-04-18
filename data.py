"""Data access layer for market data providers.

Default provider uses yfinance, but interfaces are designed so providers can be
swapped with institutional data sources later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import pandas as pd
import yfinance as yf


class PriceDataProvider(Protocol):
    """Protocol for market data providers."""

    def fetch_daily_prices(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return daily OHLCV prices with at least an adjusted close field."""


@dataclass
class YFinanceProvider:
    """yfinance-based data provider for daily index data."""

    auto_adjust: bool = False

    def fetch_daily_prices(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        # yfinance end date is exclusive; pad by one day to include desired end.
        end_inclusive = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
        raw = yf.download(
            tickers=symbol,
            start=pd.Timestamp(start).strftime("%Y-%m-%d"),
            end=end_inclusive.strftime("%Y-%m-%d"),
            auto_adjust=self.auto_adjust,
            progress=False,
            actions=False,
            interval="1d",
            threads=True,
        )

        if raw.empty:
            raise ValueError(f"No data returned for symbol={symbol}")

        raw = raw.sort_index()
        raw.index = pd.to_datetime(raw.index, utc=True).tz_convert(None)

        # Harmonize adjusted close field regardless of provider output.
        if "Adj Close" in raw.columns:
            raw["adj_close"] = raw["Adj Close"]
        elif "Close" in raw.columns:
            # If auto-adjusted close isn't present, close may already be adjusted.
            raw["adj_close"] = raw["Close"]
        else:
            raise ValueError("Provider output missing both 'Adj Close' and 'Close'.")

        cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "adj_close"] if c in raw.columns]
        prices = raw[cols].copy()
        prices = prices.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        # Drop invalid observations and de-duplicate index.
        prices = prices[~prices.index.duplicated(keep="last")]
        prices = prices.dropna(subset=["adj_close"])

        return prices


def get_sp500_daily(
    start: str | pd.Timestamp = "2000-01-01",
    end: Optional[str | pd.Timestamp] = None,
    provider: Optional[PriceDataProvider] = None,
) -> pd.DataFrame:
    """Fetch cleaned daily S&P 500 data indexed by date.

    Parameters
    ----------
    start:
        Inclusive start date.
    end:
        Inclusive end date. Defaults to today's date in UTC.
    provider:
        Optional provider implementation. Defaults to YFinanceProvider.
    """

    provider = provider or YFinanceProvider()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp.utcnow().normalize() if end is None else pd.Timestamp(end)

    prices = provider.fetch_daily_prices("^GSPC", start=start_ts, end=end_ts)

    # Ensure daily index is trading-day only and clean any missing/infinite values.
    prices = prices.replace([pd.NA, float("inf"), float("-inf")], pd.NA)
    prices = prices.dropna(subset=["adj_close"])

    # Daily close-to-close returns.
    prices["daily_return"] = prices["adj_close"].pct_change().fillna(0.0)

    return prices
