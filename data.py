"""Data access layer for market data providers.

Default provider uses yfinance, but interfaces are designed so providers can be
swapped with institutional data sources later.
"""

from __future__ import annotations

import time
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
    max_retries: int = 3
    retry_delay_seconds: float = 1.5

    @staticmethod
    def _flatten_columns(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Flatten yfinance MultiIndex columns and normalize names."""
        if isinstance(raw.columns, pd.MultiIndex):
            flattened: list[str] = []
            for col in raw.columns:
                # yfinance often uses (field, ticker) for multi-level headers.
                if isinstance(col, tuple):
                    if len(col) > 1 and str(col[0]) in {"Open", "High", "Low", "Close", "Adj Close", "Volume"}:
                        flattened.append(str(col[0]))
                    elif len(col) > 1 and str(col[1]) in {"Open", "High", "Low", "Close", "Adj Close", "Volume"}:
                        flattened.append(str(col[1]))
                    else:
                        flattened.append("_".join(str(x) for x in col if x))
                else:
                    flattened.append(str(col))
            raw = raw.copy()
            raw.columns = flattened

        # If there are duplicate OHLC names after flattening, keep the first.
        if raw.columns.duplicated().any():
            raw = raw.loc[:, ~raw.columns.duplicated(keep="first")]

        # Some yfinance responses include columns with ticker suffixes; normalize.
        normalized = {}
        for col in raw.columns:
            name = str(col)
            for base in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                if name == base or name.startswith(f"{base}_") or name.endswith(f"_{symbol}"):
                    normalized[col] = base
                    break
        if normalized:
            raw = raw.rename(columns=normalized)

        return raw

    def _download_once(self, symbol: str, start: pd.Timestamp, end_inclusive: pd.Timestamp) -> pd.DataFrame:
        raw = yf.download(
            tickers=symbol,
            start=pd.Timestamp(start).strftime("%Y-%m-%d"),
            end=end_inclusive.strftime("%Y-%m-%d"),
            auto_adjust=self.auto_adjust,
            progress=False,
            actions=False,
            interval="1d",
            threads=False,
            group_by="column",
        )
        return self._flatten_columns(raw, symbol)

    def fetch_daily_prices(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        # yfinance end date is exclusive; pad by one day to include desired end.
        end_inclusive = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)

        raw = pd.DataFrame()
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self._download_once(symbol=symbol, start=start, end_inclusive=end_inclusive)
                if not raw.empty:
                    break
            except Exception as exc:  # noqa: BLE001 - yfinance raises non-stable exception types.
                last_err = exc

            # Backoff for transient provider/rate-limit errors.
            if attempt < self.max_retries:
                time.sleep(self.retry_delay_seconds * attempt)

        if raw.empty:
            # Secondary fallback path via Ticker.history, which can succeed when download is throttled.
            try:
                hist = yf.Ticker(symbol).history(
                    start=pd.Timestamp(start).strftime("%Y-%m-%d"),
                    end=end_inclusive.strftime("%Y-%m-%d"),
                    auto_adjust=self.auto_adjust,
                    interval="1d",
                    actions=False,
                )
                raw = self._flatten_columns(hist, symbol)
            except Exception as exc:  # noqa: BLE001
                last_err = exc

        if raw.empty:
            if last_err is not None:
                raise ValueError(f"No data returned for symbol={symbol}. Last provider error: {last_err}") from last_err
            raise ValueError(f"No data returned for symbol={symbol}")

        raw = raw.sort_index()
        raw.index = pd.to_datetime(raw.index, utc=True).tz_convert(None)

        # Harmonize adjusted close field regardless of provider output.
        if "Adj Close" in raw.columns:
            raw["adj_close"] = pd.to_numeric(raw["Adj Close"], errors="coerce")
        elif "Close" in raw.columns:
            # If auto-adjusted close isn't present, close may already be adjusted.
            raw["adj_close"] = pd.to_numeric(raw["Close"], errors="coerce")
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
        if "adj_close" not in prices.columns:
            raise ValueError("Normalized output missing required column 'adj_close'.")
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
    end_ts = pd.Timestamp.now(tz="UTC").normalize() if end is None else pd.Timestamp(end)

    prices = provider.fetch_daily_prices("^GSPC", start=start_ts, end=end_ts)

    # Ensure daily index is trading-day only and clean any missing/infinite values.
    prices = prices.replace([pd.NA, float("inf"), float("-inf")], pd.NA)
    prices = prices.dropna(subset=["adj_close"])

    # Daily close-to-close returns.
    prices["daily_return"] = prices["adj_close"].pct_change().fillna(0.0)

    return prices
