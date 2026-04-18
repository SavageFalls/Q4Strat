from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


DEFAULT_SYMBOL = "^GSPC"
DEFAULT_START = "2000-01-01"


@dataclass(frozen=True)
class MarketDataRequest:
    symbol: str = DEFAULT_SYMBOL
    start: str = DEFAULT_START
    end: Optional[str] = None


class MarketDataProvider(ABC):
    name: str

    @abstractmethod
    def fetch(self, request: MarketDataRequest) -> pd.DataFrame:
        raise NotImplementedError


class YFinanceProvider(MarketDataProvider):
    name = "yfinance"

    def fetch(self, request: MarketDataRequest) -> pd.DataFrame:
        tz_cache_dir = Path("cache") / "yfinance_tz"
        tz_cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(tz_cache_dir.resolve()))
        frame = yf.download(
            tickers=request.symbol,
            start=request.start,
            end=request.end,
            auto_adjust=False,
            actions=False,
            progress=False,
            repair=True,
        )
        if frame.empty:
            raise ValueError(f"No data returned for {request.symbol}.")
        return frame


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(part) for part in frame.columns.get_level_values(0)]
    return frame


def clean_market_data(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _flatten_columns(raw.copy())
    frame.index = pd.to_datetime(frame.index)

    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_convert(None)

    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    frame = frame.rename(columns=rename_map)

    if "adj_close" not in frame.columns:
        if "close" not in frame.columns:
            raise ValueError("Adjusted close and close columns are both missing.")
        frame["adj_close"] = frame["close"]

    if frame["adj_close"].isna().any() and "close" in frame.columns:
        frame["adj_close"] = frame["adj_close"].fillna(frame["close"])

    frame = frame.dropna(subset=["adj_close"])
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    frame = frame.dropna(subset=["adj_close"])

    frame["daily_return"] = frame["adj_close"].pct_change(fill_method=None).fillna(0.0)
    frame["year"] = frame.index.year
    frame["quarter"] = "Q" + frame.index.quarter.astype(str)

    return frame


def _cache_path(request: MarketDataRequest, cache_dir: Path) -> Path:
    end = request.end or "latest"
    filename = f"{request.symbol.replace('^', '')}_{request.start}_{end}.csv"
    return cache_dir / filename


def load_market_data(
    request: Optional[MarketDataRequest] = None,
    provider: Optional[MarketDataProvider] = None,
    cache_dir: str | Path = "cache",
) -> pd.DataFrame:
    request = request or MarketDataRequest()
    provider = provider or YFinanceProvider()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(request, cache_dir)

    if cache_path.exists():
        cache_modified = pd.Timestamp(cache_path.stat().st_mtime, unit="s").date()
        if cache_modified >= pd.Timestamp.today().date():
            cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
            cached.index = pd.to_datetime(cached.index)
            return clean_market_data(cached)

    latest_error: Optional[Exception] = None
    try:
        fresh = clean_market_data(provider.fetch(request))
        fresh.to_csv(cache_path, index_label="date")
        return fresh
    except Exception as exc:
        latest_error = exc

    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
        cached.index = pd.to_datetime(cached.index)
        return clean_market_data(cached)

    raise RuntimeError(f"Unable to load market data via {provider.name}.") from latest_error


def load_sp500_data(
    start: str = DEFAULT_START,
    end: Optional[str] = None,
    provider: Optional[MarketDataProvider] = None,
    cache_dir: str | Path = "cache",
) -> pd.DataFrame:
    request = MarketDataRequest(symbol=DEFAULT_SYMBOL, start=start, end=end)
    return load_market_data(request=request, provider=provider, cache_dir=cache_dir)
