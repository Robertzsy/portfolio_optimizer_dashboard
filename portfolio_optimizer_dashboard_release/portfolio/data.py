from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


TRADING_DAYS = 252


@dataclass(frozen=True)
class PriceBundle:
    prices: pd.DataFrame
    source: str


def normalize_tickers(raw: str) -> list[str]:
    tickers = [item.strip().upper() for item in raw.replace("\n", ",").split(",")]
    return [ticker for ticker in tickers if ticker]


def download_adjusted_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    auto_adjust: bool = True,
) -> PriceBundle:
    if not tickers:
        raise ValueError("Please provide at least one ticker.")

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data.empty:
        raise ValueError("No price data returned. Check tickers, date range, or network access.")

    if isinstance(data.columns, pd.MultiIndex):
        field = "Close" if auto_adjust else "Adj Close"
        if field not in data.columns.get_level_values(0):
            field = "Close"
        prices = data[field].copy()
    else:
        prices = data[["Close"]].copy() if "Close" in data else data.copy()
        prices.columns = tickers[:1]

    prices = prices.sort_index().dropna(axis=1, how="all").ffill().dropna()
    prices = prices.loc[:, prices.nunique(dropna=True) > 1]

    if prices.empty:
        raise ValueError("Downloaded prices are empty after cleaning.")

    return PriceBundle(prices=prices, source="Yahoo Finance via yfinance")


def make_sample_prices(tickers: list[str] | None = None, seed: int = 7, years: int = 6) -> PriceBundle:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=TRADING_DAYS * years)
    sample_tickers = tickers or ["SPY", "QQQ", "IWM", "EFA", "TLT", "GLD"]
    n_assets = len(sample_tickers)
    base_mu = np.array([0.09, 0.12, 0.08, 0.075, 0.035, 0.055, 0.065, 0.10])
    base_vol = np.array([0.17, 0.23, 0.22, 0.19, 0.13, 0.16, 0.20, 0.24])
    annual_mu = np.resize(base_mu, n_assets)
    annual_vol = np.resize(base_vol, n_assets)

    market_loadings = rng.uniform(0.45, 0.85, size=n_assets)
    defensive = np.array([("TLT" in ticker or "BOND" in ticker) for ticker in sample_tickers], dtype=float)
    market_loadings = market_loadings * (1 - 0.55 * defensive)
    corr = 0.18 + np.outer(market_loadings, market_loadings) * 0.75
    corr = np.clip(corr, -0.35, 0.92)
    np.fill_diagonal(corr, 1.0)

    daily_cov = np.outer(annual_vol, annual_vol) * corr / TRADING_DAYS
    daily_mu = annual_mu / TRADING_DAYS
    returns = rng.multivariate_normal(daily_mu, daily_cov, size=len(dates))
    prices = 100 * pd.DataFrame(1 + returns, index=dates, columns=sample_tickers).cumprod()
    return PriceBundle(prices=prices, source="Synthetic sample data")


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")


def annualized_mean_cov(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    return returns.mean() * TRADING_DAYS, returns.cov() * TRADING_DAYS
