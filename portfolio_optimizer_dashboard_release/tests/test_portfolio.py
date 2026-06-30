import numpy as np
import pandas as pd

from portfolio.backtest import rolling_backtest
from portfolio.data import annualized_mean_cov, daily_returns, make_sample_prices
from portfolio.optimization import efficient_frontier, optimize_max_sharpe, optimize_min_variance


def test_optimizers_return_valid_weights():
    prices = make_sample_prices(seed=11, years=3).prices
    returns = daily_returns(prices)
    mu, cov = annualized_mean_cov(returns)

    max_sharpe = optimize_max_sharpe(mu, cov, risk_free_rate=0.02, max_weight=0.5)
    min_var = optimize_min_variance(mu, cov, risk_free_rate=0.02, max_weight=0.5)

    assert np.isclose(max_sharpe.weights.sum(), 1)
    assert np.isclose(min_var.weights.sum(), 1)
    assert (max_sharpe.weights >= -1e-8).all()
    assert (min_var.weights <= 0.500001).all()


def test_frontier_and_backtest_have_outputs():
    prices = make_sample_prices(seed=19, years=4).prices
    returns = daily_returns(prices)
    mu, cov = annualized_mean_cov(returns)

    frontier = efficient_frontier(mu, cov, risk_free_rate=0.02, max_weight=0.6, points=20)
    result = rolling_backtest(returns, "Maximum Sharpe", risk_free_rate=0.02, lookback_days=126, max_weight=0.6)

    assert isinstance(frontier, pd.DataFrame)
    assert not frontier.empty
    assert not result.equity.empty
    assert "Optimized" in result.equity.columns


def test_sample_prices_follow_requested_tickers():
    tickers = ["2800.HK", "3033.HK", "3067.HK", "2822.HK"]
    prices = make_sample_prices(tickers=tickers, seed=23, years=2).prices

    assert list(prices.columns) == tickers
    assert prices.shape[0] > 100
