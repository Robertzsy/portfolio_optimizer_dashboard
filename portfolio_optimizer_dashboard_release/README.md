# Portfolio Optimizer & Backtesting Dashboard

An interactive Streamlit dashboard for portfolio construction, Markowitz mean-variance optimization, efficient frontier visualization, capital market line analysis, and rolling-window backtesting.

## Features

- Fetches historical adjusted prices with `yfinance`.
- Supports a built-in synthetic sample universe for offline demos.
- Includes a beginner guide, glossary, asset presets, and bilingual Chinese/English UI.
- Adds AI portfolio commentary with a DeepSeek preset and custom OpenAI-compatible provider settings.
- Remembers sidebar settings locally across app restarts.
- Computes annualized expected returns and covariance from daily returns.
- Solves constrained long-only optimization with `scipy.optimize.minimize`.
- Produces maximum Sharpe and minimum variance portfolios.
- Draws random feasible portfolios, the efficient frontier, and the capital market line.
- Runs rolling-window backtests against equal-weight and optional benchmark portfolios.
- Includes a compact Black-Litterman absolute-view module for investor return views.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If network access is unavailable, keep **Use sample data** enabled in the sidebar.

## Beginner Workflow

1. Keep **Use sample data** enabled for the first run.
2. Open **Quick Start Guide** at the top of the page.
3. Choose an **Asset preset** in the sidebar, then review the tickers.
4. Compare the maximum Sharpe and minimum variance portfolios on the optimizer tab.
5. Move to the backtest tab to compare optimized, equal-weight, and benchmark performance.

## AI Analysis

Use the standalone **AI Analysis** section above the optimizer/backtest tabs after the page has loaded. In the sidebar, choose:

- **DeepSeek**: prefilled with `https://api.deepseek.com` and `deepseek-chat`.
- **Custom**: enter any OpenAI-compatible Base URL, model name, and API key.

The app sends a compact snapshot of the current assets, optimized weights, metrics, and rolling backtest summary. API keys are only held in the current Streamlit session and are not written to project files.

Advanced settings allow you to set temperature, output tokens up to `100000`, and a custom system prompt. The default system prompt positions the model as a professional market analyst and portfolio strategist.

## Local Memory

The dashboard automatically saves sidebar settings to `.dashboard_settings.json` in the project folder and restores them on the next launch. This includes asset presets, tickers, model assumptions, backtest settings, AI provider settings, and separate Chinese/English system prompts.

API keys are not saved by default. Enable **Remember API Key** only if you want the key stored locally on this machine.

## Suggested Live Tickers

- US ETFs: `SPY, QQQ, IWM, EFA, TLT, GLD`
- Hong Kong examples: `2800.HK, 3033.HK, 3067.HK, 2822.HK`
- Benchmark examples: `^GSPC`, `^IXIC`, `2800.HK`

## Methodology

The optimizer uses the classic Markowitz setup:

- Expected return: annualized mean of daily returns.
- Risk: annualized covariance matrix.
- Constraint: weights sum to 1.
- Bounds: long-only weights with a configurable maximum single-asset cap.
- Maximum Sharpe objective: maximize `(portfolio return - risk-free rate) / portfolio volatility`.
- Minimum variance objective: minimize portfolio variance.
- Efficient frontier: minimize variance for a grid of target returns.

The backtest recalculates the optimized portfolio on each rebalance date using only the trailing lookback window, then holds those weights until the next rebalance. This avoids look-ahead bias in the allocation step.

## Notes

This is an educational and research dashboard, not investment advice. Real portfolio work should include transaction costs, taxes, liquidity limits, corporate action checks, position-level constraints, and independent data validation.
