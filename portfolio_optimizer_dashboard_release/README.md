# Portfolio Optimizer & Backtesting Dashboard

An interactive Streamlit dashboard for portfolio construction, Markowitz mean-variance optimization, efficient frontier visualization, capital market line analysis, and rolling-window backtesting.

## Why This Project Matters

Portfolio construction is a core task in wealth management, asset allocation, and investment advisory work. In practice, investors rarely care about a single asset in isolation; they care about how multiple assets work together, how much risk they take, and whether the return they receive is reasonable for that risk. This project turns the abstract Markowitz mean-variance framework into an interactive decision-support tool that connects theory, data, optimization, backtesting, and AI-assisted interpretation.

The dashboard helps users move from "which assets should I buy?" to a more professional question: "what portfolio mix best matches a given risk-return objective, and how stable was that mix historically?"

## Expected Outcomes

With this project, users can:

- Understand the risk-return tradeoff among selected stocks or ETFs.
- Compare maximum Sharpe, minimum variance, equal-weight, and benchmark portfolios.
- Visualize the efficient frontier and capital market line interactively.
- Run rolling-window backtests to see whether an optimized allocation remains robust over time.
- Test how constraints such as maximum single-asset weight change the allocation result.
- Add investor views through a simplified Black-Litterman workflow.
- Use AI analysis to summarize allocation logic, concentration risk, drawdowns, and improvement ideas.

## Practical Value

This project has practical value for several real-world scenarios:

- **Wealth management demonstrations**: explain portfolio allocation to clients with charts instead of static formulas.
- **Investment research**: quickly compare candidate ETF universes and identify concentration or volatility issues.
- **Risk communication**: show that higher expected return often comes with higher volatility and drawdown risk.
- **Strategy evaluation**: use rolling backtests to avoid relying only on one-period optimized weights.
- **Education and interviews**: demonstrate understanding of Markowitz optimization, Sharpe ratio, efficient frontier, backtesting, and AI-assisted financial analysis in one coherent application.
- **Extensibility**: the project can be expanded with transaction costs, tax assumptions, sector constraints, factor exposure, alternative optimizers, or more AI providers.

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
