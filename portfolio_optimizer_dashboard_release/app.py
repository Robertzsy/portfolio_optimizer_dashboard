from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from portfolio.backtest import rolling_backtest
from portfolio.ai_analysis import AIProviderConfig, build_portfolio_prompt, default_system_prompt, generate_ai_analysis
from portfolio.black_litterman import apply_absolute_views
from portfolio.data import annualized_mean_cov, daily_returns, download_adjusted_prices, make_sample_prices, normalize_tickers
from portfolio.optimization import (
    efficient_frontier,
    optimize_max_sharpe,
    optimize_min_variance,
    random_portfolios,
)


I18N = {
    "中文": {
        "page_title": "投资组合优化与回测仪表盘",
        "caption": "马科维茨均值-方差优化、有效前沿、资本市场线、夏普比率最大化和滚动窗口回测。",
        "language": "语言 / Language",
        "universe": "资产池",
        "use_sample": "使用样例数据",
        "tickers": "股票 / ETF",
        "start_date": "开始日期",
        "end_date": "结束日期",
        "assumptions": "模型假设",
        "risk_free": "无风险利率",
        "max_weight": "单一资产最大权重",
        "random_portfolios": "随机组合数量",
        "backtest": "回测",
        "rolling_strategy": "滚动优化策略",
        "lookback": "回看窗口，交易日",
        "rebalance": "调仓频率",
        "benchmark": "基准指数",
        "investor_views": "投资者观点",
        "use_views": "使用 Black-Litterman 观点",
        "loading": "正在加载价格并计算组合...",
        "bl_views": "Black-Litterman 绝对收益观点",
        "annual_view": "年度收益观点",
        "view_confidence": "观点置信度",
        "prior_assets": "先验资产数",
        "source": "数据来源",
        "assets": "资产数量",
        "max_sharpe": "最大夏普",
        "min_volatility": "最低波动率",
        "optimizer_tab": "组合优化",
        "backtest_tab": "回测",
        "prices_tab": "价格与收益",
        "inputs_tab": "模型输入",
        "max_sharpe_portfolio": "最大夏普组合",
        "min_variance_portfolio": "最小方差组合",
        "weight": "权重",
        "growth_title": "1 美元增长曲线",
        "rolling_weights": "滚动优化权重",
        "normalized_price": "标准化价格指数",
        "expected_return": "预期收益",
        "volatility": "波动率",
        "return": "收益率",
        "sharpe": "夏普比率",
        "annualized_volatility": "年化波动率",
        "annualized_return": "年化收益率",
        "random_trace": "随机组合",
        "frontier_trace": "有效前沿",
        "asset_trace": "单项资产",
        "cml": "资本市场线",
        "strategy_max_sharpe": "最大夏普",
        "strategy_min_variance": "最小方差",
        "monthly": "每月",
        "quarterly": "每季度",
        "benchmark_failed": "基准下载失败",
        "fallback_info": "请先尝试使用样例数据。若使用真实数据，请检查 ticker、日期范围和网络连接。",
        "source_sample": "样例模拟数据",
        "source_yahoo": "Yahoo Finance via yfinance",
        "beginner_mode": "显示新手引导",
        "getting_started": "新手快速上手",
        "step_1_title": "1. 先选资产",
        "step_1_body": "新手建议先保留样例数据，熟悉页面后再输入真实 ticker。",
        "step_2_title": "2. 设置约束",
        "step_2_body": "无风险利率影响夏普比率；单一资产最大权重用于避免组合过度集中。",
        "step_3_title": "3. 看优化结果",
        "step_3_body": "左侧有效前沿展示风险与收益的权衡，星标是最大夏普组合，菱形是最小方差组合。",
        "step_4_title": "4. 做滚动回测",
        "step_4_body": "回测会用过去窗口估计参数，然后持有到下一次调仓，适合观察策略是否稳定。",
        "preset": "资产预设",
        "preset_global": "全球 ETF 示例",
        "preset_us_growth": "美股成长/防御",
        "preset_hk": "港股 ETF 示例",
        "preset_custom": "自定义",
        "workflow_tip": "建议流程",
        "workflow_body": "先用样例数据理解图表，再切换真实数据；先看最大夏普和最小方差的权重差异，再进入回测页比较长期表现。",
        "glossary": "术语速查",
        "glossary_return": "预期收益：用历史日收益年化得到的收益估计。",
        "glossary_vol": "波动率：组合年化风险，越高代表收益波动越大。",
        "glossary_sharpe": "夏普比率：每承担一单位风险获得的超额收益。",
        "glossary_frontier": "有效前沿：在给定风险下收益最高、或给定收益下风险最低的组合集合。",
        "glossary_cml": "资本市场线：引入无风险资产后，从无风险利率连接到最大夏普组合的线。",
        "guided_note": "页面中所有金额表现均以指数化增长展示，便于比较策略相对表现。",
        "quick_summary": "当前设置概览",
        "date_range": "数据区间",
        "asset_list": "资产列表",
        "tips": "提示",
        "sample_tip": "样例数据适合课堂演示，不依赖网络。",
        "live_tip": "真实数据来自 yfinance，ticker 写法需符合 Yahoo Finance。",
        "optimizer_hint": "点图例可以隐藏/显示曲线；将鼠标放到点上可查看收益、波动率和夏普比率。",
        "backtest_hint": "回测结果更关注长期稳定性，而不是单次优化的漂亮权重。",
        "inputs_hint": "模型输入页用于检查收益率和协方差是否合理。",
        "help_tickers": "用逗号分隔，例如 SPY, QQQ, TLT。港股通常带 .HK。",
        "help_rf": "可以填当前短债或货币基金收益率的近似值。",
        "help_max_weight": "值越低，组合越分散；值越高，优化器越容易集中到少数资产。",
        "help_random": "数量越大，散点更细，但计算也更慢。",
        "help_lookback": "较短窗口更敏感，较长窗口更平滑。",
        "help_views": "开启后，可手动调整对某些资产年化收益的主观看法。",
        "ai_tab": "AI 分析",
        "ai_settings": "AI 分析设置",
        "ai_provider": "AI 厂商",
        "ai_provider_deepseek": "DeepSeek",
        "ai_provider_custom": "自定义",
        "ai_custom_name": "自定义厂商名称",
        "ai_sdk": "SDK 类型",
        "ai_base_url": "Base URL",
        "ai_api_key": "API Key",
        "ai_model": "模型名称",
        "ai_temperature": "温度",
        "ai_max_tokens": "最大输出 tokens",
        "ai_config_tip": "API Key 只保存在当前页面会话中，不会写入项目文件。",
        "ai_deepseek_note": "DeepSeek 兼容 OpenAI SDK，默认 Base URL 为 https://api.deepseek.com。",
        "ai_generate": "生成 AI 分析",
        "ai_generating": "AI 正在分析当前组合...",
        "ai_result": "AI 分析结果",
        "ai_snapshot": "发送给 AI 的数据摘要",
        "ai_need_key": "请先在侧边栏填写 API Key。",
        "ai_not_advice": "AI 输出仅用于学习和研究辅助，不构成投资建议。",
        "ai_intro": "这里会把当前资产、优化权重、关键风险收益指标和滚动回测摘要发送给你选择的 AI 模型，让它生成结构化解读。",
        "ai_custom_help": "自定义厂商需提供 OpenAI-compatible Base URL、模型名和 API Key。",
        "ai_advanced": "高级设置",
        "ai_system_prompt": "System Prompt",
        "ai_system_prompt_help": "用于设定 AI 的角色、分析边界和输出风格。留空时使用默认专业市场分析师提示词。",
        "ai_section_help": "AI 分析是独立板块，基于当前页面已计算出的组合优化和回测摘要生成专业解读。",
    },
    "English": {
        "page_title": "Portfolio Optimizer & Backtesting Dashboard",
        "caption": "Markowitz mean-variance optimization, efficient frontier, capital market line, Sharpe maximization, and rolling-window backtesting.",
        "language": "Language / 语言",
        "universe": "Universe",
        "use_sample": "Use sample data",
        "tickers": "Stocks / ETFs",
        "start_date": "Start date",
        "end_date": "End date",
        "assumptions": "Assumptions",
        "risk_free": "Risk-free rate",
        "max_weight": "Max single asset weight",
        "random_portfolios": "Random portfolios",
        "backtest": "Backtest",
        "rolling_strategy": "Rolling strategy",
        "lookback": "Lookback window, trading days",
        "rebalance": "Rebalance frequency",
        "benchmark": "Benchmark",
        "investor_views": "Investor Views",
        "use_views": "Use Black-Litterman views",
        "loading": "Loading prices and calculating portfolios...",
        "bl_views": "Black-Litterman Absolute Views",
        "annual_view": "Annual View",
        "view_confidence": "View confidence",
        "prior_assets": "Prior assets",
        "source": "Data source",
        "assets": "Assets",
        "max_sharpe": "Max Sharpe",
        "min_volatility": "Min Volatility",
        "optimizer_tab": "Optimizer",
        "backtest_tab": "Backtest",
        "prices_tab": "Prices & Returns",
        "inputs_tab": "Model Inputs",
        "max_sharpe_portfolio": "Maximum Sharpe Portfolio",
        "min_variance_portfolio": "Minimum Variance Portfolio",
        "weight": "Weight",
        "growth_title": "Growth of $1",
        "rolling_weights": "Rolling Optimized Weights",
        "normalized_price": "Normalized Price Index",
        "expected_return": "Expected Return",
        "volatility": "Volatility",
        "return": "Return",
        "sharpe": "Sharpe",
        "annualized_volatility": "Annualized volatility",
        "annualized_return": "Annualized return",
        "random_trace": "Random portfolios",
        "frontier_trace": "Efficient frontier",
        "asset_trace": "Assets",
        "cml": "Capital market line",
        "strategy_max_sharpe": "Maximum Sharpe",
        "strategy_min_variance": "Minimum Variance",
        "monthly": "Monthly",
        "quarterly": "Quarterly",
        "benchmark_failed": "Benchmark download failed",
        "fallback_info": "Try the sample data toggle first. For live data, verify the ticker symbols and network access.",
        "source_sample": "Synthetic sample data",
        "source_yahoo": "Yahoo Finance via yfinance",
        "beginner_mode": "Show beginner guide",
        "getting_started": "Quick Start Guide",
        "step_1_title": "1. Choose assets",
        "step_1_body": "Beginners can keep sample data on first, then switch to live tickers later.",
        "step_2_title": "2. Set constraints",
        "step_2_body": "The risk-free rate affects Sharpe ratios; the max asset weight prevents over-concentration.",
        "step_3_title": "3. Read the optimizer",
        "step_3_body": "The frontier shows the risk-return tradeoff. The star is max Sharpe; the diamond is minimum variance.",
        "step_4_title": "4. Run the backtest",
        "step_4_body": "The backtest estimates parameters from the trailing window, then holds until the next rebalance.",
        "preset": "Asset preset",
        "preset_global": "Global ETF sample",
        "preset_us_growth": "US growth/defensive",
        "preset_hk": "Hong Kong ETF sample",
        "preset_custom": "Custom",
        "workflow_tip": "Suggested workflow",
        "workflow_body": "Start with sample data, then switch to live data. Compare max Sharpe vs. minimum variance weights before checking long-term backtests.",
        "glossary": "Glossary",
        "glossary_return": "Expected return: annualized estimate from historical daily returns.",
        "glossary_vol": "Volatility: annualized risk; higher values mean larger return swings.",
        "glossary_sharpe": "Sharpe ratio: excess return earned per unit of risk.",
        "glossary_frontier": "Efficient frontier: portfolios with the best return for a given risk, or lowest risk for a given return.",
        "glossary_cml": "Capital market line: the line from the risk-free rate to the max Sharpe portfolio.",
        "guided_note": "Performance is shown as indexed growth for easier strategy comparison.",
        "quick_summary": "Current Setup",
        "date_range": "Date range",
        "asset_list": "Asset list",
        "tips": "Tips",
        "sample_tip": "Sample data is good for demos and does not require network access.",
        "live_tip": "Live data comes from yfinance; tickers must match Yahoo Finance symbols.",
        "optimizer_hint": "Click legend items to hide/show traces; hover points to inspect return, volatility, and Sharpe.",
        "backtest_hint": "Backtests are about long-run stability, not just attractive one-shot weights.",
        "inputs_hint": "Use model inputs to check whether returns and covariance look reasonable.",
        "help_tickers": "Separate tickers with commas, e.g. SPY, QQQ, TLT. Hong Kong tickers usually end with .HK.",
        "help_rf": "A short-term Treasury or money-market yield is a common rough input.",
        "help_max_weight": "Lower values diversify more; higher values allow the optimizer to concentrate.",
        "help_random": "More points make the cloud denser, but calculation takes longer.",
        "help_lookback": "Shorter windows react faster; longer windows are smoother.",
        "help_views": "When enabled, you can edit subjective annual return views.",
        "ai_tab": "AI Analysis",
        "ai_settings": "AI Analysis Settings",
        "ai_provider": "AI provider",
        "ai_provider_deepseek": "DeepSeek",
        "ai_provider_custom": "Custom",
        "ai_custom_name": "Custom provider name",
        "ai_sdk": "SDK type",
        "ai_base_url": "Base URL",
        "ai_api_key": "API Key",
        "ai_model": "Model name",
        "ai_temperature": "Temperature",
        "ai_max_tokens": "Max output tokens",
        "ai_config_tip": "The API key stays in the current page session and is not written to project files.",
        "ai_deepseek_note": "DeepSeek is compatible with the OpenAI SDK. The default Base URL is https://api.deepseek.com.",
        "ai_generate": "Generate AI analysis",
        "ai_generating": "AI is analyzing the current portfolio...",
        "ai_result": "AI Analysis Result",
        "ai_snapshot": "Data summary sent to AI",
        "ai_need_key": "Please enter an API key in the sidebar first.",
        "ai_not_advice": "AI output is for learning and research support only, not investment advice.",
        "ai_intro": "This sends the current assets, optimized weights, key risk-return metrics, and rolling backtest summary to your selected AI model for structured interpretation.",
        "ai_custom_help": "Custom providers need an OpenAI-compatible Base URL, model name, and API key.",
        "ai_advanced": "Advanced settings",
        "ai_system_prompt": "System Prompt",
        "ai_system_prompt_help": "Sets the AI role, analysis boundaries, and output style. Leave unchanged to use the default professional market analyst prompt.",
        "ai_section_help": "AI analysis is a standalone section based on the currently computed optimization and backtest summary.",
    },
}


st.set_page_config(
    page_title="Portfolio Optimizer | 投资组合优化",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


ASSET_PRESETS = {
    "global": "SPY, QQQ, IWM, EFA, TLT, GLD",
    "us_growth": "SPY, QQQ, XLV, XLU, TLT, GLD",
    "hk": "2800.HK, 3033.HK, 3067.HK, 2822.HK",
    "custom": "SPY, QQQ, IWM, EFA, TLT, GLD",
}

AI_PROVIDER_DEFAULTS = {
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "custom": {
        "name": "Custom OpenAI-compatible",
        "base_url": "https://api.example.com/v1",
        "model": "your-model-name",
    },
}

SETTINGS_FILE = Path(__file__).with_name(".dashboard_settings.json")
DATE_SETTING_KEYS = {"start_date", "end_date"}
PERSISTED_WIDGET_KEYS = [
    "language",
    "beginner_mode",
    "use_sample",
    "asset_preset_key",
    "ticker_input",
    "start_date",
    "end_date",
    "risk_free_rate",
    "max_weight",
    "random_count",
    "strategy",
    "lookback_days",
    "rebalance_frequency",
    "benchmark_ticker",
    "use_views",
    "ai_provider_key",
    "ai_provider_name",
    "ai_sdk_type",
    "ai_base_url",
    "ai_model",
    "ai_api_key",
    "remember_api_key",
    "ai_temperature",
    "ai_max_tokens",
    "ai_system_prompt_cn",
    "ai_system_prompt_en",
]


def _serialize_setting(value):
    if isinstance(value, date):
        return value.isoformat()
    return value


def _deserialize_setting(key: str, value):
    if key in DATE_SETTING_KEYS and isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return value
    return value


def load_persisted_settings() -> None:
    if st.session_state.get("_settings_loaded"):
        return
    st.session_state["_settings_loaded"] = True
    if not SETTINGS_FILE.exists():
        return
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    st.session_state["_persisted_settings"] = data
    for key, value in data.items():
        if key in PERSISTED_WIDGET_KEYS and key not in st.session_state:
            st.session_state[key] = _deserialize_setting(key, value)


def save_persisted_settings() -> None:
    data = dict(st.session_state.get("_persisted_settings", {}))
    for key in PERSISTED_WIDGET_KEYS:
        if key not in st.session_state:
            continue
        if key == "ai_api_key" and not st.session_state.get("remember_api_key", False):
            continue
        data[key] = _serialize_setting(st.session_state[key])
    if not st.session_state.get("remember_api_key", False):
        data.pop("ai_api_key", None)
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    st.session_state["_persisted_settings"] = data


def normalize_asset_preset_key(value: str | None) -> str:
    label_aliases = {
        "全球 ETF 示例": "global",
        "Global ETF sample": "global",
        "美股成长/防御": "us_growth",
        "US growth/defensive": "us_growth",
        "港股 ETF 示例": "hk",
        "Hong Kong ETF sample": "hk",
        "自定义": "custom",
        "Custom": "custom",
    }
    if value in ASSET_PRESETS:
        return value
    return label_aliases.get(value or "", "global")


def normalize_ai_provider_key(value: str | None) -> str:
    label_aliases = {
        "DeepSeek": "deepseek",
        "自定义": "custom",
        "Custom": "custom",
    }
    if value in AI_PROVIDER_DEFAULTS:
        return value
    return label_aliases.get(value or "", "deepseek")


def initialize_widget_state(language: str) -> None:
    persisted = st.session_state.get("_persisted_settings", {})
    today = date.today()
    default_start = today - timedelta(days=365 * 6)
    st.session_state.setdefault("language", persisted.get("language", "中文"))
    st.session_state.setdefault("beginner_mode", persisted.get("beginner_mode", True))
    st.session_state.setdefault("use_sample", persisted.get("use_sample", True))

    asset_preset_key = normalize_asset_preset_key(st.session_state.get("asset_preset_key", "global"))
    st.session_state["asset_preset_key"] = asset_preset_key
    st.session_state.setdefault("ticker_input", persisted.get("ticker_input", ASSET_PRESETS[asset_preset_key]))
    st.session_state.setdefault("start_date", _deserialize_setting("start_date", persisted.get("start_date", default_start)))
    st.session_state.setdefault("end_date", _deserialize_setting("end_date", persisted.get("end_date", today)))
    st.session_state.setdefault("risk_free_rate", persisted.get("risk_free_rate", 0.03))
    st.session_state.setdefault("max_weight", persisted.get("max_weight", 0.45))
    st.session_state.setdefault("random_count", persisted.get("random_count", 2500))
    st.session_state.setdefault("strategy", persisted.get("strategy", "Maximum Sharpe"))
    st.session_state.setdefault("lookback_days", persisted.get("lookback_days", 252))
    st.session_state.setdefault("rebalance_frequency", persisted.get("rebalance_frequency", "ME"))
    st.session_state.setdefault("benchmark_ticker", persisted.get("benchmark_ticker", "^GSPC"))
    st.session_state.setdefault("use_views", persisted.get("use_views", False))

    ai_provider_key = normalize_ai_provider_key(st.session_state.get("ai_provider_key", "deepseek"))
    st.session_state["ai_provider_key"] = ai_provider_key
    provider_defaults = AI_PROVIDER_DEFAULTS[ai_provider_key]
    st.session_state.setdefault("ai_provider_name", persisted.get("ai_provider_name", provider_defaults["name"]))
    st.session_state.setdefault("ai_sdk_type", persisted.get("ai_sdk_type", "OpenAI-compatible"))
    st.session_state.setdefault("ai_base_url", persisted.get("ai_base_url", provider_defaults["base_url"]))
    st.session_state.setdefault("ai_model", persisted.get("ai_model", provider_defaults["model"]))
    st.session_state.setdefault("remember_api_key", persisted.get("remember_api_key", False))
    st.session_state.setdefault("ai_api_key", persisted.get("ai_api_key", ""))
    st.session_state.setdefault("ai_temperature", persisted.get("ai_temperature", 1.0))
    st.session_state.setdefault("ai_max_tokens", persisted.get("ai_max_tokens", 4000))

    old_prompt = st.session_state.pop("ai_system_prompt", None)
    st.session_state.setdefault("ai_system_prompt_cn", old_prompt or persisted.get("ai_system_prompt_cn", default_system_prompt("中文")))
    st.session_state.setdefault("ai_system_prompt_en", persisted.get("ai_system_prompt_en", default_system_prompt("English")))


def sync_asset_preset() -> None:
    preset_key = normalize_asset_preset_key(st.session_state.get("asset_preset_key", "global"))
    st.session_state["asset_preset_key"] = preset_key
    st.session_state["ticker_input"] = ASSET_PRESETS[preset_key]


def sync_ai_provider_defaults() -> None:
    provider_key = normalize_ai_provider_key(st.session_state.get("ai_provider_key", "deepseek"))
    st.session_state["ai_provider_key"] = provider_key
    defaults = AI_PROVIDER_DEFAULTS[provider_key]
    st.session_state["ai_provider_name"] = defaults["name"]
    st.session_state["ai_base_url"] = defaults["base_url"]
    st.session_state["ai_model"] = defaults["model"]


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-border: #d9e2ec;
            --app-muted: #5f6c7b;
            --app-panel: #f7fafc;
            --app-accent: #2563eb;
            --app-accent-soft: #e8f0ff;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #eef4f8 100%);
            border-right: 1px solid var(--app-border);
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--app-border);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stMetric"] label {
            color: var(--app-muted);
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--app-border);
            border-radius: 8px;
            background: #ffffff;
        }

        .guide-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 10px 0 6px;
        }

        .guide-card {
            min-height: 132px;
            border: 1px solid var(--app-border);
            border-radius: 8px;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            padding: 14px;
        }

        .guide-card strong {
            color: #0f172a;
            display: block;
            margin-bottom: 8px;
            font-size: 0.96rem;
        }

        .guide-card span {
            color: var(--app-muted);
            line-height: 1.45;
            font-size: 0.9rem;
        }

        .note-panel {
            border-left: 4px solid var(--app-accent);
            background: var(--app-accent-soft);
            border-radius: 8px;
            padding: 12px 14px;
            color: #1e3a8a;
            margin: 10px 0 16px;
        }

        .glossary-list {
            margin: 8px 0 0;
            padding-left: 18px;
            color: var(--app-muted);
            line-height: 1.55;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 10px 16px;
        }

        @media (max-width: 900px) {
            .guide-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 560px) {
            .guide-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def render_beginner_guide(labels: dict[str, str]) -> None:
    with st.expander(labels["getting_started"], expanded=True):
        cards = [
            ("step_1_title", "step_1_body"),
            ("step_2_title", "step_2_body"),
            ("step_3_title", "step_3_body"),
            ("step_4_title", "step_4_body"),
        ]
        html = '<div class="guide-grid">'
        for title_key, body_key in cards:
            html += f'<div class="guide-card"><strong>{labels[title_key]}</strong><span>{labels[body_key]}</span></div>'
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
        st.markdown(f'<div class="note-panel"><strong>{labels["workflow_tip"]}</strong><br>{labels["workflow_body"]}</div>', unsafe_allow_html=True)

        with st.expander(labels["glossary"], expanded=False):
            glossary = [
                labels["glossary_return"],
                labels["glossary_vol"],
                labels["glossary_sharpe"],
                labels["glossary_frontier"],
                labels["glossary_cml"],
            ]
            st.markdown("<ul class='glossary-list'>" + "".join(f"<li>{item}</li>" for item in glossary) + "</ul>", unsafe_allow_html=True)
            st.caption(labels["guided_note"])


def styled_line_chart(frame: pd.DataFrame, title: str, labels: dict[str, str]):
    fig = px.line(frame, title=title, template="plotly_white")
    fig.update_layout(
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text=labels["return"])
    return fig


def compact_weights(weights: pd.Series, limit: int = 10) -> dict[str, str]:
    selected = weights.sort_values(ascending=False)
    selected = selected[selected > 0.0005].head(limit)
    return {asset: pct(weight) for asset, weight in selected.items()}


def build_ai_snapshot(
    prices: pd.DataFrame,
    expected_returns: pd.Series,
    max_sharpe,
    min_var,
    backtest_metrics: pd.DataFrame | None,
    risk_free: float,
    lookback_days: int,
    rebalance_frequency: str,
) -> dict:
    snapshot = {
        "assets": list(prices.columns),
        "observations": int(len(prices)),
        "date_start": str(prices.index.min().date()),
        "date_end": str(prices.index.max().date()),
        "risk_free_rate": pct(risk_free),
        "lookback_trading_days": int(lookback_days),
        "rebalance_frequency": rebalance_frequency,
        "expected_returns": {asset: pct(value) for asset, value in expected_returns.items()},
        "max_sharpe": {
            "expected_return": pct(max_sharpe.expected_return),
            "volatility": pct(max_sharpe.volatility),
            "sharpe": round(float(max_sharpe.sharpe), 3),
            "weights": compact_weights(max_sharpe.weights),
        },
        "minimum_variance": {
            "expected_return": pct(min_var.expected_return),
            "volatility": pct(min_var.volatility),
            "sharpe": round(float(min_var.sharpe), 3),
            "weights": compact_weights(min_var.weights),
        },
    }
    if backtest_metrics is not None and not backtest_metrics.empty:
        snapshot["backtest_metrics"] = backtest_metrics.map(pct).to_dict(orient="index")
    return snapshot


def translate_source(source: str, labels: dict[str, str]) -> str:
    if source == "Synthetic sample data":
        return labels["source_sample"]
    if source == "Yahoo Finance via yfinance":
        return labels["source_yahoo"]
    return source


@st.cache_data(show_spinner=False)
def load_prices(tickers: tuple[str, ...], start: str, end: str, use_sample: bool):
    if use_sample:
        return make_sample_prices(list(tickers))
    return download_adjusted_prices(list(tickers), start=start, end=end)


@st.cache_data(show_spinner=False)
def load_benchmark(ticker: str, start: str, end: str):
    bundle = download_adjusted_prices([ticker], start=start, end=end)
    returns = daily_returns(bundle.prices)
    return returns.iloc[:, 0].rename(ticker)


def frontier_chart(
    frontier: pd.DataFrame,
    randoms: pd.DataFrame,
    assets: pd.DataFrame,
    max_sharpe,
    min_var,
    risk_free: float,
    labels: dict[str, str],
):
    fig = go.Figure(layout=dict(template="plotly_white"))
    fig.add_trace(
        go.Scatter(
            x=randoms["Volatility"],
            y=randoms["Return"],
            mode="markers",
            marker=dict(size=5, color=randoms["Sharpe"], colorscale="Viridis", showscale=True, colorbar=dict(title=labels["sharpe"])),
            name=labels["random_trace"],
            hovertemplate=f"{labels['volatility']} %{{x:.2%}}<br>{labels['return']} %{{y:.2%}}<br>{labels['sharpe']} %{{marker.color:.2f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frontier["Volatility"],
            y=frontier["Return"],
            mode="lines",
            line=dict(width=4, color="#e4572e"),
            name=labels["frontier_trace"],
            hovertemplate=f"{labels['volatility']} %{{x:.2%}}<br>{labels['return']} %{{y:.2%}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=assets["Volatility"],
            y=assets["Return"],
            text=assets.index,
            mode="markers+text",
            textposition="top center",
            marker=dict(size=10, color="#1f77b4"),
            name=labels["asset_trace"],
            hovertemplate=f"%{{text}}<br>{labels['volatility']} %{{x:.2%}}<br>{labels['return']} %{{y:.2%}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[max_sharpe.volatility],
            y=[max_sharpe.expected_return],
            mode="markers",
            marker=dict(size=16, symbol="star", color="#2ca02c"),
            name=labels["max_sharpe"],
            hovertemplate=f"{labels['max_sharpe']}<br>{labels['volatility']} %{{x:.2%}}<br>{labels['return']} %{{y:.2%}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_var.volatility],
            y=[min_var.expected_return],
            mode="markers",
            marker=dict(size=14, symbol="diamond", color="#9467bd"),
            name=labels["min_variance_portfolio"],
            hovertemplate=f"{labels['min_variance_portfolio']}<br>{labels['volatility']} %{{x:.2%}}<br>{labels['return']} %{{y:.2%}}<extra></extra>",
        )
    )
    cml_x = [0, max(frontier["Volatility"].max(), randoms["Volatility"].max()) * 1.05]
    cml_y = [risk_free, risk_free + max_sharpe.sharpe * cml_x[1]]
    fig.add_trace(
        go.Scatter(
            x=cml_x,
            y=cml_y,
            mode="lines",
            line=dict(width=2, dash="dash", color="#111111"),
            name=labels["cml"],
        )
    )
    fig.update_layout(
        height=620,
        xaxis_title=labels["annualized_volatility"],
        yaxis_title=labels["annualized_return"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=45, b=10),
        hovermode="closest",
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig

load_persisted_settings()
apply_theme()

with st.sidebar:
    language = st.radio(I18N["中文"]["language"], ["中文", "English"], horizontal=True, key="language")
    t = I18N[language]
    initialize_widget_state(language)
    beginner_mode = st.toggle(t["beginner_mode"], key="beginner_mode")

st.title(t["page_title"])
st.caption(t["caption"])

if beginner_mode:
    render_beginner_guide(t)

with st.sidebar:
    st.header(t["universe"])
    use_sample = st.toggle(t["use_sample"], key="use_sample")
    preset_labels = {
        "global": t["preset_global"],
        "us_growth": t["preset_us_growth"],
        "hk": t["preset_hk"],
        "custom": t["preset_custom"],
    }
    st.selectbox(
        t["preset"],
        list(ASSET_PRESETS),
        key="asset_preset_key",
        format_func=lambda key: preset_labels[key],
        on_change=sync_asset_preset,
    )
    raw_tickers = st.text_area(t["tickers"], key="ticker_input", height=86, help=t["help_tickers"])
    tickers = normalize_tickers(raw_tickers)

    start = st.date_input(t["start_date"], key="start_date")
    end = st.date_input(t["end_date"], key="end_date")

    st.header(t["assumptions"])
    risk_free = st.number_input(t["risk_free"], min_value=0.0, max_value=0.2, step=0.005, format="%.3f", help=t["help_rf"], key="risk_free_rate")
    max_weight = st.slider(t["max_weight"], min_value=0.1, max_value=1.0, step=0.05, help=t["help_max_weight"], key="max_weight")
    random_count = st.slider(t["random_portfolios"], min_value=500, max_value=8000, step=500, help=t["help_random"], key="random_count")

    st.header(t["backtest"])
    strategy_labels = {
        "Maximum Sharpe": t["strategy_max_sharpe"],
        "Minimum Variance": t["strategy_min_variance"],
    }
    strategy = st.selectbox(t["rolling_strategy"], list(strategy_labels), format_func=lambda key: strategy_labels[key], key="strategy")
    lookback_days = st.slider(t["lookback"], min_value=126, max_value=756, step=21, help=t["help_lookback"], key="lookback_days")
    rebalance_labels = {"ME": t["monthly"], "QE": t["quarterly"]}
    rebalance_frequency = st.selectbox(t["rebalance"], list(rebalance_labels), format_func=lambda key: rebalance_labels[key], key="rebalance_frequency")
    benchmark_ticker = st.text_input(t["benchmark"], key="benchmark_ticker")

    st.header(t["investor_views"])
    use_views = st.toggle(t["use_views"], help=t["help_views"], key="use_views")

    with st.expander(t["tips"], expanded=beginner_mode):
        st.write(t["sample_tip"] if use_sample else t["live_tip"])

    st.header(t["ai_settings"])
    ai_provider_labels = {
        "deepseek": t["ai_provider_deepseek"],
        "custom": t["ai_provider_custom"],
    }
    ai_provider_key = st.selectbox(
        t["ai_provider"],
        list(AI_PROVIDER_DEFAULTS),
        key="ai_provider_key",
        format_func=lambda key: ai_provider_labels[key],
        on_change=sync_ai_provider_defaults,
    )
    ai_sdk_type = st.selectbox(t["ai_sdk"], ["OpenAI-compatible"], key="ai_sdk_type")
    if ai_provider_key == "deepseek":
        ai_provider_name = "DeepSeek"
        ai_base_url = st.text_input(t["ai_base_url"], key="ai_base_url")
        ai_model = st.text_input(t["ai_model"], key="ai_model")
        st.caption(t["ai_deepseek_note"])
    else:
        ai_provider_name = st.text_input(t["ai_custom_name"], key="ai_provider_name")
        ai_base_url = st.text_input(t["ai_base_url"], key="ai_base_url", help=t["ai_custom_help"])
        ai_model = st.text_input(t["ai_model"], key="ai_model")
    ai_api_key = st.text_input(t["ai_api_key"], type="password", key="ai_api_key")
    remember_api_key = st.toggle("记住 API Key / Remember API Key", key="remember_api_key")
    with st.expander(t["ai_advanced"], expanded=False):
        ai_temperature = st.slider(t["ai_temperature"], min_value=0.0, max_value=2.0, step=0.1, key="ai_temperature")
        ai_max_tokens = st.number_input(t["ai_max_tokens"], min_value=400, max_value=100000, step=1000, key="ai_max_tokens")
        ai_system_prompt_key = "ai_system_prompt_en" if language == "English" else "ai_system_prompt_cn"
        ai_system_prompt = st.text_area(
            t["ai_system_prompt"],
            key=ai_system_prompt_key,
            height=220,
            help=t["ai_system_prompt_help"],
        )
    st.caption(t["ai_config_tip"])

save_persisted_settings()

try:
    with st.spinner(t["loading"]):
        bundle = load_prices(tuple(tickers), str(start), str(end), use_sample)
        prices = bundle.prices
        returns = daily_returns(prices)
        expected_returns, covariance = annualized_mean_cov(returns)

        view_table = pd.DataFrame({"Ticker": expected_returns.index, "Annual View": expected_returns.values})
        views: dict[str, float] = {}
        confidence = 0.6
        if use_views:
            st.subheader(t["bl_views"])
            left, right = st.columns([2, 1])
            with left:
                edited = st.data_editor(
                    view_table,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
                        "Annual View": st.column_config.NumberColumn(t["annual_view"], format="%.3f", min_value=-0.5, max_value=0.8, step=0.01),
                    },
                )
            with right:
                confidence = st.slider(t["view_confidence"], min_value=0.1, max_value=1.0, value=0.6, step=0.05)
                st.metric(t["prior_assets"], len(expected_returns))
            views = dict(zip(edited["Ticker"], edited["Annual View"]))
            expected_returns = apply_absolute_views(expected_returns, covariance, views, confidence=confidence)

        max_sharpe = optimize_max_sharpe(expected_returns, covariance, risk_free, max_weight)
        min_var = optimize_min_variance(expected_returns, covariance, risk_free, max_weight)
        frontier = efficient_frontier(expected_returns, covariance, risk_free, max_weight, points=70)
        randoms = random_portfolios(expected_returns, covariance, risk_free, n=random_count)
        asset_stats = pd.DataFrame(
            {
                "Return": expected_returns,
                "Volatility": returns.std() * (252 ** 0.5),
            }
        )

    top = st.columns(4)
    top[0].metric(t["source"], translate_source(bundle.source, t))
    top[1].metric(t["assets"], len(prices.columns))
    top[2].metric(t["max_sharpe"], f"{max_sharpe.sharpe:.2f}", delta=pct(max_sharpe.expected_return))
    top[3].metric(t["min_volatility"], pct(min_var.volatility), delta=pct(min_var.expected_return))

    with st.expander(t["quick_summary"], expanded=False):
        summary_left, summary_right = st.columns(2)
        summary_left.write(f"**{t['date_range']}**: {start} - {end}")
        summary_left.write(f"**{t['asset_list']}**: {', '.join(prices.columns)}")
        summary_right.write(f"**{t['risk_free']}**: {pct(risk_free)}")
        summary_right.write(f"**{t['max_weight']}**: {pct(max_weight)}")

    def adjust_views(mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
        return apply_absolute_views(mu, cov, views, confidence=confidence) if use_views else mu

    st.subheader(t["ai_tab"])
    with st.container(border=True):
        st.caption(t["ai_intro"])
        st.info(t["ai_not_advice"])
        st.caption(t["ai_section_help"])
        ai_config = AIProviderConfig(
            provider_name=ai_provider_name,
            sdk_type=ai_sdk_type,
            base_url=ai_base_url,
            api_key=ai_api_key,
            model=ai_model,
            temperature=ai_temperature,
            max_tokens=int(ai_max_tokens),
        )
        if st.button(t["ai_generate"], type="primary"):
            if not ai_api_key:
                st.warning(t["ai_need_key"])
            else:
                try:
                    with st.spinner(t["ai_generating"]):
                        ai_bt = rolling_backtest(
                            returns,
                            strategy=strategy,
                            risk_free_rate=risk_free,
                            lookback_days=lookback_days,
                            rebalance_frequency=rebalance_frequency,
                            max_weight=max_weight,
                            benchmark_returns=None,
                            view_adjuster=adjust_views if use_views else None,
                        )
                        snapshot = build_ai_snapshot(
                            prices=prices,
                            expected_returns=expected_returns,
                            max_sharpe=max_sharpe,
                            min_var=min_var,
                            backtest_metrics=ai_bt.metrics,
                            risk_free=risk_free,
                            lookback_days=lookback_days,
                            rebalance_frequency=rebalance_labels[rebalance_frequency],
                        )
                        messages = build_portfolio_prompt(snapshot, language, system_prompt=ai_system_prompt)
                        st.session_state["last_ai_snapshot"] = snapshot
                        st.session_state["last_ai_analysis"] = generate_ai_analysis(ai_config, messages)
                except Exception as exc:
                    st.error(str(exc))

        if st.session_state.get("last_ai_analysis"):
            st.subheader(t["ai_result"])
            st.markdown(st.session_state["last_ai_analysis"])

        if st.session_state.get("last_ai_snapshot"):
            with st.expander(t["ai_snapshot"], expanded=False):
                st.json(st.session_state["last_ai_snapshot"])

    tabs = st.tabs([t["optimizer_tab"], t["backtest_tab"], t["prices_tab"], t["inputs_tab"]])

    with tabs[0]:
        st.caption(t["optimizer_hint"])
        st.plotly_chart(frontier_chart(frontier, randoms, asset_stats, max_sharpe, min_var, risk_free, t), width="stretch")
        w1, w2 = st.columns(2)
        with w1:
            st.subheader(t["max_sharpe_portfolio"])
            st.dataframe(
                max_sharpe.weights.sort_values(ascending=False).map(pct).to_frame(t["weight"]),
                width="stretch",
            )
            st.bar_chart(max_sharpe.weights.sort_values(ascending=False))
        with w2:
            st.subheader(t["min_variance_portfolio"])
            st.dataframe(
                min_var.weights.sort_values(ascending=False).map(pct).to_frame(t["weight"]),
                width="stretch",
            )
            st.bar_chart(min_var.weights.sort_values(ascending=False))

    with tabs[1]:
        st.caption(t["backtest_hint"])
        benchmark_returns = None
        if benchmark_ticker and not use_sample:
            try:
                benchmark_returns = load_benchmark(benchmark_ticker.strip(), str(start), str(end))
            except Exception as exc:
                st.warning(f"{t['benchmark_failed']}: {exc}")

        bt = rolling_backtest(
            returns,
            strategy=strategy,
            risk_free_rate=risk_free,
            lookback_days=lookback_days,
            rebalance_frequency=rebalance_frequency,
            max_weight=max_weight,
            benchmark_returns=benchmark_returns,
            view_adjuster=adjust_views if use_views else None,
        )
        st.plotly_chart(styled_line_chart(bt.equity, t["growth_title"], t), width="stretch")
        st.dataframe(bt.metrics.map(pct), width="stretch")
        if not bt.weights.empty:
            weight_fig = px.area(bt.weights, title=t["rolling_weights"], template="plotly_white")
            weight_fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            st.plotly_chart(weight_fig, width="stretch")

    with tabs[2]:
        st.caption(t["guided_note"])
        normalized = prices / prices.iloc[0]
        st.plotly_chart(styled_line_chart(normalized, t["normalized_price"], t), width="stretch")
        st.dataframe(returns.tail(20).style.format("{:.2%}"), width="stretch")

    with tabs[3]:
        st.caption(t["inputs_hint"])
        inputs = pd.DataFrame({t["expected_return"]: expected_returns, t["volatility"]: returns.std() * (252 ** 0.5)})
        st.dataframe(inputs.map(pct), width="stretch")
        st.dataframe(covariance.style.format("{:.2%}"), width="stretch")

except Exception as exc:
    st.error(str(exc))
    st.info(t["fallback_info"])
