from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AIProviderConfig:
    provider_name: str
    sdk_type: str
    base_url: str
    api_key: str
    model: str
    temperature: float = 1.0
    max_tokens: int = 4000


def default_system_prompt(language: str) -> str:
    if language == "English":
        return (
            "You are a senior professional market analyst and portfolio strategist. "
            "Analyze the provided portfolio optimization, risk-return metrics, asset weights, and backtest data as comprehensively as possible. "
            "Discuss allocation logic, concentration risk, diversification, macro and market sensitivity, volatility, drawdown, Sharpe ratio quality, backtest robustness, hidden assumptions, and practical improvement paths. "
            "Use a rigorous, institutional research tone. Do not invent data that is not provided, do not promise returns, and clearly state that the analysis is for research support only and not investment advice."
        )
    return (
        "你是一名资深专业市场分析师和投资组合策略师。"
        "请基于用户提供的组合优化结果、风险收益指标、资产权重和回测数据，尽可能全面地进行专业分析。"
        "重点讨论资产配置逻辑、集中度风险、分散化效果、宏观与市场敏感性、波动率、回撤、夏普比率质量、回测稳健性、隐含假设和可执行的改进路径。"
        "请使用严谨的机构研究口吻。不要编造未提供的数据，不要承诺收益，并明确说明分析仅用于研究辅助，不构成投资建议。"
    )


def build_portfolio_prompt(
    snapshot: dict[str, Any],
    language: str,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    system = system_prompt or default_system_prompt(language)
    if language == "English":
        user = f"""
Please produce a comprehensive professional portfolio analysis using the following structure:
1. Executive summary
2. Asset allocation and weight interpretation
3. Risk-return profile and Sharpe ratio assessment
4. Efficient frontier and capital market line interpretation
5. Rolling backtest performance, robustness, and possible overfitting concerns
6. Portfolio concentration, diversification, and factor/market exposure risks
7. Scenario sensitivities and stress points to monitor
8. Practical improvement recommendations and next research steps

Data snapshot:
{snapshot}
"""
    else:
        user = f"""
请按照以下结构输出一份全面、专业的投资组合分析：
1. 核心结论摘要
2. 资产配置与权重解读
3. 风险收益特征与夏普比率评估
4. 有效前沿与资本市场线解读
5. 滚动回测表现、稳健性与潜在过拟合问题
6. 组合集中度、分散化效果与因子/市场暴露风险
7. 情景敏感性与需要监控的压力点
8. 可执行的改进建议与后续研究方向

数据快照：
{snapshot}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_ai_analysis(config: AIProviderConfig, messages: list[dict[str, str]]) -> str:
    if config.sdk_type != "OpenAI-compatible":
        raise ValueError("Only OpenAI-compatible SDK is supported in this version.")
    if not config.api_key:
        raise ValueError("API key is required.")
    if not config.base_url:
        raise ValueError("Base URL is required.")
    if not config.model:
        raise ValueError("Model name is required.")

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenAI SDK is not installed. Run setup_env.bat or pip install -r requirements.txt.") from exc

    client = OpenAI(api_key=config.api_key, base_url=config.base_url.rstrip("/"))
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stream=False,
    )
    return response.choices[0].message.content or ""
